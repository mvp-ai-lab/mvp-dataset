"""Chainable iterator dataset API for mvp-dataset."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace

from ..utils.sharding import assign_items
from .context import RuntimeContext
from .resume import (
    RESUME_STATE_VERSION,
    ResumeStateError,
    StatefulIterator,
    StatefulStage,
    UnsupportedResume,
    stable_fingerprint,
)
from .stages import (
    _AssembleStage,
    _BatchStage,
    _MapStage,
    _SelectStage,
    _ShuffleStage,
    _UnbatchStage,
    torch_iterabledataset_class,
)
from .types import Assembler, SourceKind, SourceStore, StageSpec


@dataclass(frozen=True, slots=True)
class Dataset(torch_iterabledataset_class()):
    """Chainable iterable dataset built from local shard sources.

    A :class:`Dataset` is immutable: every transformation returns a new dataset
    instance while leaving the previous one unchanged. Source data is loaded
    lazily during iteration, then passed through the appended iterator stages in
    declaration order.
    """

    context: RuntimeContext

    _source_kind: SourceKind
    _source: SourceStore
    _stages: tuple[StageSpec, ...]
    _resample: bool
    _iter_source_stream: Callable
    _resume_state: dict[str, object] | None = None

    def _build_source_stream(self, *, context: RuntimeContext) -> Iterable[object]:
        source_shard_stream = assign_items(self._source, context=context, resample=self._resample)
        return self._iter_source_stream(source_shard_stream)

    def _append_stage(self, spec: StageSpec) -> Dataset:
        return dataclass_replace(self, _stages=self._stages + (spec,), _resume_state=None)

    def _source_fingerprint(self) -> dict[str, object]:
        return {
            "kind": self._source_kind,
            "resample": self._resample,
            "iter_class": (
                f"{self._iter_source_stream.__class__.__module__}.{self._iter_source_stream.__class__.__qualname__}"
            ),
            "iter_config": repr(self._iter_source_stream),
            "store": repr(self._source),
        }

    def _stages_fingerprint(self) -> list[dict[str, object]]:
        return [
            {
                "kind": spec.kind,
                "apply_class": f"{spec.apply.__class__.__module__}.{spec.apply.__class__.__qualname__}",
                "apply_config": repr(spec.apply),
            }
            for spec in self._stages
        ]

    def _pipeline_fingerprint(self) -> str:
        payload = {
            "source": self._source_fingerprint(),
            "stages": self._stages_fingerprint(),
        }
        return stable_fingerprint(payload)

    def state_dict(self) -> dict[str, object]:
        """Return the resumable state for future pipeline outputs."""

        stage_states: list[dict[str, object]] = []
        for index, spec in enumerate(self._stages):
            if not isinstance(spec.apply, StatefulStage):
                msg = f"[UnsupportedResume] stage kind={spec.kind!r} index={index}"
                raise UnsupportedResume(msg)
            stage_states.append(
                {
                    "kind": spec.kind,
                    "fingerprint": spec.apply.fingerprint(),
                    "state": spec.apply.state_dict(),
                }
            )

        source_iter = self._iter_source_stream
        if not isinstance(source_iter, StatefulIterator):
            msg = f"[UnsupportedResume] source kind={self._source_kind!r}"
            raise UnsupportedResume(msg)

        return {
            "version": RESUME_STATE_VERSION,
            "runtime_fingerprint": self.context.fingerprint(),
            "pipeline_fingerprint": self._pipeline_fingerprint(),
            "output_step": 0,
            "source": {
                "kind": self._source_kind,
                "fingerprint": source_iter.fingerprint(),
                "state": source_iter.state_dict(),
            },
            "stages": stage_states,
        }

    def load_state_dict(self, state: dict[str, object]) -> Dataset:
        """Return a dataset with validated resume state attached."""

        if not isinstance(state, dict):
            msg = "[InvalidResumeState] state must be a dict"
            raise ResumeStateError(msg)
        version = state.get("version")
        if version != RESUME_STATE_VERSION:
            msg = f"[InvalidResumeStateVersion] expected={RESUME_STATE_VERSION} got={version!r}"
            raise ResumeStateError(msg)

        runtime_fingerprint = state.get("runtime_fingerprint")
        if not isinstance(runtime_fingerprint, str):
            msg = "[InvalidResumeState] runtime_fingerprint must be a string"
            raise ResumeStateError(msg)
        if runtime_fingerprint != self.context.fingerprint():
            msg = "[ResumeRuntimeMismatch] runtime fingerprint does not match"
            raise ResumeStateError(msg)

        pipeline_fingerprint = state.get("pipeline_fingerprint")
        if not isinstance(pipeline_fingerprint, str):
            msg = "[InvalidResumeState] pipeline_fingerprint must be a string"
            raise ResumeStateError(msg)
        if pipeline_fingerprint != self._pipeline_fingerprint():
            msg = "[ResumePipelineMismatch] pipeline fingerprint does not match"
            raise ResumeStateError(msg)

        source = state.get("source")
        if not isinstance(source, dict):
            msg = "[InvalidResumeState] source must be a dict"
            raise ResumeStateError(msg)
        stages = state.get("stages")
        if not isinstance(stages, list):
            msg = "[InvalidResumeState] stages must be a list"
            raise ResumeStateError(msg)

        return dataclass_replace(self, _resume_state=state)

    def map(self, fn: Callable[[object], object]) -> Dataset:
        """Append a lazy map stage.

        Args:
            fn: Callable applied to each sample yielded by the upstream stage.

        Returns:
            A new dataset that applies ``fn`` lazily during iteration.
        """

        spec = StageSpec(kind="map", apply=_MapStage(fn))
        return self._append_stage(spec)

    def shuffle(self, buffer_size: int, initial: int | None = None) -> Dataset:
        """Append a deterministic sample-level shuffle stage.

        Args:
            buffer_size: Maximum number of samples to keep in the randomization
                buffer.
            initial: Minimum number of buffered samples before the stage starts
                yielding values. Defaults to ``buffer_size``.

        Returns:
            A new dataset with bounded-memory shuffling applied lazily.
        """

        spec = StageSpec(
            kind="shuffle",
            apply=_ShuffleStage(context=self.context, buffer_size=buffer_size, initial=initial),
        )
        return self._append_stage(spec)

    def select(self, fields: list[str] | tuple[str, ...]) -> Dataset:
        """Append a lazy field projection stage.

        The stage keeps requested user fields and preserves internal metadata
        keys such as ``__key__``.
        """

        selected_fields = tuple(fields)

        spec = StageSpec(
            kind="select",
            apply=_SelectStage(selected_fields),
        )
        return self._append_stage(spec)

    def batch(
        self,
        batch_size: int,
        drop_last: bool = False,
        collate_fn: Callable[[list[object]], object] | None = None,
    ) -> Dataset:
        """Append a batching stage.

        Args:
            batch_size: Number of samples per yielded batch.
            drop_last: Whether to drop the final incomplete batch.
            collate_fn: Optional callable that transforms each list of samples
                into a user-defined batch object.

        Returns:
            A new dataset that yields batches instead of individual samples.
        """

        spec = StageSpec(
            kind="batch",
            apply=_BatchStage(
                batch_size=batch_size,
                drop_last=drop_last,
                collate_fn=collate_fn,
            ),
        )
        return self._append_stage(spec)

    def assemble(
        self,
        factory: Callable[[RuntimeContext], Assembler[object, object]],
        *,
        drop_last: bool = False,
    ) -> Dataset:
        """Append a stateful assembly stage.

        Args:
            factory: Callable that builds one fresh assembler for each iterator
                execution. The assembler may consume multiple upstream samples
                before yielding one or more downstream outputs.
            drop_last: Whether to discard unfinished tail state instead of
                delegating it to the assembler's final flush.

        Returns:
            A new dataset that assembles the upstream sample stream lazily.
        """

        spec = StageSpec(
            kind="assemble",
            apply=_AssembleStage(factory=factory, context=self.context, drop_last=drop_last),
        )
        return self._append_stage(spec)

    def unbatch(self) -> Dataset:
        """Append an unbatching stage.

        Returns:
            A new dataset that expands list, tuple, or dict-style batches back
            into individual samples during iteration.
        """

        spec = StageSpec(
            kind="unbatch",
            apply=_UnbatchStage(),
        )
        return self._append_stage(spec)

    def __iter__(self) -> Iterator[object]:
        """Materialize and run the full lazy pipeline."""
        context = RuntimeContext.from_runtime(base=self.context)
        stream: Iterable[object] = self._build_source_stream(context=context)
        for spec in self._stages:
            stream = spec.apply(stream)
        yield from stream

    @classmethod
    def from_source(cls, source_kind: SourceKind, *args, **kwargs) -> Dataset:
        """Construct a dataset from a supported source type.

        See the relevant source-specific classmethod constructors for details.
        """
        if source_kind == "tars":
            from ..sources.tar.dataset import TarDataset

            return TarDataset.from_source(*args, **kwargs)
        if source_kind == "jsonl":
            from ..sources.jsonl.dataset import JsonlDataset

            return JsonlDataset.from_source(*args, **kwargs)
        if source_kind == "parquet":
            from ..sources.parquet.dataset import ParquetDataset

            return ParquetDataset.from_source(*args, **kwargs)
        if source_kind == "lance":
            from ..sources.lance.dataset import LanceDataset

            return LanceDataset.from_source(*args, **kwargs)
        msg = f"[UnsupportedSourceKind] source_kind={source_kind!r}"
        raise ValueError(msg)
