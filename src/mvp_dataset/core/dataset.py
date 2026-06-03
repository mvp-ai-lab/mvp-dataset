"""Chainable iterator dataset API for mvp-dataset."""

from __future__ import annotations

import warnings
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


class DatasetIterator:
    """Materialized iterator for one Dataset pipeline execution."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.context = RuntimeContext.from_runtime(base=dataset.context)
        self.num_yielded = 0

        source = dataset._build_source_stream(context=self.context)
        if not isinstance(source, StatefulIterator):
            msg = f"[UnsupportedResume] source kind={dataset._source_kind!r}"
            raise UnsupportedResume(msg)
        self.source = source

        stage_resume_states = self._load_resume_state(dataset._resume_state)

        stream: Iterable[object] = self.source
        self.stages: list[object] = []
        for spec in dataset._stages:
            stream = spec.apply(stream)
            stage = stream if isinstance(stream, StatefulStage) else spec.apply
            self.stages.append(stage)
        if stage_resume_states is not None:
            self._load_stage_resume_state(stage_resume_states)
        self.stream = iter(stream)

    def __iter__(self) -> DatasetIterator:
        return self

    def __next__(self) -> object:
        item = next(self.stream)
        self.num_yielded += 1
        return item

    def state_dict(self) -> dict[str, object]:
        stage_states: list[dict[str, object]] = []
        for index, (spec, stage) in enumerate(zip(self.dataset._stages, self.stages, strict=True)):
            if not isinstance(stage, StatefulStage):
                msg = f"[UnsupportedResume] stage kind={spec.kind!r} index={index}"
                raise UnsupportedResume(msg)
            stage_states.append(
                {
                    "kind": spec.kind,
                    "fingerprint": stage.fingerprint(),
                    "state": stage.state_dict(),
                }
            )

        return {
            "version": RESUME_STATE_VERSION,
            "runtime_fingerprint": self.dataset.context.fingerprint(),
            "pipeline_fingerprint": self.dataset._pipeline_fingerprint(),
            "num_yielded": self.num_yielded,
            "source": {
                "kind": self.dataset._source_kind,
                "fingerprint": self.source.fingerprint(),
                "state": self.source.state_dict(),
            },
            "stages": stage_states,
        }

    def _load_resume_state(self, state: dict[str, object] | None) -> list[object] | None:
        if state is None:
            return None

        num_yielded = state.get("num_yielded")
        if not isinstance(num_yielded, int) or num_yielded < 0:
            msg = "[InvalidResumeState] num_yielded must be a non-negative integer"
            raise ResumeStateError(msg)
        stages = state.get("stages")
        if not isinstance(stages, list):
            msg = "[InvalidResumeState] stages must be a list"
            raise ResumeStateError(msg)
        if len(stages) != len(self.dataset._stages):
            msg = "[ResumeStageMismatch] stage count does not match"
            raise ResumeStateError(msg)

        source_state = state.get("source")
        if not isinstance(source_state, dict):
            msg = "[InvalidResumeState] source must be a dict"
            raise ResumeStateError(msg)
        if source_state.get("fingerprint") != self.source.fingerprint():
            msg = "[ResumeSourceMismatch] source fingerprint does not match"
            raise ResumeStateError(msg)
        raw_source_state = source_state.get("state")
        if not isinstance(raw_source_state, dict):
            msg = "[InvalidResumeState] source.state must be a dict"
            raise ResumeStateError(msg)
        self.source.load_state_dict(raw_source_state)
        self.num_yielded = num_yielded
        return stages

    def _load_stage_resume_state(self, stages: list[object]) -> None:
        for index, (spec, stage, stage_state) in enumerate(zip(self.dataset._stages, self.stages, stages, strict=True)):
            if not isinstance(stage_state, dict):
                msg = "[InvalidResumeState] stage must be a dict"
                raise ResumeStateError(msg)
            if stage_state.get("kind") != spec.kind:
                msg = f"[ResumeStageMismatch] stage kind does not match index={index}"
                raise ResumeStateError(msg)
            if not isinstance(stage, StatefulStage):
                msg = f"[UnsupportedResume] stage kind={spec.kind!r} index={index}"
                raise UnsupportedResume(msg)
            if stage_state.get("fingerprint") != stage.fingerprint():
                msg = f"[ResumeStageMismatch] stage fingerprint does not match index={index}"
                raise ResumeStateError(msg)
            raw_stage_state = stage_state.get("state")
            if not isinstance(raw_stage_state, dict):
                msg = "[InvalidResumeState] stage.state must be a dict"
                raise ResumeStateError(msg)
            stage.load_state_dict(raw_stage_state)


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
    # TODO: rename it as factory?
    _iter_source_stream: Callable | None
    _resume_state: dict[str, object] | None = None

    def _build_source_stream(self, *, context: RuntimeContext) -> Iterable[object]:
        assert self._iter_source_stream is not None, "source stream factory is required"
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
        warnings.warn(
            "Dataset.state_dict() creates a fresh initial iterator state. "
            "Use iterator.state_dict() to checkpoint an in-progress iteration.",
            UserWarning,
            stacklevel=2,
        )
        return DatasetIterator(self).state_dict()

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

        warnings.warn(
            "Dataset.load_state_dict() stores pending resume state. "
            "The source cursor is restored when iter(dataset) creates a DatasetIterator.",
            UserWarning,
            stacklevel=2,
        )
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
        return DatasetIterator(self)

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
