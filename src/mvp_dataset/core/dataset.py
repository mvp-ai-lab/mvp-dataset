"""Chainable iterator dataset API for mvp-dataset."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace

from .context import RuntimeContext
from .iterator import DatasetIterator
from .resume import RESUME_STATE_VERSION, ResumeStateError, stable_fingerprint
from .stages import (
    _AssembleStage,
    _BatchStage,
    _MapStage,
    _SelectStage,
    _ShuffleStage,
    _UnbatchStage,
)
from .torch_compat import TorchIterableDataset
from .types import Assembler, Consumer, SourceKind, StageSpec


@dataclass(frozen=True, slots=True)
class Dataset(TorchIterableDataset):
    """Chainable iterable dataset built from local shard sources.

    A :class:`Dataset` is immutable: every transformation returns a new dataset
    instance while leaving the previous one unchanged. Source data is loaded
    lazily during iteration, then passed through the appended iterator stages in
    declaration order.
    """

    context: RuntimeContext

    _source_kind: SourceKind
    _source: object
    _stages: tuple[StageSpec, ...]
    _resample: bool
    _resume_state: dict[str, object] | None = None

    def _build_source_stream(self, *, context: RuntimeContext) -> Iterable[object]:
        """Build the source iterator for a runtime context."""
        msg = f"[UnsupportedSourceKind] source kind {self._source_kind!r} does not implement iteration"
        raise NotImplementedError(msg)

    def _append_stage(self, spec: StageSpec) -> Dataset:
        """Return a new dataset with one additional stage."""
        return dataclass_replace(self, _stages=self._stages + (spec,), _resume_state=None)

    def _source_fingerprint(self) -> dict[str, object]:
        """Return the source portion of the pipeline fingerprint."""
        msg = f"[UnsupportedSourceKind] source kind {self._source_kind!r} does not implement fingerprinting"
        raise NotImplementedError(msg)

    def _stages_fingerprint(self) -> list[dict[str, object]]:
        """Return the stage portion of the pipeline fingerprint."""
        result: list[dict[str, object]] = []
        for spec in self._stages:
            fingerprint = getattr(spec.apply, "fingerprint", None)
            if callable(fingerprint):
                result.append({"kind": spec.kind, "fingerprint": fingerprint()})
            else:
                result.append(
                    {
                        "kind": spec.kind,
                        "apply_class": f"{spec.apply.__class__.__module__}.{spec.apply.__class__.__qualname__}",
                        "apply_config": repr(spec.apply),
                    }
                )
        return result

    def _pipeline_fingerprint(self) -> str:
        """Return the combined source and stage fingerprint."""
        payload = {
            "source": self._source_fingerprint(),
            "stages": self._stages_fingerprint(),
        }
        return stable_fingerprint(payload)

    def state_dict(self) -> dict[str, object]:
        """Return the resumable state for future pipeline outputs.

        Returns:
            A dictionary that can be passed to load_state_dict()."""
        warnings.warn(
            "Dataset.state_dict() creates a fresh initial iterator state. "
            "Use iterator.state_dict() to checkpoint an in-progress iteration.",
            UserWarning,
            stacklevel=2,
        )
        return DatasetIterator(self).state_dict()

    def load_state_dict(self, state: dict[str, object]) -> Dataset:
        """Return a dataset with validated resume state attached.

        Args:
            state: Resume state dictionary to validate and load.

        Returns:
            None."""

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
            fn: Callable applied to each upstream sample.

        Returns:
            A new dataset with the map stage appended."""

        spec = StageSpec(kind="map", apply=_MapStage(fn))
        return self._append_stage(spec)

    def shuffle(self, buffer_size: int, initial: int | None = None) -> Dataset:
        """Append a deterministic sample-level shuffle stage.

        Args:
            buffer_size: Maximum number of items kept in the shuffle buffer.
            initial: Minimum buffered item count before shuffle starts yielding.

        Returns:
            A new object with the shuffle stage appended."""

        spec = StageSpec(
            kind="shuffle",
            apply=_ShuffleStage(context=self.context, buffer_size=buffer_size, initial=initial),
        )
        return self._append_stage(spec)

    def select(self, fields: list[str] | tuple[str, ...]) -> Dataset:
        """Append a lazy field projection stage.

        Args:
            fields: Field names to keep in each dictionary sample.

        Returns:
            A new dataset with the select stage appended."""

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
            batch_size: Number of samples to group into each batch.
            drop_last: Whether to discard the final incomplete batch.
            collate_fn: Optional callable used to convert a list of samples into one batch.

        Returns:
            A new object with the batch stage appended."""

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
            factory: Callable that builds a fresh assembler for the runtime context.
            drop_last: Whether to discard the final incomplete batch.

        Returns:
            A new object with the assemble stage appended."""

        spec = StageSpec(
            kind="assemble",
            apply=_AssembleStage(factory=factory, context=self.context, drop_last=drop_last),
        )
        return self._append_stage(spec)

    def unbatch(self) -> Dataset:
        """Append an unbatching stage.

        Returns:
            A new object with the unbatch stage appended."""

        spec = StageSpec(
            kind="unbatch",
            apply=_UnbatchStage(),
        )
        return self._append_stage(spec)

    def consume(self, factory: Callable[[RuntimeContext], Consumer]) -> object:
        """Consume this pipeline eagerly and return a user-defined result.

        Args:
            factory: Callable that builds a consumer for the resolved runtime context.

        Returns:
            The result returned by ``consumer.finish()`` after the stream ends or
            ``consumer.push(item)`` returns False."""

        context = RuntimeContext.from_runtime(base=self.context)
        consumer = factory(context)
        for item in DatasetIterator(self, context=context):
            if consumer.push(item) is False:
                break
        return consumer.finish()

    def __iter__(self) -> Iterator[object]:
        """Materialize and run the full lazy pipeline."""
        return DatasetIterator(self)

    @classmethod
    def from_source(cls, source_kind: SourceKind, *args, **kwargs) -> Dataset:
        """Construct a dataset from a supported source type.

        Args:
            source_kind: Source backend name.
            args: Positional arguments forwarded to the source constructor.
            kwargs: Keyword arguments forwarded to the source constructor.

        Returns:
            A dataset configured for the requested source."""
        if source_kind == "tar":
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
