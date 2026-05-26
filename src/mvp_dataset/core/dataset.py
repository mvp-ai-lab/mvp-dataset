"""Chainable iterator dataset API for mvp-dataset."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace

from ..utils.sharding import assign_items
from .context import RuntimeContext
from .resume import RESUME_TOKEN_KEY, make_resume_state, validate_resume_state
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
    _last_resume_token: object | None = None
    _resume_step: int = 0

    def _build_source_stream(self, *, context: RuntimeContext) -> Iterable[object]:
        source_shard_stream = assign_items(self._source, context=context, resample=self._resample)
        resume_cursor = self._resume_state.get("source_cursor") if self._resume_state is not None else None
        return self._iter_source_stream(source_shard_stream, resume_cursor=resume_cursor)

    def _append_stage(self, spec: StageSpec) -> Dataset:
        return dataclass_replace(self, _stages=self._stages + (spec,))

    def _resume_source_payload(self) -> dict[str, object]:
        return {
            "source": self._source,
            "resample": self._resample,
            "reader": self._iter_source_stream,
        }

    def _validate_resume_supported(self) -> None:
        return None

    def _track_source_resume(self, stream: Iterable[object]) -> Iterator[object]:
        for sample in stream:
            if isinstance(sample, dict) and RESUME_TOKEN_KEY in sample:
                object.__setattr__(self, "_last_resume_token", sample[RESUME_TOKEN_KEY])
            yield sample

    def state_dict(self) -> dict[str, object]:
        """Return a checkpointable state for the next dataset item."""

        self._validate_resume_supported()
        context = RuntimeContext.from_runtime(base=self.context)
        source_cursor = self._last_resume_token
        if source_cursor is None and self._resume_state is not None:
            source_cursor = self._resume_state.get("source_cursor")
        return make_resume_state(
            source_kind=self._source_kind,
            source=self._resume_source_payload(),
            stages=self._stages,
            context=context,
            source_cursor=source_cursor,
            step=self._resume_step,
        )

    def load_state_dict(self, state: dict[str, object]) -> Dataset:
        """Load a previously returned dataset resume state."""

        self._validate_resume_supported()
        context = RuntimeContext.from_runtime(base=self.context)
        validate_resume_state(
            state,
            source_kind=self._source_kind,
            source=self._resume_source_payload(),
            stages=self._stages,
            context=context,
        )
        object.__setattr__(self, "_resume_state", dict(state))
        object.__setattr__(self, "_last_resume_token", state.get("source_cursor"))
        object.__setattr__(self, "_resume_step", int(state.get("step", 0)))
        return self

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
        stream: Iterable[object] = self._track_source_resume(self._build_source_stream(context=context))
        for spec in self._stages:
            stream = spec.apply(stream)
        for sample in stream:
            object.__setattr__(self, "_resume_step", self._resume_step + 1)
            yield sample

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
