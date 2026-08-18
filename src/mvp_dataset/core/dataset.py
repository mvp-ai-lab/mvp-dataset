"""Chainable iterator dataset API for mvp-dataset."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace

from .context import RuntimeContext
from .iterator import DatasetIterator
from .resume import (
    check_identity,
    checkpoint_from_active_iter,
    parse_checkpoint,
    warn_if_iterator_replaced,
)
from .stages import (
    _AssembleStage,
    _BatchStage,
    _MapStage,
    _SelectStage,
    _ShuffleStage,
    _UnbatchStage,
)
from .torch_compat import TorchIterableDataset
from .types import Assembler, Consumer, FingerprintProvider, SourceKind, StageSpec


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
    _pending_state: object | None = None
    _active_iter: object | None = None

    def _build_source_stream(self, *, context: RuntimeContext) -> Iterable[object]:
        """Build the source iterator for a runtime context."""
        msg = f"[UnsupportedSourceKind] source kind {self._source_kind!r} does not implement iteration"
        raise NotImplementedError(msg)

    def _append_stage(self, spec: StageSpec) -> Dataset:
        """Return a new dataset with one additional stage."""
        return dataclass_replace(self, _stages=self._stages + (spec,), _pending_state=None, _active_iter=None)

    def _source_identity(self) -> dict[str, object]:
        """Return the source portion of the pipeline identity."""
        raise NotImplementedError(
            f"[UnsupportedSourceKind] source kind {self._source_kind!r} does not implement identity"
        )

    def identity(self) -> dict[str, object]:
        """Return the process-stable identity of this pipeline configuration."""
        return {
            "runtime": self.context.identity(),
            "source": self._source_identity(),
            "stages": [spec.apply.identity() for spec in self._stages],
            "loader": None,
        }

    def state_dict(self) -> dict[str, object]:
        """Return identity plus the active iterator's live state, if any."""
        return checkpoint_from_active_iter(self.identity(), self._active_iter)

    def load_state_dict(self, blob: dict[str, object]) -> None:
        """Validate identity and stage pending live state on this dataset."""
        expected_identity, state = parse_checkpoint(blob)
        check_identity(expected_identity, self.identity())
        object.__setattr__(self, "_pending_state", state)

    def load_live_state(self, state: object) -> None:
        """Stage inner live state without an identity check."""
        object.__setattr__(self, "_pending_state", state)

    def _peek_pending_state(self) -> object | None:
        """Return pending live state without consuming it."""
        return self._pending_state

    def _take_pending_state(self) -> object | None:
        """Return and clear pending live state."""
        pending = self._pending_state
        object.__setattr__(self, "_pending_state", None)
        return pending

    def map(self, fn: Callable[[object], object]) -> Dataset:
        """Append a lazy map stage.

        Args:
            fn: Callable applied to each upstream sample.

        Returns:
            A new dataset with the map stage appended."""

        spec = StageSpec(kind="map", apply=_MapStage(fn))
        return self._append_stage(spec)

    def filter(self, predicate: str | Sequence[str], *, index: dict[str, object] | None = None) -> Dataset:
        """Return a source-filtered dataset when supported by the backend.

        Args:
            predicate: Backend-native filter expression or batched expressions.
            index: Optional backend filter-index configuration.

        Returns:
            A new filtered dataset."""
        _ = predicate, index
        msg = f"[UnsupportedFilter] source kind={self._source_kind!r}"
        raise NotImplementedError(msg)

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

    def snapshot(self, fingerprint_provider: FingerprintProvider | None = None) -> Dataset:
        """Materialize this finite pipeline into a reusable Lance snapshot.

        The snapshot is built lazily during its first iteration. A cache hit
        reads the materialized Lance source without executing this upstream
        pipeline.

        Args:
            fingerprint_provider: Optional zero-argument callable returning a
                stable cache identity. The upstream pipeline fingerprint is
                used when omitted.

        Returns:
            A new dataset whose source is the materialized snapshot.
        """
        from ..sources.snapshot.dataset import SnapshotDataset

        return SnapshotDataset.from_upstream(self, fingerprint_provider=fingerprint_provider)

    def split(self, fractions: Sequence[float]) -> tuple[Dataset, ...]:
        """Partition this dataset into disjoint subsets covering all data.

        Each returned dataset reads only its own data.

        The default implementation treats each ``_source`` element as one
        equally weighted unit. Sources override this when they partition at a
        different granularity or do not support subsetting.

        Args:
            fractions: Split weights, normalized internally (``[0.8, 0.2]`` and
                ``[8, 2]`` are equivalent).

        Returns:
            One dataset per fraction, in the input order."""
        from .subset import split_units

        return split_units(self, [1.0] * len(self._source), fractions)

    def sample(self, fraction: float, *, seed: int = 0) -> Dataset:
        """Return a dataset over a seeded random subset of this dataset.

        Sampling is without replacement and cannot oversample
        (``0 < fraction <= 1``). It is reproducible for a given ``seed``.
        Resampled rounds preserve subset membership and may only change its order.

        The default implementation treats each ``_source`` element as one equally
        weighted unit; sources override this for other granularities or to opt out.

        Args:
            fraction: Fraction of the dataset to keep, in ``(0, 1]``.
            seed: Seed controlling which data is kept.

        Returns:
            A new dataset reading only the sampled data."""
        from .subset import sample_units

        return sample_units(self, [1.0] * len(self._source), fraction, seed)

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
        warn_if_iterator_replaced(self._active_iter)
        iterator = DatasetIterator(self)
        object.__setattr__(self, "_active_iter", iterator)
        return iterator

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
        if source_kind == "mixed":
            from ..sources.mixed.dataset import MixedDataset

            return MixedDataset.from_source(*args, **kwargs)
        msg = f"[UnsupportedSourceKind] source_kind={source_kind!r}"
        raise ValueError(msg)
