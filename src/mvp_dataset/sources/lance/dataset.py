"""Lance dataset source configuration."""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace

from mvp_dataset.core.context import RuntimeContext
from mvp_dataset.core.dataset import Dataset
from mvp_dataset.core.resume import stable_fingerprint
from mvp_dataset.core.stages import _AssembleStage
from mvp_dataset.core.subset import split_offsets
from mvp_dataset.core.types import ShardInput, StageSpec

from .config import resolve_lance_source_config
from .iterator import _LanceSourceIterator
from .order import ChunkShuffleConfig, ChunkShuffleInput, resolve_chunk_shuffle_config
from .refs import (
    LanceResolveRefFactory,
    attach_lance_ref_columns,
    resolve_ref_index_config,
    validate_ref_names,
)
from .source import list_lance_sources
from .types import (
    LanceRefIndexConfigInput,
    LanceRefResolverConfig,
    LanceSelection,
    LanceShuffleMode,
)


@dataclass(frozen=True, slots=True)
class LanceDataset(Dataset):
    """Dataset configuration for Lance datasets."""

    _shuffle_mode: LanceShuffleMode = "none"
    _columns: tuple[str, ...] | None = None
    _read_batch_size: int = 1024
    _chunk_shuffle: ChunkShuffleConfig | None = None
    _selection: LanceSelection | None = None

    @classmethod
    def from_source(
        cls,
        shards: ShardInput | Sequence[ShardInput],
        context: RuntimeContext | None = None,
        resample: bool = False,
        columns: Sequence[str] | None = None,
        read_batch_size: int = 1024,
        shuffle_mode: LanceShuffleMode = "none",
        chunk_shuffle: ChunkShuffleInput = None,
        ref_columns: dict[str, dict[str, str]] | None = None,
    ):
        """Build a dataset from local Lance dataset paths.

        Args:
            shards: Input shard path or paths.
            context: Runtime context used for sharding and deterministic randomness.
            resample: Whether to repeat the source indefinitely across rounds.
            columns: Column names to read from the source.
            read_batch_size: Number of row indexes aggregated into one Lance read call.
            shuffle_mode: Source-level shuffle mode.
            chunk_shuffle: Optional chunk shuffle configuration used when ``shuffle_mode="chunk"``.
                Supported keys are:
                ``chunk_size``: positive integer number of rows in each chunk. Defaults to ``250000``.
                ``k``: positive integer number of chunks in each local shuffle window. Defaults to ``8``.
                ``row_order``: ``"permuted"`` to shuffle rows inside chunks, or ``"sequential"`` to keep rows
                sequential inside each shuffled chunk. Defaults to ``"permuted"``.
            ref_columns: Optional Lance reference column configuration. It maps output column names to dicts with:
                ``uri``: reference Lance dataset URI, or a list of URIs.
                ``key_column``: column in the reference dataset containing lookup keys.
                ``value_column``: column in the reference dataset containing resolved values.

        Returns:
            A dataset configured for the requested source."""
        if read_batch_size <= 0:
            msg = "[InvalidLanceReadBatchSize] read_batch_size must be a positive integer"
            raise ValueError(msg)
        if shuffle_mode not in ("none", "global", "chunk"):
            msg = f"[InvalidLanceShuffleMode] expected none, global, or chunk, got {shuffle_mode!r}"
            raise ValueError(msg)
        if chunk_shuffle is not None and shuffle_mode != "chunk":
            msg = "[InvalidLanceChunkShuffle] chunk_shuffle requires shuffle_mode='chunk'"
            raise ValueError(msg)
        resolved_chunk_shuffle = resolve_chunk_shuffle_config(chunk_shuffle) if shuffle_mode == "chunk" else None
        runtime_context = RuntimeContext.from_runtime() if context is None else context
        normalized_shards, resolved_ref_columns = resolve_lance_source_config(shards, ref_columns)
        # 1. Merge all shards into a single source spec, validating that they can be read together.
        sources = list_lance_sources(
            normalized_shards,
        )
        # 2. Attach reference column configs to the source spec, if any.
        source = attach_lance_ref_columns(sources[0], resolved_ref_columns)
        sources = [source]

        return cls(
            context=runtime_context,
            _source=sources,
            _resample=resample,
            _source_kind="lance",
            _stages=(),
            _shuffle_mode=shuffle_mode,
            _columns=tuple(columns) if columns is not None else None,
            _read_batch_size=read_batch_size,
            _chunk_shuffle=resolved_chunk_shuffle,
        )

    def _build_source_stream(self, *, context: RuntimeContext) -> Iterable[object]:
        """Build the source iterator for a runtime context."""
        if len(self._source) != 1:
            msg = "[InvalidLanceSource] exactly one merged Lance source is required"
            raise RuntimeError(msg)
        return _LanceSourceIterator(
            source=self._source[0],
            context=context,
            resample=self._resample,
            columns=self._columns,
            read_batch_size=self._read_batch_size,
            source_fingerprint=stable_fingerprint(self._source_fingerprint()),
            shuffle_mode=self._shuffle_mode,
            chunk_config=self._chunk_shuffle,
            selection=self._selection,
        )

    def _source_fingerprint(self) -> dict[str, object]:
        """Return the source portion of the pipeline fingerprint."""
        source = self._source[0]
        return {
            "kind": "lance",
            "resample": self._resample,
            "shuffle_mode": self._shuffle_mode,
            "chunk_shuffle": self._chunk_shuffle_fingerprint(),
            "selection": self._selection_fingerprint(),
            "iter": {
                "columns": list(self._columns) if self._columns else None,
                "read_batch_size": self._read_batch_size,
            },
            "datasets": [
                {
                    "uri": dataset.uri,
                    "num_rows": dataset.num_rows,
                    "row_offset": dataset.row_offset,
                }
                for dataset in source.datasets
            ],
            "ref_columns": [
                {
                    "column": ref.column,
                    "uri": ref.uri,
                    "key_column": ref.key_column,
                    "value_column": ref.value_column,
                    "index_uri": ref.index_uri,
                    "index_offsets_path": ref.index_offsets_path,
                    "index_entries_path": ref.index_entries_path,
                }
                for ref in source.ref_columns
            ],
        }

    def _chunk_shuffle_fingerprint(self) -> dict[str, int] | None:
        """Return chunk shuffle fingerprint payload."""
        config = self._chunk_shuffle
        if config is None:
            return None
        return {
            "chunk_size": config.chunk_size,
            "k": config.k,
            "row_order": config.row_order,
        }

    def _selection_fingerprint(self) -> dict[str, object] | None:
        """Return the split/sample selection fingerprint payload."""
        selection = self._selection
        if selection is None:
            return None
        return {
            "seed": selection.seed,
            "start": selection.start,
            "count": selection.count,
            "total": selection.total,
        }

    def split(self, fractions: Sequence[float]) -> tuple[Dataset, ...]:
        """Partition the Lance dataset into disjoint row subsets covering all rows.

        Splitting is row-exact and efficient: each split reads only its own rows
        through ``take()``. Splits are contiguous source windows; source-level
        shuffle can randomize the physical membership.

        Args:
            fractions: Split weights, normalized internally.

        Returns:
            One dataset per fraction, in input order."""
        if self._selection is not None:
            msg = (
                "[UnsupportedNestedLanceSubset] apply split()/sample() on the base Lance "
                "dataset before other subset operations"
            )
            raise ValueError(msg)
        total = self._source[0].total_rows
        offsets = split_offsets(total, fractions)
        return tuple(
            dataclass_replace(
                self,
                _selection=LanceSelection(
                    start=offsets[index],
                    count=offsets[index + 1] - offsets[index],
                    total=total,
                ),
                _resume_state=None,
            )
            for index in range(len(offsets) - 1)
        )

    def sample(self, fraction: float, *, seed: int = 0) -> Dataset:
        """Return a dataset over a seeded random row subset of the Lance dataset.

        Sampling is row-exact and efficient: only the sampled rows are read. It is
        without replacement and cannot oversample (``0 < fraction <= 1``).

        Args:
            fraction: Fraction of rows to keep, in ``(0, 1]``.
            seed: Seed controlling which rows are kept.

        Returns:
            A new dataset reading only the sampled rows."""
        if self._selection is not None:
            msg = (
                "[UnsupportedNestedLanceSubset] apply split()/sample() on the base Lance "
                "dataset before other subset operations"
            )
            raise ValueError(msg)
        fraction = float(fraction)
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            msg = f"[InvalidSampleFraction] fraction must be in (0, 1], got={fraction!r}"
            raise ValueError(msg)
        total = self._source[0].total_rows
        count = round(fraction * total)
        selection = LanceSelection(start=0, count=count, total=total, seed=seed)
        return dataclass_replace(self, _selection=selection, _resume_state=None)

    def resolve_ref(
        self,
        ref_names: Sequence[str],
        *,
        resolve_batch_size: int = 1024,
        context: RuntimeContext | None = None,
        index: LanceRefIndexConfigInput = None,
    ) -> Dataset:
        """Append a lazy stage that resolves configured Lance reference columns.

        Args:
            ref_names: Reference column names to resolve.
            resolve_batch_size: Number of source samples to collect before resolving reference values.
            context: Runtime context used for sharding and deterministic randomness.
            index: Optional reference index configuration. Supported keys are:
                ``scope``: where workers coordinate index building. Use ``"shared"`` for one
                builder across all ranks, ``"node_local"`` for one builder per node, or
                ``"process"`` for each process to build independently. Defaults to ``"shared"``.
                ``build_strategy``: missing-index build strategy. Use ``"auto"``, ``"in_memory"``,
                or ``"bucketed"``. Defaults to ``"auto"``, which uses ``"in_memory"`` for small
                sources and ``"bucketed"`` for large sources.
                ``bucket_count``: positive integer number of hash buckets used by ``"bucketed"``
                builds. Defaults to ``4096``.
                Index cache files are stored under the main Lance dataset by default. Set
                ``MVP_DATASET_LANCE_REF_INDEX_CACHE_DIR`` to use another cache directory.

        Returns:
            A dataset that resolves the requested Lance reference columns."""
        if resolve_batch_size <= 0:
            msg = "[InvalidLanceRefResolveBatchSize] resolve_batch_size must be a positive integer"
            raise ValueError(msg)

        ref_names = validate_ref_names(self._source[0], ref_names)
        source = self._source[0]
        ref_index = resolve_ref_index_config(index)

        factory = LanceResolveRefFactory(
            source=source,
            ref_names=ref_names,
            config=LanceRefResolverConfig(
                resolve_batch_size=resolve_batch_size,
                index=ref_index,
            ),
        )
        stage_context = self.context if context is None else context
        spec = StageSpec(
            kind="assemble",
            apply=_AssembleStage(factory=factory, context=stage_context),
        )
        return self._append_stage(spec)
