"""Lance dataset source configuration."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from mvp_dataset.core.context import RuntimeContext
from mvp_dataset.core.dataset import Dataset
from mvp_dataset.core.resume import stable_fingerprint
from mvp_dataset.core.stages import _AssembleStage
from mvp_dataset.core.types import ShardInput, StageSpec

from .config import resolve_lance_source_config
from .iterator import _LanceSourceIterator
from .reader import list_lance_sources
from .refs import LanceResolveRefFactory, attach_lance_ref_columns, validate_ref_names
from .types import LanceRefIndexScope, LanceShuffleMode


@dataclass(frozen=True, slots=True)
class LanceDataset(Dataset):
    """Dataset configuration for Lance datasets."""

    _shuffle_mode: LanceShuffleMode = "none"
    _columns: tuple[str, ...] | None = None
    _batch_size: int = 1024
    _ref_index_scope: LanceRefIndexScope | None = None

    @classmethod
    def from_source(
        cls,
        shards: ShardInput | Sequence[ShardInput],
        context: RuntimeContext | None = None,
        resample: bool = False,
        columns: Sequence[str] | None = None,
        batch_size: int = 1024,
        shuffle_mode: LanceShuffleMode = "none",
        ref_columns: dict[str, dict[str, str]] | None = None,
        ref_index_scope: LanceRefIndexScope | None = None,
    ):
        """Build a dataset from local Lance dataset paths.

        Args:
            shards: Input shard path or paths.
            context: Runtime context used for sharding and deterministic randomness.
            resample: Whether to repeat the source indefinitely across rounds.
            columns: Column names to read from the source.
            batch_size: Number of samples to group into each batch.
            shuffle_mode: Source-level shuffle mode.
            ref_columns: Lance reference column configuration.
            ref_index_scope: Scope that controls where Lance reference indexes are stored.

        Returns:
            A dataset configured for the requested source."""
        if batch_size <= 0:
            msg = "[InvalidLanceBatchSize] batch_size must be a positive integer"
            raise ValueError(msg)

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
            _batch_size=batch_size,
            _ref_index_scope=ref_index_scope,
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
            batch_size=self._batch_size,
            source_fingerprint=stable_fingerprint(self._source_fingerprint()),
            shuffle_mode=self._shuffle_mode,
        )

    def _source_fingerprint(self) -> dict[str, object]:
        """Return the source portion of the pipeline fingerprint."""
        source = self._source[0]
        return {
            "kind": "lance",
            "resample": self._resample,
            "shuffle_mode": self._shuffle_mode,
            "ref_index_scope": self._ref_index_scope,
            "iter": {
                "columns": list(self._columns) if self._columns else None,
                "batch_size": self._batch_size,
            },
            "datasets": [
                {
                    "uri": dataset.uri,
                    "num_rows": dataset.num_rows,
                    "row_offset": dataset.row_offset,
                    "fragment_ids": list(dataset.fragment_ids),
                    "fragment_row_counts": list(dataset.fragment_row_counts),
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

    def resolve_ref(
        self,
        ref_names: Sequence[str],
        *,
        batch_size: int = 1024,
        context: RuntimeContext | None = None,
        ref_index_scope: LanceRefIndexScope | None = None,
    ) -> Dataset:
        """Append a lazy stage that resolves configured Lance reference columns.

        Args:
            ref_names: Reference column names to resolve.
            batch_size: Number of samples to group into each batch.
            context: Runtime context used for sharding and deterministic randomness.
            ref_index_scope: Scope that controls where Lance reference indexes are stored.

        Returns:
            A dataset that resolves the requested Lance reference columns."""
        if batch_size <= 0:
            msg = "[InvalidLanceRefBatchSize] batch_size must be a positive integer"
            raise ValueError(msg)

        ref_names = validate_ref_names(self._source[0], ref_names)
        source = self._source[0]

        factory = LanceResolveRefFactory(
            source=source,
            ref_names=ref_names,
            batch_size=batch_size,
            ref_index_scope=ref_index_scope if ref_index_scope is not None else self._ref_index_scope,
        )
        stage_context = self.context if context is None else context
        spec = StageSpec(
            kind="assemble",
            apply=_AssembleStage(factory=factory, context=stage_context),
        )
        return self._append_stage(spec)
