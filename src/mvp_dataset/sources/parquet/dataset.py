"""Parquet dataset source configuration."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from mvp_dataset.core.context import RuntimeContext
from mvp_dataset.core.dataset import Dataset
from mvp_dataset.core.resume import stable_fingerprint
from mvp_dataset.core.types import ShardInput
from mvp_dataset.utils.url import normalize_paths

from .chunks import list_parquet_chunks
from .iterator import _ParquetSourceIterator
from .reader import resolve_parquet_batch_size
from .types import ParquetShuffleMode


@dataclass(frozen=True, slots=True)
class ParquetDataset(Dataset):
    """Dataset configuration for Parquet shards."""

    _columns: tuple[str, ...] | None = None
    _use_threads: bool = True
    _shuffle_mode: ParquetShuffleMode = "chunk_aware"

    @classmethod
    def from_source(
        cls,
        shards: ShardInput | Sequence[ShardInput],
        context: RuntimeContext | None = None,
        resample: bool = False,
        min_row_groups_per_chunk: int = 1,
        columns: Sequence[str] | None = None,
        use_threads: bool = True,
        shuffle_mode: ParquetShuffleMode = "chunk_aware",
    ):
        """Build a dataset from local Parquet file paths.

        Args:
            shards: Input shard path or paths.
            context: Runtime context used for sharding and deterministic randomness.
            resample: Whether to repeat the source indefinitely across rounds.
            min_row_groups_per_chunk: Minimum row groups combined into one Parquet chunk.
            columns: Column names to read from the source.
            use_threads: Whether the reader may use threaded decoding.
            shuffle_mode: Source-level shuffle mode.

        Returns:
            A dataset configured for the requested source."""
        runtime_context = RuntimeContext.from_runtime() if context is None else context
        if shuffle_mode == "global":
            msg = "[UnsupportedParquetShuffleMode] shuffle_mode='global'"
            raise ValueError(msg)
        if shuffle_mode not in ("none", "chunk_aware"):
            msg = f"[InvalidParquetShuffleMode] expected none or chunk_aware, got={shuffle_mode!r}"
            raise ValueError(msg)
        resolve_parquet_batch_size()
        normalized_shards = normalize_paths(shards)
        if not all(path.endswith(".parquet") for path in normalized_shards):
            msg = f"[InvalidSourceType] expected .parquet inputs, got={normalized_shards!r}"
            raise ValueError(msg)
        chunks = list_parquet_chunks(
            normalized_shards,
            min_row_groups_per_chunk=min_row_groups_per_chunk,
            min_chunks=runtime_context.total_slots,
        )

        return cls(
            context=runtime_context,
            _source=chunks,
            _resample=resample,
            _source_kind="parquet",
            _stages=(),
            _columns=tuple(columns) if columns is not None else None,
            _use_threads=use_threads,
            _shuffle_mode=shuffle_mode,
        )

    def _build_source_stream(self, *, context: RuntimeContext) -> Iterable[object]:
        """Build the source iterator for a runtime context."""
        return _ParquetSourceIterator(
            chunks=self._source,
            context=context,
            resample=self._resample,
            columns=self._columns,
            use_threads=self._use_threads,
            source_fingerprint=stable_fingerprint(self._source_fingerprint()),
            shuffle_mode=self._shuffle_mode,
        )

    def _source_fingerprint(self) -> dict[str, object]:
        """Return the source portion of the pipeline fingerprint."""
        return {
            "kind": "parquet",
            "resample": self._resample,
            "shuffle_mode": self._shuffle_mode,
            "iter": {
                "columns": list(self._columns) if self._columns else None,
                "use_threads": self._use_threads,
            },
            "chunks": [
                {
                    "path": chunk.path,
                    "row_groups": list(chunk.row_groups),
                    "row_offset": chunk.row_offset,
                    "num_rows": chunk.num_rows,
                    "row_group_offsets": list(chunk.row_group_offsets),
                    "row_group_num_rows": list(chunk.row_group_num_rows),
                }
                for chunk in self._source
            ],
        }
