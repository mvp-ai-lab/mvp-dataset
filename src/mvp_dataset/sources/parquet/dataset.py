from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from mvp_dataset.core.context import RuntimeContext
from mvp_dataset.core.dataset import Dataset
from mvp_dataset.core.resume import stable_fingerprint
from mvp_dataset.core.types import ShardInput
from mvp_dataset.utils.url import normalize_paths

from .fragments import list_parquet_fragments
from .iterator import _ParquetSourceIterator
from .reader import resolve_parquet_batch_size
from .types import ParquetShuffleMode


@dataclass(frozen=True, slots=True)
class ParquetDataset(Dataset):
    _columns: tuple[str, ...] | None = None
    _use_threads: bool = True
    _shuffle_mode: ParquetShuffleMode = "fragment_aware"

    @classmethod
    def from_source(
        cls,
        shards: ShardInput | Sequence[ShardInput],
        context: RuntimeContext | None = None,
        resample: bool = False,
        min_row_groups_per_fragment: int = 1,
        columns: Sequence[str] | None = None,
        use_threads: bool = True,
        shuffle_mode: ParquetShuffleMode = "fragment_aware",
    ):
        """Build a dataset from local Parquet file paths.

        Args:
            shards: One or more file paths, glob specs, or brace-expansion specs.
            context: Optional execution context. If omitted, inferred from runtime.
            resample: Whether to loop shards indefinitely across rounds.
            min_row_groups_per_fragment: Minimum number of consecutive parquet
                row groups to merge into one schedulable fragment. Defaults to
                ``1``, which creates one fragment per row group unless
                ``min_fragments`` fallback logic re-splits the source.
            columns: Optional list of column names to read.
            use_threads: Whether to use multi-threaded Arrow reads.
            shuffle_mode: ``"fragment_aware"`` shuffles fragments by round,
                ``"none"`` reads fragments in original order. ``"global"`` is
                not supported for Parquet row access.

        Arrow record batch size is controlled via the
        ``MVP_DATASET_PARQUET_BATCH_SIZE`` environment variable.

        Returns:
            A dataset whose source is the list of parquet fragments.

        Raises:
            ValueError: If any input path does not end with ``.parquet``.
        """
        runtime_context = RuntimeContext.from_runtime() if context is None else context
        if shuffle_mode == "global":
            msg = "[UnsupportedParquetShuffleMode] shuffle_mode='global'"
            raise ValueError(msg)
        if shuffle_mode not in ("none", "fragment_aware"):
            msg = f"[InvalidParquetShuffleMode] expected none or fragment_aware, got={shuffle_mode!r}"
            raise ValueError(msg)
        resolve_parquet_batch_size()
        normalized_shards = normalize_paths(shards)
        if not all(path.endswith(".parquet") for path in normalized_shards):
            msg = f"[InvalidSourceType] expected .parquet inputs, got={normalized_shards!r}"
            raise ValueError(msg)
        fragments = list_parquet_fragments(
            normalized_shards,
            min_row_groups_per_fragment=min_row_groups_per_fragment,
            min_fragments=runtime_context.total_slots,
        )

        return cls(
            context=runtime_context,
            _source=fragments,
            _resample=resample,
            _source_kind="parquet",
            _stages=(),
            _columns=tuple(columns) if columns is not None else None,
            _use_threads=use_threads,
            _shuffle_mode=shuffle_mode,
        )

    def _build_source_stream(self, *, context: RuntimeContext) -> Iterable[object]:
        return _ParquetSourceIterator(
            fragments=self._source,
            context=context,
            resample=self._resample,
            columns=self._columns,
            use_threads=self._use_threads,
            source_fingerprint=stable_fingerprint(self._source_fingerprint()),
            shuffle_mode=self._shuffle_mode,
        )

    def _source_fingerprint(self) -> dict[str, object]:
        return {
            "kind": "parquet",
            "resample": self._resample,
            "shuffle_mode": self._shuffle_mode,
            "iter": {
                "columns": list(self._columns) if self._columns else None,
                "use_threads": self._use_threads,
            },
            "fragments": [
                {
                    "path": fragment.path,
                    "row_groups": list(fragment.row_groups),
                    "row_offset": fragment.row_offset,
                    "num_rows": fragment.num_rows,
                    "row_group_offsets": list(fragment.row_group_offsets),
                    "row_group_num_rows": list(fragment.row_group_num_rows),
                }
                for fragment in self._source
            ],
        }
