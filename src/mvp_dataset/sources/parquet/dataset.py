from collections.abc import Sequence
from dataclasses import dataclass

from mvp_dataset.core.context import RuntimeContext
from mvp_dataset.core.dataset import Dataset
from mvp_dataset.core.types import ShardInput
from mvp_dataset.utils.url import normalize_paths

from .utils import iter_parquets, list_parquet_fragments, resolve_parquet_batch_size


@dataclass(frozen=True, slots=True)
class _ParquetSourceIter:
    columns: Sequence[str] | None = None
    use_threads: bool = True

    def __call__(self, fragment_stream):
        return iter_parquets(
            fragment_stream,
            columns=self.columns,
            use_threads=self.use_threads,
        )


class ParquetDataset(Dataset):
    @classmethod
    def from_source(
        cls,
        shards: ShardInput | Sequence[ShardInput],
        context: RuntimeContext | None = None,
        resample: bool = False,
        min_row_groups_per_fragment: int = 1,
        columns: Sequence[str] | None = None,
        use_threads: bool = True,
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

        Returns:
            A dataset whose source is the list of parquet fragments.

        Raises:
            ValueError: If any input path does not end with ``.parquet``.
        """
        runtime_context = RuntimeContext.from_runtime() if context is None else context
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
            _iter_source_stream=_ParquetSourceIter(
                columns=columns,
                use_threads=use_threads,
            ),
        )
