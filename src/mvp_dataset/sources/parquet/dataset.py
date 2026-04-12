from collections.abc import Sequence

from mvp_dataset.core.context import RuntimeContext
from mvp_dataset.core.dataset import Dataset
from mvp_dataset.core.types import ShardInput
from mvp_dataset.utils.url import normalize_paths

from .utils import iter_parquets, list_parquet_fragments


class ParquetDataset(Dataset):
    @classmethod
    def from_source(
        cls,
        shards: ShardInput | Sequence[ShardInput],
        context: RuntimeContext | None = None,
        resample: bool = False,
        columns: Sequence[str] | None = None,
        batch_size: int = 65536,
        use_threads: bool = True,
    ):
        """Build a dataset from local Parquet file paths.

        Args:
            shards: One or more file paths, glob specs, or brace-expansion specs.
            context: Optional execution context. If omitted, inferred from runtime.
            resample: Whether to loop shards indefinitely across rounds.
            columns: Optional list of column names to read.
            batch_size: Number of rows per Arrow batch during iteration.
            use_threads: Whether to use multi-threaded Arrow reads.

        Returns:
            A dataset whose source is the list of parquet fragments.

        Raises:
            ValueError: If any input path does not end with ``.parquet``.
        """
        runtime_context = RuntimeContext.from_runtime() if context is None else context
        normalized_shards = normalize_paths(shards)
        if not all(path.endswith(".parquet") for path in normalized_shards):
            msg = f"[InvalidSourceType] expected .parquet inputs, got={normalized_shards!r}"
            raise ValueError(msg)
        fragments = list_parquet_fragments(
            normalized_shards,
            min_fragments=runtime_context.total_slots,
        )

        def _iter_source(fragment_stream):
            return iter_parquets(
                fragment_stream,
                columns=columns,
                batch_size=batch_size,
                use_threads=use_threads,
            )

        return cls(
            context=runtime_context,
            _source=fragments,
            _resample=resample,
            _source_kind="parquet",
            _stages=(),
            _iter_source_stream=_iter_source,
        )
