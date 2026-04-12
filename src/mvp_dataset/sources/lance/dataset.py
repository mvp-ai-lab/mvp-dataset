from collections.abc import Sequence

from mvp_dataset.core.context import RuntimeContext
from mvp_dataset.core.dataset import Dataset
from mvp_dataset.core.types import ShardInput
from mvp_dataset.utils.url import normalize_paths

from .utils import iter_lances, list_lance_fragments


class LanceDataset(Dataset):
    @classmethod
    def from_source(
        cls,
        shards: ShardInput | Sequence[ShardInput],
        context: RuntimeContext | None = None,
        resample: bool = False,
        columns: Sequence[str] | None = None,
        batch_size: int = 65536,
    ):
        """Build a dataset from local Lance dataset paths.

        Args:
            shards: One or more Lance dataset URIs or directory paths.
            context: Optional execution context. If omitted, inferred from runtime.
            resample: Whether to loop shards indefinitely across rounds.
            columns: Optional list of column names to read.
            batch_size: Number of rows per Arrow batch during iteration.

        Returns:
            A dataset whose source is the list of lance fragments.
        """
        runtime_context = RuntimeContext.from_runtime() if context is None else context
        normalized_shards = normalize_paths(shards)
        fragments = list_lance_fragments(
            normalized_shards,
            min_fragments=runtime_context.total_slots,
        )

        def _iter_source(fragment_stream):
            return iter_lances(
                fragment_stream,
                columns=columns,
                batch_size=batch_size,
            )

        return cls(
            context=runtime_context,
            _source=fragments,
            _resample=resample,
            _source_kind="lance",
            _stages=(),
            _iter_source_stream=_iter_source,
        )
