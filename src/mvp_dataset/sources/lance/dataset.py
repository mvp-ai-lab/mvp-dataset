from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from mvp_dataset.core.context import RuntimeContext
from mvp_dataset.core.dataset import Dataset
from mvp_dataset.core.types import ShardInput
from mvp_dataset.utils.url import normalize_paths

from .utils import (
    LanceSourceSpec,
    assign_items,
    attach_lance_ref_columns,
    iter_lance,
    list_lance_sources,
)


@dataclass(frozen=True, slots=True)
class _LanceSourceIter:
    source: LanceSourceSpec
    columns: Sequence[str] | None = None
    batch_size: int = 65536
    load_in_memory: bool = False

    handles_sharding = True

    def __call__(self, source_stream):
        return iter_lance(
            self.source,
            source_stream,
            columns=self.columns,
            batch_size=self.batch_size,
            load_in_memory=self.load_in_memory,
        )


@dataclass(frozen=True, slots=True)
class LanceDataset(Dataset):
    _global_shuffle: bool = False
    _load_in_memory: bool = False

    @classmethod
    def from_source(
        cls,
        shards: ShardInput | Sequence[ShardInput],
        context: RuntimeContext | None = None,
        resample: bool = False,
        columns: Sequence[str] | None = None,
        batch_size: int = 65536,
        global_shuffle: bool = False,
        load_in_memory: bool = False,
        ref_columns: dict[str, dict[str, str]] | None = None,
    ):
        """Build a dataset from local Lance dataset paths.

        Args:
            shards: One or more Lance dataset URIs or directory paths.
            context: Optional execution context. If omitted, inferred from runtime.
            resample: Whether to loop shards indefinitely across rounds.
            columns: Optional list of column names to read.
            batch_size: Number of rows per Arrow batch during iteration.
            global_shuffle: Whether to shuffle rows globally across all datasets.
            load_in_memory: Whether to load entire datasets into memory (recommended
                            if you provide a metadata lance dataset
                            and link other data via reference columns).
            ref_columns: Optional mapping of source column names to explicit Lance
                         reference configs containing uri, key_column, and value_column.
        Returns:
            A lance dataset.
        """
        runtime_context = RuntimeContext.from_runtime() if context is None else context
        normalized_shards = normalize_paths(shards)
        sources = list_lance_sources(
            normalized_shards,
        )
        source = attach_lance_ref_columns(sources[0], ref_columns)
        sources = [source]

        return cls(
            context=runtime_context,
            _source=sources,
            _resample=resample,
            _source_kind="lance",
            _stages=(),
            _iter_source_stream=_LanceSourceIter(
                source=source,
                columns=columns,
                batch_size=batch_size,
                load_in_memory=load_in_memory,
            ),
            _global_shuffle=global_shuffle,
            _load_in_memory=load_in_memory,
        )

    def _build_source_stream(self, *, context: RuntimeContext) -> Iterable[object]:
        assert len(self._source) == 1, "Multiple Lance sources are not supported in this implementation"
        source_shard_stream = assign_items(
            self._source,
            context=context,
            resample=self._resample,
            shuffle=self._global_shuffle,
        )
        return self._iter_source_stream(source_shard_stream)

    def shuffle(self, *args, **kwargs) -> Dataset:
        raise NotImplementedError("LanceDataset.shuffle() is not supported.")
