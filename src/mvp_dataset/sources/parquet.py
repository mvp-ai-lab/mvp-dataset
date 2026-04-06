"""Parquet source implementation for row-group-parallel sample iteration."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Final

import pyarrow.parquet as pq

from ..core.types import PathLikeStr, Sample

_DEFAULT_BATCH_SIZE: Final[int] = 65536
_FRAGMENT_SUFFIX: Final[str] = "@@rg="


@dataclass(frozen=True, slots=True)
class ParquetFragment:
    """One schedulable parquet row-group fragment."""

    path: str
    row_group: int
    row_offset: int
    num_rows: int

    @property
    def cache_key(self) -> str:
        """Stable shard identifier used by cache manifests and routing."""

        return f"{self.path}{_FRAGMENT_SUFFIX}{self.row_group}"


def list_parquet_fragments(
    shard_paths: Sequence[PathLikeStr],
) -> list[ParquetFragment]:
    """Expand parquet files into schedulable row-group fragments."""

    fragments: list[ParquetFragment] = []
    for shard_path in shard_paths:
        shard = str(shard_path)
        parquet_file = pq.ParquetFile(shard)
        row_offset = 0
        for row_group in range(parquet_file.num_row_groups):
            num_rows = parquet_file.metadata.row_group(row_group).num_rows
            fragments.append(
                ParquetFragment(
                    path=shard,
                    row_group=row_group,
                    row_offset=row_offset,
                    num_rows=num_rows,
                )
            )
            row_offset += num_rows
    return fragments


def iter_parquet(
    fragment: ParquetFragment,
    *,
    columns: Sequence[str] | None = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    use_threads: bool = True,
) -> Iterator[Sample]:
    """Iterate one parquet row-group fragment and yield one sample dict per row."""

    parquet_file = pq.ParquetFile(fragment.path)
    index_in_file = fragment.row_offset

    for record_batch in parquet_file.iter_batches(
        batch_size=batch_size,
        row_groups=[fragment.row_group],
        columns=columns,
        use_threads=use_threads,
    ):
        column_names = record_batch.schema.names
        columns_data = [record_batch.column(i) for i in range(record_batch.num_columns)]
        for batch_row_index in range(record_batch.num_rows):
            sample: Sample = {
                name: columns_data[column_index][batch_row_index].as_py()
                for column_index, name in enumerate(column_names)
            }
            sample["__file__"] = fragment.path
            sample["__index_in_file__"] = index_in_file
            sample["__key__"] = f"{fragment.path}:{index_in_file}"
            yield sample
            index_in_file += 1


def iter_parquets(
    fragments: Iterator[ParquetFragment],
    *,
    columns: Sequence[str] | None = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    use_threads: bool = True,
) -> Iterator[Sample]:
    """Iterate parquet row-group fragments in order and yield row samples."""

    for fragment in fragments:
        yield from iter_parquet(
            fragment,
            columns=columns,
            batch_size=batch_size,
            use_threads=use_threads,
        )
