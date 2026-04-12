"""Parquet source implementation for row-group-parallel sample iteration."""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Final

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from ...core.types import PathLikeStr, Sample

_DEFAULT_BATCH_SIZE: Final[int] = 65536
_FRAGMENT_SUFFIX: Final[str] = "@@rg="
_LOCAL_FILESYSTEM = pafs.LocalFileSystem()


@dataclass(frozen=True, slots=True)
class ParquetFragment:
    """One schedulable parquet row-group fragment."""

    path: str
    row_groups: tuple[int, ...]
    row_offset: int
    num_rows: int

    @property
    def cache_key(self) -> str:
        """Stable shard identifier used by cache manifests and routing."""

        rg_spec = (
            str(self.row_groups[0]) if len(self.row_groups) == 1 else f"{self.row_groups[0]}-{self.row_groups[-1]}"
        )
        return f"{self.path}{_FRAGMENT_SUFFIX}{rg_spec}"


def list_parquet_fragments(
    shard_paths: Sequence[PathLikeStr],
    *,
    min_fragments: int = 0,
) -> list[ParquetFragment]:
    """Expand parquet files into schedulable row-group fragments.

    Row groups with fewer than *min_rows_per_fragment* rows are merged with
    subsequent row groups until the threshold is reached.  When the resulting
    fragment count is below *min_fragments*, the threshold is lowered to 0 and
    fragments are re-split so that every row group becomes its own fragment.
    """
    min_rows_per_fragment = int(os.environ.get("MVP_DATASET_PARQUET_MIN_ROWS_PER_FRAGMENT", "5000"))
    fragments = _collect_parquet_fragments(shard_paths, min_rows_per_fragment)
    if len(fragments) < min_fragments:
        fragments = _collect_parquet_fragments(shard_paths, 0)
    return fragments


def _collect_parquet_fragments(
    shard_paths: Sequence[PathLikeStr],
    min_rows_per_fragment: int,
) -> list[ParquetFragment]:
    fragments: list[ParquetFragment] = []
    for shard_path in shard_paths:
        shard = str(shard_path)
        parquet_file = pq.ParquetFile(shard)
        row_offset = 0
        pending_groups: list[int] = []
        pending_rows = 0
        for row_group in range(parquet_file.num_row_groups):
            num_rows = parquet_file.metadata.row_group(row_group).num_rows
            pending_groups.append(row_group)
            pending_rows += num_rows
            if pending_rows >= min_rows_per_fragment:
                fragments.append(
                    ParquetFragment(
                        path=shard,
                        row_groups=tuple(pending_groups),
                        row_offset=row_offset,
                        num_rows=pending_rows,
                    )
                )
                row_offset += pending_rows
                pending_groups = []
                pending_rows = 0
        if pending_groups:
            fragments.append(
                ParquetFragment(
                    path=shard,
                    row_groups=tuple(pending_groups),
                    row_offset=row_offset,
                    num_rows=pending_rows,
                )
            )
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

    for record_batch in _iter_record_batches(
        parquet_file,
        fragment,
        columns=columns,
        batch_size=batch_size,
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


def _iter_record_batches(
    parquet_file: pq.ParquetFile,
    fragment: ParquetFragment,
    *,
    columns: Sequence[str] | None,
    batch_size: int,
    use_threads: bool,
) -> Iterator[pa.RecordBatch]:
    """Stream record batches for one fragment without materializing whole row groups."""

    if _requires_dataset_scanner(parquet_file.schema_arrow, columns):
        yield from _iter_record_batches_via_scanner(
            fragment,
            columns=columns,
            batch_size=batch_size,
            use_threads=use_threads,
        )
        return

    yield from parquet_file.iter_batches(
        batch_size=batch_size,
        row_groups=list(fragment.row_groups),
        columns=columns,
        use_threads=use_threads,
    )


def _requires_dataset_scanner(schema: pa.Schema, columns: Sequence[str] | None) -> bool:
    """Use Scanner for nested columns to avoid ParquetFile.iter_batches conversion limits."""

    selected_columns = schema.names if columns is None else columns
    for column_name in selected_columns:
        field_index = schema.get_field_index(column_name)
        if field_index < 0:
            return True
        if pa.types.is_nested(schema.field(field_index).type):
            return True
    return False


def _iter_record_batches_via_scanner(
    fragment: ParquetFragment,
    *,
    columns: Sequence[str] | None,
    batch_size: int,
    use_threads: bool,
) -> Iterator[pa.RecordBatch]:
    """Stream one parquet fragment through Scanner with bounded readahead."""

    parquet_fragment = ds.ParquetFileFormat().make_fragment(
        fragment.path,
        filesystem=_LOCAL_FILESYSTEM,
        row_groups=list(fragment.row_groups),
    )
    scanner = ds.Scanner.from_fragment(
        parquet_fragment,
        columns=columns,
        batch_size=batch_size,
        batch_readahead=1,
        fragment_readahead=1,
        use_threads=use_threads,
    )
    yield from scanner.to_batches()


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
