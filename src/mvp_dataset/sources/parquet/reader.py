"""Parquet row readers."""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from typing import Final

import pyarrow.parquet as pq

from ...core.types import Sample
from .chunks import ParquetChunk

_DEFAULT_BATCH_SIZE: Final[int] = 65536
_BATCH_SIZE_ENV_VAR: Final[str] = "MVP_DATASET_PARQUET_BATCH_SIZE"


def resolve_parquet_batch_size(batch_size: int | None = None) -> int:
    """Return a valid Parquet batch size.

    Args:
        batch_size: Number of samples to group into each batch.

    Returns:
        A positive Parquet batch size."""
    if batch_size is None:
        raw_value = os.environ.get(_BATCH_SIZE_ENV_VAR)
        if raw_value is None:
            batch_size = _DEFAULT_BATCH_SIZE
        else:
            try:
                batch_size = int(raw_value)
            except ValueError as exc:
                msg = f"[InvalidParquetBatchSize] {_BATCH_SIZE_ENV_VAR} must be an integer, got {raw_value!r}"
                raise ValueError(msg) from exc
    if batch_size <= 0:
        msg = f"[InvalidParquetBatchSize] {_BATCH_SIZE_ENV_VAR} must be >= 1, got {batch_size}"
        raise ValueError(msg)
    return batch_size


def iter_parquet(
    chunk: ParquetChunk,
    *,
    columns: Sequence[str] | None = None,
    batch_size: int | None = None,
    use_threads: bool = True,
    row_group_index: int | None = None,
) -> Iterator[Sample]:
    """Iterate one parquet row-group chunk and yield one sample dict per row.

    Args:
        chunk: Parquet chunk to read.
        columns: Column names to read from the source.
        batch_size: Number of samples to group into each batch.
        use_threads: Whether the reader may use threaded decoding.
        row_group_index: Optional row group index to read within the chunk.

    Returns:
        An iterator over sample dictionaries from one Parquet chunk."""

    resolved_batch_size = resolve_parquet_batch_size(batch_size)
    parquet_file = pq.ParquetFile(chunk.path)
    row_groups = list(chunk.row_groups)
    index_in_file = chunk.row_offset
    if row_group_index is not None:
        row_groups = [chunk.row_groups[row_group_index]]
        index_in_file = chunk.row_group_offsets[row_group_index]

    for record_batch in parquet_file.iter_batches(
        batch_size=resolved_batch_size,
        row_groups=row_groups,
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
            sample["__file__"] = chunk.path
            sample["__index_in_file__"] = index_in_file
            sample["__key__"] = f"{chunk.path}:{index_in_file}"
            yield sample
            index_in_file += 1


def iter_parquets(
    chunks: Iterator[ParquetChunk],
    *,
    columns: Sequence[str] | None = None,
    batch_size: int | None = None,
    use_threads: bool = True,
) -> Iterator[Sample]:
    """Iterate parquet row-group chunks in order and yield row samples.

    Args:
        chunks: Parquet chunks to read.
        columns: Column names to read from the source.
        batch_size: Number of samples to group into each batch.
        use_threads: Whether the reader may use threaded decoding.

    Returns:
        An iterator over sample dictionaries from all chunks."""

    for chunk in chunks:
        yield from iter_parquet(
            chunk,
            columns=columns,
            batch_size=batch_size,
            use_threads=use_threads,
        )
