"""Lance table read helpers for ref columns."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pyarrow as pa
import pyarrow.dataset as pads

from mvp_dataset.core.types import Sample

REF_INDEX_BUILD_BATCH_SIZE = 65536


def _read_table_rows(
    dataset_handle: object,
    row_indices: Sequence[int],
    *,
    columns: Sequence[str] | None = None,
) -> list[Sample]:
    take_indices = pa.array(row_indices, type=pa.int64())
    if isinstance(dataset_handle, pa.Table):
        table = dataset_handle
        if columns is not None:
            table = table.select(columns)
        table = table.take(take_indices)
    elif isinstance(dataset_handle, pads.Dataset):
        table = dataset_handle.take(take_indices, columns=columns)
    else:
        table = dataset_handle.take(row_indices, columns=columns)
    return table.to_pylist()


def _iter_table_record_batches(
    dataset_handle: object,
    *,
    columns: Sequence[str],
    batch_size: int = REF_INDEX_BUILD_BATCH_SIZE,
) -> Iterable[pa.RecordBatch]:
    if isinstance(dataset_handle, pa.Table):
        table = dataset_handle.select(columns)
        yield from table.to_batches(max_chunksize=batch_size)
        return
    if isinstance(dataset_handle, pads.Dataset):
        yield from dataset_handle.to_batches(columns=columns, batch_size=batch_size)
        return

    to_batches = getattr(dataset_handle, "to_batches", None)
    if callable(to_batches):
        yield from to_batches(columns=columns, batch_size=batch_size)
        return

    scanner = dataset_handle.scanner(columns=columns, batch_size=batch_size, scan_in_order=True)
    yield from scanner.to_batches()
