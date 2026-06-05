"""Main Lance source listing, sharding, and row reading."""

from __future__ import annotations

from collections.abc import Sequence

import lance
import pyarrow as pa
import pyarrow.dataset as pads

from mvp_dataset.core.types import PathLikeStr, Sample

from .types import LanceDatasetSpec, LanceIndexItem, LanceSourceSpec


def list_lance_sources(dataset_uris: Sequence[PathLikeStr]) -> list[LanceSourceSpec]:
    """Resolve Lance dataset URIs into one slot-aware source configuration.

    Args:
        dataset_uris: Lance dataset URIs to inspect.

    Returns:
        Lance source metadata for each dataset URI."""

    datasets: list[LanceDatasetSpec] = []
    row_offset = 0
    for uri in dataset_uris:
        ds = lance.dataset(str(uri))
        fragments = tuple(ds.get_fragments())
        fragment_ids = tuple(fragment.fragment_id for fragment in fragments)
        fragment_row_counts = tuple(int(fragment.count_rows()) for fragment in fragments)
        num_rows = ds.count_rows()
        datasets.append(
            LanceDatasetSpec(
                uri=str(uri),
                num_rows=num_rows,
                row_offset=row_offset,
                fragment_ids=fragment_ids,
                fragment_row_counts=fragment_row_counts,
            )
        )
        row_offset += num_rows

    if not datasets:
        msg = "[EmptyLanceSource] at least one Lance dataset URI is required"
        raise ValueError(msg)

    return [LanceSourceSpec(datasets=list(datasets))]


def _read_batch(
    source: LanceSourceSpec,
    batch_indexes: Sequence[LanceIndexItem],
    *,
    columns: Sequence[str] | None = None,
):
    """Read a batch of rows from Lance datasets."""
    if not batch_indexes:
        return []

    # 1. Group row indexes by dataset.
    per_dataset_indices: dict[int, list[int]] = {}
    for index_item in batch_indexes:
        per_dataset_indices.setdefault(index_item.dataset_i, []).append(index_item.local_index)

    # 2. Read rows from each dataset.
    per_dataset_rows: dict[int, list[Sample]] = {}
    take_indices_type = pa.int64()
    for dataset_i, local_indices in per_dataset_indices.items():
        dataset = source.datasets[dataset_i]
        dataset_handle = dataset.handle if dataset.handle is not None else lance.dataset(dataset.uri)

        if isinstance(dataset_handle, pa.Table):
            table = dataset_handle
            if columns is not None:
                table = table.select(columns)
            table = table.take(pa.array(local_indices, type=take_indices_type))
        elif isinstance(dataset_handle, pads.Dataset):
            table = dataset_handle.take(
                pa.array(local_indices, type=take_indices_type),
                columns=columns,
            )
        else:
            table = dataset_handle.take(local_indices, columns=columns)

        per_dataset_rows[dataset_i] = table.to_pylist()

    per_dataset_offsets = {dataset_i: 0 for dataset_i in per_dataset_indices}

    # 3. Restore original index order and yield samples with metadata.
    batch: list[Sample] = []
    for index_item in batch_indexes:
        dataset = source.datasets[index_item.dataset_i]
        dataset_offset = per_dataset_offsets[index_item.dataset_i]
        sample = dict(per_dataset_rows[index_item.dataset_i][dataset_offset])
        sample["__file__"] = dataset.uri
        sample["__local_index__"] = index_item.local_index
        sample["__global_index__"] = index_item.global_index
        sample["__key__"] = f"{dataset.uri}:{index_item.local_index}"
        batch.append(sample)
        per_dataset_offsets[index_item.dataset_i] += 1

    return batch
