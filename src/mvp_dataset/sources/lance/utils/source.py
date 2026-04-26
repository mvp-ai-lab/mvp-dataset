"""Main Lance source listing, sharding, and row reading."""

from __future__ import annotations

import bisect
from collections.abc import Iterable, Sequence

import lance
import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads

from mvp_dataset.core.context import RuntimeContext
from mvp_dataset.core.types import PathLikeStr, Sample

from .types import LanceDatasetSpec, LanceIndexItem, LanceSourceSpec


def list_lance_sources(dataset_uris: Sequence[PathLikeStr]) -> list[LanceSourceSpec]:
    """Resolve Lance dataset URIs into one slot-aware source configuration."""

    datasets: list[LanceDatasetSpec] = []
    row_offset = 0
    for uri in dataset_uris:
        ds = lance.dataset(str(uri))
        fragment_ids = tuple(fragment.fragment_id for fragment in ds.get_fragments())
        num_rows = ds.count_rows()
        datasets.append(
            LanceDatasetSpec(
                uri=str(uri),
                num_rows=num_rows,
                row_offset=row_offset,
                fragment_ids=fragment_ids,
            )
        )
        row_offset += num_rows

    if not datasets:
        msg = "[EmptyLanceSource] at least one Lance dataset URI is required"
        raise ValueError(msg)

    return [LanceSourceSpec(datasets=list(datasets))]


def assign_items(
    source: Sequence[LanceSourceSpec],
    *,
    context: RuntimeContext,
    resample: bool,
    shuffle: bool = False,
) -> Iterable[LanceIndexItem]:
    """Yield slot-assigned row indexes for a single logical Lance source.

    ``LanceDataset`` treats one or more physical Lance datasets as a single
    concatenated row space.  This iterator walks that global row space, applies
    runtime slot sharding, and translates each selected global row into the
    dataset-local coordinates needed by the batch reader.

    Args:
        source: Sequence containing exactly one ``LanceSourceSpec``. The spec may
            itself reference multiple physical Lance datasets; their row offsets
            define the global row numbering.
        context: Runtime context that provides the current slot, total slot
            count, and deterministic seed.
        resample: When true, repeat the source forever. Each repeat is called a
            round.
        shuffle: When true, shuffle global row indexes before each round using
            ``context.seed`` as the deterministic seed base.

    Yields:
        ``LanceIndexItem`` objects containing ``global_index`` in the
        concatenated source, plus ``dataset_i`` and ``local_index`` for reading
        the row from its physical Lance dataset.

    Raises:
        AssertionError: If more than one ``LanceSourceSpec`` is provided.
    """

    assert len(source) == 1, "Multiple Lance sources are not supported in this implementation"
    source_spec: LanceSourceSpec = source[0]

    round_index = 0
    global_index_list = np.arange(source_spec.total_rows)
    shuffle_rng = np.random.default_rng(context.seed + round_index)

    slot = context.slot
    total_slots = context.total_slots

    while True:
        if shuffle:
            shuffle_rng.shuffle(global_index_list)

        for i, global_index in enumerate(global_index_list):
            if i % total_slots != slot:
                continue

            dataset_i = (
                bisect.bisect_right(
                    [dataset.row_offset for dataset in source_spec.datasets],
                    global_index,
                )
                - 1
            )
            dataset = source_spec.datasets[dataset_i]
            local_index = int(global_index - dataset.row_offset)
            yield LanceIndexItem(
                dataset_i=dataset_i,
                local_index=local_index,
                global_index=int(global_index),
            )

        if not resample:
            break
        round_index += 1


def _read_batch(
    source: LanceSourceSpec,
    batch_indexes: Sequence[LanceIndexItem],
    *,
    columns: Sequence[str] | None = None,
):
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


def iter_lance(
    source: LanceSourceSpec,
    index_stream: Iterable[LanceIndexItem],
    *,
    columns: Sequence[str] | None = None,
    batch_size: int = 1024,
    load_in_memory: bool = False,
):
    """Read Lance rows from an index stream and yield sample dictionaries.

    This is the source-reader half of ``LanceDataset``.  The upstream
    ``index_stream`` decides ordering, sharding, resampling, and optional global
    shuffle; this function only groups those row indexes into read batches,
    fetches rows from the referenced Lance datasets, and restores the original
    index-stream order before yielding samples.

    Reference columns are intentionally left unresolved here.  They remain as
    keys or key lists from the main Lance dataset until a later
    ``resolve_ref(...)`` stage resolves them.

    Args:
        source: One logical Lance source. It may contain multiple physical Lance
            datasets with global row offsets.
        index_stream: Iterable of ``LanceIndexItem`` values, usually produced by
            ``assign_items``.
        columns: Optional projection passed through to Lance row reads. Metadata
            fields added by this reader are always present in yielded samples.
        batch_size: Maximum number of row indexes to read per Lance batch.
        load_in_memory: When true, load each physical Lance dataset into an
            in-memory Arrow dataset before serving row reads.

    Yields:
        Dict samples containing projected user columns plus ``__file__``,
        ``__local_index__``, ``__global_index__``, and ``__key__`` metadata.
    """

    for dataset_i, dataset in enumerate(source.datasets):
        ds_handle = lance.dataset(dataset.uri)
        if load_in_memory:
            ds_handle = pads.InMemoryDataset(ds_handle.to_table())
        source.datasets[dataset_i] = LanceDatasetSpec(
            uri=dataset.uri,
            num_rows=dataset.num_rows,
            row_offset=dataset.row_offset,
            fragment_ids=dataset.fragment_ids,
            handle=ds_handle,
        )

    batch_indexes: list[LanceIndexItem] = []
    for index_item in index_stream:
        batch_indexes.append(index_item)
        if len(batch_indexes) >= batch_size:
            batch = _read_batch(source, batch_indexes, columns=columns)
            yield from batch
            batch_indexes.clear()

    if batch_indexes:
        batch = _read_batch(source, batch_indexes, columns=columns)
        yield from batch
