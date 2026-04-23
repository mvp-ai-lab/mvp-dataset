"""Lance source implementation for slot-aware shuffled sample iteration."""

from __future__ import annotations

import bisect
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import lance
import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads

from ...core.context import RuntimeContext
from ...core.types import PathLikeStr, Sample


@dataclass(frozen=True, slots=True)
class LanceDatasetSpec:
    """Resolved metadata for one Lance dataset URI."""

    uri: str
    num_rows: int
    row_offset: int
    fragment_ids: tuple[int, ...]
    handle: object | None = None


@dataclass(frozen=True, slots=True)
class LanceSourceSpec:
    """One schedulable Lance source configuration."""

    datasets: list[LanceDatasetSpec]

    @property
    def total_rows(self) -> int:
        return sum(dataset.num_rows for dataset in self.datasets)

    @property
    def total_fragments(self) -> int:
        return sum(len(dataset.fragment_ids) for dataset in self.datasets)


@dataclass(frozen=True, slots=True)
class LanceIndexItem:
    dataset_i: int
    local_index: int
    global_index: int


def list_lance_sources(dataset_uris: Sequence[PathLikeStr]) -> list[LanceSourceSpec]:
    """Resolve Lance dataset URIs into one slot-aware source configuration."""

    # 1. Resolve URIs into metadata specs, accumulating row offsets for global indexing.
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
) -> Iterable[tuple[LanceSourceSpec, int, int]]:
    assert len(source) == 1, "Multiple Lance sources are not supported in this implementation"
    source: LanceSourceSpec = source[0]

    # 1. Yield globally slot-assigned indexes for all rows across all datasets.
    round_index = 0
    global_index_list = np.arange(source.total_rows)
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
                    [dataset.row_offset for dataset in source.datasets],
                    global_index,
                )
                - 1
            )
            dataset = source.datasets[dataset_i]
            local_index = global_index - dataset.row_offset
            yield LanceIndexItem(
                dataset_i=dataset_i,
                local_index=local_index,
                global_index=global_index,
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
    # 1. group batch indexes by dataset
    if not batch_indexes:
        return []

    per_dataset_indices: dict[int, list[int]] = {}
    for index_item in batch_indexes:
        per_dataset_indices.setdefault(index_item.dataset_i, []).append(index_item.local_index)

    # 2. read each dataset batch
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
            table = dataset_handle.to_table(columns=columns)
            table = table.take(pa.array(local_indices, type=take_indices_type))
        else:
            table = dataset_handle.take(local_indices, columns=columns)

        per_dataset_rows[dataset_i] = table.to_pylist()

    # 3. restore global order within batch
    per_dataset_offsets = {dataset_i: 0 for dataset_i in per_dataset_indices}

    # 4. concatenate into one batch and return
    batch: list[Sample] = []
    for index_item in batch_indexes:
        dataset = source.datasets[index_item.dataset_i]
        dataset_offset = per_dataset_offsets[index_item.dataset_i]
        sample = dict(per_dataset_rows[index_item.dataset_i][dataset_offset])
        sample["__file__"] = dataset.uri
        sample["__index_in_file__"] = index_item.local_index
        sample["__key__"] = f"{dataset.uri}:{index_item.local_index}"
        batch.append(sample)
        per_dataset_offsets[index_item.dataset_i] += 1

    return batch


def iter_lance(
    source: LanceSourceSpec,
    index_stream: Iterable[LanceIndexItem],
    *,
    columns: Sequence[str] | None = None,
    batch_size: int = 65536,
    load_in_memory: bool = False,
):
    # 1. Load all dataset into memory if requested.
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
            yield from _read_batch(source, batch_indexes, columns=columns)
            batch_indexes.clear()
    if batch_indexes:
        yield from _read_batch(source, batch_indexes, columns=columns)
