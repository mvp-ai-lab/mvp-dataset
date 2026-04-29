"""Main Lance source listing, sharding, and row reading."""

from __future__ import annotations

import bisect
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import lance
import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads

from mvp_dataset.core.context import RuntimeContext
from mvp_dataset.core.types import PathLikeStr, Sample

from .types import LanceDatasetSpec, LanceIndexItem, LanceShuffleMode, LanceSourceSpec

FRAGMENT_SHUFFLE_BLOCK_SIZE = 64


@dataclass(frozen=True, slots=True)
class _FragmentRowSpan:
    dataset_i: int
    local_row_offset: int
    num_rows: int


def list_lance_sources(dataset_uris: Sequence[PathLikeStr]) -> list[LanceSourceSpec]:
    """Resolve Lance dataset URIs into one slot-aware source configuration."""

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


def _iter_fragment_aware_shuffled_items(
    source: LanceSourceSpec,
    *,
    context: RuntimeContext,
    round_index: int,
) -> Iterable[LanceIndexItem]:
    spans: list[_FragmentRowSpan] = []
    for dataset_i, dataset in enumerate(source.datasets):
        local_row_offset = 0
        for row_count in dataset.fragment_row_counts or (dataset.num_rows,):
            row_count = int(row_count)
            if row_count > 0:
                spans.append(
                    _FragmentRowSpan(
                        dataset_i=dataset_i,
                        local_row_offset=local_row_offset,
                        num_rows=row_count,
                    )
                )
            local_row_offset += row_count

    if not spans:
        return

    # Split large fragments only when needed so every slot can receive work.
    if len(spans) < context.total_slots:
        split_spans: list[_FragmentRowSpan] = []
        base_parts, extra_parts = divmod(context.total_slots, len(spans))
        for span_i, span in enumerate(spans):
            parts = min(span.num_rows, base_parts + (1 if span_i < extra_parts else 0))
            chunk_size, extra_rows = divmod(span.num_rows, parts)
            local_row_offset = span.local_row_offset
            for part_i in range(parts):
                num_rows = chunk_size + (1 if part_i < extra_rows else 0)
                split_spans.append(
                    _FragmentRowSpan(
                        dataset_i=span.dataset_i,
                        local_row_offset=local_row_offset,
                        num_rows=num_rows,
                    )
                )
                local_row_offset += num_rows
        spans = split_spans

    rng = np.random.default_rng(context.seed + round_index)
    slot_spans: list[list[tuple[_FragmentRowSpan, int]]] = [[] for _ in range(context.total_slots)]
    slot_row_counts = [0] * context.total_slots
    for span_i in rng.permutation(len(spans)):
        span = spans[int(span_i)]
        slot = min(range(context.total_slots), key=slot_row_counts.__getitem__)
        row_seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        slot_spans[slot].append((span, row_seed))
        slot_row_counts[slot] += span.num_rows

    assigned_spans = slot_spans[context.slot]
    row_offsets_by_span: list[np.ndarray] = []
    blocks: list[tuple[int, int, int]] = []
    for assigned_span_i, (span, row_seed) in enumerate(assigned_spans):
        row_offsets_by_span.append(np.random.default_rng(row_seed).permutation(span.num_rows))
        for start in range(0, span.num_rows, FRAGMENT_SHUFFLE_BLOCK_SIZE):
            stop = min(start + FRAGMENT_SHUFFLE_BLOCK_SIZE, span.num_rows)
            blocks.append((assigned_span_i, start, stop))

    for block_i in rng.permutation(len(blocks)):
        assigned_span_i, start, stop = blocks[int(block_i)]
        span, _row_seed = assigned_spans[assigned_span_i]
        row_offsets = row_offsets_by_span[assigned_span_i]
        dataset = source.datasets[span.dataset_i]
        for row_offset in row_offsets[start:stop]:
            local_index = span.local_row_offset + int(row_offset)
            yield LanceIndexItem(
                dataset_i=span.dataset_i,
                local_index=local_index,
                global_index=int(dataset.row_offset + local_index),
            )


def assign_items(
    source: Sequence[LanceSourceSpec],
    *,
    context: RuntimeContext,
    resample: bool,
    shuffle_mode: LanceShuffleMode = "none",
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
        shuffle_mode: ``"none"`` preserves source row order, ``"global"`` uses
            an exact row-level global shuffle, and ``"fragment_aware"`` shuffles
            fragment/chunk spans while keeping each slot close to fewer
            fragments.

    Yields:
        ``LanceIndexItem`` objects containing ``global_index`` in the
        concatenated source, plus ``dataset_i`` and ``local_index`` for reading
        the row from its physical Lance dataset.

    Raises:
        AssertionError: If more than one ``LanceSourceSpec`` is provided.
    """

    assert len(source) == 1, "Multiple Lance sources are not supported in this implementation"
    if shuffle_mode not in ("none", "global", "fragment_aware"):
        msg = f"[InvalidLanceShuffleMode] expected none, global, or fragment_aware, got {shuffle_mode!r}"
        raise ValueError(msg)
    source_spec: LanceSourceSpec = source[0]

    round_index = 0
    slot = context.slot
    total_slots = context.total_slots
    global_index_list = np.arange(source_spec.total_rows)

    while True:
        if shuffle_mode == "fragment_aware":
            yield from _iter_fragment_aware_shuffled_items(
                source_spec,
                context=context,
                round_index=round_index,
            )
        else:
            if shuffle_mode == "global":
                global_index_list = np.random.default_rng(context.seed + round_index).permutation(
                    source_spec.total_rows
                )
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
            fragment_row_counts=dataset.fragment_row_counts,
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
