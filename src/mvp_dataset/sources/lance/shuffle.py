"""Lance source index order helpers."""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np

from mvp_dataset.core.context import RuntimeContext

from .types import LanceIndexItem, LanceShuffleMode, LanceSourceSpec

_FRAGMENT_SHUFFLE_BLOCK_SIZE = 64
_MASK64 = (1 << 64) - 1
_FEISTEL_ROUNDS = 8
_MAX_CYCLE_WALK_ATTEMPTS = 1024


@dataclass(frozen=True, slots=True)
class _FragmentRowSpan:
    """Contiguous row span for a Lance fragment."""

    dataset_i: int
    local_row_offset: int
    num_rows: int


def _mix64(value: int) -> int:
    """Mix a 64-bit integer for deterministic pseudo-random indexing."""
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return value ^ (value >> 31)


def _feistel(value: int, *, bits: int, seed: int) -> int:
    """Permute a bounded integer with a Feistel network."""
    left_bits = bits // 2
    right_bits = bits - left_bits
    left_mask = (1 << left_bits) - 1
    right_mask = (1 << right_bits) - 1

    left = value >> right_bits
    right = value & right_mask
    for round_i in range(_FEISTEL_ROUNDS):
        if round_i % 2 == 0:
            mixed = _mix64(left ^ seed ^ ((round_i + 1) * 0x9E3779B97F4A7C15))
            right = (right ^ mixed) & right_mask
        else:
            mixed = _mix64(right ^ seed ^ ((round_i + 1) * 0x9E3779B97F4A7C15))
            left = (left ^ mixed) & left_mask
    return (left << right_bits) | right


def permute_index(position: int, *, total_rows: int, seed: int) -> int:
    """Return a deterministic permutation value for one index.

    Args:
        position: Logical position to permute.
        total_rows: Total number of rows in the permutation domain.
        seed: Base random seed.

    Returns:
        The permuted index."""
    if total_rows <= 0:
        msg = "total_rows must be positive"
        raise ValueError(msg)
    if position < 0 or position >= total_rows:
        msg = "position out of range"
        raise ValueError(msg)
    if total_rows == 1:
        return 0

    bits = max(2, (total_rows - 1).bit_length())
    value = position
    for _ in range(_MAX_CYCLE_WALK_ATTEMPTS):
        value = _feistel(value, bits=bits, seed=seed)
        if value < total_rows:
            return value

    msg = "[LanceGlobalShuffleCycleWalkLimit] exceeded max cycle-walk attempts"
    raise RuntimeError(msg)


def lance_round_size(
    source: LanceSourceSpec,
    context: RuntimeContext,
    shuffle_mode: LanceShuffleMode,
    fragment_order: list[LanceIndexItem],
) -> int:
    """Return the number of items assigned to one runtime slot.

    Args:
        source: Lance source specification.
        context: Runtime context used for sharding and deterministic randomness.
        shuffle_mode: Source-level shuffle mode.
        fragment_order: Precomputed fragment-aware Lance row order.

    Returns:
        Number of items assigned to the current runtime slot."""
    if shuffle_mode in ("none", "global"):
        if context.slot >= source.total_rows:
            return 0
        return ((source.total_rows - 1 - context.slot) // context.total_slots) + 1
    if shuffle_mode == "fragment_aware":
        return len(fragment_order)

    msg = f"[InvalidLanceShuffleMode] expected none, global, or fragment_aware, got {shuffle_mode!r}"
    raise ValueError(msg)


def lance_index_at(
    source: LanceSourceSpec,
    context: RuntimeContext,
    shuffle_mode: LanceShuffleMode,
    round_index: int,
    position_in_round: int,
    fragment_order: list[LanceIndexItem],
) -> LanceIndexItem:
    """Return the Lance row location for one logical slot position.

    Args:
        source: Lance source specification.
        context: Runtime context used for sharding and deterministic randomness.
        shuffle_mode: Source-level shuffle mode.
        round_index: Resampling round index.
        position_in_round: Position within the current source round.
        fragment_order: Precomputed fragment-aware Lance row order.

    Returns:
        The physical Lance row location for the logical position."""
    global_position = context.slot + position_in_round * context.total_slots
    if shuffle_mode == "none":
        return _index_item(source, global_position)
    if shuffle_mode == "global":
        global_index = permute_index(global_position, total_rows=source.total_rows, seed=context.seed + round_index)
        return _index_item(source, global_index)
    if shuffle_mode == "fragment_aware":
        return fragment_order[position_in_round]

    msg = f"[InvalidLanceShuffleMode] expected none, global, or fragment_aware, got {shuffle_mode!r}"
    raise ValueError(msg)


def fragment_aware_index_order(
    source: LanceSourceSpec,
    context: RuntimeContext,
    round_index: int,
) -> list[LanceIndexItem]:
    """Return deterministic fragment-aware row order for one slot.

    Args:
        source: Lance source specification.
        context: Runtime context used for sharding and deterministic randomness.
        round_index: Resampling round index.

    Returns:
        A deterministic row order for the current runtime slot."""
    spans = _fragment_spans(source)
    if not spans:
        return []

    if len(spans) < context.total_slots:
        spans = _split_spans_for_slots(spans, context.total_slots)

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
        for start in range(0, span.num_rows, _FRAGMENT_SHUFFLE_BLOCK_SIZE):
            stop = min(start + _FRAGMENT_SHUFFLE_BLOCK_SIZE, span.num_rows)
            blocks.append((assigned_span_i, start, stop))

    index_order: list[LanceIndexItem] = []
    for block_i in rng.permutation(len(blocks)):
        assigned_span_i, start, stop = blocks[int(block_i)]
        span, _row_seed = assigned_spans[assigned_span_i]
        row_offsets = row_offsets_by_span[assigned_span_i]
        dataset = source.datasets[span.dataset_i]
        for row_offset in row_offsets[start:stop]:
            local_index = span.local_row_offset + int(row_offset)
            index_order.append(
                LanceIndexItem(
                    dataset_i=span.dataset_i,
                    local_index=local_index,
                    global_index=int(dataset.row_offset + local_index),
                )
            )
    return index_order


def _fragment_spans(source: LanceSourceSpec) -> list[_FragmentRowSpan]:
    """Return row spans for all Lance fragments."""
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
    return spans


def _split_spans_for_slots(spans: list[_FragmentRowSpan], total_slots: int) -> list[_FragmentRowSpan]:
    """Split fragment spans across runtime slots."""
    split_spans: list[_FragmentRowSpan] = []
    base_parts, extra_parts = divmod(total_slots, len(spans))
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
    return split_spans


def _index_item(source: LanceSourceSpec, global_index: int) -> LanceIndexItem:
    """Return a Lance index item for one global row index."""
    dataset_i = bisect.bisect_right([dataset.row_offset for dataset in source.datasets], global_index) - 1
    dataset = source.datasets[dataset_i]
    local_index = int(global_index - dataset.row_offset)
    return LanceIndexItem(dataset_i=dataset_i, local_index=local_index, global_index=int(global_index))
