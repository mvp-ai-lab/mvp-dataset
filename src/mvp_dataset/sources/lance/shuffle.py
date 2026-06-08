"""Lance source index order helpers."""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np

from mvp_dataset.core.context import RuntimeContext

from .types import LanceIndexItem, LanceShuffleMode, LanceSourceSpec

DEFAULT_CHUNK_AWARE_SHUFFLE_CHUNK_SIZE = 250_000
DEFAULT_CHUNK_AWARE_SHUFFLE_K = 8
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


@dataclass(frozen=True, slots=True)
class ChunkAwareShuffleOrder:
    """Compact chunk-level order for chunk-aware Lance shuffle."""

    chunk_size: int
    k: int
    chunk_order: tuple[int, ...]
    window_offsets: tuple[int, ...]


def _mix_seed(*parts: int) -> int:
    seed = 0xCBF29CE484222325
    for part in parts:
        seed ^= int(part) & _MASK64
        seed = (seed * 0x100000001B3) & _MASK64
    return seed


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
    chunk_order: ChunkAwareShuffleOrder | None = None,
) -> int:
    """Return the number of items assigned to one runtime slot.

    Args:
        source: Lance source specification.
        context: Runtime context used for sharding and deterministic randomness.
        shuffle_mode: Source-level shuffle mode.
        fragment_order: Precomputed fragment-aware Lance row order.

    Returns:
        Number of items assigned to the current runtime slot."""
    if shuffle_mode in ("none", "global", "chunk_aware"):
        if context.slot >= source.total_rows:
            return 0
        return ((source.total_rows - 1 - context.slot) // context.total_slots) + 1
    if shuffle_mode == "fragment_aware":
        return len(fragment_order)

    msg = f"[InvalidLanceShuffleMode] expected none, global, chunk_aware, or fragment_aware, got {shuffle_mode!r}"
    raise ValueError(msg)


def lance_index_at(
    source: LanceSourceSpec,
    context: RuntimeContext,
    shuffle_mode: LanceShuffleMode,
    round_index: int,
    position_in_round: int,
    fragment_order: list[LanceIndexItem],
    chunk_order: ChunkAwareShuffleOrder | None = None,
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
    if shuffle_mode == "chunk_aware":
        if chunk_order is None:
            msg = "[InvalidLanceChunkAwareShuffle] chunk order is required"
            raise ValueError(msg)
        global_index = _chunk_aware_global_index_at(
            source,
            global_position=global_position,
            round_index=round_index,
            seed=context.seed,
            order=chunk_order,
        )
        return _index_item(source, global_index)
    if shuffle_mode == "fragment_aware":
        return fragment_order[position_in_round]

    msg = f"[InvalidLanceShuffleMode] expected none, global, chunk_aware, or fragment_aware, got {shuffle_mode!r}"
    raise ValueError(msg)


def chunk_aware_index_order(
    source: LanceSourceSpec,
    context: RuntimeContext,
    round_index: int,
    *,
    chunk_size: int = DEFAULT_CHUNK_AWARE_SHUFFLE_CHUNK_SIZE,
    k: int = DEFAULT_CHUNK_AWARE_SHUFFLE_K,
) -> ChunkAwareShuffleOrder:
    """Return compact chunk ordering metadata for one chunk-aware round.

    Args:
        source: Lance source specification.
        context: Runtime context used for deterministic randomness.
        round_index: Resampling round index.
        chunk_size: Number of global rows in one chunk.
        k: Number of adjacent shuffled chunks in one active window.

    Returns:
        Chunk-aware order metadata for position-based lookup."""
    _validate_chunk_aware_config(chunk_size=chunk_size, k=k)
    if source.total_rows <= 0:
        return ChunkAwareShuffleOrder(chunk_size=chunk_size, k=k, chunk_order=(), window_offsets=(0,))

    num_chunks = (source.total_rows + chunk_size - 1) // chunk_size
    rng = np.random.default_rng(_mix_seed(context.seed, round_index, 0))
    chunk_order = tuple(int(chunk_id) for chunk_id in rng.permutation(num_chunks))

    window_offsets = [0]
    for window_start in range(0, num_chunks, k):
        window_size = sum(
            _chunk_row_count(source, chunk_order[order_i], chunk_size=chunk_size)
            for order_i in range(window_start, min(window_start + k, num_chunks))
        )
        window_offsets.append(window_offsets[-1] + window_size)

    return ChunkAwareShuffleOrder(
        chunk_size=chunk_size,
        k=k,
        chunk_order=chunk_order,
        window_offsets=tuple(window_offsets),
    )


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


def _chunk_aware_global_index_at(
    source: LanceSourceSpec,
    *,
    global_position: int,
    round_index: int,
    seed: int,
    order: ChunkAwareShuffleOrder,
) -> int:
    """Map one logical global stream position to one source global row."""
    if global_position < 0 or global_position >= source.total_rows:
        msg = "[InvalidLanceChunkAwareShuffle] global position is out of range"
        raise ValueError(msg)

    window_i = bisect.bisect_right(order.window_offsets, global_position) - 1
    window_start_offset = order.window_offsets[window_i]
    window_size = order.window_offsets[window_i + 1] - window_start_offset
    local_position = global_position - window_start_offset
    shuffled_position = permute_index(
        local_position,
        total_rows=window_size,
        seed=_mix_seed(seed, round_index, 1, window_i),
    )

    chunk_order_start = window_i * order.k
    chunk_order_stop = min(chunk_order_start + order.k, len(order.chunk_order))
    for order_i in range(chunk_order_start, chunk_order_stop):
        chunk_id = order.chunk_order[order_i]
        row_count = _chunk_row_count(source, chunk_id, chunk_size=order.chunk_size)
        if shuffled_position < row_count:
            return chunk_id * order.chunk_size + shuffled_position
        shuffled_position -= row_count

    msg = "[InvalidLanceChunkAwareShuffle] failed to map chunk-aware position"
    raise RuntimeError(msg)


def _validate_chunk_aware_config(*, chunk_size: int, k: int) -> None:
    if chunk_size <= 0:
        msg = f"[InvalidLanceChunkAwareShuffle] chunk_size must be > 0, got {chunk_size}"
        raise ValueError(msg)
    if k <= 0:
        msg = f"[InvalidLanceChunkAwareShuffle] k must be > 0, got {k}"
        raise ValueError(msg)


def _chunk_row_count(source: LanceSourceSpec, chunk_id: int, *, chunk_size: int) -> int:
    start = chunk_id * chunk_size
    if start >= source.total_rows:
        return 0
    return min(chunk_size, source.total_rows - start)


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
