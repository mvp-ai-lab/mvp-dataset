"""Lance logical order to physical row indexes."""

from __future__ import annotations

import bisect
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np

from mvp_dataset.core.context import RuntimeContext

from .types import LanceIndexItem, LanceShuffleMode, LanceSource

DEFAULT_CHUNK_SHUFFLE_CHUNK_SIZE: Final[int] = 250_000
DEFAULT_CHUNK_SHUFFLE_K: Final[int] = 8
DEFAULT_CHUNK_ROW_ORDER: Final[str] = "permuted"
_MASK64: Final[int] = (1 << 64) - 1
_FEISTEL_ROUNDS: Final[int] = 8
_MAX_CYCLE_WALK_ATTEMPTS: Final[int] = 1024

ChunkRowOrder = Literal["permuted", "sequential"]


@dataclass(frozen=True, slots=True)
class ChunkShuffleConfig:
    """Chunk shuffle configuration."""

    chunk_size: int
    k: int
    row_order: ChunkRowOrder


ChunkShuffleInput = Mapping[str, object] | ChunkShuffleConfig | None


@dataclass(frozen=True, slots=True)
class _ChunkRoundOrder:
    """Cached chunk order for one source round."""

    chunk_size: int
    k: int
    chunk_order: tuple[int, ...]
    chunk_offsets: tuple[int, ...]
    window_offsets: tuple[int, ...]


def resolve_chunk_shuffle_config(config: ChunkShuffleInput = None) -> ChunkShuffleConfig:
    """Return validated chunk shuffle settings."""
    if isinstance(config, ChunkShuffleConfig):
        return _validate_chunk_shuffle_config(config)
    if config is not None and not isinstance(config, Mapping):
        msg = "[InvalidLanceChunkShuffle] chunk_shuffle must be a mapping"
        raise TypeError(msg)

    raw_config = {} if config is None else dict(config)
    allowed_keys = {"chunk_size", "k", "row_order"}
    unknown_keys = sorted(set(raw_config) - allowed_keys)
    if unknown_keys:
        msg = f"[InvalidLanceChunkShuffle] unknown config key(s): {', '.join(unknown_keys)}"
        raise ValueError(msg)

    chunk_size = int(raw_config.get("chunk_size", DEFAULT_CHUNK_SHUFFLE_CHUNK_SIZE))
    k = int(raw_config.get("k", DEFAULT_CHUNK_SHUFFLE_K))
    row_order = raw_config.get("row_order", DEFAULT_CHUNK_ROW_ORDER)
    return _validate_chunk_shuffle_config(ChunkShuffleConfig(chunk_size=chunk_size, k=k, row_order=row_order))


def _validate_chunk_shuffle_config(config: ChunkShuffleConfig) -> ChunkShuffleConfig:
    chunk_size = config.chunk_size
    k = config.k
    row_order = config.row_order
    if chunk_size <= 0:
        msg = f"[InvalidLanceChunkShuffle] chunk_size must be > 0, got {chunk_size}"
        raise ValueError(msg)
    if k <= 0:
        msg = f"[InvalidLanceChunkShuffle] k must be > 0, got {k}"
        raise ValueError(msg)
    if row_order not in ("permuted", "sequential"):
        msg = f"[InvalidLanceChunkShuffle] row_order must be permuted or sequential, got {row_order!r}"
        raise ValueError(msg)
    return config


def build_lance_index_order(
    source: LanceSource,
    context: RuntimeContext,
    shuffle_mode: LanceShuffleMode,
    *,
    chunk_config: ChunkShuffleConfig | None = None,
) -> LanceIndexOrder:
    """Build the index order implementation for a Lance shuffle mode."""
    if shuffle_mode == "none":
        return SequentialIndexOrder(source=source, context=context)
    if shuffle_mode == "global":
        return GlobalShuffleIndexOrder(source=source, context=context)
    if shuffle_mode == "chunk":
        config = resolve_chunk_shuffle_config(chunk_config)
        return ChunkShuffleIndexOrder(source=source, context=context, config=config)

    msg = f"[InvalidLanceShuffleMode] expected none, global, or chunk, got {shuffle_mode!r}"
    raise ValueError(msg)


def map_global_indexes(source: LanceSource, global_indexes: Sequence[int]) -> list[LanceIndexItem]:
    """Map global row indexes to dataset-local Lance index items."""
    if not global_indexes:
        return []

    dataset_indices = (
        np.searchsorted(
            np.asarray(source.row_offsets, dtype=np.int64),
            np.asarray(global_indexes, dtype=np.int64),
            side="right",
        )
        - 1
    )
    items: list[LanceIndexItem] = []
    for global_index, dataset_i in zip(global_indexes, dataset_indices, strict=True):
        dataset_index = int(dataset_i)
        dataset = source.datasets[dataset_index]
        local_index = int(global_index - dataset.row_offset)
        items.append(
            LanceIndexItem(
                dataset_i=dataset_index,
                local_index=local_index,
                global_index=int(global_index),
            )
        )
    return items


def _map_sorted_global_indexes(source: LanceSource, global_indexes: Sequence[int]) -> list[LanceIndexItem]:
    """Map sorted global indexes with one forward dataset scan."""
    if not global_indexes:
        return []

    dataset_i = bisect.bisect_right(source.row_offsets, global_indexes[0]) - 1
    items: list[LanceIndexItem] = []
    for global_index in global_indexes:
        while dataset_i + 1 < len(source.datasets) and global_index >= source.row_offsets[dataset_i + 1]:
            dataset_i += 1
        dataset = source.datasets[dataset_i]
        items.append(
            LanceIndexItem(
                dataset_i=dataset_i,
                local_index=int(global_index - dataset.row_offset),
                global_index=int(global_index),
            )
        )
    return items


class LanceIndexOrder(ABC):
    """Map logical source positions to Lance row indexes in batches."""

    def __init__(self, *, source: LanceSource, context: RuntimeContext) -> None:
        self.source = source
        self.context = context

    def round_size(self, round_index: int) -> int:
        """Return item count assigned to this runtime slot in one round."""
        _ = round_index
        if self.context.slot >= self.source.total_rows:
            return 0
        return ((self.source.total_rows - 1 - self.context.slot) // self.context.total_slots) + 1

    @abstractmethod
    def items(self, round_index: int, start: int, count: int) -> list[LanceIndexItem]:
        """Return physical Lance indexes for a logical position range."""

    def _global_positions(self, start: int, count: int) -> list[int]:
        return [self.context.slot + position * self.context.total_slots for position in range(start, start + count)]


class SequentialIndexOrder(LanceIndexOrder):
    """Non-shuffled deterministic Lance order."""

    def items(self, round_index: int, start: int, count: int) -> list[LanceIndexItem]:
        """Return sequential Lance indexes for a logical position range."""
        _ = round_index
        return _map_sorted_global_indexes(self.source, self._global_positions(start, count))


class GlobalShuffleIndexOrder(LanceIndexOrder):
    """Deterministic global Lance shuffle."""

    def items(self, round_index: int, start: int, count: int) -> list[LanceIndexItem]:
        """Return globally shuffled Lance indexes for a logical position range."""
        global_indexes = [
            permute_index(position, total_rows=self.source.total_rows, seed=self.context.seed + round_index)
            for position in self._global_positions(start, count)
        ]
        return map_global_indexes(self.source, global_indexes)


class ChunkShuffleIndexOrder(LanceIndexOrder):
    """Deterministic chunk-window Lance shuffle."""

    def __init__(self, *, source: LanceSource, context: RuntimeContext, config: ChunkShuffleConfig) -> None:
        super().__init__(source=source, context=context)
        self.config = config
        self._round_index: int | None = None
        self._round_order: _ChunkRoundOrder | None = None

    def items(self, round_index: int, start: int, count: int) -> list[LanceIndexItem]:
        """Return chunk-shuffled Lance indexes for a logical position range."""
        order = self._order_for_round(round_index)
        global_indexes = [
            self._global_index_at(position, round_index=round_index, order=order)
            for position in self._global_positions(start, count)
        ]
        return map_global_indexes(self.source, global_indexes)

    def _order_for_round(self, round_index: int) -> _ChunkRoundOrder:
        if self._round_index == round_index and self._round_order is not None:
            return self._round_order

        chunk_size = self.config.chunk_size
        k = self.config.k
        if self.source.total_rows <= 0:
            order = _ChunkRoundOrder(
                chunk_size=chunk_size,
                k=k,
                chunk_order=(),
                chunk_offsets=(0,),
                window_offsets=(0,),
            )
        else:
            num_chunks = (self.source.total_rows + chunk_size - 1) // chunk_size
            rng = np.random.default_rng(_mix_seed(self.context.seed, round_index, 0))
            chunk_order = tuple(int(chunk_id) for chunk_id in rng.permutation(num_chunks))
            chunk_offsets = [0]
            for chunk_id in chunk_order:
                chunk_offsets.append(chunk_offsets[-1] + self._chunk_row_count(chunk_id))
            window_offsets = [0]
            for window_start in range(0, num_chunks, k):
                window_size = sum(
                    self._chunk_row_count(chunk_order[order_i])
                    for order_i in range(window_start, min(window_start + k, num_chunks))
                )
                window_offsets.append(window_offsets[-1] + window_size)
            order = _ChunkRoundOrder(
                chunk_size=chunk_size,
                k=k,
                chunk_order=chunk_order,
                chunk_offsets=tuple(chunk_offsets),
                window_offsets=tuple(window_offsets),
            )

        self._round_index = round_index
        self._round_order = order
        return order

    def _global_index_at(self, global_position: int, *, round_index: int, order: _ChunkRoundOrder) -> int:
        if global_position < 0 or global_position >= self.source.total_rows:
            msg = "[InvalidLanceChunkShuffle] global position is out of range"
            raise ValueError(msg)
        if self.config.row_order == "sequential":
            order_i = bisect.bisect_right(order.chunk_offsets, global_position) - 1
            row_offset = global_position - order.chunk_offsets[order_i]
            chunk_id = order.chunk_order[order_i]
            return chunk_id * order.chunk_size + row_offset

        window_i = bisect.bisect_right(order.window_offsets, global_position) - 1
        window_start_offset = order.window_offsets[window_i]
        window_size = order.window_offsets[window_i + 1] - window_start_offset
        local_position = global_position - window_start_offset
        shuffled_position = permute_index(
            local_position,
            total_rows=window_size,
            seed=_mix_seed(self.context.seed, round_index, 1, window_i),
        )

        chunk_order_start = window_i * order.k
        chunk_order_stop = min(chunk_order_start + order.k, len(order.chunk_order))
        for order_i in range(chunk_order_start, chunk_order_stop):
            chunk_id = order.chunk_order[order_i]
            row_count = self._chunk_row_count(chunk_id)
            if shuffled_position < row_count:
                return chunk_id * order.chunk_size + shuffled_position
            shuffled_position -= row_count

        msg = "[InvalidLanceChunkShuffle] failed to map chunk-shuffled position"
        raise RuntimeError(msg)

    def _chunk_row_count(self, chunk_id: int) -> int:
        start = chunk_id * self.config.chunk_size
        if start >= self.source.total_rows:
            return 0
        return min(self.config.chunk_size, self.source.total_rows - start)


def permute_index(position: int, *, total_rows: int, seed: int) -> int:
    """Return a deterministic permutation value for one index."""
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


def _mix_seed(*parts: int) -> int:
    seed = 0xCBF29CE484222325
    for part in parts:
        seed ^= int(part) & _MASK64
        seed = (seed * 0x100000001B3) & _MASK64
    return seed


def _mix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return value ^ (value >> 31)


def _feistel(value: int, *, bits: int, seed: int) -> int:
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
