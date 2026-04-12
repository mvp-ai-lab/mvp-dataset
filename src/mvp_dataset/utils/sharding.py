"""Distributed shard splitting utilities."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence

from ..core.context import RuntimeContext
from ..log import get_logger


def select_items_for_slots[T](
    items: Sequence[T],
    *,
    slot_ids: Sequence[int],
    total_slots: int,
    seed: int,
) -> list[T]:
    """Select the deterministic one-round item assignment for a set of slots."""
    ordered = list(items)
    if len(ordered) == 0:
        return []
    if total_slots <= 0:
        msg = f"[InvalidSlotCount] total_slots must be > 0, got {total_slots}"
        raise ValueError(msg)

    slot_id_set = set(slot_ids)
    if not slot_id_set:
        return []

    rng = random.Random(seed)
    rng.shuffle(ordered)
    return [item for index, item in enumerate(ordered) if index % total_slots in slot_id_set]


def iter_items[T](items: Sequence[T], context: RuntimeContext, resample: bool = False) -> Iterator[T]:
    """Yield slot-assigned items in deterministic rounds.

    Args:
        items: Input items to be sharded across all global slots.
        context: Runtime context providing slot and total slot counts.
        resample: If ``True``, emit infinite rounds with reshuffle each round.
    """

    ordered_base = list(items)
    if len(ordered_base) == 0:
        msg = f"[InsufficientItemsForSlot] items=0 total_slots={context.total_slots} slot={context.slot}"
        raise ValueError(msg)

    slot = context.slot
    total_slots = context.total_slots

    def _rounds() -> Iterator[T]:
        round_index = 0
        while resample or round_index == 0:
            ordered = list(ordered_base)
            rng = random.Random(context.seed + round_index)
            rng.shuffle(ordered)
            yield from ordered
            round_index += 1

    for i, item in enumerate(_rounds()):
        if i % total_slots == slot:
            get_logger().debug(
                "<MVP Dataset - rank %d> yielding %s slot=%d total_slots=%d seed=%d",
                context.rank,
                item,
                slot,
                total_slots,
                context.seed,
            )
            yield item
