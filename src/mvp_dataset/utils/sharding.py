"""Distributed shard splitting utilities."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from typing import cast

from ..core.types import RuntimeContext


def iter_items[T](
    items: Sequence[T], context: RuntimeContext, resample: bool = False, grouped: bool = False
) -> Iterator[T]:
    """Yield slot-assigned items in deterministic rounds.

    Args:
        items: Input items to be sharded across all global slots.
        context: Runtime context providing slot and total slot counts.
        resample: If ``True``, emit infinite rounds with reshuffle each round.
        grouped: If ``True``, each item is expected to be a mutable group
            (typically ``list[Sample]``). Group contents are shuffled first,
            then groups are shuffled, flattened, and re-chunked.
    """

    ordered_base = list(items)
    if len(ordered_base) == 0:
        msg = (
            "[InsufficientItemsForSlot] "
            f"items=0 total_slots={context.total_slots} slot={context.slot}"
        )
        raise ValueError(msg)

    slot = context.slot
    total_slots = context.total_slots

    def _item_generator(max_round: int = -1) -> Iterator[T]:
        round_index = 0

        while True:
            ordered = list(ordered_base)
            rng = random.Random(context.seed + round_index)
            if grouped:
                ordered_groups = cast(list[list[object]], ordered)
                for group in ordered_groups:
                    rng.shuffle(group)
            rng.shuffle(ordered)

            if grouped:
                ordered_groups = cast(list[list[object]], ordered)
                flatten_ordered: list[object] = []
                for group in ordered_groups:
                    flatten_ordered.extend(group)
                if not flatten_ordered:
                    round_index += 1
                    if 0 <= max_round <= round_index:
                        break
                    continue

                # Regroup by slot to keep round-level ordering stable.
                regrouped: list[list[object]] = []
                per_slot_n = (len(flatten_ordered) + total_slots - 1) // total_slots
                for i in range(0, len(flatten_ordered), per_slot_n):
                    regrouped.append(flatten_ordered[i : i + per_slot_n])
                ordered = cast(list[T], regrouped)

            yield from ordered
            round_index += 1
            if 0 <= max_round <= round_index:
                break

    i = 0
    for item in _item_generator(-1 if resample else 1):
        if i % total_slots == slot:
            yield item
        i += 1
