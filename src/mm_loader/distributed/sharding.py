"""Distributed shard splitting utilities."""

from __future__ import annotations

import random
from collections.abc import Sequence

from ..core.types import RuntimeContext


def split_items[T](items: Sequence[T], context: RuntimeContext) -> list[T]:
    """Split any sequence for data-parallel loading.

    The algorithm is fixed and deterministic:
    1. Shuffle all items using ``seed + epoch``.
    2. Assign each shuffled shard by round-robin over global slots.

    Slot formula:
    ``slot = rank * num_workers + worker_id``

    Selection formula:
    ``index % (world_size * num_workers) == slot``
    """

    ordered = list(items)
    rng = random.Random(context.seed + context.epoch)
    rng.shuffle(ordered)

    slot = context.slot
    total_slots = context.total_slots
    return [item for index, item in enumerate(ordered) if index % total_slots == slot]


def split_shards(shards: Sequence[str], context: RuntimeContext) -> list[str]:
    """Split shard paths for data-parallel loading."""

    return split_items(shards, context)
