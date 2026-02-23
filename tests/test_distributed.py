"""Tests for distributed shard splitting behavior."""

from __future__ import annotations

from mm_loader.core import RuntimeContext
from mm_loader.distributed import split_shards


def _snapshot_assignments(shards: list[str], *, epoch: int, seed: int) -> list[list[str]]:
    assignments: list[list[str]] = []
    for rank in range(2):
        for worker_id in range(2):
            context = RuntimeContext(
                rank=rank,
                world_size=2,
                worker_id=worker_id,
                num_workers=2,
                epoch=epoch,
                seed=seed,
            )
            assignments.append(split_shards(shards, context))
    return assignments


def test_split_shards_is_disjoint_and_complete() -> None:
    shards = [f"shard-{index:03d}.tar" for index in range(16)]
    assignments = _snapshot_assignments(shards, epoch=1, seed=7)

    for left_index, left in enumerate(assignments):
        for right in assignments[left_index + 1 :]:
            assert set(left).isdisjoint(right)

    merged = set().union(*(set(part) for part in assignments))
    assert merged == set(shards)


def test_split_shards_is_reproducible_and_epoch_changes_order() -> None:
    shards = [f"shard-{index:03d}.tar" for index in range(16)]

    first = _snapshot_assignments(shards, epoch=3, seed=11)
    second = _snapshot_assignments(shards, epoch=3, seed=11)
    third = _snapshot_assignments(shards, epoch=4, seed=11)

    assert first == second
    assert first != third
