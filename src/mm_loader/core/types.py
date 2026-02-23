"""Core data types used by mm-loader."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

Sample = dict[str, object]


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    """Execution context for distributed and multi-worker data loading.

    The context is intentionally framework-agnostic and can be created directly or
    derived from common environment variables used by distributed training launchers.
    """

    rank: int = 0
    world_size: int = 1
    worker_id: int = 0
    num_workers: int = 1
    epoch: int = 0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            msg = f"world_size must be > 0, got {self.world_size}"
            raise ValueError(msg)
        if self.num_workers <= 0:
            msg = f"num_workers must be > 0, got {self.num_workers}"
            raise ValueError(msg)
        if self.rank < 0 or self.rank >= self.world_size:
            msg = f"rank must be in [0, {self.world_size}), got {self.rank}"
            raise ValueError(msg)
        if self.worker_id < 0 or self.worker_id >= self.num_workers:
            msg = f"worker_id must be in [0, {self.num_workers}), got {self.worker_id}"
            raise ValueError(msg)
        if self.epoch < 0:
            msg = f"epoch must be >= 0, got {self.epoch}"
            raise ValueError(msg)

    @property
    def slot(self) -> int:
        """Return the global worker slot used for data parallel sharding."""

        return self.rank * self.num_workers + self.worker_id

    @property
    def total_slots(self) -> int:
        """Return the total number of global slots across all nodes/workers."""

        return self.world_size * self.num_workers

    @property
    def sample_shuffle_seed(self) -> int:
        """Return the deterministic seed for sample-level shuffle."""

        return self.seed + self.epoch + self.slot

    @classmethod
    def from_env(
        cls,
        *,
        epoch: int = 0,
        seed: int = 0,
        env: Mapping[str, str] | None = None,
    ) -> RuntimeContext:
        """Build a context from common environment variables.

        Environment variables:
        - ``RANK`` and ``WORLD_SIZE`` for multi-node rank info.
        - ``WORKER`` and ``NUM_WORKERS`` for worker info.

        Missing variables fall back to single-process defaults.
        """

        source = os.environ if env is None else env
        rank = int(source.get("RANK", "0"))
        world_size = int(source.get("WORLD_SIZE", "1"))
        worker_id = int(source.get("WORKER", "0"))
        num_workers = int(source.get("NUM_WORKERS", "1"))
        return cls(
            rank=rank,
            world_size=world_size,
            worker_id=worker_id,
            num_workers=num_workers,
            epoch=epoch,
            seed=seed,
        )
