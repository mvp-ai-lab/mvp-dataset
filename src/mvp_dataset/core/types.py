"""Core data types used by mvp-dataset."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Shared aliases
# ---------------------------------------------------------------------------

PathLikeStr = str | os.PathLike[str]
"""Path representation accepted by public APIs."""

Sample = dict[str, object]
"""In-memory sample mapping exchanged across pipeline stages."""

GroupedSample = list[Sample]
"""A group of samples sharing the same grouping key."""

ShardInput = PathLikeStr
"""Shard path input accepted by dataset constructors."""

PathResolver = Callable[[PathLikeStr], PathLikeStr]
"""Callable that maps a main shard path to a related shard path."""

SidecarSpec = tuple[str, PathResolver]
"""Sidecar join specification as ``(name, path_resolver)``."""

RefFieldSpec = tuple[str, PathLikeStr]
"""Reference resolver spec as ``(field_name, base_dir)``."""

Stage = Callable[[Iterable[object]], Iterable[object]]
"""One lazy transformation stage in the iterator pipeline."""

def _read_torch_runtime_values() -> tuple[int | None, int | None, int | None, int | None]:
    """Read rank/world_size/worker_id/num_workers from PyTorch runtime if available."""

    rank: int | None = None
    world_size: int | None = None
    worker_id: int | None = None
    num_workers: int | None = None

    try:
        torch_dist = importlib.import_module("torch.distributed")
        if torch_dist.is_available() and torch_dist.is_initialized():
            rank = int(torch_dist.get_rank())
            world_size = int(torch_dist.get_world_size())
    except ModuleNotFoundError:
        pass

    try:
        torch_utils_data = importlib.import_module("torch.utils.data")
        worker_info = torch_utils_data.get_worker_info()
        if worker_info is not None:
            worker_id = int(worker_info.id)
            num_workers = int(worker_info.num_workers)
    except ModuleNotFoundError:
        pass

    return rank, world_size, worker_id, num_workers


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

        return self.seed + self.slot

    @classmethod
    def from_runtime(
        cls,
        *,
        seed: int = 0,
        env: Mapping[str, str] | None = None,
        prefer_torch: bool = True,
    ) -> RuntimeContext:
        """Build a context from runtime sources (torch > env > defaults)."""

        source = os.environ if env is None else env
        rank = int(source.get("RANK", "0"))
        world_size = int(source.get("WORLD_SIZE", "1"))
        worker_id = int(source.get("WORKER", "0"))
        num_workers = int(source.get("NUM_WORKERS", "1"))

        if prefer_torch:
            torch_rank, torch_world_size, torch_worker_id, torch_num_workers = (
                _read_torch_runtime_values()
            )
            if torch_rank is not None:
                rank = torch_rank
            if torch_world_size is not None:
                world_size = torch_world_size
            if torch_worker_id is not None:
                worker_id = torch_worker_id
            if torch_num_workers is not None:
                num_workers = torch_num_workers

        return cls(
            rank=rank,
            world_size=world_size,
            worker_id=worker_id,
            num_workers=num_workers,
            seed=seed,
        )

    def resolve_current_process(self) -> RuntimeContext:
        """Return a new context with any dynamic values resolved for the current process.

        This is useful when the context is created in a parent process and needs to be
        resolved separately in each worker process to get correct worker_id/num_workers.
        """

        return self.from_runtime(seed=self.seed)
