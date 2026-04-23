"""Runtime execution context for distributed and multi-worker data loading."""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace

from .mesh import DataLoadMesh, resolve_data_load_mesh


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

    For hybrid parallelism (e.g. TP + FSDP + DDP), set :attr:`mesh` to a
    :class:`DataLoadMesh` that identifies the data-parallel dimensions.  When a
    mesh is present, :attr:`slot` and :attr:`total_slots` are derived from the
    mesh's ``dp_rank`` / ``dp_size`` instead of the global ``rank`` /
    ``world_size``, so that model-parallel co-members (e.g. TP ranks) receive
    identical shards while data-parallel ranks receive distinct ones.
    """

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    local_world_size: int = 1
    node_rank: int = 0
    num_nodes: int = 1
    worker_id: int = 0
    num_workers: int = 1
    epoch: int = 0  # TODO: check where we need this?
    seed: int = 0
    mesh: DataLoadMesh | None = None

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
        if self.local_world_size <= 0:
            msg = f"local_world_size must be > 0, got {self.local_world_size}"
            raise ValueError(msg)
        if self.local_rank < 0 or self.local_rank >= self.local_world_size:
            msg = f"local_rank must be in [0, {self.local_world_size}), got {self.local_rank}"
            raise ValueError(msg)
        if self.num_nodes <= 0:
            msg = f"num_nodes must be > 0, got {self.num_nodes}"
            raise ValueError(msg)
        if self.node_rank < 0 or self.node_rank >= self.num_nodes:
            msg = f"node_rank must be in [0, {self.num_nodes}), got {self.node_rank}"
            raise ValueError(msg)
        if self.worker_id < 0 or self.worker_id >= self.num_workers:
            msg = f"worker_id must be in [0, {self.num_workers}), got {self.worker_id}"
            raise ValueError(msg)

    def __hash__(self) -> int:
        # Exclude mesh: DeviceMesh objects may not be hashable and the numeric
        # fields already uniquely identify the context for caching purposes.
        return hash(
            (
                self.rank,
                self.world_size,
                self.local_rank,
                self.local_world_size,
                self.node_rank,
                self.num_nodes,
                self.worker_id,
                self.num_workers,
                self.epoch,
                self.seed,
            )
        )

    @property
    def slot(self) -> int:
        """Return the global worker slot used for data parallel sharding.

        When a :class:`DataLoadMesh` is set, the slot is computed from the
        mesh's ``dp_rank`` so that model-parallel co-members share the same slot
        and therefore receive the same shards.
        """

        dp_rank = self.mesh.dp_rank if self.mesh is not None else self.rank
        return dp_rank * self.num_workers + self.worker_id

    @property
    def total_slots(self) -> int:
        """Return the total number of global slots across all nodes/workers.

        When a :class:`DataLoadMesh` is set, uses the mesh's ``dp_size`` instead
        of the global ``world_size``.
        """

        dp_size = self.mesh.dp_size if self.mesh is not None else self.world_size
        return dp_size * self.num_workers

    @property
    def sample_shuffle_seed(self) -> int:
        """Return the deterministic seed for sample-level shuffle."""

        return self.seed + self.slot

    @classmethod
    def from_runtime(
        cls,
        *,
        base: RuntimeContext | None = None,
        seed: int = 0,
        env: Mapping[str, str] | None = None,
        prefer_torch: bool = True,
        mesh: DataLoadMesh | None = None,
        device_mesh: object | None = None,
        dp_dims: str | Sequence[str] | None = None,
    ) -> RuntimeContext:
        """Build a context from runtime sources (torch > env > defaults).

        When *base* is provided, ``seed``, ``epoch``, and ``mesh`` are
        inherited from it unless explicitly overridden.  This replaces the
        former ``resolve_current_process`` pattern: call
        ``RuntimeContext.from_runtime(base=ctx)`` inside a worker process to
        re-read live ``rank``/``world_size``/``worker_id``/``num_workers``
        while preserving the caller's seed, epoch, and mesh.

        Args:
            base: Optional existing context to inherit ``seed``, ``epoch``,
                and ``mesh`` from.
            seed: Base random seed. Ignored when *base* is provided.
            env: Optional environment mapping to read launcher values from.
            prefer_torch: Whether to prefer live PyTorch runtime values when
                available.
            mesh: Mesh override. Falls back to ``base.mesh`` when *base* is
                set and no mesh arguments are given.
            device_mesh: Optional PyTorch ``DeviceMesh`` to wrap as a
                :class:`DataLoadMesh`.
            dp_dims: Data-parallel mesh dimensions used with *device_mesh*.
        """

        resolved_mesh = resolve_data_load_mesh(mesh=mesh, device_mesh=device_mesh, dp_dims=dp_dims)
        if base is not None:
            seed = base.seed
            if resolved_mesh is None:
                resolved_mesh = base.mesh

        source = os.environ if env is None else env
        rank = int(source.get("RANK", "0"))
        world_size = int(source.get("WORLD_SIZE", "1"))
        local_rank = int(source.get("LOCAL_RANK", str(rank)))
        local_world_size = int(source.get("LOCAL_WORLD_SIZE", str(world_size)))
        node_rank = int(
            source.get(
                "NODE_RANK",
                str(rank // local_world_size if local_world_size > 0 else 0),
            )
        )
        worker_id = int(source.get("WORKER", "0"))
        num_workers = int(source.get("NUM_WORKERS", "1"))

        if prefer_torch:
            torch_rank, torch_world_size, torch_worker_id, torch_num_workers = _read_torch_runtime_values()
            if torch_rank is not None:
                rank = torch_rank
            if torch_world_size is not None:
                world_size = torch_world_size
            if torch_worker_id is not None:
                worker_id = torch_worker_id
            if torch_num_workers is not None:
                num_workers = torch_num_workers

        result = cls(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            local_world_size=local_world_size,
            node_rank=node_rank,
            num_nodes=max(1, (world_size + local_world_size - 1) // local_world_size),
            worker_id=worker_id,
            num_workers=num_workers,
            seed=seed,
            mesh=resolved_mesh,
        )
        if base is not None:
            result = dataclass_replace(result, epoch=base.epoch)
        return result
