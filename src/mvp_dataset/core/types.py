"""Core data types used by mvp-dataset."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol
from dataclasses import replace as dataclass_replace

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


class Assembler[T, U](Protocol):
    """Stateful stream assembler that may emit outputs after consuming inputs."""

    def push(self, sample: T) -> Iterable[U]:
        """Consume one upstream sample and yield any completed outputs."""

    def finish(self, *, drop_last: bool = False) -> Iterable[U]:
        """Flush remaining state at end of stream."""
@dataclass(frozen=True, slots=True)
class DataLoadMesh:
    """Data-parallel sub-mesh specification for shard assignment.

    Wraps a PyTorch ``DeviceMesh`` and the names of its **data-parallel**
    dimensions. Ranks that share the same :attr:`dp_rank` receive identical
    data shards during iteration.

    In a typical 3-D training setup with ``(replicate, shard, tensor)``
    dimensions:

    * ``replicate`` and ``shard`` are data-parallel (each rank processes
      different data) → both should appear in *dp_dims*.
    * ``tensor`` is model-parallel (all TP co-members process the same data)
      → should be omitted from *dp_dims*.

    Args:
        device_mesh: A ``torch.distributed.DeviceMesh`` instance.
        dp_dims: Names of the data-parallel dimensions within the mesh.
    """

    device_mesh: object
    dp_dims: tuple[str, ...]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DataLoadMesh):
            return NotImplemented
        return self.device_mesh is other.device_mesh and self.dp_dims == other.dp_dims

    def __hash__(self) -> int:
        return hash((id(self.device_mesh), self.dp_dims))

    @property
    def dp_rank(self) -> int:
        """Flattened rank across all data-parallel dimensions."""

        sizes = [self.device_mesh.size(dim) for dim in self.dp_dims]
        local_ranks = [self.device_mesh.get_local_rank(dim) for dim in self.dp_dims]
        rank = 0
        for i, lr in enumerate(local_ranks):
            stride = 1
            for j in range(i + 1, len(sizes)):
                stride *= sizes[j]
            rank += lr * stride
        return rank

    @property
    def dp_size(self) -> int:
        """Product of all data-parallel dimension sizes."""

        size = 1
        for dim in self.dp_dims:
            size *= self.device_mesh.size(dim)
        return size


def _normalize_dp_dims(dp_dims: str | Sequence[str]) -> tuple[str, ...]:
    """Normalize one-or-many DP mesh dimension names into a tuple."""

    dims = (dp_dims,) if isinstance(dp_dims, str) else tuple(dp_dims)
    if not dims:
        msg = "dp_dims must not be empty"
        raise ValueError(msg)
    return dims


def _resolve_data_load_mesh(
    *,
    mesh: DataLoadMesh | None = None,
    device_mesh: object | None = None,
    dp_dims: str | Sequence[str] | None = None,
) -> DataLoadMesh | None:
    """Resolve a mesh specification into a :class:`DataLoadMesh`."""

    if mesh is not None and (device_mesh is not None or dp_dims is not None):
        msg = "pass either mesh or device_mesh/dp_dims, not both"
        raise ValueError(msg)
    if mesh is not None:
        return mesh
    if device_mesh is None and dp_dims is None:
        return None
    if device_mesh is None or dp_dims is None:
        msg = "device_mesh and dp_dims must be provided together"
        raise ValueError(msg)
    return DataLoadMesh(device_mesh=device_mesh, dp_dims=_normalize_dp_dims(dp_dims))


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
    worker_id: int = 0
    num_workers: int = 1
    epoch: int = 0
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
        if self.worker_id < 0 or self.worker_id >= self.num_workers:
            msg = f"worker_id must be in [0, {self.num_workers}), got {self.worker_id}"
            raise ValueError(msg)

    def __hash__(self) -> int:
        # Exclude mesh: DeviceMesh objects may not be hashable and the numeric
        # fields already uniquely identify the context for caching purposes.
        return hash((self.rank, self.world_size, self.worker_id, self.num_workers, self.epoch, self.seed))

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
        seed: int = 0,
        env: Mapping[str, str] | None = None,
        prefer_torch: bool = True,
        mesh: DataLoadMesh | None = None,
        device_mesh: object | None = None,
        dp_dims: str | Sequence[str] | None = None,
    ) -> RuntimeContext:
        """Build a context from runtime sources (torch > env > defaults).

        Args:
            seed: Base random seed used for deterministic shuffling.
            env: Optional environment mapping to read launcher values from.
            prefer_torch: Whether to prefer live PyTorch runtime values when
                available.
            mesh: Optional prebuilt data-loading mesh to attach.
            device_mesh: Optional PyTorch ``DeviceMesh`` to wrap as a
                :class:`DataLoadMesh`.
            dp_dims: Data-parallel mesh dimensions used with *device_mesh*.
        """

        source = os.environ if env is None else env
        rank = int(source.get("RANK", "0"))
        world_size = int(source.get("WORLD_SIZE", "1"))
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

        return cls(
            rank=rank,
            world_size=world_size,
            worker_id=worker_id,
            num_workers=num_workers,
            seed=seed,
            mesh=_resolve_data_load_mesh(mesh=mesh, device_mesh=device_mesh, dp_dims=dp_dims),
        )

    def resolve_current_process(
        self,
        *,
        env: Mapping[str, str] | None = None,
        prefer_torch: bool = True,
        mesh: DataLoadMesh | None = None,
        device_mesh: object | None = None,
        dp_dims: str | Sequence[str] | None = None,
    ) -> RuntimeContext:
        """Return a new context with any dynamic values resolved for the current process.

        This is useful when the context is created in a parent process and needs to be
        resolved separately in each worker process to get correct worker_id/num_workers.
        The context keeps its existing :attr:`epoch` and, by default, its mesh.
        Pass *mesh* or *device_mesh*/*dp_dims* to override the attached mesh.
        """

        resolved_mesh = _resolve_data_load_mesh(mesh=mesh, device_mesh=device_mesh, dp_dims=dp_dims)
        if resolved_mesh is None:
            resolved_mesh = self.mesh
        resolved = self.from_runtime(
            seed=self.seed,
            env=env,
            prefer_torch=prefer_torch,
            mesh=resolved_mesh,
        )
        return dataclass_replace(resolved, epoch=self.epoch)
