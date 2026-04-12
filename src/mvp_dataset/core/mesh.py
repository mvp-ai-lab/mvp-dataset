"""Data-parallel mesh specification for distributed data loading."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


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

    @property
    def is_dp_leader(self) -> bool:
        """Return whether the local rank is the designated writer for its DP slot.

        The leader is defined as the rank whose non-data-parallel mesh
        coordinates are all zero. This selects exactly one representative from
        each model-parallel replica group so TP co-members do not duplicate
        cache writes.
        """

        dim_names = getattr(self.device_mesh, "mesh_dim_names", None)
        if dim_names is None:
            return True

        non_dp_dims = [dim for dim in dim_names if dim not in self.dp_dims]
        return all(self.device_mesh.get_local_rank(dim) == 0 for dim in non_dp_dims)


def _normalize_dp_dims(dp_dims: str | Sequence[str]) -> tuple[str, ...]:
    """Normalize one-or-many DP mesh dimension names into a tuple."""

    dims = (dp_dims,) if isinstance(dp_dims, str) else tuple(dp_dims)
    if not dims:
        msg = "dp_dims must not be empty"
        raise ValueError(msg)
    return dims


def resolve_data_load_mesh(
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
