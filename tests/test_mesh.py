from __future__ import annotations

from dataclasses import dataclass

from mvp_dataset.core.mesh import DataLoadMesh


@dataclass(frozen=True, slots=True)
class StringOnlyDeviceMesh:
    dp_size: int = 2
    tp_size: int = 3
    rank: int = 4

    mesh_dim_names = ("dp", "tp")

    def size(self, dim: str) -> int:
        if dim == "dp":
            return self.dp_size
        if dim == "tp":
            return self.tp_size
        raise KeyError(dim)

    def get_local_rank(self, dim: str) -> int:
        if dim == "dp":
            return self.rank // self.tp_size
        if dim == "tp":
            return self.rank % self.tp_size
        raise KeyError(dim)


def test_data_load_mesh_preserves_string_dims_for_duck_typed_meshes() -> None:
    mesh = DataLoadMesh(device_mesh=StringOnlyDeviceMesh(), dp_dims=("dp",))

    assert mesh.dp_size == 2
    assert mesh.dp_rank == 1
    assert not mesh.is_dp_leader
