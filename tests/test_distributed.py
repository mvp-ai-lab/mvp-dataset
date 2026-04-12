from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from mvp_dataset import Dataset, RuntimeContext, TorchLoader

from .helpers import build_records, write_tar_shards

if importlib.util.find_spec("torch") is not None:
    import torch.distributed as dist
    import torch.multiprocessing as mp
else:
    dist = None
    mp = None

pytestmark = pytest.mark.skipif(dist is None or mp is None, reason="torch is not installed")


@dataclass(frozen=True, slots=True)
class FakeDeviceMesh:
    dp_size: int
    tp_size: int
    rank: int

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


def _distributed_reader_worker(
    rank: int,
    world_size: int,
    shards: list[str],
    init_file: str,
    output_dir: str,
) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        dataset = Dataset.from_source("tars", shards=shards)
        ids = [sample["id"].decode("utf-8") for sample in dataset]
        output_path = Path(output_dir) / f"rank_{rank}.json"
        output_path.write_text(json.dumps(ids), encoding="utf-8")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _identity_collate(batch: list[object]) -> list[object]:
    return batch


def _tp_group_reader_worker(
    rank: int,
    world_size: int,
    shards: list[str],
    init_file: str,
    output_dir: str,
) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        mesh = FakeDeviceMesh(dp_size=2, tp_size=2, rank=rank)
        context = RuntimeContext.from_runtime(device_mesh=mesh, dp_dims=("dp",))
        dataset = Dataset.from_source("tars", shards=shards, context=context)
        ids = [sample["id"].decode("utf-8") for sample in dataset]
        output_path = Path(output_dir) / f"rank_{rank}.json"
        output_path.write_text(json.dumps(ids), encoding="utf-8")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _tp_group_loader_shuffle_reader_worker(
    rank: int,
    world_size: int,
    shards: list[str],
    init_file: str,
    output_dir: str,
) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        mesh = FakeDeviceMesh(dp_size=2, tp_size=2, rank=rank)
        context = RuntimeContext.from_runtime(device_mesh=mesh, dp_dims=("dp",), seed=23)
        dataset = Dataset.from_source("tars", shards=shards, context=context)
        loader = (
            TorchLoader(
                dataset,
                num_workers=0,
                batch_size=2,
                collate_fn=_identity_collate,
            )
            .unbatch()
            .shuffle(buffer_size=3)
        )
        ids = [sample["id"].decode("utf-8") for sample in loader]
        output_path = Path(output_dir) / f"rank_{rank}.json"
        output_path.write_text(json.dumps(ids), encoding="utf-8")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _tp_group_multiworker_reader_worker(
    rank: int,
    world_size: int,
    shards: list[str],
    init_file: str,
    output_dir: str,
) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        mesh = FakeDeviceMesh(dp_size=2, tp_size=2, rank=rank)
        context = RuntimeContext.from_runtime(device_mesh=mesh, dp_dims=("dp",), seed=31)
        dataset = Dataset.from_source("tars", shards=shards, context=context).shuffle(buffer_size=2)
        loader = TorchLoader(dataset, num_workers=2, batch_size=None)
        ids = [sample["id"].decode("utf-8") for sample in loader]
        output_path = Path(output_dir) / f"rank_{rank}.json"
        output_path.write_text(json.dumps(ids), encoding="utf-8")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _tp_group_shuffle_reader_worker(
    rank: int,
    world_size: int,
    shards: list[str],
    init_file: str,
    output_dir: str,
) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        mesh = FakeDeviceMesh(dp_size=2, tp_size=2, rank=rank)
        context = RuntimeContext.from_runtime(device_mesh=mesh, dp_dims=("dp",), seed=17)
        dataset = Dataset.from_source("tars", shards=shards, context=context).shuffle(buffer_size=3)
        ids = [sample["id"].decode("utf-8") for sample in dataset]
        output_path = Path(output_dir) / f"rank_{rank}.json"
        output_path.write_text(json.dumps(ids), encoding="utf-8")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _tp_group_cache_reader_worker(
    rank: int,
    world_size: int,
    shards: list[str],
    init_file: str,
    cache_dir: str,
    output_dir: str,
) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        mesh = FakeDeviceMesh(dp_size=2, tp_size=2, rank=rank)
        context = RuntimeContext.from_runtime(device_mesh=mesh, dp_dims=("dp",), seed=29)
        dataset = Dataset.from_source("tars", shards=shards, context=context).cache(cache_dir=cache_dir)
        ids = [sample["id"].decode("utf-8") for sample in dataset]
        output_path = Path(output_dir) / f"rank_{rank}.json"
        output_path.write_text(json.dumps(ids), encoding="utf-8")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_torch_distributed_reading_shards_samples_by_rank(tmp_path) -> None:
    records = build_records(count=4)
    shards = write_tar_shards(tmp_path, records, num_shards=2)
    init_file = tmp_path / "dist_init"
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    mp.spawn(
        _distributed_reader_worker,
        args=(2, shards, str(init_file), str(output_dir)),
        nprocs=2,
        join=True,
    )

    rank0_ids = json.loads((output_dir / "rank_0.json").read_text(encoding="utf-8"))
    rank1_ids = json.loads((output_dir / "rank_1.json").read_text(encoding="utf-8"))

    assert set(rank0_ids).isdisjoint(rank1_ids)
    assert set(rank0_ids) | set(rank1_ids) == {str(record["id"]) for record in records}


def test_tp_group_members_receive_identical_data(tmp_path) -> None:
    records = build_records(count=4)
    shards = write_tar_shards(tmp_path, records, num_shards=2)
    init_file = tmp_path / "tp_dist_init"
    output_dir = tmp_path / "tp_outputs"
    output_dir.mkdir()

    mp.spawn(
        _tp_group_reader_worker,
        args=(4, shards, str(init_file), str(output_dir)),
        nprocs=4,
        join=True,
    )

    outputs = {rank: json.loads((output_dir / f"rank_{rank}.json").read_text(encoding="utf-8")) for rank in range(4)}

    assert outputs[0] == outputs[1]
    assert outputs[2] == outputs[3]
    assert set(outputs[0]).isdisjoint(outputs[2])
    assert set(outputs[0]) | set(outputs[2]) == {str(record["id"]) for record in records}


def test_tp_group_members_receive_identical_shuffled_data(tmp_path) -> None:
    records = build_records(count=8)
    shards = write_tar_shards(tmp_path, records, num_shards=4)
    init_file = tmp_path / "tp_shuffle_dist_init"
    output_dir = tmp_path / "tp_shuffle_outputs"
    output_dir.mkdir()

    mp.spawn(
        _tp_group_shuffle_reader_worker,
        args=(4, shards, str(init_file), str(output_dir)),
        nprocs=4,
        join=True,
    )

    outputs = {rank: json.loads((output_dir / f"rank_{rank}.json").read_text(encoding="utf-8")) for rank in range(4)}

    assert outputs[0] == outputs[1]
    assert outputs[2] == outputs[3]
    assert outputs[0] != outputs[2]
    assert set(outputs[0]).isdisjoint(outputs[2])
    assert set(outputs[0]) | set(outputs[2]) == {str(record["id"]) for record in records}


def test_tp_group_members_receive_identical_loader_shuffled_data(tmp_path) -> None:
    records = build_records(count=8)
    shards = write_tar_shards(tmp_path, records, num_shards=4)
    init_file = tmp_path / "tp_loader_shuffle_dist_init"
    output_dir = tmp_path / "tp_loader_shuffle_outputs"
    output_dir.mkdir()

    mp.spawn(
        _tp_group_loader_shuffle_reader_worker,
        args=(4, shards, str(init_file), str(output_dir)),
        nprocs=4,
        join=True,
    )

    outputs = {rank: json.loads((output_dir / f"rank_{rank}.json").read_text(encoding="utf-8")) for rank in range(4)}

    assert outputs[0] == outputs[1]
    assert outputs[2] == outputs[3]
    assert outputs[0] != outputs[2]
    assert set(outputs[0]).isdisjoint(outputs[2])
    assert set(outputs[0]) | set(outputs[2]) == {str(record["id"]) for record in records}


def test_tp_group_members_receive_identical_data_with_multiple_workers(tmp_path) -> None:
    records = build_records(count=8)
    shards = write_tar_shards(tmp_path, records, num_shards=4)
    init_file = tmp_path / "tp_multiworker_dist_init"
    output_dir = tmp_path / "tp_multiworker_outputs"
    output_dir.mkdir()

    mp.spawn(
        _tp_group_multiworker_reader_worker,
        args=(4, shards, str(init_file), str(output_dir)),
        nprocs=4,
        join=True,
    )

    outputs = {rank: json.loads((output_dir / f"rank_{rank}.json").read_text(encoding="utf-8")) for rank in range(4)}

    assert outputs[0] == outputs[1]
    assert outputs[2] == outputs[3]
    assert outputs[0] != outputs[2]
    assert set(outputs[0]).isdisjoint(outputs[2])
    assert set(outputs[0]) | set(outputs[2]) == {str(record["id"]) for record in records}


def test_cache_does_not_duplicate_tp_group_outputs(tmp_path) -> None:
    records = build_records(count=8)
    shards = write_tar_shards(tmp_path, records, num_shards=4)
    init_file = tmp_path / "tp_cache_dist_init"
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "tp_cache_outputs"
    output_dir.mkdir()

    mp.spawn(
        _tp_group_cache_reader_worker,
        args=(4, shards, str(init_file), str(cache_dir), str(output_dir)),
        nprocs=4,
        join=True,
    )

    outputs = {rank: json.loads((output_dir / f"rank_{rank}.json").read_text(encoding="utf-8")) for rank in range(4)}
    combined_dp_outputs = outputs[0] + outputs[2]

    assert outputs[0] == outputs[1]
    assert outputs[2] == outputs[3]
    assert outputs[0] != outputs[2]
    assert sorted(combined_dp_outputs) == sorted(str(record["id"]) for record in records)
