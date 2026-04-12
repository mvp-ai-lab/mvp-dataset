from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

from mvp_dataset import Dataset

from .helpers import build_records, write_tar_shards

if importlib.util.find_spec("torch") is not None:
    import torch.distributed as dist
    import torch.multiprocessing as mp
else:
    dist = None
    mp = None

pytestmark = pytest.mark.skipif(dist is None or mp is None, reason="torch is not installed")


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
