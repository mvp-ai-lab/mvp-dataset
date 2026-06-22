from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

from mvp_dataset import RuntimeContext, TorchLoader, UnsupportedResume
from mvp_dataset.loader.stages.balance import RankStatus, Transfer, plan_balance_chunk

if importlib.util.find_spec("torch") is not None:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
else:
    torch = None
    dist = None
    mp = None


def _status(rank: int, available: int, *, node_rank: int = 0, done: bool = True) -> RankStatus:
    return RankStatus(
        rank=rank,
        node_rank=node_rank,
        local_rank=rank,
        available=available,
        upstream_done=done,
    )


def test_balance_planner_prefers_local_items() -> None:
    plan = plan_balance_chunk(
        [_status(0, 2), _status(1, 2)],
        chunk_size=2,
        max_transfer_per_round=4,
        drop_last=True,
        topology="node",
    )

    assert plan.local_take == {0: 2, 1: 2}
    assert plan.transfers == []
    assert plan.dummy_counts == {0: 0, 1: 0}


def test_balance_planner_prefers_same_node_donors() -> None:
    plan = plan_balance_chunk(
        [
            _status(0, 4, node_rank=0),
            _status(1, 0, node_rank=0),
            _status(2, 4, node_rank=1),
            _status(3, 0, node_rank=1),
        ],
        chunk_size=1,
        max_transfer_per_round=4,
        drop_last=True,
        topology="node",
    )

    assert plan.transfers == [Transfer(src=0, dst=1, count=1), Transfer(src=2, dst=3, count=1)]


def test_balance_planner_falls_back_to_cross_node_donors() -> None:
    plan = plan_balance_chunk(
        [
            _status(0, 0, node_rank=0),
            _status(1, 0, node_rank=0),
            _status(2, 4, node_rank=1),
            _status(3, 1, node_rank=1),
        ],
        chunk_size=1,
        max_transfer_per_round=4,
        drop_last=True,
        topology="node",
    )

    assert plan.transfers == [Transfer(src=2, dst=0, count=1), Transfer(src=2, dst=1, count=1)]


def test_balance_planner_respects_chunk_size_and_transfer_limit() -> None:
    plan = plan_balance_chunk(
        [_status(0, 10), _status(1, 0)],
        chunk_size=4,
        max_transfer_per_round=2,
        drop_last=True,
        topology="none",
    )

    assert plan.local_take == {0: 2, 1: 0}
    assert plan.transfers == [Transfer(src=0, dst=1, count=2)]


def test_balance_planner_handles_tail_drop_and_dummy() -> None:
    drop_plan = plan_balance_chunk(
        [_status(0, 1), _status(1, 0)],
        chunk_size=2,
        max_transfer_per_round=4,
        drop_last=True,
        topology="none",
    )
    dummy_plan = plan_balance_chunk(
        [_status(0, 1), _status(1, 0)],
        chunk_size=2,
        max_transfer_per_round=4,
        drop_last=False,
        topology="none",
    )

    assert drop_plan.local_take == {0: 0, 1: 0}
    assert drop_plan.finished_after_chunk
    assert dummy_plan.local_take == {0: 1, 1: 0}
    assert dummy_plan.dummy_counts == {0: 0, 1: 1}
    assert dummy_plan.finished_after_chunk


def test_balance_api_validation_and_resume_rejection() -> None:
    if torch is None:
        pytest.skip("torch is not installed")

    with pytest.raises(ValueError, match="dummy_factory"):
        TorchLoader([{"x": 1}], num_workers=0).balance(drop_last=False)
    with pytest.raises(ValueError, match="buffer_size"):
        TorchLoader([{"x": 1}], num_workers=0).balance(buffer_size=1, chunk_size=2)

    loader = TorchLoader([{"x": 1}], num_workers=0).balance(buffer_size=1, chunk_size=1)
    with pytest.warns(UserWarning), pytest.raises(UnsupportedResume, match="balance"):
        loader.state_dict()


class _RankItems:
    def __init__(self, counts: list[int], *, tensor: bool = False) -> None:
        self.counts = counts
        self.tensor = tensor

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else int(worker_info.id)
        num_workers = 1 if worker_info is None else int(worker_info.num_workers)
        rank = int(os.environ["RANK"])
        for index in range(worker_id, self.counts[rank], num_workers):
            if self.tensor:
                yield {"payload": torch.tensor([rank, index], dtype=torch.long), "dummy": False}
            else:
                yield {"source_rank": rank, "index": index, "dummy": False}


def _dummy_batch(_context: RuntimeContext) -> dict[str, object]:
    return {"source_rank": -1, "index": -1, "dummy": True}


def _balance_worker(
    rank: int,
    world_size: int,
    init_file: str,
    output_dir: str,
    counts: list[int],
    drop_last: bool,
    chunk_size: int,
    num_workers: int,
    tensor: bool,
) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        loader = TorchLoader(_RankItems(counts, tensor=tensor), num_workers=num_workers).balance(
            drop_last=drop_last,
            dummy_factory=None if drop_last else _dummy_batch,
            buffer_size=max(4, chunk_size),
            chunk_size=chunk_size,
            max_transfer_per_round=8,
            topology="none",
        )
        outputs = []
        for item in loader:
            if tensor:
                outputs.append({"tensor": item["payload"].tolist(), "dummy": item["dummy"]})
            else:
                outputs.append(item)
        Path(output_dir, f"rank_{rank}.json").write_text(json.dumps(outputs), encoding="utf-8")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_balance_spawn(
    tmp_path,
    *,
    counts: list[int],
    drop_last: bool,
    chunk_size: int,
    num_workers: int = 0,
    tensor: bool = False,
) -> dict[int, list[dict[str, object]]]:
    if dist is None or mp is None:
        pytest.skip("torch distributed is not installed")
    init_file = tmp_path / "balance_init"
    output_dir = tmp_path / "balance_outputs"
    output_dir.mkdir()
    mp.spawn(
        _balance_worker,
        args=(len(counts), str(init_file), str(output_dir), counts, drop_last, chunk_size, num_workers, tensor),
        nprocs=len(counts),
        join=True,
    )
    return {
        rank: json.loads((output_dir / f"rank_{rank}.json").read_text(encoding="utf-8")) for rank in range(len(counts))
    }


@pytest.mark.skipif(dist is None or mp is None, reason="torch distributed is not installed")
def test_balance_distributed_drop_last_equalizes_outputs(tmp_path) -> None:
    outputs = _run_balance_spawn(tmp_path, counts=[5, 1], drop_last=True, chunk_size=2)

    assert [len(items) for items in outputs.values()] == [3, 3]
    assert not any(item["dummy"] for items in outputs.values() for item in items)


@pytest.mark.skipif(dist is None or mp is None, reason="torch distributed is not installed")
def test_balance_distributed_dummy_tail(tmp_path) -> None:
    outputs = _run_balance_spawn(tmp_path, counts=[5, 2], drop_last=False, chunk_size=3)

    assert [len(items) for items in outputs.values()] == [4, 4]
    assert sum(int(item["dummy"]) for items in outputs.values() for item in items) == 1


@pytest.mark.skipif(dist is None or mp is None, reason="torch distributed is not installed")
def test_balance_distributed_chunk_size_one_and_multiworker(tmp_path) -> None:
    outputs = _run_balance_spawn(tmp_path, counts=[5, 1], drop_last=True, chunk_size=1, num_workers=2)

    assert [len(items) for items in outputs.values()] == [3, 3]


@pytest.mark.skipif(dist is None or mp is None, reason="torch distributed is not installed")
def test_balance_distributed_tensor_payload(tmp_path) -> None:
    outputs = _run_balance_spawn(tmp_path, counts=[3, 1], drop_last=True, chunk_size=2, tensor=True)

    assert [len(items) for items in outputs.values()] == [2, 2]
    assert all("tensor" in item for items in outputs.values() for item in items)
