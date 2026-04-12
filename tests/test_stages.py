from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import mvp_dataset.core.dataset as dataset_module
from mvp_dataset import Dataset, RuntimeContext, TorchLoader

from .helpers import build_records, write_parquet_file


def _build_parquet_dataset(tmp_path, records: list[dict[str, object]], *, seed: int = 0) -> Dataset:
    path = write_parquet_file(tmp_path, records)
    context = RuntimeContext(seed=seed)
    return Dataset.from_source("parquet", shards=path, context=context)


def _sample_ids(stream) -> list[str]:
    return [str(sample["id"]) for sample in stream]


def add_mapped_flag(sample: object) -> object:
    if not isinstance(sample, dict):
        return sample
    return {
        **sample,
        "mapped": True,
    }


def identity_collate(batch: list[object]) -> list[object]:
    return batch


def test_map_stage_transforms_each_sample(tmp_path) -> None:
    records = build_records()
    dataset = _build_parquet_dataset(tmp_path, records).map(add_mapped_flag)

    observed = [bool(sample["mapped"]) for sample in dataset]

    assert observed == [True for _ in records]


def test_select_stage_keeps_requested_fields_and_metadata(tmp_path) -> None:
    records = build_records()
    dataset = _build_parquet_dataset(tmp_path, records).select(["id"])

    first_sample = next(iter(dataset))

    assert set(first_sample) == {"id", "__file__", "__index_in_file__", "__key__"}
    assert first_sample["id"] == records[0]["id"]


def test_shuffle_stage_is_deterministic_for_fixed_seed(tmp_path) -> None:
    records = build_records()

    shuffled_once = _sample_ids(_build_parquet_dataset(tmp_path, records, seed=7).shuffle(buffer_size=3))
    shuffled_twice = _sample_ids(_build_parquet_dataset(tmp_path, records, seed=7).shuffle(buffer_size=3))

    assert shuffled_once == shuffled_twice
    assert set(shuffled_once) == {str(record["id"]) for record in records}


def test_shuffle_stage_uses_runtime_worker_specific_seed(tmp_path, monkeypatch) -> None:
    records = build_records()
    dataset = _build_parquet_dataset(tmp_path, records, seed=11).shuffle(buffer_size=3)

    recorded_rng_outputs: list[float] = []

    def fake_shuffle_samples(data, *, buffer_size, initial, rng):
        recorded_rng_outputs.append(rng.random())
        yield from data

    monkeypatch.setattr(dataset_module, "shuffle_samples", fake_shuffle_samples)

    def worker_zero_runtime(*, base=None, **_kwargs):
        assert base is not None
        return RuntimeContext(
            rank=base.rank,
            world_size=base.world_size,
            local_rank=base.local_rank,
            local_world_size=base.local_world_size,
            node_rank=base.node_rank,
            num_nodes=base.num_nodes,
            worker_id=0,
            num_workers=2,
            epoch=base.epoch,
            seed=base.seed,
            mesh=base.mesh,
        )

    def worker_one_runtime(*, base=None, **_kwargs):
        assert base is not None
        return RuntimeContext(
            rank=base.rank,
            world_size=base.world_size,
            local_rank=base.local_rank,
            local_world_size=base.local_world_size,
            node_rank=base.node_rank,
            num_nodes=base.num_nodes,
            worker_id=1,
            num_workers=2,
            epoch=base.epoch,
            seed=base.seed,
            mesh=base.mesh,
        )

    monkeypatch.setattr(dataset_module.RuntimeContext, "from_runtime", worker_zero_runtime)
    list(dataset)

    monkeypatch.setattr(dataset_module.RuntimeContext, "from_runtime", worker_one_runtime)
    list(dataset)

    assert len(recorded_rng_outputs) == 2
    assert recorded_rng_outputs[0] != recorded_rng_outputs[1]


def test_batch_stage_groups_samples(tmp_path) -> None:
    records = build_records()
    dataset = _build_parquet_dataset(tmp_path, records).batch(2)

    observed = [[str(sample["id"]) for sample in batch] for batch in dataset]

    assert observed == [
        ["sample-0", "sample-1"],
        ["sample-2", "sample-3"],
        ["sample-4", "sample-5"],
    ]


def test_unbatch_stage_restores_original_stream(tmp_path) -> None:
    records = build_records()
    dataset = _build_parquet_dataset(tmp_path, records).batch(2).unbatch()

    assert _sample_ids(dataset) == [str(record["id"]) for record in records]


@dataclass
class PairSumAssembler:
    pending: list[dict[str, object]] = field(default_factory=list)

    def push(self, sample: dict[str, object]):
        self.pending.append(sample)
        if len(self.pending) < 2:
            return []
        left, right = self.pending
        self.pending = []
        return [
            {
                "left_id": str(left["id"]),
                "right_id": str(right["id"]),
                "sum_value": int(left["value"]) + int(right["value"]),
            }
        ]

    def finish(self, *, drop_last: bool = False):
        if drop_last or not self.pending:
            return []
        tail = self.pending.pop()
        return [
            {
                "left_id": str(tail["id"]),
                "right_id": None,
                "sum_value": int(tail["value"]),
            }
        ]


def build_pair_sum_assembler(_context: RuntimeContext) -> PairSumAssembler:
    return PairSumAssembler()


def test_dataset_stage_apply_objects_are_picklable(tmp_path) -> None:
    records = build_records()

    apply_objects = [
        _build_parquet_dataset(tmp_path, records).map(add_mapped_flag)._stages[-1].apply,
        _build_parquet_dataset(tmp_path, records).select(["id"])._stages[-1].apply,
        _build_parquet_dataset(tmp_path, records, seed=5).shuffle(buffer_size=3)._stages[-1].apply,
        _build_parquet_dataset(tmp_path, records).batch(2, collate_fn=identity_collate)._stages[-1].apply,
        _build_parquet_dataset(tmp_path, records).assemble(build_pair_sum_assembler)._stages[-1].apply,
        _build_parquet_dataset(tmp_path, records).unbatch()._stages[-1].apply,
    ]

    for apply in apply_objects:
        pickle.dumps(apply)


def test_loader_stage_objects_are_picklable(tmp_path) -> None:
    records = build_records()
    dataset = _build_parquet_dataset(tmp_path, records, seed=5)
    loader = (
        TorchLoader(dataset, num_workers=0, batch_size=2, collate_fn=identity_collate)
        .unbatch()
        .shuffle(buffer_size=3)
        .batch(2, collate_fn=identity_collate)
    )

    for stage in loader._stages:
        pickle.dumps(stage)


def test_assemble_stage_flushes_tail_by_default(tmp_path) -> None:
    records = build_records(count=5)
    dataset = _build_parquet_dataset(tmp_path, records).assemble(build_pair_sum_assembler)

    observed = list(dataset)

    assert observed == [
        {"left_id": "sample-0", "right_id": "sample-1", "sum_value": 1},
        {"left_id": "sample-2", "right_id": "sample-3", "sum_value": 5},
        {"left_id": "sample-4", "right_id": None, "sum_value": 4},
    ]


def test_assemble_stage_drop_last_discards_tail(tmp_path) -> None:
    records = build_records(count=5)
    dataset = _build_parquet_dataset(tmp_path, records).assemble(build_pair_sum_assembler, drop_last=True)

    observed = list(dataset)

    assert observed == [
        {"left_id": "sample-0", "right_id": "sample-1", "sum_value": 1},
        {"left_id": "sample-2", "right_id": "sample-3", "sum_value": 5},
    ]
