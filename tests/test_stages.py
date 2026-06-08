from __future__ import annotations

import pickle
from dataclasses import dataclass, field

from mvp_dataset import Dataset, RuntimeContext, TorchLoader

from .helpers import build_records, write_parquet_file


def _build_parquet_dataset(tmp_path, records: list[dict[str, object]], *, seed: int = 0) -> Dataset:
    path = write_parquet_file(tmp_path, records)
    return Dataset.from_source("parquet", shards=path, context=RuntimeContext(seed=seed))


def _sample_ids(stream) -> list[str]:
    return [str(sample["id"]) for sample in stream]


def add_mapped_flag(sample: object) -> object:
    if not isinstance(sample, dict):
        return sample
    return {**sample, "mapped": True}


def identity_collate(batch: list[object]) -> list[object]:
    return batch


@dataclass
class PairSumAssembler:
    pending: list[dict[str, object]] = field(default_factory=list)

    def push(self, sample: dict[str, object]) -> list[dict[str, object]]:
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

    def finish(self, *, drop_last: bool = False) -> list[dict[str, object]]:
        if drop_last or not self.pending:
            return []
        tail = self.pending.pop()
        return [{"left_id": str(tail["id"]), "right_id": None, "sum_value": int(tail["value"])}]

    def state_dict(self) -> dict[str, object]:
        return {"pending": list(self.pending)}

    def load_state_dict(self, state: dict[str, object]) -> None:
        pending = state.get("pending")
        self.pending = [] if not isinstance(pending, list) else list(pending)

    def fingerprint(self) -> str:
        return "pair-sum-assembler:v1"


def build_pair_sum_assembler(_context: RuntimeContext) -> PairSumAssembler:
    return PairSumAssembler()


def test_map_select_batch_unbatch_pipeline(tmp_path) -> None:
    records = build_records()
    dataset = _build_parquet_dataset(tmp_path, records).map(add_mapped_flag).select(["id", "mapped"]).batch(2).unbatch()

    observed = list(dataset)

    assert [sample["id"] for sample in observed] == [record["id"] for record in records]
    assert [sample["mapped"] for sample in observed] == [True for _ in records]


def test_shuffle_stage_is_deterministic_for_fixed_seed(tmp_path) -> None:
    records = build_records(count=12)

    shuffled_once = _sample_ids(_build_parquet_dataset(tmp_path, records, seed=7).shuffle(buffer_size=4))
    shuffled_twice = _sample_ids(_build_parquet_dataset(tmp_path, records, seed=7).shuffle(buffer_size=4))

    assert shuffled_once == shuffled_twice
    assert set(shuffled_once) == {str(record["id"]) for record in records}


def test_assemble_stage_flushes_tail_and_drop_last(tmp_path) -> None:
    records = build_records(count=5)
    keep_tail = list(_build_parquet_dataset(tmp_path, records).assemble(build_pair_sum_assembler))
    drop_tail = list(_build_parquet_dataset(tmp_path, records).assemble(build_pair_sum_assembler, drop_last=True))

    assert keep_tail == [
        {"left_id": "sample-0", "right_id": "sample-1", "sum_value": 1},
        {"left_id": "sample-2", "right_id": "sample-3", "sum_value": 5},
        {"left_id": "sample-4", "right_id": None, "sum_value": 4},
    ]
    assert drop_tail == keep_tail[:2]


def test_loader_stage_pipeline_is_deterministic(tmp_path) -> None:
    records = build_records(count=6)

    def build_loader() -> TorchLoader:
        return (
            TorchLoader(
                _build_parquet_dataset(tmp_path, records, seed=13),
                num_workers=0,
                batch_size=2,
                collate_fn=identity_collate,
            )
            .unbatch()
            .shuffle(buffer_size=3)
            .batch(2, collate_fn=identity_collate)
            .unbatch()
            .assemble(build_pair_sum_assembler)
        )

    observed = list(build_loader())
    repeated = list(build_loader())

    assert observed == repeated
    assert sorted(item["sum_value"] for item in observed) == sorted(item["sum_value"] for item in repeated)


def test_stage_objects_are_picklable(tmp_path) -> None:
    records = build_records()
    dataset = _build_parquet_dataset(tmp_path, records)
    loader = (
        TorchLoader(dataset, num_workers=0, batch_size=2, collate_fn=identity_collate).unbatch().shuffle(buffer_size=3)
    )
    objects = [
        dataset.map(add_mapped_flag)._stages[-1].apply,
        dataset.select(["id"])._stages[-1].apply,
        dataset.shuffle(buffer_size=3)._stages[-1].apply,
        dataset.batch(2, collate_fn=identity_collate)._stages[-1].apply,
        dataset.unbatch()._stages[-1].apply,
        dataset.assemble(build_pair_sum_assembler)._stages[-1].apply,
        *loader._stages,
    ]

    for item in objects:
        pickle.dumps(item)
