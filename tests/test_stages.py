from __future__ import annotations

from dataclasses import dataclass, field

from mvp_dataset import Dataset, RuntimeContext

from .helpers import build_records, write_parquet_file


def _build_parquet_dataset(tmp_path, records: list[dict[str, object]], *, seed: int = 0) -> Dataset:
    path = write_parquet_file(tmp_path, records)
    context = RuntimeContext(seed=seed)
    return Dataset.from_source("parquet", shards=path, context=context)


def _sample_ids(stream) -> list[str]:
    return [str(sample["id"]) for sample in stream]


def test_map_stage_transforms_each_sample(tmp_path) -> None:
    records = build_records()
    dataset = _build_parquet_dataset(tmp_path, records).map(
        lambda sample: {
            **sample,
            "double_value": int(sample["value"]) * 2,
        }
    )

    observed = [int(sample["double_value"]) for sample in dataset]

    assert observed == [int(record["value"]) * 2 for record in records]


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
