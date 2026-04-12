from __future__ import annotations

import pickle
import threading
import time
from dataclasses import dataclass, field

import pyarrow as pa
import pytest

import mvp_dataset.cache.store as store_module
import mvp_dataset.core.dataset as dataset_module
import mvp_dataset.core.stages as stages_module
import mvp_dataset.sources.parquet.utils as parquet_utils_module
from mvp_dataset import Dataset, RuntimeContext, TorchLoader
from mvp_dataset.log import reset_logger, set_logger

from .helpers import build_records, write_nested_parquet_file, write_parquet_file

_OBSERVED_CACHE_THREAD_IDS: set[int] = set()
_OBSERVED_CACHE_THREAD_IDS_LOCK = threading.Lock()
_OBSERVED_CACHE_STAGE_ONE_THREAD_IDS: set[int] = set()
_OBSERVED_CACHE_STAGE_ONE_THREAD_IDS_LOCK = threading.Lock()
_OBSERVED_CACHE_STAGE_TWO_THREAD_IDS: set[int] = set()
_OBSERVED_CACHE_STAGE_TWO_THREAD_IDS_LOCK = threading.Lock()


def _build_parquet_dataset(tmp_path, records: list[dict[str, object]], *, seed: int = 0) -> Dataset:
    path = write_parquet_file(tmp_path, records)
    context = RuntimeContext(seed=seed)
    return Dataset.from_source("parquet", shards=path, context=context)


def _sample_ids(stream) -> list[str]:
    return [str(sample["id"]) for sample in stream]


def _select_user_fields(sample: dict[str, object], fields: tuple[str, ...]) -> dict[str, object]:
    return {field: sample[field] for field in fields}


def add_mapped_flag(sample: object) -> object:
    if not isinstance(sample, dict):
        return sample
    return {
        **sample,
        "mapped": True,
    }


def identity_collate(batch: list[object]) -> list[object]:
    return batch


def record_cache_thread(sample: object) -> object:
    time.sleep(0.01)
    with _OBSERVED_CACHE_THREAD_IDS_LOCK:
        _OBSERVED_CACHE_THREAD_IDS.add(threading.get_ident())
    return sample


def record_cache_thread_stage_one(sample: object) -> object:
    time.sleep(0.01)
    with _OBSERVED_CACHE_STAGE_ONE_THREAD_IDS_LOCK:
        _OBSERVED_CACHE_STAGE_ONE_THREAD_IDS.add(threading.get_ident())
    return sample


def record_cache_thread_stage_two(sample: object) -> object:
    time.sleep(0.01)
    with _OBSERVED_CACHE_STAGE_TWO_THREAD_IDS_LOCK:
        _OBSERVED_CACHE_STAGE_TWO_THREAD_IDS.add(threading.get_ident())
    return sample


def add_chunked_array_field(sample: object) -> object:
    if not isinstance(sample, dict):
        return sample
    value = int(sample["value"])
    return {
        **sample,
        "chunked": pa.chunked_array([[value], [value + 1]]),
    }


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

    monkeypatch.setattr(stages_module, "shuffle_samples", fake_shuffle_samples)

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


def test_assemble_stage_uses_runtime_worker_specific_context(tmp_path, monkeypatch) -> None:
    records = build_records(count=4)
    dataset = _build_parquet_dataset(tmp_path, records, seed=11).assemble(build_pair_sum_assembler)

    recorded_worker_ids: list[int] = []

    def recording_factory(context: RuntimeContext) -> PairSumAssembler:
        recorded_worker_ids.append(context.worker_id)
        return PairSumAssembler()

    dataset = _build_parquet_dataset(tmp_path, records, seed=11).assemble(recording_factory)

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

    assert recorded_worker_ids == [0, 1]


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


@pytest.mark.parametrize(
    ("expected_stage_kind", "builder"),
    [
        ("batch", lambda dataset: dataset.batch(2)),
        ("batch", lambda dataset: dataset.batch(2).unbatch()),
    ],
)
def test_cache_rejects_non_one_to_one_pre_cache_stages(tmp_path, expected_stage_kind: str, builder) -> None:
    records = build_records()
    dataset = _build_parquet_dataset(tmp_path, records)

    with pytest.raises(ValueError, match=rf"\[CacheError\] '{expected_stage_kind}' cannot appear before \.cache\(\)"):
        builder(dataset).cache(cache_dir=str(tmp_path / "cache"))


def test_cache_accepts_one_to_one_pre_cache_stages(tmp_path) -> None:
    records = build_records()
    dataset = _build_parquet_dataset(tmp_path, records).map(add_mapped_flag).select(["id", "mapped"])

    cached = dataset.cache(cache_dir=str(tmp_path / "cache"))

    observed = [sample for sample in cached]

    assert len(observed) == len(records)
    assert all(sample["mapped"] is True for sample in observed)


def test_cache_preserves_shuffle_stream_semantics(tmp_path) -> None:
    records = build_records()
    uncached = _build_parquet_dataset(tmp_path, records, seed=13).shuffle(buffer_size=3)
    cached = (
        _build_parquet_dataset(tmp_path, records, seed=13)
        .shuffle(buffer_size=3)
        .cache(cache_dir=str(tmp_path / "cache"))
    )

    uncached_samples = [_select_user_fields(sample, ("id", "text", "value")) for sample in uncached]
    cached_samples = [_select_user_fields(sample, ("id", "text", "value")) for sample in cached]

    assert cached_samples == uncached_samples


def test_cache_handles_nested_parquet_columns_with_scanner_fallback(tmp_path, monkeypatch) -> None:
    records = [
        {
            "id": f"sample-{index}",
            "meta": {"index": index, "parity": "even" if index % 2 == 0 else "odd"},
            "tags": [f"tag-{index}", f"group-{index % 2}"],
        }
        for index in range(6)
    ]
    path = write_nested_parquet_file(tmp_path, records, row_group_size=2)
    context = RuntimeContext(seed=13)
    scanner_used = False
    original = parquet_utils_module._iter_record_batches_via_scanner

    def _recording_scanner(*args, **kwargs):
        nonlocal scanner_used
        scanner_used = True
        yield from original(*args, **kwargs)

    monkeypatch.setattr(parquet_utils_module, "_iter_record_batches_via_scanner", _recording_scanner)

    uncached = Dataset.from_source("parquet", shards=path, context=context)
    cached = Dataset.from_source("parquet", shards=path, context=context).cache(cache_dir=str(tmp_path / "cache"))

    uncached_samples = [_select_user_fields(sample, ("id", "meta", "tags")) for sample in uncached]
    cached_samples = [_select_user_fields(sample, ("id", "meta", "tags")) for sample in cached]

    assert scanner_used is True
    assert cached_samples == uncached_samples


def test_cache_normalizes_chunked_array_field_values(tmp_path) -> None:
    records = build_records()
    cached = (
        _build_parquet_dataset(tmp_path, records).map(add_chunked_array_field).cache(cache_dir=str(tmp_path / "cache"))
    )

    observed = [_select_user_fields(sample, ("id", "chunked")) for sample in cached]

    assert observed == [
        {
            "id": f"sample-{index}",
            "chunked": [index, index + 1],
        }
        for index in range(len(records))
    ]


def test_cache_schema_uses_large_offsets_for_nested_variable_width_fields() -> None:
    images_type = store_module._infer_arrow_type(
        [
            [{"bytes": b"a", "path": None}],
            [{"bytes": b"b", "path": None}],
        ]
    )
    img_size_type = store_module._infer_arrow_type(
        [
            [[1, 2]],
            None,
            [[3, 4]],
        ]
    )

    assert images_type == pa.large_list(
        pa.struct(
            [
                pa.field("bytes", pa.large_binary()),
                pa.field("path", pa.null()),
            ]
        )
    )
    assert img_size_type == pa.large_list(pa.large_list(pa.int64()))


def test_cache_warns_when_shuffle_is_not_fingerprinted(tmp_path) -> None:
    records = build_records()
    warnings: list[str] = []

    class _CapturingLogger:
        def debug(self, _msg, *args, **kwargs):
            return None

        def info(self, _msg, *args, **kwargs):
            return None

        def warning(self, msg, *args, **kwargs):
            warnings.append(str(msg % args if args else msg))
            return None

        def error(self, _msg, *args, **kwargs):
            return None

    set_logger(_CapturingLogger())
    try:
        _build_parquet_dataset(tmp_path, records, seed=13).shuffle(buffer_size=3).cache(
            cache_dir=str(tmp_path / "cache")
        )
    finally:
        reset_logger()

    assert len(warnings) == 1
    assert "not included in cache fingerprinting" in warnings[0]
    assert "shuffle" in warnings[0]


def test_cache_preserves_assemble_stream_semantics(tmp_path) -> None:
    records = build_records(count=5)
    uncached = _build_parquet_dataset(tmp_path, records).assemble(build_pair_sum_assembler)
    cached = (
        _build_parquet_dataset(tmp_path, records)
        .assemble(build_pair_sum_assembler)
        .cache(cache_dir=str(tmp_path / "cache"))
    )

    uncached_samples = [_select_user_fields(sample, ("left_id", "right_id", "sum_value")) for sample in uncached]
    cached_samples = [_select_user_fields(sample, ("left_id", "right_id", "sum_value")) for sample in cached]

    assert cached_samples == uncached_samples


def test_assemble_drop_last_changes_cache_fingerprint(tmp_path) -> None:
    records = build_records(count=5)
    cache_dir = tmp_path / "cache"

    cached_keep_tail = (
        _build_parquet_dataset(tmp_path, records)
        .assemble(build_pair_sum_assembler, drop_last=False)
        .cache(cache_dir=str(cache_dir))
    )
    cached_drop_tail = (
        _build_parquet_dataset(tmp_path, records)
        .assemble(build_pair_sum_assembler, drop_last=True)
        .cache(cache_dir=str(cache_dir))
    )

    assert cached_keep_tail._cache_spec is not None
    assert cached_drop_tail._cache_spec is not None
    assert cached_keep_tail._cache_spec.plan_fingerprint != cached_drop_tail._cache_spec.plan_fingerprint


def test_cache_preserves_assemble_stream_semantics_with_parallel_prefix(tmp_path) -> None:
    records = build_records(count=8)
    uncached = _build_parquet_dataset(tmp_path, records).map(add_mapped_flag).assemble(build_pair_sum_assembler)
    cached = (
        _build_parquet_dataset(tmp_path, records)
        .map(add_mapped_flag)
        .assemble(build_pair_sum_assembler)
        .cache(cache_dir=str(tmp_path / "cache"), cache_num_workers=4)
    )

    uncached_samples = [_select_user_fields(sample, ("left_id", "right_id", "sum_value")) for sample in uncached]
    cached_samples = [_select_user_fields(sample, ("left_id", "right_id", "sum_value")) for sample in cached]

    assert cached_samples == uncached_samples


def test_cache_parallelizes_map_prefix_with_multiple_threads(tmp_path) -> None:
    _OBSERVED_CACHE_THREAD_IDS.clear()

    records = build_records(count=24)
    cached = (
        _build_parquet_dataset(tmp_path, records)
        .map(record_cache_thread)
        .cache(cache_dir=str(tmp_path / "cache"), cache_num_workers=4)
    )

    observed = [_select_user_fields(sample, ("id", "text", "value")) for sample in cached]

    assert observed == [
        {
            "id": record["id"],
            "text": record["text"],
            "value": record["value"],
        }
        for record in records
    ]
    assert len(_OBSERVED_CACHE_THREAD_IDS) >= 2


def test_cache_preserves_stream_semantics_when_only_map_prefix_parallelizes(tmp_path) -> None:
    records = build_records(count=8)
    uncached = (
        _build_parquet_dataset(tmp_path, records, seed=17)
        .map(add_mapped_flag)
        .select(["id", "value", "mapped"])
        .shuffle(buffer_size=3)
        .map(add_mapped_flag)
        .assemble(build_pair_sum_assembler)
        .map(add_mapped_flag)
    )
    cached = (
        _build_parquet_dataset(tmp_path, records, seed=17)
        .map(add_mapped_flag)
        .select(["id", "value", "mapped"])
        .shuffle(buffer_size=3)
        .map(add_mapped_flag)
        .assemble(build_pair_sum_assembler)
        .map(add_mapped_flag)
        .cache(cache_dir=str(tmp_path / "cache"), cache_num_workers=4)
    )

    uncached_samples = [
        _select_user_fields(sample, ("left_id", "right_id", "sum_value", "mapped")) for sample in uncached
    ]
    cached_samples = [_select_user_fields(sample, ("left_id", "right_id", "sum_value", "mapped")) for sample in cached]

    assert cached_samples == uncached_samples


def test_cache_runs_only_map_prefix_in_parallel(tmp_path) -> None:
    _OBSERVED_CACHE_STAGE_ONE_THREAD_IDS.clear()
    _OBSERVED_CACHE_STAGE_TWO_THREAD_IDS.clear()

    records = build_records(count=24)
    uncached = _build_parquet_dataset(tmp_path, records, seed=19).shuffle(buffer_size=3)
    cached = (
        _build_parquet_dataset(tmp_path, records, seed=19)
        .map(record_cache_thread_stage_one)
        .shuffle(buffer_size=3)
        .map(record_cache_thread_stage_two)
        .cache(cache_dir=str(tmp_path / "cache"), cache_num_workers=4)
    )

    expected = [_select_user_fields(sample, ("id", "text", "value")) for sample in uncached]
    observed = [_select_user_fields(sample, ("id", "text", "value")) for sample in cached]

    assert observed == expected
    assert len(_OBSERVED_CACHE_STAGE_ONE_THREAD_IDS) >= 2
    assert len(_OBSERVED_CACHE_STAGE_TWO_THREAD_IDS) == 1


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
