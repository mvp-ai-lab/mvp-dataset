from __future__ import annotations

import shutil
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import lance
import pyarrow as pa
import pytest

from mvp_dataset import Dataset, RuntimeContext
from mvp_dataset.core.iterator import DatasetIterator

from .helpers import build_records, write_jsonl_file, write_lance_dataset


@dataclass
class _PairAssembler:
    pending: dict[str, object] | None = None

    def push(self, sample: object):
        assert isinstance(sample, dict)
        if self.pending is None:
            self.pending = sample
            return ()
        first = self.pending
        self.pending = None
        return ({"values": [first["value"], sample["value"]]},)

    def finish(self, *, drop_last: bool = False):
        if self.pending is None or drop_last:
            self.pending = None
            return ()
        pending = self.pending
        self.pending = None
        return ({"values": [pending["value"]]},)

    def state_dict(self) -> dict[str, object]:
        return {"pending": self.pending}

    def load_state_dict(self, state: dict[str, object]) -> None:
        pending = state["pending"]
        assert pending is None or isinstance(pending, dict)
        self.pending = pending

    def fingerprint(self) -> str:
        return "pair-assembler-v1"


def _make_pair_assembler(_context: RuntimeContext) -> _PairAssembler:
    return _PairAssembler()


def test_snapshot_is_lazy_and_reuses_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MVP_DATASET_CACHE_DIR", str(tmp_path / "cache"))
    source = write_lance_dataset(tmp_path, build_records(count=4))
    calls = 0

    def transform(sample: object) -> object:
        nonlocal calls
        calls += 1
        assert isinstance(sample, dict)
        return {**sample, "processed": True}

    dataset = Dataset.from_source("lance", source).map(transform).snapshot()
    assert calls == 0

    first = list(dataset)
    assert calls == 4
    shutil.rmtree(source)
    second = list(dataset)
    assert calls == 4
    assert first == second
    assert [sample["value"] for sample in first] == [0, 1, 2, 3]
    assert all(sample["processed"] is True for sample in first)
    assert [sample["__source_key__"] for sample in first] == [f"{source}:{index}" for index in range(4)]
    assert all(sample["__source_file__"] == source for sample in first)
    assert all(sample["__file__"] != source for sample in first)


def test_snapshot_repartitions_global_order_for_current_context(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MVP_DATASET_CACHE_DIR", str(tmp_path / "cache"))
    source = write_lance_dataset(tmp_path, build_records(count=7))
    upstream = Dataset.from_source("lance", source, context=RuntimeContext(rank=1, world_size=2))
    snapshot = upstream.snapshot()

    with ThreadPoolExecutor(max_workers=2) as executor:
        rank0, rank1 = executor.map(
            lambda rank: list(DatasetIterator(snapshot, context=RuntimeContext(rank=rank, world_size=2))),
            range(2),
        )
    merged = list(DatasetIterator(snapshot, context=RuntimeContext()))

    assert [sample["value"] for sample in merged] == [0, 2, 4, 6, 1, 3, 5]
    assert [sample["value"] for sample in rank0] == [0, 4, 1, 5]
    assert [sample["value"] for sample in rank1] == [2, 6, 3]


def test_snapshot_custom_fingerprint_controls_reuse(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MVP_DATASET_CACHE_DIR", str(tmp_path / "cache"))
    source = write_lance_dataset(tmp_path, build_records(count=2))
    first_calls = 0
    second_calls = 0

    def first_map(sample: object) -> object:
        nonlocal first_calls
        first_calls += 1
        assert isinstance(sample, dict)
        return {**sample, "version": "first"}

    def second_map(sample: object) -> object:
        nonlocal second_calls
        second_calls += 1
        assert isinstance(sample, dict)
        return {**sample, "version": "second"}

    first = Dataset.from_source("lance", source).map(first_map).snapshot(lambda: "shared")
    assert {sample["version"] for sample in first} == {"first"}
    assert first_calls == 2

    reused = Dataset.from_source("lance", source).map(second_map).snapshot(lambda: "shared")
    assert {sample["version"] for sample in reused} == {"first"}
    assert second_calls == 0

    rebuilt = Dataset.from_source("lance", source).map(second_map).snapshot(lambda: "changed")
    assert {sample["version"] for sample in rebuilt} == {"second"}
    assert second_calls == 2


def test_snapshot_pipeline_fingerprint_invalidates_changed_map(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MVP_DATASET_CACHE_DIR", str(tmp_path / "cache"))
    source = write_lance_dataset(tmp_path, build_records(count=2))
    second_calls = 0

    def first_map(sample: object) -> object:
        assert isinstance(sample, dict)
        return {**sample, "version": "first"}

    def second_map(sample: object) -> object:
        nonlocal second_calls
        second_calls += 1
        assert isinstance(sample, dict)
        return {**sample, "version": "second"}

    first = Dataset.from_source("lance", source).map(first_map).snapshot()
    second = Dataset.from_source("lance", source).map(second_map).snapshot()

    assert {sample["version"] for sample in first} == {"first"}
    assert {sample["version"] for sample in second} == {"second"}
    assert second_calls == 2


def test_snapshot_supports_post_snapshot_map_assemble_and_batch(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MVP_DATASET_CACHE_DIR", str(tmp_path / "cache"))
    source = write_lance_dataset(tmp_path, build_records(count=5))

    dataset = (
        Dataset.from_source("lance", source)
        .snapshot()
        .map(lambda sample: {**sample, "value": sample["value"] + 10})
        .assemble(_make_pair_assembler)
        .batch(2)
    )

    assert list(dataset) == [
        [{"values": [10, 11]}, {"values": [12, 13]}],
        [{"values": [14]}],
    ]


def test_snapshot_supports_empty_stream(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MVP_DATASET_CACHE_DIR", str(tmp_path / "cache"))
    source = tmp_path / "empty.lance"
    lance.write_dataset(pa.table({"value": pa.array([], type=pa.int64())}), source)

    assert list(Dataset.from_source("lance", source).snapshot()) == []


def test_snapshot_materializes_non_lance_upstream(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MVP_DATASET_CACHE_DIR", str(tmp_path / "cache"))
    source = write_jsonl_file(tmp_path, build_records(count=3))

    outputs = list(Dataset.from_source("jsonl", source).snapshot())

    assert [sample["value"] for sample in outputs] == [0, 1, 2]


def test_snapshot_round_trips_torch_tensors(tmp_path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("MVP_DATASET_CACHE_DIR", str(tmp_path / "cache"))
    source = write_lance_dataset(tmp_path, build_records(count=1))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def add_tensors(sample: object) -> object:
        assert isinstance(sample, dict)
        return {
            **sample,
            "tensor": torch.arange(6, dtype=torch.float16, device=device).reshape(2, 3),
            "nested": {"mask": torch.tensor([True, False], dtype=torch.bool, device=device)},
        }

    output = next(iter(Dataset.from_source("lance", source).map(add_tensors).snapshot(lambda: "tensor-v1")))

    assert isinstance(output["tensor"], torch.Tensor)
    assert output["tensor"].dtype == torch.float16
    assert tuple(output["tensor"].shape) == (2, 3)
    assert output["tensor"].device.type == device.type
    assert torch.equal(output["tensor"], torch.arange(6, dtype=torch.float16, device=device).reshape(2, 3))
    assert isinstance(output["nested"]["mask"], torch.Tensor)
    assert output["nested"]["mask"].dtype == torch.bool
    assert output["nested"]["mask"].device.type == device.type


def test_snapshot_concurrent_readers_build_upstream_once(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MVP_DATASET_CACHE_DIR", str(tmp_path / "cache"))
    source = write_lance_dataset(tmp_path, build_records(count=4))
    calls = 0

    def transform(sample: object) -> object:
        nonlocal calls
        calls += 1
        return sample

    snapshot = Dataset.from_source("lance", source).map(transform).snapshot()
    with ThreadPoolExecutor(max_workers=2) as executor:
        outputs = list(
            executor.map(
                lambda rank: list(DatasetIterator(snapshot, context=RuntimeContext(rank=rank, world_size=2))),
                range(2),
            )
        )

    assert calls == 4
    assert sorted(sample["value"] for output in outputs for sample in output) == [0, 1, 2, 3]


def test_snapshot_cache_hit_reuses_parts_under_new_topology(tmp_path, monkeypatch) -> None:
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("MVP_DATASET_CACHE_DIR", str(cache_root))
    source = write_lance_dataset(tmp_path, build_records(count=6))
    calls = 0

    def transform(sample: object) -> object:
        nonlocal calls
        calls += 1
        return sample

    snapshot = Dataset.from_source("lance", source).map(transform).snapshot()
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(
            executor.map(
                lambda rank: list(DatasetIterator(snapshot, context=RuntimeContext(rank=rank, world_size=2))),
                range(2),
            )
        )
    assert calls == 6
    shutil.rmtree(source)

    outputs = list(DatasetIterator(snapshot, context=RuntimeContext(rank=0, world_size=3)))

    assert calls == 6
    assert [sample["value"] for sample in outputs] == [0, 1]
    part_paths = sorted(cache_root.glob("*/dataset-snapshot-v1/*/slot-*/data.lance"))
    assert [path.parent.name for path in part_paths] == ["slot-00000000", "slot-00000001"]


def test_snapshot_source_supports_resume(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MVP_DATASET_CACHE_DIR", str(tmp_path / "cache"))
    source = write_lance_dataset(tmp_path, build_records(count=5))
    snapshot = Dataset.from_source("lance", source).snapshot()
    iterator = iter(snapshot)
    prefix = [next(iterator), next(iterator)]
    state = iterator.state_dict()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        resumed = snapshot.load_state_dict(state)

    assert prefix + list(resumed) == list(snapshot)


def test_snapshot_split_uses_materialized_row_space(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MVP_DATASET_CACHE_DIR", str(tmp_path / "cache"))
    source = write_lance_dataset(tmp_path, build_records(count=7))
    snapshot = Dataset.from_source("lance", source).snapshot()

    first, second = snapshot.split([1, 1])

    assert [sample["value"] for sample in first] == [0, 1, 2, 3]
    assert [sample["value"] for sample in second] == [4, 5, 6]
    with pytest.raises(ValueError, match="UnsupportedNestedSnapshotSubset"):
        first.sample(0.5)


def test_snapshot_sample_is_seeded_and_row_exact(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MVP_DATASET_CACHE_DIR", str(tmp_path / "cache"))
    source = write_lance_dataset(tmp_path, build_records(count=10))
    snapshot = Dataset.from_source("lance", source).snapshot()

    first = [sample["value"] for sample in snapshot.sample(0.4, seed=7)]
    repeated = [sample["value"] for sample in snapshot.sample(0.4, seed=7)]
    changed = [sample["value"] for sample in snapshot.sample(0.4, seed=8)]

    assert len(first) == 4
    assert len(set(first)) == 4
    assert first == repeated
    assert first != changed


def test_snapshot_rejects_invalid_configuration(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MVP_DATASET_CACHE_DIR", str(tmp_path / "cache"))
    source = write_lance_dataset(tmp_path, build_records(count=1))
    dataset = Dataset.from_source("lance", source)

    with pytest.raises(ValueError, match="finite pipeline"):
        Dataset.from_source("lance", source, resample=True).snapshot()
    with pytest.raises(ValueError, match="non-empty string"):
        dataset.snapshot(lambda: "")
    with pytest.raises(TypeError, match="must be callable"):
        dataset.snapshot("invalid")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="expected dict sample"):
        list(dataset.map(lambda sample: sample["value"]).snapshot())
    with pytest.raises(ValueError, match="SnapshotSourceMetadataConflict"):
        list(dataset.map(lambda sample: {**sample, "__source_key__": "existing"}).snapshot(lambda: "conflict"))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        resumed = dataset.load_state_dict(dataset.state_dict())
    with pytest.raises(ValueError, match="pending resume state"):
        resumed.snapshot()
