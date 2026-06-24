from __future__ import annotations

import json

import pytest

from mvp_dataset import Dataset
from mvp_dataset.core.context import RuntimeContext

from .helpers import (
    build_records,
    normalize_sample,
    write_lance_dataset,
    write_parquet_file,
    write_tar_shards,
)

_SOURCE_CASES = ["tar", "jsonl", "parquet", "lance"]


def _write_jsonl_shards(root, records: list[dict[str, object]], *, num_shards: int) -> list[str]:
    root.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for shard_index in range(num_shards):
        path = root / f"shard_{shard_index:06d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records[shard_index::num_shards]:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        paths.append(str(path))
    return paths


def _build_source(tmp_path, source_kind: str, records: list[dict[str, object]]):
    if source_kind == "tar":
        return write_tar_shards(tmp_path, records, num_shards=10)
    if source_kind == "jsonl":
        return _write_jsonl_shards(tmp_path / "jsonl", records, num_shards=10)
    if source_kind == "parquet":
        return write_parquet_file(tmp_path, records, row_group_size=2)
    if source_kind == "lance":
        pytest.importorskip("lance")
        return write_lance_dataset(tmp_path, records, max_rows_per_file=5)
    raise AssertionError(f"unexpected source_kind={source_kind!r}")


def _values(dataset: Dataset) -> list[int]:
    return [int(normalize_sample(sample)["value"]) for sample in dataset]


@pytest.mark.parametrize("source_kind", _SOURCE_CASES)
def test_split_is_disjoint_and_complete(tmp_path, source_kind: str) -> None:
    records = build_records(count=100)
    source = _build_source(tmp_path, source_kind, records)

    train, val = Dataset.from_source(source_kind, shards=source).split([0.8, 0.2])
    train_values = _values(train)
    val_values = _values(val)

    assert set(train_values).isdisjoint(val_values)
    assert sorted(train_values + val_values) == [record["value"] for record in records]
    assert len(train_values) > len(val_values)


@pytest.mark.parametrize("source_kind", _SOURCE_CASES)
def test_split_is_reproducible(tmp_path, source_kind: str) -> None:
    records = build_records(count=100)
    source = _build_source(tmp_path, source_kind, records)
    dataset = Dataset.from_source(source_kind, shards=source)

    first = set(_values(dataset.split([0.8, 0.2])[0]))
    same = set(_values(dataset.split([0.8, 0.2])[0]))

    assert first == same


@pytest.mark.parametrize("source_kind", _SOURCE_CASES)
def test_split_fractions_are_normalized(tmp_path, source_kind: str) -> None:
    records = build_records(count=100)
    source = _build_source(tmp_path, source_kind, records)
    dataset = Dataset.from_source(source_kind, shards=source)

    ratio_values = set(_values(dataset.split([8, 2])[0]))
    fraction_values = set(_values(dataset.split([0.8, 0.2])[0]))

    assert ratio_values == fraction_values


def test_split_supports_more_than_two_parts(tmp_path) -> None:
    records = build_records(count=100)
    source = _build_source(tmp_path, "lance", records)

    train, val, test = Dataset.from_source("lance", shards=source).split([0.8, 0.1, 0.1])
    train_values, val_values, test_values = _values(train), _values(val), _values(test)

    assert (len(train_values), len(val_values), len(test_values)) == (80, 10, 10)
    assert sorted(train_values + val_values + test_values) == [record["value"] for record in records]


def test_lance_split_is_row_exact(tmp_path) -> None:
    records = build_records(count=100)
    source = _build_source(tmp_path, "lance", records)

    train, val = Dataset.from_source("lance", shards=source).split([0.8, 0.2])

    assert len(_values(train)) == 80
    assert len(_values(val)) == 20


def test_lance_split_membership_uses_source_order(tmp_path) -> None:
    records = build_records(count=100)
    source = _build_source(tmp_path, "lance", records)
    dataset = Dataset.from_source("lance", shards=source, shuffle_mode="none")

    train, val = dataset.split([0.8, 0.2])

    assert _values(train) == list(range(80))
    assert _values(val) == list(range(80, 100))


def test_lance_split_membership_can_follow_source_seed(tmp_path) -> None:
    records = build_records(count=100)
    source = _build_source(tmp_path, "lance", records)

    seed7 = set(
        _values(
            Dataset.from_source("lance", shards=source, context=RuntimeContext(seed=7), shuffle_mode="global").split(
                [0.8, 0.2]
            )[0]
        )
    )
    seed13 = set(
        _values(
            Dataset.from_source("lance", shards=source, context=RuntimeContext(seed=13), shuffle_mode="global").split(
                [0.8, 0.2]
            )[0]
        )
    )

    assert len(seed7) == 80
    assert len(seed13) == 80
    assert seed7 != seed13


@pytest.mark.parametrize("source_kind", _SOURCE_CASES)
def test_sample_returns_reproducible_subset(tmp_path, source_kind: str) -> None:
    records = build_records(count=100)
    source = _build_source(tmp_path, source_kind, records)
    dataset = Dataset.from_source(source_kind, shards=source)
    full = set(record["value"] for record in records)

    first = _values(dataset.sample(0.5, seed=7))
    same = _values(dataset.sample(0.5, seed=7))

    assert first == same
    assert set(first) <= full
    assert 0 < len(first) < len(records)


def test_lance_sample_count_is_exact(tmp_path) -> None:
    records = build_records(count=100)
    source = _build_source(tmp_path, "lance", records)

    sampled = _values(Dataset.from_source("lance", shards=source).sample(0.2, seed=1))

    assert len(sampled) == 20
    assert len(set(sampled)) == 20


@pytest.mark.parametrize("source_kind", _SOURCE_CASES)
def test_sample_rejects_oversample(tmp_path, source_kind: str) -> None:
    source = _build_source(tmp_path, source_kind, build_records(count=20))
    dataset = Dataset.from_source(source_kind, shards=source)

    with pytest.raises(ValueError, match="InvalidSampleFraction"):
        dataset.sample(1.5)
    with pytest.raises(ValueError, match="InvalidSampleFraction"):
        dataset.sample(0.0)


def test_split_rejects_invalid_fractions(tmp_path) -> None:
    source = _build_source(tmp_path, "lance", build_records(count=20))
    dataset = Dataset.from_source("lance", shards=source)

    with pytest.raises(ValueError, match="InvalidSplitFractions"):
        dataset.split([0.8, -0.2])


def test_unit_source_split_requires_enough_shards(tmp_path, monkeypatch) -> None:
    records = build_records(count=40)
    source = _write_jsonl_shards(tmp_path / "jsonl", records, num_shards=4)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    dataset = Dataset.from_source("jsonl", shards=source, context=RuntimeContext(world_size=2))

    # 4 shards, 20%% -> ~1 shard for val, but 2 slots are required.
    with pytest.raises(ValueError, match="InsufficientSubsetUnits"):
        dataset.split([0.8, 0.2])


def test_mixed_source_rejects_subset(tmp_path) -> None:
    records = build_records(count=20)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    left = Dataset.from_source("jsonl", shards=_write_jsonl_shards(tmp_path / "a", records, num_shards=3))
    right = Dataset.from_source("jsonl", shards=_write_jsonl_shards(tmp_path / "b", records, num_shards=3))
    mixed = Dataset.from_source("mixed", {"left": left, "right": right})

    with pytest.raises(ValueError, match="UnsupportedSubsetSource"):
        mixed.split([0.8, 0.2])
    with pytest.raises(ValueError, match="UnsupportedSubsetSource"):
        mixed.sample(0.5)


def test_lance_rejects_nested_subset(tmp_path) -> None:
    source = _build_source(tmp_path, "lance", build_records(count=100))
    train, _ = Dataset.from_source("lance", shards=source).split([0.8, 0.2])

    with pytest.raises(ValueError, match="UnsupportedNestedLanceSubset"):
        train.sample(0.5)


def test_lance_sample_distributed_union_matches_single_process(tmp_path, monkeypatch) -> None:
    records = build_records(count=100)
    source = _build_source(tmp_path, "lance", records)

    def read_rank(world_size: int, rank: int) -> list[int]:
        monkeypatch.setenv("WORLD_SIZE", str(world_size))
        monkeypatch.setenv("RANK", str(rank))
        dataset = Dataset.from_source(
            "lance", shards=source, context=RuntimeContext(rank=rank, world_size=world_size, seed=5)
        )
        return _values(dataset.sample(0.4, seed=2))

    single = set(read_rank(1, 0))
    sharded = set(read_rank(2, 0)) | set(read_rank(2, 1))

    assert len(single) == 40
    assert sharded == single


def test_lance_split_resume_matches_continuation(tmp_path) -> None:
    source = _build_source(tmp_path, "lance", build_records(count=100))
    train, _ = Dataset.from_source("lance", shards=source, shuffle_mode="global").split([0.8, 0.2])

    iterator = iter(train)
    consumed = [int(normalize_sample(next(iterator))["value"]) for _ in range(10)]
    state = iterator.state_dict()

    resumed = train.load_state_dict(state)
    remainder = _values(resumed)

    full = _values(train)
    assert consumed + remainder == full
