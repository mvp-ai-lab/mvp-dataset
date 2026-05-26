from __future__ import annotations

import importlib.util
import json

import pytest

from mvp_dataset import Dataset, TorchLoader

from .helpers import (
    build_records,
    normalize_sample,
    write_jsonl_file,
    write_lance_dataset,
    write_lance_table,
    write_parquet_file,
    write_tar_shards,
)


def identity_collate(batch: list[object]) -> list[object]:
    return batch


def _build_source(tmp_path, source_kind: str, records: list[dict[str, object]]) -> str | list[str]:
    if source_kind == "tars":
        return write_tar_shards(tmp_path, records, num_shards=2)
    if source_kind == "jsonl":
        return write_jsonl_file(tmp_path, records)
    if source_kind == "parquet":
        return write_parquet_file(tmp_path, records, row_group_size=2)
    if source_kind == "lance":
        return write_lance_dataset(tmp_path, records)
    raise AssertionError(f"unexpected source_kind={source_kind!r}")


@pytest.mark.parametrize(
    ("source_kind", "requires_lance"),
    [
        ("tars", False),
        ("jsonl", False),
        ("parquet", False),
        ("lance", True),
    ],
)
def test_dataset_resume_matches_uninterrupted_read(tmp_path, source_kind: str, requires_lance: bool) -> None:
    if requires_lance and importlib.util.find_spec("lance") is None:
        pytest.skip("lance is not installed")

    records = build_records(count=8)
    source = _build_source(tmp_path, source_kind, records)

    full = [normalize_sample(sample) for sample in Dataset.from_source(source_kind, shards=source)]

    partial_dataset = Dataset.from_source(source_kind, shards=source)
    partial_iter = iter(partial_dataset)
    prefix = [normalize_sample(next(partial_iter)) for _ in range(3)]
    state = partial_dataset.state_dict()

    resumed_dataset = Dataset.from_source(source_kind, shards=source)
    resumed_dataset.load_state_dict(state)
    resumed = [normalize_sample(sample) for sample in resumed_dataset]

    assert prefix + resumed == full


def test_torch_loader_resume_matches_uninterrupted_batches(tmp_path) -> None:
    records = build_records(count=8)
    source = write_parquet_file(tmp_path, records, row_group_size=2)

    full_loader = TorchLoader(
        Dataset.from_source("parquet", shards=source),
        num_workers=0,
        batch_size=2,
        collate_fn=identity_collate,
    )
    full = [[normalize_sample(sample) for sample in batch] for batch in full_loader]

    partial_loader = TorchLoader(
        Dataset.from_source("parquet", shards=source),
        num_workers=0,
        batch_size=2,
        collate_fn=identity_collate,
    )
    partial_iter = iter(partial_loader)
    prefix = [[normalize_sample(sample) for sample in next(partial_iter)] for _ in range(2)]
    state = partial_loader.state_dict()

    resumed_loader = TorchLoader(
        Dataset.from_source("parquet", shards=source),
        num_workers=0,
        batch_size=2,
        collate_fn=identity_collate,
    )
    resumed_loader.load_state_dict(state)
    resumed = [[normalize_sample(sample) for sample in batch] for batch in resumed_loader]

    assert prefix + resumed == full
    assert state["step"] == 2


def test_resume_rejects_changed_source_fingerprint(tmp_path) -> None:
    records = build_records(count=4)
    source = write_jsonl_file(tmp_path, records)
    dataset = Dataset.from_source("jsonl", shards=source)
    next(iter(dataset))
    state = dataset.state_dict()

    with open(source, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": "new", "text": "new", "value": 99}, ensure_ascii=True) + "\n")

    with pytest.raises(ValueError, match="source_fingerprint"):
        Dataset.from_source("jsonl", shards=source).load_state_dict(state)


def test_resume_rejects_changed_reader_config(tmp_path) -> None:
    records = build_records(count=4)
    source = write_parquet_file(tmp_path, records, row_group_size=2)
    dataset = Dataset.from_source("parquet", shards=source, columns=("id", "text", "value"))
    next(iter(dataset))
    state = dataset.state_dict()

    with pytest.raises(ValueError, match="source_fingerprint"):
        Dataset.from_source("parquet", shards=source, columns=("id", "value")).load_state_dict(state)


def test_resume_rejects_changed_jsonl_ref_fields(tmp_path) -> None:
    records = build_records(count=4)
    source = write_jsonl_file(tmp_path, records)
    dataset = Dataset.from_source("jsonl", shards=source)
    next(iter(dataset))
    state = dataset.state_dict()

    with pytest.raises(ValueError, match="source_fingerprint"):
        Dataset.from_source("jsonl", shards=source, ref_fields=(("payload", str(tmp_path)),)).load_state_dict(state)


def test_resume_rejects_runtime_change(tmp_path, monkeypatch) -> None:
    records = build_records(count=4)
    source = write_jsonl_file(tmp_path, records)
    dataset = Dataset.from_source("jsonl", shards=source)
    next(iter(dataset))
    state = dataset.state_dict()

    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="runtime_fingerprint"):
        Dataset.from_source("jsonl", shards=source).load_state_dict(state)


def test_resume_rejects_unsupported_shuffle_stage(tmp_path) -> None:
    records = build_records(count=4)
    source = write_jsonl_file(tmp_path, records)
    dataset = Dataset.from_source("jsonl", shards=source).shuffle(buffer_size=2)

    with pytest.raises(ValueError, match="UnsupportedResumeStage"):
        dataset.state_dict()


def test_lance_resume_matches_uninterrupted_multipart_read(tmp_path) -> None:
    if importlib.util.find_spec("lance") is None:
        pytest.skip("lance is not installed")

    records = build_records(count=10)
    first = write_lance_table(tmp_path, "first.lance", records[:4])
    second = write_lance_table(tmp_path, "second.lance", records[4:])
    source = [first, second]

    full = [normalize_sample(sample) for sample in Dataset.from_source("lance", shards=source)]

    partial_dataset = Dataset.from_source("lance", shards=source)
    partial_iter = iter(partial_dataset)
    prefix = [normalize_sample(next(partial_iter)) for _ in range(6)]
    state = partial_dataset.state_dict()

    resumed_dataset = Dataset.from_source("lance", shards=source)
    resumed_dataset.load_state_dict(state)
    resumed = [normalize_sample(sample) for sample in resumed_dataset]

    assert prefix + resumed == full


def test_lance_resume_rejects_shuffle_mode(tmp_path) -> None:
    if importlib.util.find_spec("lance") is None:
        pytest.skip("lance is not installed")

    source = write_lance_dataset(tmp_path, build_records(count=4))
    dataset = Dataset.from_source("lance", shards=source, shuffle_mode="global")

    with pytest.raises(ValueError, match="UnsupportedResumeSource"):
        dataset.state_dict()


def test_lance_resume_rejects_changed_ref_index_scope(tmp_path) -> None:
    if importlib.util.find_spec("lance") is None:
        pytest.skip("lance is not installed")

    source = write_lance_dataset(tmp_path, build_records(count=4))
    dataset = Dataset.from_source("lance", shards=source, ref_index_scope="shared")
    next(iter(dataset))
    state = dataset.state_dict()

    with pytest.raises(ValueError, match="source_fingerprint"):
        Dataset.from_source("lance", shards=source, ref_index_scope="process").load_state_dict(state)
