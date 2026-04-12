from __future__ import annotations

import importlib.util

import pytest

from mvp_dataset import Dataset

from .helpers import (
    build_records,
    normalize_sample,
    write_jsonl_file,
    write_lance_dataset,
    write_parquet_file,
    write_tar_shards,
)


def _build_source(tmp_path, source_kind: str, records: list[dict[str, object]]) -> str | list[str]:
    if source_kind == "tars":
        return write_tar_shards(tmp_path, records, num_shards=1)
    if source_kind == "jsonl":
        return write_jsonl_file(tmp_path, records)
    if source_kind == "parquet":
        return write_parquet_file(tmp_path, records)
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
def test_single_process_reading_across_sources(tmp_path, source_kind: str, requires_lance: bool) -> None:
    if requires_lance and importlib.util.find_spec("lance") is None:
        pytest.skip("lance is not installed")

    records = build_records()
    source = _build_source(tmp_path, source_kind, records)

    dataset = Dataset.from_source(source_kind, shards=source)
    observed = [normalize_sample(sample) for sample in dataset]

    assert observed == records
