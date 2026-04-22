from __future__ import annotations

import importlib.util
import pickle

import pytest

from mvp_dataset import Dataset
from mvp_dataset.sources.parquet.utils import list_parquet_fragments

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


@pytest.mark.parametrize(
    ("source_kind", "requires_lance"),
    [
        ("tars", False),
        ("jsonl", False),
        ("parquet", False),
        ("lance", True),
    ],
)
def test_source_datasets_are_picklable(tmp_path, source_kind: str, requires_lance: bool) -> None:
    if requires_lance and importlib.util.find_spec("lance") is None:
        pytest.skip("lance is not installed")

    records = build_records()
    source = _build_source(tmp_path, source_kind, records)

    dataset = Dataset.from_source(source_kind, shards=source)

    pickle.dumps(dataset)


def test_list_parquet_fragments_groups_by_row_group_count(tmp_path) -> None:
    records = build_records(count=6)
    path = write_parquet_file(tmp_path, records, row_group_size=2)

    fragments = list_parquet_fragments([path], min_row_groups_per_fragment=2)

    assert [fragment.row_groups for fragment in fragments] == [(0, 1), (2,)]
    assert [fragment.row_offset for fragment in fragments] == [0, 4]
    assert [fragment.num_rows for fragment in fragments] == [4, 2]


def test_list_parquet_fragments_falls_back_to_one_row_group_per_fragment(tmp_path) -> None:
    records = build_records(count=6)
    path = write_parquet_file(tmp_path, records, row_group_size=2)

    fragments = list_parquet_fragments([path], min_row_groups_per_fragment=2, min_fragments=3)

    assert [fragment.row_groups for fragment in fragments] == [(0,), (1,), (2,)]
    assert [fragment.row_offset for fragment in fragments] == [0, 2, 4]
    assert [fragment.num_rows for fragment in fragments] == [2, 2, 2]


def test_list_parquet_fragments_preserves_shard_order_across_parallel_metadata_collection(
    tmp_path,
) -> None:
    records = build_records(count=6)
    shard_a_root = tmp_path / "shard_a"
    shard_b_root = tmp_path / "shard_b"
    shard_a_root.mkdir()
    shard_b_root.mkdir()

    shard_a = write_parquet_file(shard_a_root, records, row_group_size=2)
    shard_b = write_parquet_file(shard_b_root, records, row_group_size=2)

    fragments = list_parquet_fragments([shard_a, shard_b], min_row_groups_per_fragment=2)

    assert [fragment.path for fragment in fragments] == [shard_a, shard_a, shard_b, shard_b]
    assert [fragment.row_groups for fragment in fragments] == [(0, 1), (2,), (0, 1), (2,)]
    assert [fragment.row_offset for fragment in fragments] == [0, 4, 0, 4]
    assert [fragment.num_rows for fragment in fragments] == [4, 2, 4, 2]


def test_parquet_dataset_from_source_accepts_min_row_groups_per_fragment(tmp_path) -> None:
    records = build_records(count=6)
    path = write_parquet_file(tmp_path, records, row_group_size=2)

    dataset = Dataset.from_source(
        "parquet",
        shards=path,
        min_row_groups_per_fragment=2,
    )

    assert [fragment.row_groups for fragment in dataset._source] == [(0, 1), (2,)]
