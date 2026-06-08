from __future__ import annotations

import pickle

import pytest

from mvp_dataset import Dataset
from mvp_dataset.core.context import RuntimeContext
from mvp_dataset.sources.parquet.fragments import list_parquet_fragments
from mvp_dataset.sources.parquet.reader import resolve_parquet_batch_size

from .helpers import (
    build_records,
    normalize_sample,
    write_jsonl_file,
    write_lance_dataset,
    write_parquet_file,
    write_tar_shards,
)


def _build_source(tmp_path, source_kind: str, records: list[dict[str, object]]) -> str | list[str]:
    if source_kind == "tar":
        return write_tar_shards(tmp_path, records, num_shards=3)
    if source_kind == "jsonl":
        return write_jsonl_file(tmp_path, records)
    if source_kind == "parquet":
        return write_parquet_file(tmp_path, records, row_group_size=2)
    if source_kind == "lance":
        pytest.importorskip("lance")
        return write_lance_dataset(tmp_path, records, max_rows_per_file=2)
    raise AssertionError(f"unexpected source_kind={source_kind!r}")


_SOURCE_CASES = ["tar", "jsonl", "parquet", "lance"]


@pytest.mark.parametrize("source_kind", _SOURCE_CASES)
def test_single_process_reading_across_sources(tmp_path, source_kind: str) -> None:
    records = build_records()
    source = _build_source(tmp_path, source_kind, records)

    observed = [normalize_sample(sample) for sample in Dataset.from_source(source_kind, shards=source)]

    assert sorted(observed, key=lambda sample: int(sample["value"])) == records


@pytest.mark.parametrize("source_kind", _SOURCE_CASES)
def test_source_datasets_are_picklable(tmp_path, source_kind: str) -> None:
    source = _build_source(tmp_path, source_kind, build_records())

    pickle.dumps(Dataset.from_source(source_kind, shards=source))


@pytest.mark.parametrize("source_kind", _SOURCE_CASES)
def test_sources_shard_across_ranks(tmp_path, monkeypatch, source_kind: str) -> None:
    records = build_records(count=8)
    source = _build_source(tmp_path, source_kind, records)

    def read_rank(rank: int) -> list[dict[str, object]]:
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("RANK", str(rank))
        dataset = Dataset.from_source(source_kind, shards=source, context=RuntimeContext(seed=17), shuffle_mode="none")
        return [normalize_sample(sample) for sample in dataset]

    observed_rank0 = read_rank(0)
    observed_rank1 = read_rank(1)
    combined = sorted(observed_rank0 + observed_rank1, key=lambda sample: int(sample["value"]))

    assert combined == records
    assert {sample["id"] for sample in observed_rank0}.isdisjoint({sample["id"] for sample in observed_rank1})


@pytest.mark.parametrize(
    ("source_kind", "explicit_mode", "metadata_key"),
    [
        ("jsonl", "shard_aware", "__index_in_file__"),
        ("tar", "shard_aware", "__key__"),
        ("parquet", "fragment_aware", "__index_in_file__"),
    ],
)
def test_default_shuffle_mode_matches_explicit_mode(
    tmp_path,
    source_kind: str,
    explicit_mode: str,
    metadata_key: str,
) -> None:
    source = _build_source(tmp_path, source_kind, build_records(count=8))
    context = RuntimeContext(seed=19)

    default = Dataset.from_source(source_kind, shards=source, context=context)
    explicit = Dataset.from_source(source_kind, shards=source, context=context, shuffle_mode=explicit_mode)

    assert [sample[metadata_key] for sample in default] == [sample[metadata_key] for sample in explicit]


@pytest.mark.parametrize(
    ("source_kind", "pattern"),
    [
        ("jsonl", r"\[UnsupportedJsonlShuffleMode\]"),
        ("tar", r"\[UnsupportedTarShuffleMode\]"),
        ("parquet", r"\[UnsupportedParquetShuffleMode\]"),
    ],
)
def test_global_shuffle_is_only_supported_by_lance(tmp_path, source_kind: str, pattern: str) -> None:
    source = _build_source(tmp_path, source_kind, build_records())

    with pytest.raises(ValueError, match=pattern):
        Dataset.from_source(source_kind, shards=source, shuffle_mode="global")


def test_parquet_fragment_grouping_and_batch_size_env(tmp_path, monkeypatch) -> None:
    path = write_parquet_file(tmp_path, build_records(count=6), row_group_size=2)

    grouped = list_parquet_fragments([path], min_row_groups_per_fragment=2)
    fallback = list_parquet_fragments([path], min_row_groups_per_fragment=2, min_fragments=3)
    monkeypatch.setenv("MVP_DATASET_PARQUET_BATCH_SIZE", "1234")

    assert [fragment.row_groups for fragment in grouped] == [(0, 1), (2,)]
    assert [fragment.num_rows for fragment in grouped] == [4, 2]
    assert [fragment.row_groups for fragment in fallback] == [(0,), (1,), (2,)]
    assert resolve_parquet_batch_size() == 1234


def test_lance_global_shuffle_is_deterministic_and_sharded(tmp_path, monkeypatch) -> None:
    pytest.importorskip("lance")

    records = build_records(count=10)
    path = write_lance_dataset(tmp_path, records)
    context = RuntimeContext(seed=17)

    first = [
        sample["__global_index__"]
        for sample in Dataset.from_source("lance", shards=path, context=context, shuffle_mode="global")
    ]
    second = [
        sample["__global_index__"]
        for sample in Dataset.from_source("lance", shards=path, context=context, shuffle_mode="global")
    ]

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    rank0 = [
        sample["__global_index__"]
        for sample in Dataset.from_source("lance", shards=path, context=context, shuffle_mode="global")
    ]
    monkeypatch.setenv("RANK", "1")
    rank1 = [
        sample["__global_index__"]
        for sample in Dataset.from_source("lance", shards=path, context=context, shuffle_mode="global")
    ]

    assert first == second
    assert sorted(first) == list(range(len(records)))
    assert sorted(rank0 + rank1) == list(range(len(records)))
    assert set(rank0).isdisjoint(rank1)


def test_lance_dataset_chunk_aware_shuffle_reads_all_rows_once(tmp_path, monkeypatch) -> None:
    pytest.importorskip("lance")

    records = build_records(count=12)
    path = write_lance_dataset(tmp_path, records)
    rank0 = Dataset.from_source(
        "lance",
        shards=path,
        context=RuntimeContext(seed=109),
        shuffle_mode="chunk_aware",
        chunk_aware_shuffle_chunk_size=3,
        chunk_aware_shuffle_k=2,
    )
    rank1 = Dataset.from_source(
        "lance",
        shards=path,
        context=RuntimeContext(seed=109),
        shuffle_mode="chunk_aware",
        chunk_aware_shuffle_chunk_size=3,
        chunk_aware_shuffle_k=2,
    )

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    observed_rank0 = [normalize_sample(sample) for sample in rank0]
    monkeypatch.setenv("RANK", "1")
    observed_rank1 = [normalize_sample(sample) for sample in rank1]
    observed = observed_rank0 + observed_rank1

    assert sorted(observed, key=lambda sample: int(sample["value"])) == records
    assert observed != records


def test_lance_chunk_aware_shuffle_validates_parameters(tmp_path) -> None:
    pytest.importorskip("lance")

    path = write_lance_dataset(tmp_path, build_records(count=4))

    with pytest.raises(ValueError, match="chunk_size"):
        Dataset.from_source(
            "lance",
            shards=path,
            shuffle_mode="chunk_aware",
            chunk_aware_shuffle_chunk_size=0,
        )

    with pytest.raises(ValueError, match="k"):
        Dataset.from_source(
            "lance",
            shards=path,
            shuffle_mode="chunk_aware",
            chunk_aware_shuffle_k=0,
        )


def test_lance_fragment_aware_shuffle_reads_all_rows_once(tmp_path) -> None:
    pytest.importorskip("lance")

    records = build_records(count=8)
    path = write_lance_dataset(tmp_path, records, max_rows_per_file=2)
    dataset = Dataset.from_source("lance", shards=path, context=RuntimeContext(seed=11), shuffle_mode="fragment_aware")

    observed = [normalize_sample(sample) for sample in dataset]

    assert sorted(observed, key=lambda sample: str(sample["id"])) == records
