from __future__ import annotations

import pickle

import pytest

from mvp_dataset import Dataset
from mvp_dataset.core.context import RuntimeContext
from mvp_dataset.sources.parquet.chunks import list_parquet_chunks
from mvp_dataset.sources.parquet.reader import resolve_parquet_batch_size

from .helpers import (
    build_records,
    normalize_sample,
    write_jsonl_file,
    write_lance_dataset,
    write_lance_table,
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
        ("parquet", "chunk_aware", "__index_in_file__"),
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


def test_parquet_chunk_grouping_and_batch_size_env(tmp_path, monkeypatch) -> None:
    path = write_parquet_file(tmp_path, build_records(count=6), row_group_size=2)

    grouped = list_parquet_chunks([path], min_row_groups_per_chunk=2)
    fallback = list_parquet_chunks([path], min_row_groups_per_chunk=2, min_chunks=3)
    monkeypatch.setenv("MVP_DATASET_PARQUET_BATCH_SIZE", "1234")

    assert [chunk.row_groups for chunk in grouped] == [(0, 1), (2,)]
    assert [chunk.num_rows for chunk in grouped] == [4, 2]
    assert [chunk.row_groups for chunk in fallback] == [(0,), (1,), (2,)]
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


def test_lance_merges_multiple_datasets_into_global_row_space(tmp_path) -> None:
    pytest.importorskip("lance")

    records_a = build_records(count=3)
    records_b = [{**record, "id": f"b-{record['id']}"} for record in build_records(count=2)]
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    path_a = write_lance_dataset(root_a, records_a)
    path_b = write_lance_dataset(root_b, records_b)

    observed = list(
        Dataset.from_source(
            "lance",
            shards=[path_a, path_b],
            read_batch_size=2,
            shuffle_mode="none",
        )
    )

    assert [sample["id"] for sample in observed] == [sample["id"] for sample in records_a + records_b]
    assert [sample["__global_index__"] for sample in observed] == [0, 1, 2, 3, 4]
    assert [sample["__local_index__"] for sample in observed] == [0, 1, 2, 0, 1]
    assert [sample["__file__"] for sample in observed] == [path_a, path_a, path_a, path_b, path_b]


@pytest.mark.parametrize("shuffle_mode", ["none", "global", "chunk"])
def test_lance_read_batch_size_does_not_change_output(tmp_path, shuffle_mode: str) -> None:
    pytest.importorskip("lance")

    records = build_records(count=11)
    path = write_lance_dataset(tmp_path, records)
    context = RuntimeContext(seed=23)
    chunk_shuffle = {"chunk_size": 3, "k": 2, "row_order": "permuted"} if shuffle_mode == "chunk" else None

    small = list(
        Dataset.from_source(
            "lance",
            shards=path,
            context=context,
            read_batch_size=1,
            shuffle_mode=shuffle_mode,
            chunk_shuffle=chunk_shuffle,
        )
    )
    large = list(
        Dataset.from_source(
            "lance",
            shards=path,
            context=context,
            read_batch_size=5,
            shuffle_mode=shuffle_mode,
            chunk_shuffle=chunk_shuffle,
        )
    )

    assert small == large


def test_lance_chunk_shuffle_can_read_rows_sequentially_within_chunks(tmp_path) -> None:
    pytest.importorskip("lance")

    records = build_records(count=11)
    path = write_lance_dataset(tmp_path, records)

    observed = [
        sample["__global_index__"]
        for sample in Dataset.from_source(
            "lance",
            shards=path,
            context=RuntimeContext(seed=109),
            shuffle_mode="chunk",
            chunk_shuffle={"chunk_size": 3, "k": 2, "row_order": "sequential"},
        )
    ]

    assert sorted(observed) == list(range(len(records)))
    assert observed != list(range(len(records)))
    runs: list[list[int]] = []
    for index in observed:
        if runs and index == runs[-1][-1] + 1:
            runs[-1].append(index)
        else:
            runs.append([index])
    for run in runs:
        assert run[0] % 3 == 0
        assert len(run) <= 3
        assert run == list(range(run[0], run[0] + len(run)))


def test_lance_dataset_chunk_shuffle_reads_all_rows_once(tmp_path, monkeypatch) -> None:
    pytest.importorskip("lance")

    records = build_records(count=12)
    path = write_lance_dataset(tmp_path, records)
    chunk_shuffle = {"chunk_size": 3, "k": 2}
    rank0 = Dataset.from_source(
        "lance",
        shards=path,
        context=RuntimeContext(seed=109),
        shuffle_mode="chunk",
        chunk_shuffle=chunk_shuffle,
    )
    rank1 = Dataset.from_source(
        "lance",
        shards=path,
        context=RuntimeContext(seed=109),
        shuffle_mode="chunk",
        chunk_shuffle=chunk_shuffle,
    )

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    observed_rank0 = [normalize_sample(sample) for sample in rank0]
    monkeypatch.setenv("RANK", "1")
    observed_rank1 = [normalize_sample(sample) for sample in rank1]
    observed = observed_rank0 + observed_rank1

    assert sorted(observed, key=lambda sample: int(sample["value"])) == records
    assert observed != records


def test_lance_chunk_shuffle_validates_parameters(tmp_path) -> None:
    pytest.importorskip("lance")

    path = write_lance_dataset(tmp_path, build_records(count=4))

    with pytest.raises(ValueError, match="chunk_size"):
        Dataset.from_source(
            "lance",
            shards=path,
            shuffle_mode="chunk",
            chunk_shuffle={"chunk_size": 0},
        )

    with pytest.raises(ValueError, match="k"):
        Dataset.from_source(
            "lance",
            shards=path,
            shuffle_mode="chunk",
            chunk_shuffle={"k": 0},
        )

    with pytest.raises(ValueError, match="row_order"):
        Dataset.from_source(
            "lance",
            shards=path,
            shuffle_mode="chunk",
            chunk_shuffle={"row_order": "unknown"},
        )

    with pytest.raises(ValueError, match="unknown config key"):
        Dataset.from_source(
            "lance",
            shards=path,
            shuffle_mode="chunk",
            chunk_shuffle={"unknown": 1},
        )


def test_lance_rejects_unknown_shuffle_mode(tmp_path) -> None:
    pytest.importorskip("lance")

    path = write_lance_dataset(tmp_path, build_records(count=4))

    with pytest.raises(ValueError, match=r"\[InvalidLanceShuffleMode\]"):
        Dataset.from_source("lance", shards=path, shuffle_mode="unknown")


@pytest.mark.parametrize("shuffle_mode", ["global", "chunk"])
def test_lance_bucketed_ref_index_loads_with_shuffle_modes(tmp_path, shuffle_mode: str) -> None:
    pytest.importorskip("lance")

    main_records = [
        {"id": f"sample-{index}", "text": f"text-{index}", "value": index, "image_ref": f"img-{index % 4}"}
        for index in range(12)
    ]
    ref_records = [{"image_id": f"img-{index}", "image_value": f"resolved-{index}"} for index in range(4)]
    main_path = write_lance_table(tmp_path, "main.lance", main_records)
    ref_path = write_lance_table(tmp_path, "refs.lance", ref_records)

    dataset = Dataset.from_source(
        "lance",
        shards=main_path,
        context=RuntimeContext(seed=37),
        read_batch_size=3,
        shuffle_mode=shuffle_mode,
        chunk_shuffle={"chunk_size": 3, "k": 2} if shuffle_mode == "chunk" else None,
        ref_columns={
            "image_ref": {
                "uri": ref_path,
                "key_column": "image_id",
                "value_column": "image_value",
            }
        },
    ).resolve_ref(
        ["image_ref"],
        resolve_batch_size=2,
        index={
            "scope": "process",
            "build_strategy": "bucketed",
            "bucket_count": 2,
        },
    )

    observed = sorted(dataset, key=lambda sample: int(sample["value"]))

    assert [sample["image_ref"] for sample in observed] == [f"resolved-{index % 4}" for index in range(12)]


def test_lance_ref_index_cache_dir_env(tmp_path, monkeypatch) -> None:
    pytest.importorskip("lance")

    main_path = write_lance_table(
        tmp_path,
        "main.lance",
        [{"id": f"sample-{index}", "image_ref": f"img-{index}"} for index in range(3)],
    )
    ref_path = write_lance_table(
        tmp_path,
        "refs.lance",
        [{"image_id": f"img-{index}", "image_value": f"resolved-{index}"} for index in range(3)],
    )
    cache_dir = tmp_path / "ref-index-cache"
    monkeypatch.setenv("MVP_DATASET_LANCE_REF_INDEX_CACHE_DIR", str(cache_dir))

    dataset = Dataset.from_source(
        "lance",
        shards=main_path,
        ref_columns={
            "image_ref": {
                "uri": ref_path,
                "key_column": "image_id",
                "value_column": "image_value",
            }
        },
    ).resolve_ref(
        ["image_ref"],
        resolve_batch_size=2,
        index={"scope": "process", "build_strategy": "in_memory"},
    )

    assert [sample["image_ref"] for sample in dataset] == [f"resolved-{index}" for index in range(3)]
    assert list(cache_dir.glob("ref-index-*"))
    assert not (tmp_path / "main.lance" / "_mvp_ref_index").exists()
