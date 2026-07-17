from __future__ import annotations

import json
import multiprocessing
import os
import pickle
import queue
import traceback

import pytest

from mvp_dataset import Dataset
from mvp_dataset.core.context import RuntimeContext

from .helpers import (
    build_records,
    write_jsonl_file,
    write_lance_dataset,
    write_lance_table,
)


def _values(dataset: Dataset) -> list[int]:
    return [int(sample["value"]) for sample in dataset]


def _distributed_filter_worker(
    source: str,
    cache_dir: str,
    filter_arg,
    rank: int,
    world_size: int,
    worker_id: int,
    num_workers: int,
    results,
) -> None:
    try:
        os.environ.update(
            RANK=str(rank),
            WORLD_SIZE=str(world_size),
            LOCAL_RANK=str(rank),
            LOCAL_WORLD_SIZE=str(world_size),
            WORKER=str(worker_id),
            NUM_WORKERS=str(num_workers),
            MVP_DATASET_LANCE_FILTER_INDEX_CACHE_DIR=cache_dir,
        )
        from mvp_dataset.sources.lance import filter as lance_filter

        lance_filter.FILTER_INDEX_WAIT_TIMEOUT_SECONDS = 20
        built_parts = []
        build_parts = lance_filter._build_parts

        def tracked_build_parts(entries, predicate_groups):
            built_parts.extend(path.name for path, _, _ in entries)
            return build_parts(entries, predicate_groups)

        lance_filter._build_parts = tracked_build_parts
        dataset = Dataset.from_source(
            "lance",
            source,
            context=RuntimeContext(
                rank=rank,
                world_size=world_size,
                local_rank=rank,
                local_world_size=world_size,
                worker_id=worker_id,
                num_workers=num_workers,
            ),
        ).filter(filter_arg)
        results.put(("ok", {"values": _values(dataset), "built_parts": built_parts}))
    except Exception:
        results.put(("error", traceback.format_exc()))


def test_lance_filter_uses_source_order_and_projection(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MVP_DATASET_LANCE_FILTER_INDEX_CACHE_DIR", str(tmp_path / "cache"))
    source = write_lance_dataset(tmp_path, build_records(20), max_rows_per_file=3)

    rows = list(Dataset.from_source("lance", source, columns=["id"]).filter("value >= 4").filter("value < 9"))

    assert [row["id"] for row in rows] == [f"sample-{index}" for index in range(4, 9)]
    assert all("value" not in row for row in rows)
    assert [row["__global_index__"] for row in rows] == list(range(4, 9))


def test_lance_filter_batches_are_or_combined_deduplicated_and_source_ordered(tmp_path, monkeypatch) -> None:
    import lance

    monkeypatch.setenv("MVP_DATASET_LANCE_FILTER_INDEX_CACHE_DIR", str(tmp_path / "cache"))
    source = write_lance_dataset(tmp_path, build_records(20), max_rows_per_file=3)
    lance.dataset(source).create_scalar_index("value", "BTREE")
    filters = []
    scanner = lance.LanceDataset.scanner

    def tracked_scanner(self, *args, **kwargs):
        if kwargs.get("filter") is not None:
            filters.append((kwargs["filter"], kwargs.get("use_scalar_index")))
        return scanner(self, *args, **kwargs)

    monkeypatch.setattr(lance.LanceDataset, "scanner", tracked_scanner)

    dataset = Dataset.from_source("lance", source).filter(
        [
            "value IN (7, 1, 4)",
            "value IN (4, 2, 7)",
            "value = 18",
        ]
    )

    assert _values(dataset) == [1, 2, 4, 7, 18]
    assert len(filters) == 3
    assert {item for item, _ in filters} == {"value IN (7, 1, 4)", "value IN (4, 2, 7)", "value = 18"}
    assert all(use_scalar_index is True for _, use_scalar_index in filters)


def test_lance_filter_batches_preserve_and_across_calls(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MVP_DATASET_LANCE_FILTER_INDEX_CACHE_DIR", str(tmp_path / "cache"))
    source = write_lance_dataset(tmp_path, build_records(20), max_rows_per_file=3)

    dataset = Dataset.from_source("lance", source).filter(["value < 5", "value > 15"]).filter("value % 2 = 0")

    assert _values(dataset) == [0, 2, 4, 16, 18]

    dataset = (
        Dataset.from_source("lance", source)
        .filter(["value < 10", "value > 15"])
        .filter(["value % 2 = 0", "value = 17"])
    )

    assert _values(dataset) == [0, 2, 4, 6, 8, 16, 17, 18]


def test_lance_filter_batches_cross_bitmap_chunk_boundary(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MVP_DATASET_LANCE_FILTER_INDEX_CACHE_DIR", str(tmp_path / "cache"))
    source = write_lance_dataset(tmp_path, build_records(65_540))

    dataset = Dataset.from_source("lance", source).filter(["value IN (0, 65535)", "value IN (65536, 65539)"])

    assert _values(dataset) == [0, 65_535, 65_536, 65_539]


@pytest.mark.parametrize("shuffle_mode", ["none", "global", "chunk"])
def test_lance_filter_supports_source_shuffle(tmp_path, monkeypatch, shuffle_mode: str) -> None:
    monkeypatch.setenv("MVP_DATASET_LANCE_FILTER_INDEX_CACHE_DIR", str(tmp_path / "cache"))
    source = write_lance_dataset(tmp_path, build_records(30), max_rows_per_file=4)
    chunk_shuffle = {"chunk_size": 4, "k": 2} if shuffle_mode == "chunk" else None

    observed = _values(
        Dataset.from_source(
            "lance",
            source,
            context=RuntimeContext(seed=17),
            shuffle_mode=shuffle_mode,
            chunk_shuffle=chunk_shuffle,
        ).filter("value % 2 = 0")
    )

    assert sorted(observed) == list(range(0, 30, 2))
    if shuffle_mode == "none":
        assert observed == list(range(0, 30, 2))
    else:
        assert observed != list(range(0, 30, 2))


def test_lance_filter_split_sample_and_resume(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MVP_DATASET_LANCE_FILTER_INDEX_CACHE_DIR", str(tmp_path / "cache"))
    source = write_lance_dataset(tmp_path, build_records(40), max_rows_per_file=5)
    dataset = Dataset.from_source("lance", source, shuffle_mode="global").filter("value >= 10")

    left, right = dataset.split([2, 1])
    assert len(_values(left)) == 20
    assert len(_values(right)) == 10
    sampled = _values(dataset.sample(0.2, seed=7))
    assert len(sampled) == 6
    assert set(sampled) <= set(range(10, 40))

    iterator = iter(left)
    consumed = [int(next(iterator)["value"]) for _ in range(4)]
    state = iterator.state_dict()
    with pytest.warns(UserWarning, match="Dataset.load_state_dict"):
        resumed = left.load_state_dict(state)
    assert consumed + _values(resumed) == _values(left)


def test_lance_filter_preserves_multi_source_global_indexes(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MVP_DATASET_LANCE_FILTER_INDEX_CACHE_DIR", str(tmp_path / "cache"))
    left_root = tmp_path / "left"
    right_root = tmp_path / "right"
    left_root.mkdir()
    right_root.mkdir()
    left = write_lance_dataset(left_root, build_records(6), max_rows_per_file=2)
    right_records = [{**row, "id": f"right-{row['id']}", "value": int(row["value"]) + 6} for row in build_records(6)]
    right = write_lance_dataset(right_root, right_records, max_rows_per_file=2)

    rows = list(Dataset.from_source("lance", [left, right]).filter(["value IN (4, 6, 8)", "value IN (5, 7)"]))

    assert [row["value"] for row in rows] == [4, 5, 6, 7, 8]
    assert [row["__global_index__"] for row in rows] == [4, 5, 6, 7, 8]
    assert [row["__local_index__"] for row in rows] == [4, 5, 0, 1, 2]


def test_lance_filter_resolve_ref_uses_original_global_indexes(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MVP_DATASET_LANCE_FILTER_INDEX_CACHE_DIR", str(tmp_path / "filter-cache"))
    monkeypatch.setenv("MVP_DATASET_LANCE_REF_INDEX_CACHE_DIR", str(tmp_path / "ref-cache"))
    main = write_lance_table(
        tmp_path,
        "main.lance",
        [{"id": index, "value": index, "image": f"image-{index}"} for index in range(8)],
    )
    refs = write_lance_table(
        tmp_path,
        "refs.lance",
        [{"key": f"image-{index}", "payload": f"bytes-{index}"} for index in range(8)],
    )
    dataset = Dataset.from_source(
        "lance",
        main,
        ref_columns={"image": {"uri": refs, "key_column": "key", "value_column": "payload"}},
    ).filter("value IN (1, 4, 7)")

    rows = list(dataset.resolve_ref(["image"], index={"scope": "process"}))

    assert [row["__global_index__"] for row in rows] == [1, 4, 7]
    assert [row["image"] for row in rows] == ["bytes-1", "bytes-4", "bytes-7"]


def test_lance_filter_cache_is_picklable_and_reused(tmp_path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("MVP_DATASET_LANCE_FILTER_INDEX_CACHE_DIR", str(cache_dir))
    source = write_lance_dataset(tmp_path, build_records(20), max_rows_per_file=2)
    dataset = Dataset.from_source("lance", source).filter("value >= 5")

    pickle.dumps(dataset)
    assert _values(dataset) == list(range(5, 20))
    manifest = next(cache_dir.rglob("manifest.json"))
    part_mtimes = {path.name: path.stat().st_mtime_ns for path in manifest.parent.glob("part-*.i64")}
    assert len(part_mtimes) > 1

    assert _values(dataset) == list(range(5, 20))
    assert {path.name: path.stat().st_mtime_ns for path in manifest.parent.glob("part-*.i64")} == part_mtimes
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    assert metadata["offsets"][-1] == 15


def test_lance_filter_warm_cache_skips_fragment_planning(tmp_path, monkeypatch) -> None:
    from mvp_dataset.sources.lance import filter as lance_filter

    monkeypatch.setenv("MVP_DATASET_LANCE_FILTER_INDEX_CACHE_DIR", str(tmp_path / "cache"))
    source = write_lance_dataset(tmp_path, build_records(12), max_rows_per_file=2)
    dataset = Dataset.from_source("lance", source).filter("value >= 5")
    assert _values(dataset) == list(range(5, 12))

    monkeypatch.setattr(lance_filter, "_plan_parts", lambda _source: pytest.fail("warm cache was replanned"))
    assert _values(dataset) == list(range(5, 12))


def test_lance_filter_rebuilds_truncated_part(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MVP_DATASET_LANCE_FILTER_INDEX_CACHE_DIR", str(tmp_path / "cache"))
    source = write_lance_dataset(tmp_path, build_records(20), max_rows_per_file=2)
    dataset = Dataset.from_source("lance", source).filter("value >= 5")
    assert _values(dataset) == list(range(5, 20))

    manifest_path = next((tmp_path / "cache").rglob("manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    part = next(part for part in manifest["parts"] if part["count"] > 0)
    part_path = manifest_path.parent / part["file"]
    with part_path.open("r+b") as handle:
        handle.truncate(part_path.stat().st_size - 8)

    assert _values(dataset) == list(range(5, 20))


def test_lance_source_without_filter_does_not_enumerate_fragments(tmp_path, monkeypatch) -> None:
    import lance

    source = write_lance_dataset(tmp_path, build_records(8), max_rows_per_file=2)
    monkeypatch.setattr(lance.LanceDataset, "get_fragments", lambda _self: pytest.fail("fragments were enumerated"))

    assert _values(Dataset.from_source("lance", source)) == list(range(8))


def test_lance_filter_handles_empty_results(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MVP_DATASET_LANCE_FILTER_INDEX_CACHE_DIR", str(tmp_path / "cache"))
    source = write_lance_dataset(tmp_path, build_records(8), max_rows_per_file=2)

    dataset = Dataset.from_source("lance", source).filter("value < 0")

    assert list(dataset) == []
    assert list(dataset.sample(0.5)) == []


def test_lance_filter_validates_predicate_for_zero_fragment_source(tmp_path, monkeypatch) -> None:
    import lance
    import pyarrow as pa

    monkeypatch.setenv("MVP_DATASET_LANCE_FILTER_INDEX_CACHE_DIR", str(tmp_path / "cache"))
    source = tmp_path / "empty.lance"
    lance.write_dataset(pa.table({"value": pa.array([], type=pa.int64())}), source)

    assert list(Dataset.from_source("lance", str(source)).filter("value > 0")) == []
    with pytest.raises(OSError, match="No field named missing"):
        list(Dataset.from_source("lance", str(source)).filter("missing > 0"))


def test_lance_filter_and_ref_index_use_discovered_version(tmp_path, monkeypatch) -> None:
    import lance
    import pyarrow as pa

    monkeypatch.setenv("MVP_DATASET_LANCE_FILTER_INDEX_CACHE_DIR", str(tmp_path / "filter-cache"))
    monkeypatch.setenv("MVP_DATASET_LANCE_REF_INDEX_CACHE_DIR", str(tmp_path / "ref-cache"))
    main = write_lance_table(
        tmp_path,
        "main.lance",
        [{"id": index, "value": index, "image": f"image-{index}"} for index in range(8)],
    )
    refs = write_lance_table(
        tmp_path,
        "refs.lance",
        [{"key": f"image-{index}", "payload": f"bytes-{index}"} for index in range(8)],
    )
    dataset = (
        Dataset.from_source(
            "lance",
            main,
            ref_columns={"image": {"uri": refs, "key_column": "key", "value_column": "payload"}},
        )
        .filter("value >= 0")
        .resolve_ref(["image"], index={"scope": "process"})
    )
    lance.write_dataset(pa.Table.from_pylist([{"id": 8, "value": 8, "image": "image-8"}]), main, mode="append")

    rows = list(dataset)
    assert [row["value"] for row in rows] == list(range(8))
    assert [row["image"] for row in rows] == [f"bytes-{index}" for index in range(8)]


def test_lance_ref_value_source_uses_discovered_version(tmp_path, monkeypatch) -> None:
    import lance
    import pyarrow as pa

    monkeypatch.setenv("MVP_DATASET_LANCE_REF_INDEX_CACHE_DIR", str(tmp_path / "ref-cache"))
    main = write_lance_table(
        tmp_path,
        "main.lance",
        [{"id": index, "image": f"image-{index}"} for index in range(4)],
    )
    refs = write_lance_table(
        tmp_path,
        "refs.lance",
        [{"key": f"image-{index}", "payload": f"bytes-{index}"} for index in range(4)],
    )
    dataset = Dataset.from_source(
        "lance",
        main,
        ref_columns={"image": {"uri": refs, "key_column": "key", "value_column": "payload"}},
    ).resolve_ref(["image"], index={"scope": "process"})
    reversed_refs = [{"key": f"image-{index}", "payload": f"changed-{index}"} for index in reversed(range(4))]
    lance.write_dataset(pa.Table.from_pylist(reversed_refs), refs, mode="overwrite")

    rows = list(dataset)
    assert [row["image"] for row in rows] == [f"bytes-{index}" for index in range(4)]


def test_lance_filter_rejects_deleted_rows(tmp_path, monkeypatch) -> None:
    import lance

    monkeypatch.setenv("MVP_DATASET_LANCE_FILTER_INDEX_CACHE_DIR", str(tmp_path / "cache"))
    source = write_lance_dataset(tmp_path, build_records(8), max_rows_per_file=2)
    lance.dataset(source).delete("value = 1")

    with pytest.raises(ValueError, match="UnsupportedLanceFilterDeletedRows"):
        list(Dataset.from_source("lance", source).filter("value >= 0"))


def test_lance_filter_validates_api_and_placement(tmp_path) -> None:
    source = write_lance_dataset(tmp_path, build_records(8))

    with pytest.raises(NotImplementedError, match="UnsupportedFilter"):
        Dataset.from_source("jsonl", write_jsonl_file(tmp_path, build_records(2))).filter("value > 0")
    with pytest.raises(TypeError, match="predicate must be a string"):
        Dataset.from_source("lance", source).filter(1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="predicate must be non-empty"):
        Dataset.from_source("lance", source).filter("  ")
    with pytest.raises(ValueError, match="predicate sequence must be non-empty"):
        Dataset.from_source("lance", source).filter([])
    with pytest.raises(TypeError, match="every predicate must be a string"):
        Dataset.from_source("lance", source).filter(["value > 0", 1])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="predicate must be non-empty"):
        Dataset.from_source("lance", source).filter(["value > 0", "  "])
    with pytest.raises(ValueError, match="unknown config key"):
        Dataset.from_source("lance", source).filter("value > 0", index={"unknown": True})
    with pytest.raises(ValueError, match="InvalidLanceFilterIndexScope"):
        Dataset.from_source("lance", source).filter("value > 0", index={"scope": "unknown"})
    with pytest.raises(ValueError, match="InvalidFilterPlacement"):
        Dataset.from_source("lance", source).map(lambda row: row).filter("value > 0")
    with pytest.raises(ValueError, match="InvalidFilterPlacement"):
        Dataset.from_source("lance", source).split([1, 1])[0].filter("value > 0")


def test_lance_filter_requires_cache_dir_for_uri_source(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MVP_DATASET_LANCE_FILTER_INDEX_CACHE_DIR", raising=False)
    source = write_lance_dataset(tmp_path, build_records(8))

    with pytest.raises(ValueError, match="MissingLanceFilterIndexCacheDir"):
        list(Dataset.from_source("lance", f"file://{source}").filter("value > 0"))


@pytest.mark.parametrize(
    "builders",
    [
        ((0, 2, 0, 1), (1, 2, 0, 1)),
        ((0, 1, 0, 2), (0, 1, 1, 2)),
    ],
    ids=["ranks", "workers"],
)
@pytest.mark.parametrize(
    ("filter_arg", "expected"),
    [
        ("value >= 5", list(range(5, 24))),
        (["value % 3 = 0", "value % 5 = 0"], [value for value in range(24) if value % 3 == 0 or value % 5 == 0]),
    ],
    ids=["single", "batched"],
)
def test_lance_filter_builds_shared_parts_across_builders(tmp_path, builders, filter_arg, expected) -> None:
    source = write_lance_dataset(tmp_path, build_records(24), max_rows_per_file=2)
    cache_dir = tmp_path / "cache"
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    processes = [
        context.Process(
            target=_distributed_filter_worker,
            args=(source, str(cache_dir), filter_arg, rank, world_size, worker_id, num_workers, results),
        )
        for rank, world_size, worker_id, num_workers in builders
    ]

    try:
        for process in processes:
            process.start()
        outputs = [results.get(timeout=30) for _ in processes]
        for process in processes:
            process.join(timeout=30)
        assert all(process.exitcode == 0 for process in processes)
    except queue.Empty:
        pytest.fail("distributed filter workers timed out")
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join()

    errors = [payload for status, payload in outputs if status == "error"]
    assert not errors, "\n".join(errors)
    values = [value for status, payload in outputs if status == "ok" for value in payload["values"]]
    assert sorted(values) == expected
    assert all(payload["built_parts"] for status, payload in outputs if status == "ok")
    assert len(list(cache_dir.rglob("part-*.i64"))) > 1
    assert len(list(cache_dir.rglob("manifest.json"))) == 1
