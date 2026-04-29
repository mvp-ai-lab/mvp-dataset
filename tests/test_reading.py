from __future__ import annotations

import importlib.util
import json
import pickle

import numpy as np
import pytest

from mvp_dataset import Dataset
from mvp_dataset.core.context import RuntimeContext
from mvp_dataset.sources.parquet.utils import (
    list_parquet_fragments,
    resolve_parquet_batch_size,
)

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


def test_resolve_parquet_batch_size_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("MVP_DATASET_PARQUET_BATCH_SIZE", "1234")

    assert resolve_parquet_batch_size() == 1234


def test_lance_assign_items_shuffle_uses_context_seed() -> None:
    pytest.importorskip("lance")

    from mvp_dataset.sources.lance.utils import (
        LanceDatasetSpec,
        LanceSourceSpec,
        assign_items,
    )

    source = [
        LanceSourceSpec(
            datasets=(
                LanceDatasetSpec(uri="a", num_rows=6, row_offset=0, fragment_ids=()),
                LanceDatasetSpec(uri="b", num_rows=4, row_offset=6, fragment_ids=()),
            )
        )
    ]
    context = RuntimeContext(seed=17)

    np.random.seed(123)
    first = [item.global_index for item in assign_items(source, context=context, resample=False, shuffle_mode="global")]
    np.random.seed(999)
    second = [
        item.global_index for item in assign_items(source, context=context, resample=False, shuffle_mode="global")
    ]

    assert first == second
    assert sorted(first) == list(range(10))


def test_lance_assign_items_shuffle_prefers_fragment_affinity_across_slots() -> None:
    pytest.importorskip("lance")

    from mvp_dataset.sources.lance.utils import (
        LanceDatasetSpec,
        LanceSourceSpec,
        assign_items,
    )

    source = [
        LanceSourceSpec(
            datasets=[
                LanceDatasetSpec(
                    uri="a",
                    num_rows=8,
                    row_offset=0,
                    fragment_ids=(10, 11, 12, 13),
                    fragment_row_counts=(2, 2, 2, 2),
                )
            ]
        )
    ]

    rank0 = [
        item.global_index
        for item in assign_items(
            source,
            context=RuntimeContext(rank=0, world_size=2, seed=23),
            resample=False,
            shuffle_mode="fragment_aware",
        )
    ]
    rank1 = [
        item.global_index
        for item in assign_items(
            source,
            context=RuntimeContext(rank=1, world_size=2, seed=23),
            resample=False,
            shuffle_mode="fragment_aware",
        )
    ]

    assert sorted(rank0 + rank1) == list(range(8))
    assert set(rank0).isdisjoint(rank1)
    assert len(rank0) == len(rank1) == 4
    assert len({index // 2 for index in rank0}) == 2
    assert len({index // 2 for index in rank1}) == 2


def test_lance_assign_items_shuffle_splits_single_fragment_across_slots() -> None:
    pytest.importorskip("lance")

    from mvp_dataset.sources.lance.utils import (
        LanceDatasetSpec,
        LanceSourceSpec,
        assign_items,
    )

    source = [
        LanceSourceSpec(
            datasets=[
                LanceDatasetSpec(
                    uri="a",
                    num_rows=8,
                    row_offset=0,
                    fragment_ids=(10,),
                    fragment_row_counts=(8,),
                )
            ]
        )
    ]

    ranks = [
        [
            item.global_index
            for item in assign_items(
                source,
                context=RuntimeContext(rank=rank, world_size=4, seed=31),
                resample=False,
                shuffle_mode="fragment_aware",
            )
        ]
        for rank in range(4)
    ]

    assert sorted(index for rank_items in ranks for index in rank_items) == list(range(8))
    assert [len(rank_items) for rank_items in ranks] == [2, 2, 2, 2]


def test_lance_assign_items_shuffle_mixes_blocks_across_assigned_fragments() -> None:
    pytest.importorskip("lance")

    from mvp_dataset.sources.lance.utils import (
        LanceDatasetSpec,
        LanceSourceSpec,
        assign_items,
    )

    source = [
        LanceSourceSpec(
            datasets=[
                LanceDatasetSpec(
                    uri="a",
                    num_rows=1024,
                    row_offset=0,
                    fragment_ids=(10, 11, 12, 13),
                    fragment_row_counts=(256, 256, 256, 256),
                )
            ]
        )
    ]

    observed = [
        item.global_index
        for item in assign_items(
            source,
            context=RuntimeContext(seed=41),
            resample=False,
            shuffle_mode="fragment_aware",
        )
    ]
    fragment_runs: list[tuple[int, int]] = []
    last_fragment = None
    for index in observed:
        fragment = index // 256
        if fragment != last_fragment:
            fragment_runs.append((fragment, 1))
            last_fragment = fragment
        else:
            fragment_runs[-1] = (fragment_runs[-1][0], fragment_runs[-1][1] + 1)

    assert sorted(observed) == list(range(1024))
    assert max(run_length for _, run_length in fragment_runs) < 256
    assert all(sum(1 for fragment, _ in fragment_runs if fragment == fragment_i) > 1 for fragment_i in range(4))


def test_lance_dataset_global_shuffle_shards_rows_across_slots(tmp_path, monkeypatch) -> None:
    if importlib.util.find_spec("lance") is None:
        pytest.skip("lance is not installed")

    records = build_records(count=8)
    path = write_lance_dataset(tmp_path, records)
    rank0 = Dataset.from_source("lance", shards=path, context=RuntimeContext(seed=5), shuffle_mode="global")
    rank1 = Dataset.from_source("lance", shards=path, context=RuntimeContext(seed=5), shuffle_mode="global")

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    observed_rank0 = [normalize_sample(sample) for sample in rank0]
    monkeypatch.setenv("RANK", "1")
    observed_rank1 = [normalize_sample(sample) for sample in rank1]
    shuffled_indices = np.random.default_rng(5).permutation(len(records)).tolist()

    assert observed_rank0 == [records[index] for index in shuffled_indices[0::2]]
    assert observed_rank1 == [records[index] for index in shuffled_indices[1::2]]


def test_lance_dataset_fragment_aware_shuffle_reads_all_rows_once(tmp_path) -> None:
    if importlib.util.find_spec("lance") is None:
        pytest.skip("lance is not installed")

    records = build_records(count=8)
    path = write_lance_dataset(tmp_path, records, max_rows_per_file=2)
    dataset = Dataset.from_source(
        "lance",
        shards=path,
        context=RuntimeContext(seed=11),
        shuffle_mode="fragment_aware",
    )

    observed = [normalize_sample(sample) for sample in dataset]

    assert sorted(observed, key=lambda sample: str(sample["id"])) == records


def test_lance_dataset_none_shuffle_preserves_slot_stride(tmp_path, monkeypatch) -> None:
    if importlib.util.find_spec("lance") is None:
        pytest.skip("lance is not installed")

    records = build_records(count=8)
    path = write_lance_dataset(tmp_path, records)
    dataset = Dataset.from_source("lance", shards=path, context=RuntimeContext(seed=9), shuffle_mode="none")

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    observed_rank0 = [normalize_sample(sample) for sample in dataset]
    monkeypatch.setenv("RANK", "1")
    observed_rank1 = [normalize_sample(sample) for sample in dataset]

    assert observed_rank0 == records[0::2]
    assert observed_rank1 == records[1::2]


def test_lance_ref_columns_resolve_via_prebuilt_index(tmp_path, monkeypatch) -> None:
    if importlib.util.find_spec("lance") is None:
        pytest.skip("lance is not installed")

    monkeypatch.chdir(tmp_path)
    meta_path = write_lance_table(
        tmp_path,
        "meta.lance",
        [
            {"id": "sample-0", "image": "image-1", "value": 0},
            {"id": "sample-1", "image": "image-0", "value": 1},
            {"id": "sample-2", "image": None, "value": 2},
        ],
    )
    image_path = write_lance_table(
        tmp_path,
        "image.lance",
        [
            {"image_id": "image-0", "image": b"zero"},
            {"image_id": "image-1", "image": b"one"},
        ],
    )

    dataset = Dataset.from_source(
        "lance",
        shards=meta_path,
        batch_size=2,
        ref_columns={"image": {"uri": image_path, "key_column": "image_id", "value_column": "image"}},
    )

    observed = list(dataset)
    index_paths = list((tmp_path / "meta.lance" / "_mvp_ref_index").glob("ref-index-*"))
    manifest_mtime_ns = (index_paths[0] / "metadata.json").stat().st_mtime_ns
    second_observed = list(dataset)
    second_index_paths = list((tmp_path / "meta.lance" / "_mvp_ref_index").glob("ref-index-*"))
    manifest = json.loads((index_paths[0] / "metadata.json").read_text(encoding="utf-8"))

    assert [sample["image"] for sample in observed] == [b"one", b"zero", None]
    assert [sample["image"] for sample in second_observed] == [b"one", b"zero", None]
    assert [sample["__index_in_file__"] for sample in observed] == [0, 1, 2]
    assert len(index_paths) == 1
    assert second_index_paths == index_paths
    assert manifest["refs"]["image"]["kind"] == "csr_row_index"
    assert (index_paths[0] / manifest["refs"]["image"]["offsets_file"]).exists()
    assert (index_paths[0] / manifest["refs"]["image"]["entries_file"]).exists()
    assert manifest["refs"]["image"]["ref_dataset"]["uri"] == image_path
    assert (index_paths[0] / "metadata.json").exists()
    assert (index_paths[0] / "metadata.json").stat().st_mtime_ns == manifest_mtime_ns


def test_lance_ref_columns_reuse_ref_dataset_handle_during_apply(tmp_path, monkeypatch) -> None:
    if importlib.util.find_spec("lance") is None:
        pytest.skip("lance is not installed")

    from mvp_dataset.sources.lance import utils as lance_utils

    monkeypatch.chdir(tmp_path)
    meta_path = write_lance_table(
        tmp_path,
        "meta.lance",
        [
            {"id": "sample-0", "image": "image-0"},
            {"id": "sample-1", "image": "image-1"},
            {"id": "sample-2", "image": "image-0"},
        ],
    )
    image_path = write_lance_table(
        tmp_path,
        "image.lance",
        [
            {"image_id": "image-0", "image": b"zero"},
            {"image_id": "image-1", "image": b"one"},
        ],
    )
    source = lance_utils.attach_lance_ref_columns(
        lance_utils.list_lance_sources([meta_path])[0],
        {"image": {"uri": image_path, "key_column": "image_id", "value_column": "image"}},
    )
    prepared = lance_utils.prepare_ref_indexes(source)
    index_handle = prepared.ref_columns[0].index_handle

    assert isinstance(index_handle, dict)
    assert "value_dataset" in index_handle

    real_dataset = lance_utils.lance.dataset

    def fail_if_ref_dataset_reopened(uri, *args, **kwargs):
        if str(uri) == image_path:
            raise AssertionError("ref dataset should be reused while applying ref columns")
        return real_dataset(uri, *args, **kwargs)

    monkeypatch.setattr(lance_utils.lance, "dataset", fail_if_ref_dataset_reopened)
    batch_indexes = [
        lance_utils.LanceIndexItem(dataset_i=0, local_index=index, global_index=index) for index in range(3)
    ]
    batch = lance_utils._read_batch(prepared, batch_indexes, columns=None)

    lance_utils._apply_ref_columns(prepared, batch, batch_indexes, columns=None)

    assert [sample["image"] for sample in batch] == [b"zero", b"one", b"zero"]


def test_lance_ref_columns_share_unified_index(tmp_path, monkeypatch) -> None:
    if importlib.util.find_spec("lance") is None:
        pytest.skip("lance is not installed")

    monkeypatch.chdir(tmp_path)
    meta_path = write_lance_table(
        tmp_path,
        "meta.lance",
        [
            {"id": "sample-0", "image": "image-1", "label": "label-0"},
            {"id": "sample-1", "image": "image-0", "label": "label-1"},
        ],
    )
    image_path = write_lance_table(
        tmp_path,
        "image.lance",
        [
            {"image_id": "image-0", "image": b"zero"},
            {"image_id": "image-1", "image": b"one"},
        ],
    )
    label_path = write_lance_table(
        tmp_path,
        "label.lance",
        [
            {"label_id": "label-0", "label": "cat"},
            {"label_id": "label-1", "label": "dog"},
        ],
    )

    dataset = Dataset.from_source(
        "lance",
        shards=meta_path,
        ref_columns={
            "image": {"uri": image_path, "key_column": "image_id", "value_column": "image"},
            "label": {"uri": label_path, "key_column": "label_id", "value_column": "label"},
        },
    )

    observed = list(dataset)
    index_paths = list((tmp_path / "meta.lance" / "_mvp_ref_index").glob("ref-index-*"))
    manifest = json.loads((index_paths[0] / "metadata.json").read_text(encoding="utf-8"))

    assert [(sample["image"], sample["label"]) for sample in observed] == [(b"one", "cat"), (b"zero", "dog")]
    assert len(index_paths) == 1
    assert set(manifest["refs"]) == {"image", "label"}
    assert manifest["refs"]["image"]["kind"] == "csr_row_index"
    assert manifest["refs"]["label"]["kind"] == "csr_row_index"
    assert manifest["refs"]["image"]["ref_dataset"]["uri"] == image_path
    assert manifest["refs"]["label"]["ref_dataset"]["uri"] == label_path


def test_lance_ref_columns_resolve_multi_value_refs(tmp_path, monkeypatch) -> None:
    if importlib.util.find_spec("lance") is None:
        pytest.skip("lance is not installed")

    monkeypatch.chdir(tmp_path)
    meta_path = write_lance_table(
        tmp_path,
        "meta.lance",
        [
            {"id": "sample-0", "images": ["image-1", "image-0"]},
            {"id": "sample-1", "images": []},
            {"id": "sample-2", "images": ["image-0"]},
        ],
    )
    image_path = write_lance_table(
        tmp_path,
        "image.lance",
        [
            {"image_id": "image-0", "image": b"zero"},
            {"image_id": "image-1", "image": b"one"},
        ],
    )

    dataset = Dataset.from_source(
        "lance",
        shards=meta_path,
        ref_columns={"images": {"uri": image_path, "key_column": "image_id", "value_column": "image"}},
    )

    observed = list(dataset)

    assert [sample["images"] for sample in observed] == [[b"one", b"zero"], [], [b"zero"]]


def test_lance_ref_columns_respect_projection(tmp_path, monkeypatch) -> None:
    if importlib.util.find_spec("lance") is None:
        pytest.skip("lance is not installed")

    monkeypatch.chdir(tmp_path)
    meta_path = write_lance_table(tmp_path, "meta.lance", [{"id": "sample-0", "image": "image-0", "value": 0}])
    image_path = write_lance_table(tmp_path, "image.lance", [{"image_id": "image-0", "image": b"zero"}])

    dataset = Dataset.from_source(
        "lance",
        shards=meta_path,
        columns=["id", "value"],
        ref_columns={"image": {"uri": image_path, "key_column": "image_id", "value_column": "image"}},
    )

    observed = list(dataset)

    assert observed[0]["id"] == "sample-0"
    assert "image" not in observed[0]


def test_lance_ref_columns_missing_key_raises(tmp_path, monkeypatch) -> None:
    if importlib.util.find_spec("lance") is None:
        pytest.skip("lance is not installed")

    monkeypatch.chdir(tmp_path)
    meta_path = write_lance_table(tmp_path, "meta.lance", [{"id": "sample-0", "image": "missing", "value": 0}])
    image_path = write_lance_table(tmp_path, "image.lance", [{"image_id": "image-0", "image": b"zero"}])

    dataset = Dataset.from_source(
        "lance",
        shards=meta_path,
        ref_columns={"image": {"uri": image_path, "key_column": "image_id", "value_column": "image"}},
    )

    with pytest.raises(KeyError, match="MissingLanceRefKey"):
        list(dataset)
