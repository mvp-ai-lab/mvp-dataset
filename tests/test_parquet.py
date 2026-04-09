"""Tests for parquet-backed Dataset sources."""

from __future__ import annotations

from pathlib import Path

import pytest

from mvp_dataset import Dataset


def _make_parquet(path: Path, rows: list[dict]) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def test_from_parquet_yields_metadata_and_nested_values(tmp_path):
    shard = tmp_path / "train-00000.parquet"
    _make_parquet(
        shard,
        [
            {
                "messages": [{"from": "human", "value": "describe"}],
                "images": [{"bytes": b"img-1", "path": "images/a.jpg"}],
            },
            {
                "messages": [{"from": "gpt", "value": "done"}],
                "images": [{"bytes": b"img-2", "path": "images/b.jpg"}],
            },
        ],
    )

    samples = list(Dataset.from_parquet([str(shard)]))

    assert [sample["__index_in_file__"] for sample in samples] == [0, 1]
    assert all(sample["__file__"] == str(shard) for sample in samples)
    assert samples[0]["__key__"] == f"{shard}:0"
    assert samples[0]["messages"][0]["from"] == "human"
    assert samples[0]["images"][0]["bytes"] == b"img-1"
    assert samples[1]["images"][0]["path"] == "images/b.jpg"


def test_from_source_dispatches_to_parquet(tmp_path):
    shard = tmp_path / "train-00000.parquet"
    _make_parquet(shard, [{"text": "hello"}])

    dataset = Dataset.from_source([str(shard)])
    samples = list(dataset)

    assert dataset.source_kind == "parquet"
    assert samples[0]["text"] == "hello"
    assert samples[0]["__index_in_file__"] == 0



def test_from_parquet_recursive_glob_matches_symlinked_directories(tmp_path):
    stage1_dir = tmp_path / "stage1"
    stage1_dir.mkdir()
    stage1_shard = stage1_dir / "train-00000.parquet"
    _make_parquet(stage1_shard, [{"text": "stage1"}])

    external_stage2_dir = tmp_path.parent / f"{tmp_path.name}_external_stage2"
    external_stage2_dir.mkdir()
    coyo_dir = external_stage2_dir / "coyo"
    coyo_dir.mkdir()
    stage2_shard = coyo_dir / "train-00000.parquet"
    _make_parquet(stage2_shard, [{"text": "stage2"}])

    stage2_link = tmp_path / "stage2"
    try:
        stage2_link.symlink_to(external_stage2_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    samples = list(Dataset.from_parquet(str(tmp_path / "**/*.parquet")))

    assert {sample["text"] for sample in samples} == {"stage1", "stage2"}
    expected_stage2_shard = stage2_link / "coyo" / "train-00000.parquet"
    assert {sample["__file__"] for sample in samples} == {str(stage1_shard), str(expected_stage2_shard)}

def test_parquet_cache_reuses_warmup_results(tmp_path):
    shard = tmp_path / "train-00000.parquet"
    _make_parquet(
        shard,
        [
            {
                "messages": [{"from": "human", "value": "describe"}],
                "images": [{"bytes": b"img-1", "path": "images/a.jpg"}],
            },
            {
                "messages": [{"from": "gpt", "value": "done"}],
                "images": [{"bytes": b"img-2", "path": "images/b.jpg"}],
            },
        ],
    )

    call_count = [0]

    def summarize(sample):
        call_count[0] += 1
        return {
            "image_size": len(sample["images"][0]["bytes"]),
            "turns": len(sample["messages"]),
        }

    dataset = Dataset.from_parquet([str(shard)]).map(summarize).cache(show_progress=False)

    first = list(dataset)
    assert call_count[0] == 2

    call_count[0] = 0
    second = list(dataset)
    assert call_count[0] == 0
    assert first == second
    assert (tmp_path / ".cache" / "train-00000.parquet.manifest.json").is_file()
