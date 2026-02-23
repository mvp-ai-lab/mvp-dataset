"""Tests for tar shard reading and sample grouping."""

from __future__ import annotations

from pathlib import Path

import pytest
from helpers import create_tar_shard

from mm_loader.sources import iter_tar_records


def test_iter_tar_records_groups_fields_by_key(tmp_path: Path) -> None:
    shard_path = tmp_path / "part-000.tar"
    create_tar_shard(
        shard_path,
        rows=[
            ("000001", {"jpg": b"rgb-1", "txt": b"caption-1"}),
            ("000002", {"jpg": b"rgb-2"}),
        ],
        meta_files={"__meta__": b"ignored"},
    )

    samples = list(iter_tar_records(shard_path))

    assert len(samples) == 2
    assert samples[0]["__key__"] == "000001"
    assert samples[0]["__shard__"] == str(shard_path)
    assert samples[0]["__index_in_shard__"] == 0
    assert samples[0]["jpg"] == b"rgb-1"
    assert samples[0]["txt"] == b"caption-1"

    assert samples[1]["__key__"] == "000002"
    assert samples[1]["__index_in_shard__"] == 1
    assert samples[1]["jpg"] == b"rgb-2"


def test_iter_tar_records_raises_on_duplicate_fields(tmp_path: Path) -> None:
    shard_path = tmp_path / "part-dup.tar"
    create_tar_shard(
        shard_path,
        rows=[
            ("000001", {"jpg": b"rgb-a"}),
            ("000001", {"jpg": b"rgb-b"}),
        ],
    )

    with pytest.raises(ValueError, match="duplicate field"):
        list(iter_tar_records(shard_path))
