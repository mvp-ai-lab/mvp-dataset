"""Tests for strict streaming tar-to-tar join behavior."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from helpers import create_tar_shard

from mm_loader import Dataset
from mm_loader.core import RuntimeContext


def _write_shards(
    root: Path,
    *,
    stem: str,
    keys_by_shard: Sequence[Sequence[str]],
    field: str,
    payload_prefix: str,
) -> list[str]:
    paths: list[str] = []
    for shard_index, shard_keys in enumerate(keys_by_shard):
        rows = [
            (key, {field: f"{payload_prefix}-{key}".encode()})
            for key in shard_keys
        ]
        shard_path = root / f"{stem}-{shard_index:03d}.tar"
        create_tar_shard(shard_path, rows)
        paths.append(str(shard_path))
    return paths


def test_join_tars_single_sidecar_success(tmp_path: Path) -> None:
    keys_by_shard = [["000001", "000002"], ["000003", "000004"]]
    primary_shards = _write_shards(
        tmp_path,
        stem="rgb",
        keys_by_shard=keys_by_shard,
        field="jpg",
        payload_prefix="rgb",
    )
    depth_shards = _write_shards(
        tmp_path,
        stem="depth",
        keys_by_shard=keys_by_shard,
        field="png",
        payload_prefix="depth",
    )

    context = RuntimeContext(seed=3, epoch=1)
    output = list(
        Dataset.from_tars(primary_shards, context=context).join_tars([("depth", depth_shards)])
    )

    assert len(output) == 4
    keys = {sample["__key__"] for sample in output if isinstance(sample, dict)}
    assert keys == {"000001", "000002", "000003", "000004"}
    for sample in output:
        assert isinstance(sample, dict)
        key = sample["__key__"]
        assert sample["jpg"] == f"rgb-{key}".encode()
        assert sample["png"] == f"depth-{key}".encode()


def test_join_tars_multi_sidecar_success(tmp_path: Path) -> None:
    keys_by_shard = [["000101", "000102"], ["000103"]]
    primary_shards = _write_shards(
        tmp_path,
        stem="rgb",
        keys_by_shard=keys_by_shard,
        field="jpg",
        payload_prefix="rgb",
    )
    depth_shards = _write_shards(
        tmp_path,
        stem="depth",
        keys_by_shard=keys_by_shard,
        field="dpt",
        payload_prefix="depth",
    )
    mask_shards = _write_shards(
        tmp_path,
        stem="mask",
        keys_by_shard=keys_by_shard,
        field="msk",
        payload_prefix="mask",
    )

    context = RuntimeContext(seed=5, epoch=2)
    output = list(
        Dataset.from_tars(primary_shards, context=context).join_tars(
            [("depth", depth_shards), ("mask", mask_shards)]
        )
    )

    assert len(output) == 3
    for sample in output:
        assert isinstance(sample, dict)
        key = sample["__key__"]
        assert sample["jpg"] == f"rgb-{key}".encode()
        assert sample["dpt"] == f"depth-{key}".encode()
        assert sample["msk"] == f"mask-{key}".encode()


def test_join_tars_shard_count_mismatch(tmp_path: Path) -> None:
    keys_by_shard = [["000201"], ["000202"]]
    primary_shards = _write_shards(
        tmp_path,
        stem="rgb",
        keys_by_shard=keys_by_shard,
        field="jpg",
        payload_prefix="rgb",
    )
    depth_shards = _write_shards(
        tmp_path,
        stem="depth",
        keys_by_shard=[["000201"]],
        field="png",
        payload_prefix="depth",
    )

    dataset = Dataset.from_tars(primary_shards, context=RuntimeContext())
    with pytest.raises(ValueError, match="ShardCountMismatch"):
        dataset.join_tars([("depth", depth_shards)])


def test_join_tars_key_mismatch_raises(tmp_path: Path) -> None:
    primary_shards = _write_shards(
        tmp_path,
        stem="rgb",
        keys_by_shard=[["000301", "000302"]],
        field="jpg",
        payload_prefix="rgb",
    )
    depth_shards = _write_shards(
        tmp_path,
        stem="depth",
        keys_by_shard=[["000301", "999999"]],
        field="png",
        payload_prefix="depth",
    )

    dataset = Dataset.from_tars(primary_shards, context=RuntimeContext()).join_tars(
        [("depth", depth_shards)]
    )
    with pytest.raises(ValueError, match="KeyMismatch"):
        list(dataset)


@pytest.mark.parametrize(
    ("primary_keys", "sidecar_keys", "expected_error"),
    [
        (["000401", "000402"], ["000401"], "SidecarExhaustedEarly"),
        (["000501"], ["000501", "000502"], "SidecarHasExtraRows"),
    ],
)
def test_join_tars_sidecar_exhausted_or_extra_rows(
    tmp_path: Path,
    primary_keys: list[str],
    sidecar_keys: list[str],
    expected_error: str,
) -> None:
    primary_shards = _write_shards(
        tmp_path,
        stem="rgb",
        keys_by_shard=[primary_keys],
        field="jpg",
        payload_prefix="rgb",
    )
    depth_shards = _write_shards(
        tmp_path,
        stem="depth",
        keys_by_shard=[sidecar_keys],
        field="png",
        payload_prefix="depth",
    )

    dataset = Dataset.from_tars(primary_shards, context=RuntimeContext()).join_tars(
        [("depth", depth_shards)]
    )
    with pytest.raises(ValueError, match=expected_error):
        list(dataset)


def test_join_tars_field_collision_raises(tmp_path: Path) -> None:
    keys_by_shard = [["000601", "000602"]]
    primary_shards = _write_shards(
        tmp_path,
        stem="rgb",
        keys_by_shard=keys_by_shard,
        field="png",
        payload_prefix="rgb",
    )
    depth_shards = _write_shards(
        tmp_path,
        stem="depth",
        keys_by_shard=keys_by_shard,
        field="png",
        payload_prefix="depth",
    )

    dataset = Dataset.from_tars(primary_shards, context=RuntimeContext()).join_tars(
        [("depth", depth_shards)]
    )
    with pytest.raises(ValueError, match="FieldCollision"):
        list(dataset)


def test_join_tars_must_be_first_stage(tmp_path: Path) -> None:
    keys_by_shard = [["000701"]]
    primary_shards = _write_shards(
        tmp_path,
        stem="rgb",
        keys_by_shard=keys_by_shard,
        field="jpg",
        payload_prefix="rgb",
    )
    depth_shards = _write_shards(
        tmp_path,
        stem="depth",
        keys_by_shard=keys_by_shard,
        field="png",
        payload_prefix="depth",
    )

    dataset = Dataset.from_tars(primary_shards, context=RuntimeContext()).map(lambda sample: sample)
    with pytest.raises(ValueError, match="JoinOrder"):
        dataset.join_tars([("depth", depth_shards)])


def test_join_tars_with_distributed_split(tmp_path: Path) -> None:
    keys_by_shard = [[f"{index:06d}"] for index in range(6)]
    primary_shards = _write_shards(
        tmp_path,
        stem="rgb",
        keys_by_shard=keys_by_shard,
        field="jpg",
        payload_prefix="rgb",
    )
    depth_shards = _write_shards(
        tmp_path,
        stem="depth",
        keys_by_shard=keys_by_shard,
        field="png",
        payload_prefix="depth",
    )

    outputs: dict[int, list[object]] = {}
    for rank in (0, 1):
        context = RuntimeContext(
            rank=rank,
            world_size=2,
            worker_id=0,
            num_workers=1,
            seed=17,
            epoch=1,
        )
        dataset = Dataset.from_tars(primary_shards, context=context).join_tars(
            [("depth", depth_shards)]
        )
        outputs[rank] = list(dataset)

    rank0_keys = {sample["__key__"] for sample in outputs[0] if isinstance(sample, dict)}
    rank1_keys = {sample["__key__"] for sample in outputs[1] if isinstance(sample, dict)}
    assert rank0_keys.isdisjoint(rank1_keys)
    assert rank0_keys | rank1_keys == {f"{index:06d}" for index in range(6)}

    for rank_samples in outputs.values():
        for sample in rank_samples:
            assert isinstance(sample, dict)
            key = sample["__key__"]
            assert sample["jpg"] == f"rgb-{key}".encode()
            assert sample["png"] == f"depth-{key}".encode()
