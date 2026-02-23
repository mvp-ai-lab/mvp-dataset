"""End-to-end tests for the dataset pipeline."""

from __future__ import annotations

from pathlib import Path

from helpers import create_tar_shard

from mm_loader import Dataset
from mm_loader.core import RuntimeContext
from mm_loader.distributed import split_shards
from mm_loader.sources import iter_tar_records


def _build_demo_shards(root: Path) -> list[str]:
    shards: list[str] = []
    sample_index = 0

    for shard_id in range(6):
        rows: list[tuple[str, dict[str, bytes]]] = []
        for _ in range(3):
            key = f"{sample_index:06d}"
            rows.append((key, {"txt": str(sample_index).encode("utf-8")}))
            sample_index += 1

        shard_path = root / f"part-{shard_id:03d}.tar"
        create_tar_shard(shard_path, rows)
        shards.append(str(shard_path))

    return shards


def _decode_sample(sample: object) -> object:
    assert isinstance(sample, dict)
    txt_payload = sample.get("txt")
    assert isinstance(txt_payload, bytes)

    decoded = dict(sample)
    decoded["value"] = int(txt_payload.decode("utf-8"))
    decoded["mapped"] = True
    return decoded


def test_dataset_pipeline_e2e_with_split_shuffle_map_batch_unbatch(tmp_path: Path) -> None:
    shards = _build_demo_shards(tmp_path)
    context = RuntimeContext(rank=1, world_size=2, worker_id=0, num_workers=1, epoch=2, seed=23)

    selected_shards = split_shards(shards, context)
    expected_values: list[int] = []
    for shard in selected_shards:
        for sample in iter_tar_records(shard):
            txt_payload = sample.get("txt")
            assert isinstance(txt_payload, bytes)
            expected_values.append(int(txt_payload.decode("utf-8")))

    dataset = (
        Dataset.from_tars(shards, context=context)
        .shuffle(buffer_size=8, initial=4)
        .map(_decode_sample)
        .batch(batch_size=5)
        .unbatch()
    )

    output = list(dataset)

    assert len(output) == len(expected_values)
    assert all(isinstance(sample, dict) for sample in output)
    assert all(sample.get("mapped") is True for sample in output if isinstance(sample, dict))

    output_values = sorted(int(sample["value"]) for sample in output if isinstance(sample, dict))
    assert output_values == sorted(expected_values)
