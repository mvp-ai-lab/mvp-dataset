from __future__ import annotations

import argparse
import io
import os
import sys
import tarfile
from functools import partial
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mvp_dataset import Dataset
from mvp_dataset.sources.tar import cache_tar_dir, cache_tar_glob, count_tar_samples

DEFAULT_DATASET_DIR = REPO_ROOT / "examples" / "demo_data"


def read_tar_member(shard_path: Path, member_name: str) -> bytes:
    with tarfile.open(shard_path) as archive:
        member = archive.getmember(member_name)
        extracted = archive.extractfile(member)
        assert extracted is not None
        return extracted.read()


def rewrite_cache_with_one_sample(cache_path: Path) -> None:
    with tarfile.open(cache_path) as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        first_key = members[0].name.split(".", 1)[0]
        kept_members = [member for member in members if member.name.startswith(f"{first_key}.")]
        payloads: list[tuple[str, bytes]] = []
        for member in kept_members:
            extracted = archive.extractfile(member)
            assert extracted is not None
            payloads.append((member.name, extracted.read()))

    temp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    with tarfile.open(temp_path, mode="w") as archive:
        for member_name, payload in payloads:
            info = tarfile.TarInfo(name=member_name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    temp_path.replace(cache_path)


def find_generated_field(sample: dict[str, object], *, key: str, suffix: str) -> str:
    matches = [
        field_name
        for field_name in sample
        if field_name.startswith(f"{key}.") and field_name.endswith(suffix)
    ]
    assert len(matches) == 1
    return matches[0]


def process_depth(
    sample: dict[str, object],
    *,
    image_dir: Path,
) -> dict[str, bytes]:
    shard_name = Path(str(sample["__shard__"])).name
    sample_key = str(sample["__key__"])
    image_bytes = read_tar_member(image_dir / shard_name, f"{sample_key}.image.png")
    return {"depth.processed.bin": b"processed:" + image_bytes[:16]}


def resolve_dataset_dirs(dataset_dir: Path) -> tuple[Path, Path]:
    image_dir = dataset_dir / "image"
    depth_dir = dataset_dir / "depth"
    return image_dir, depth_dir


def configured_dataset_dir() -> Path:
    dataset_dir = os.environ.get("MVP_DATASET_DIR")
    return Path(dataset_dir) if dataset_dir is not None else DEFAULT_DATASET_DIR


def run_select_smoke_check(dataset_dir: Path) -> None:
    image_dir, _ = resolve_dataset_dirs(dataset_dir)
    process_depth_fn = partial(process_depth, image_dir=image_dir)
    dataset = (
        Dataset.from_tars(
            str(image_dir / "shard_{000000..000003}.tar"),
            resample=False,
        )
        .map({"depth": process_depth_fn})
        .shuffle(buffer_size=8)
    )

    samples = iter(dataset)
    for _ in range(8):
        sample = next(samples)
        depth_field = find_generated_field(sample, key="depth", suffix=".processed.bin")
        assert bytes(sample[depth_field]).startswith(b"processed:")


def run_select_from_tars_check() -> None:
    run_select_smoke_check(configured_dataset_dir())


def run_map_materializes_stage_specific_caches_check() -> None:
    dataset_dir = configured_dataset_dir()
    image_dir, _ = resolve_dataset_dirs(dataset_dir)
    shard_path = image_dir / "shard_000000.tar"

    allow_calls = True
    call_counts = {"stage1": 0, "stage2": 0}

    def stage1(sample: dict[str, object]) -> dict[str, bytes]:
        nonlocal allow_calls
        if not allow_calls:
            raise AssertionError("stage1 should reuse cache on the second iteration")
        call_counts["stage1"] += 1
        assert "depth.png" not in sample
        return {"mapped.stage1.bin": b"stage1:" + bytes(sample["image.png"])}

    def stage2(sample: dict[str, object]) -> dict[str, bytes]:
        nonlocal allow_calls
        if not allow_calls:
            raise AssertionError("stage2 should reuse cache on the second iteration")
        call_counts["stage2"] += 1
        assert "depth.png" not in sample
        stage1_field = find_generated_field(sample, key="mapped", suffix=".stage1.bin")
        return {"mapped.stage2.bin": b"stage2:" + bytes(sample[stage1_field])}

    dataset = (
        Dataset.from_tars(str(shard_path), resample=False)
        .map({"mapped": stage1})
        .map({"mapped": stage2})
        .shuffle(buffer_size=8)
    )

    sample = next(iter(dataset))
    stage1_field = find_generated_field(sample, key="mapped", suffix=".stage1.bin")
    stage2_field = find_generated_field(sample, key="mapped", suffix=".stage2.bin")
    assert sample[stage1_field].startswith(b"stage1:")
    assert sample[stage2_field].startswith(b"stage2:stage1:")

    cache_paths = sorted(cache_tar_dir(shard_path).glob(Path(cache_tar_glob(shard_path, "mapped")).name))
    assert len(cache_paths) == 1
    with tarfile.open(cache_paths[0]) as archive:
        cache_fields = [member.name for member in archive.getmembers() if member.isfile()]
    assert any(name.endswith(".stage1.bin") for name in cache_fields)
    assert any(name.endswith(".stage2.bin") for name in cache_fields)

    first_call_counts = dict(call_counts)
    allow_calls = False

    second_sample = next(iter(dataset))
    assert second_sample[stage1_field] == sample[stage1_field]
    assert second_sample[stage2_field] == sample[stage2_field]
    assert call_counts == first_call_counts


def run_cache_rebuilds_when_sample_count_mismatches_check() -> None:
    dataset_dir = configured_dataset_dir()
    image_dir, _ = resolve_dataset_dirs(dataset_dir)
    shard_path = image_dir / "shard_000000.tar"

    dataset = Dataset.from_tars(str(shard_path), resample=False).map({"rebuilt": lambda sample: {"rebuilt.bin": bytes(sample["image.png"])}})
    sample = next(iter(dataset))
    rebuild_field = find_generated_field(sample, key="rebuilt", suffix=".bin")
    assert sample[rebuild_field]

    cache_paths = sorted(cache_tar_dir(shard_path).glob(Path(cache_tar_glob(shard_path, "rebuilt")).name))
    assert len(cache_paths) == 1
    rebuild_cache_path = cache_paths[0]
    original_count = count_tar_samples(shard_path)
    rewrite_cache_with_one_sample(rebuild_cache_path)
    assert count_tar_samples(rebuild_cache_path) < original_count

    rebuilt_sample = next(iter(dataset))
    rebuilt_field = find_generated_field(rebuilt_sample, key="rebuilt", suffix=".bin")
    assert rebuilt_sample[rebuilt_field]
    assert count_tar_samples(rebuild_cache_path) == original_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    args = parser.parse_args()
    os.environ["MVP_DATASET_DIR"] = str(args.dataset_dir)
    run_select_smoke_check(args.dataset_dir)
    run_map_materializes_stage_specific_caches_check()
    run_cache_rebuilds_when_sample_count_mismatches_check()
    print("test_tars checks passed")


if __name__ == "__main__":
    main()
