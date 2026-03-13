from __future__ import annotations

import shutil
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path

from mvp_dataset import Dataset


class TarMemberReader:
    def __init__(self) -> None:
        self._cache: dict[str, tuple[tarfile.TarFile, dict[str, tarfile.TarInfo]]] = {}

    def close(self) -> None:
        for archive, _members in self._cache.values():
            archive.close()
        self._cache.clear()

    def read(self, shard_path: str, member_name: str) -> bytes:
        if shard_path not in self._cache:
            archive = tarfile.open(shard_path, mode="r:*")
            members = {member.name: member for member in archive.getmembers()}
            self._cache[shard_path] = (archive, members)

        archive, members = self._cache[shard_path]
        extracted = archive.extractfile(members[member_name])
        if extracted is None:
            msg = f"failed to extract {member_name!r} from {shard_path!r}"
            raise tarfile.ExtractError(msg)
        return extracted.read()


def copy_demo_image_shards(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for shard_path in sorted(Path("examples/demo_data/image").glob("*.tar")):
        target = output_dir / shard_path.name
        shutil.copy2(shard_path, target)
        copied.append(target)
    return copied


def make_normal_preprocessor(
    depth_dir: Path,
    reader: TarMemberReader,
    preprocessor_calls: list[int],
) -> Callable[[dict[str, object]], dict[str, bytes]]:
    def normal_preprocessor(sample: dict[str, object]) -> dict[str, bytes]:
        preprocessor_calls[0] += 1
        shard_path = str(depth_dir / Path(str(sample["__shard__"])).name)
        member_name = f"{sample['__key__']}.depth.png"
        return {"normal.png": reader.read(shard_path, member_name)}

    return normal_preprocessor


def build_normal_dataset(
    shards: str | list[str],
    depth_dir: Path,
    reader: TarMemberReader,
    preprocessor_calls: list[int],
) -> Dataset:
    return Dataset.from_tars(shards).select(
        ["image", "normal"],
        preprocessors={"normal": make_normal_preprocessor(depth_dir, reader, preprocessor_calls)},
    )


def export_demo_normal_shards(output_dir: Path) -> list[Path]:
    depth_dir = Path("examples/demo_data/depth")
    source_image_shards = sorted(Path("examples/demo_data/image").glob("*.tar"))
    missing_source_shards: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for shard_path in source_image_shards:
        target = output_dir / shard_path.name
        if target.is_file():
            continue
        missing_source_shards.append(shard_path)

    if not missing_source_shards:
        return []

    reader = TarMemberReader()
    preprocessor_calls = [0]

    try:
        with tempfile.TemporaryDirectory(prefix="mvp_dataset_normal_") as tmp_dir:
            staged_image_dir = Path(tmp_dir) / "image"
            copied_shards = []
            for shard_path in missing_source_shards:
                target = staged_image_dir / shard_path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(shard_path, target)
                copied_shards.append(target)
            observed_samples = list(
                build_normal_dataset([str(path) for path in copied_shards], depth_dir, reader, preprocessor_calls)
            )

            cache_paths = [path.with_name(f"{path.stem}_normal.tar") for path in copied_shards]
            exported_paths: list[Path] = []
            for cache_path in cache_paths:
                target = output_dir / cache_path.name.replace("_normal.tar", ".tar")
                if target.exists():
                    continue
                shutil.copy2(cache_path, target)
                exported_paths.append(target)
    finally:
        reader.close()

    assert preprocessor_calls[0] == len(missing_source_shards) * 8
    assert len(observed_samples) == len(missing_source_shards) * 8
    return exported_paths


def test_select_materializes_and_reuses_cached_tar_fields(tmp_path: Path) -> None:
    copied_shards = copy_demo_image_shards(tmp_path / "image")
    shard_pattern = str((tmp_path / "image") / "shard_{000000..000003}.tar")
    depth_dir = Path("examples/demo_data/depth")
    reader = TarMemberReader()
    preprocessor_calls = [0]

    try:
        observed_runs = [list(build_normal_dataset(shard_pattern, depth_dir, reader, preprocessor_calls)) for _ in range(2)]
    finally:
        reader.close()

    assert preprocessor_calls[0] == 32
    assert len(observed_runs) == 2
    assert len(observed_runs[0]) == len(observed_runs[1]) == 32
    assert observed_runs[0] == observed_runs[1]

    first_sample = observed_runs[0][0]
    assert set(first_sample) == {"__key__", "__shard__", "__index_in_shard__", "image.png", "normal.png"}

    cache_paths = [path.with_name(f"{path.stem}_normal.tar") for path in copied_shards]
    assert all(path.is_file() for path in cache_paths)

    source_depth_shard = str(Path("examples/demo_data/depth") / Path(first_sample["__shard__"]).name)
    expected_depth = TarMemberReader()
    try:
        payload = expected_depth.read(source_depth_shard, f"{first_sample['__key__']}.depth.png")
    finally:
        expected_depth.close()
    assert first_sample["normal.png"] == payload


def main() -> None:
    output_dir = Path("examples/demo_data/normal")
    exported_paths = export_demo_normal_shards(output_dir)
    print(f"materialized {len(exported_paths)} normal shards under {output_dir}")
    for path in exported_paths:
        print(path)


if __name__ == "__main__":
    main()
