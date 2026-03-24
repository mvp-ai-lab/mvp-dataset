from __future__ import annotations

import shutil
import tarfile
from functools import partial
from pathlib import Path

from mvp_dataset import Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = REPO_ROOT / "examples" / "demo_data" / "image"
DEPTH_DIR = REPO_ROOT / "examples" / "demo_data" / "depth"


def read_tar_member(shard_path: Path, member_name: str) -> bytes:
    with tarfile.open(shard_path) as archive:
        member = archive.getmember(member_name)
        extracted = archive.extractfile(member)
        assert extracted is not None
        return extracted.read()


def load_depth_sample(sample: dict[str, object], *, depth_dir: Path) -> dict[str, bytes]:
    shard_name = Path(str(sample["__shard__"])).name
    sample_key = str(sample["__key__"])
    return {"depth.png": read_tar_member(depth_dir / shard_name, f"{sample_key}.depth.png")}


def test_select_from_tars(tmp_path: Path) -> None:
    image_dir = tmp_path / "image"
    depth_dir = tmp_path / "depth"
    shutil.copytree(IMAGE_DIR, image_dir)
    shutil.copytree(DEPTH_DIR, depth_dir)

    dataset = (
        Dataset.from_tars(
            str(image_dir / "shard_{000000..000003}.tar"),
            resample=False,
        )
        .select(
            ["image", "normal"],
            preprocessors={"depth": partial(load_depth_sample, depth_dir=depth_dir)},
        )
        .shuffle(buffer_size=8)
    )

    sample = next(iter(dataset))
    assert sample["image.png"]
    assert sample["depth.png"]
