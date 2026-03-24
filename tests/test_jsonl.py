from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mvp_dataset import Dataset
from mvp_dataset.sources.jsonl import count_jsonl_samples, jsonl_cache_dir, jsonl_cache_glob

DEFAULT_JSONL_PATH = REPO_ROOT / "examples" / "demo_data" / "samples.jsonl"


def find_generated_field(sample: dict[str, object], *, key: str, suffix: str) -> str:
    matches = [
        field_name
        for field_name in sample
        if field_name.startswith(f"{key}.") and field_name.endswith(suffix)
    ]
    assert len(matches) == 1
    return matches[0]


def run_jsonl_map_cache_check(jsonl_path: Path) -> None:
    allow_calls = True
    call_counts = {"stage1": 0, "stage2": 0}

    def stage1(sample: dict[str, object]) -> dict[str, bytes]:
        nonlocal allow_calls
        if not allow_calls:
            raise AssertionError("stage1 should reuse cache on the second iteration")
        call_counts["stage1"] += 1
        return {"mapped.stage1.bin": b"stage1:" + bytes(sample["image"][:16])}

    def stage2(sample: dict[str, object]) -> dict[str, bytes]:
        nonlocal allow_calls
        if not allow_calls:
            raise AssertionError("stage2 should reuse cache on the second iteration")
        call_counts["stage2"] += 1
        stage1_field = find_generated_field(sample, key="mapped", suffix=".stage1.bin")
        return {"mapped.stage2.bin": b"stage2:" + bytes(sample[stage1_field])}

    dataset = (
        Dataset.from_jsonl(str(jsonl_path), resample=False)
        .resolve_refs((("image", "."), ("depth", ".")))
        .map({"mapped": stage1})
        .map({"mapped": stage2})
        .shuffle(buffer_size=8)
    )

    sample = next(iter(dataset))
    stage1_field = find_generated_field(sample, key="mapped", suffix=".stage1.bin")
    stage2_field = find_generated_field(sample, key="mapped", suffix=".stage2.bin")
    assert bytes(sample[stage1_field]).startswith(b"stage1:")
    assert bytes(sample[stage2_field]).startswith(b"stage2:stage1:")

    cache_paths = sorted(jsonl_cache_dir(jsonl_path).glob(Path(jsonl_cache_glob(jsonl_path, "mapped")).name))
    assert cache_paths
    stage2_cache_path: Path | None = None
    stage2_cache_fields: list[str] | None = None
    for cache_path in cache_paths:
        with cache_path.open(encoding="utf-8") as handle:
            row = json.loads(next(handle))
        cache_fields = sorted(row["__cache_fields__"])
        if any(field_name.endswith(".stage2.bin") for field_name in cache_fields):
            stage2_cache_path = cache_path
            stage2_cache_fields = cache_fields
            break
    assert stage2_cache_path is not None
    assert stage2_cache_fields is not None
    assert any(field_name.endswith(".stage1.bin") for field_name in stage2_cache_fields)
    assert any(field_name.endswith(".stage2.bin") for field_name in stage2_cache_fields)
    assert count_jsonl_samples(stage2_cache_path) == count_jsonl_samples(jsonl_path)

    first_call_counts = dict(call_counts)
    allow_calls = False

    second_sample = next(iter(dataset))
    assert second_sample[stage1_field] == sample[stage1_field]
    assert second_sample[stage2_field] == sample[stage2_field]
    assert call_counts == first_call_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL_PATH)
    args = parser.parse_args()
    os.environ["MVP_JSONL_PATH"] = str(args.jsonl)
    run_jsonl_map_cache_check(args.jsonl)
    print("test_jsonl checks passed")


if __name__ == "__main__":
    main()
