#!/usr/bin/env python3
"""Benchmark read throughput across source formats using the Dataset API.

Generates 10 000 synthetic image+text samples, writes them into each
supported format (tar, jsonl+tar-ref, jsonl-inline, parquet, lance),
then times iteration through three pipeline configurations:

  1. source only   — raw Dataset read
  2. + map         — Dataset.map(decode_fn)
  3. + map+shuffle — Dataset.map(decode_fn).shuffle(buffer)

Usage:
    python bench_sources.py [--num-samples 10000] [--image-kb 50] [--num-shards 10]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from mvp_dataset.core.context import RuntimeContext
from mvp_dataset.core.dataset import Dataset
from mvp_dataset.sources.jsonl.dataset import JsonlDataset
from mvp_dataset.sources.lance.dataset import LanceDataset
from mvp_dataset.sources.parquet.dataset import ParquetDataset
from mvp_dataset.sources.tar.dataset import TarDataset

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def generate_samples(num_samples: int, image_kb: int, seed: int = 42) -> list[dict]:
    """Return deterministic synthetic samples with image bytes + caption."""
    import random

    rng = random.Random(seed)
    image_size = image_kb * 1024
    samples = []
    for i in range(num_samples):
        image = rng.randbytes(image_size)
        caption = f"A synthetic caption for sample {i:06d}. " + rng.choice(
            [
                "The quick brown fox jumps over the lazy dog.",
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                "Pack my box with five dozen liquor jugs.",
                "How vexingly quick daft zebras jump.",
            ]
        )
        samples.append({"image": image, "caption": caption})
    return samples


# ---------------------------------------------------------------------------
# Writers — one per format
# ---------------------------------------------------------------------------


def write_tar_shards(samples: list[dict], out_dir: Path, num_shards: int) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: list[str] = []
    per_shard = (len(samples) + num_shards - 1) // num_shards

    for shard_idx in range(num_shards):
        shard_path = out_dir / f"shard_{shard_idx:05d}.tar"
        shard_paths.append(str(shard_path))
        start = shard_idx * per_shard
        end = min(start + per_shard, len(samples))

        with tarfile.open(shard_path, "w") as tf:
            for i in range(start, end):
                key = f"{i:06d}"
                sample = samples[i]

                img_info = tarfile.TarInfo(name=f"{key}.jpg")
                img_info.size = len(sample["image"])
                tf.addfile(img_info, io.BytesIO(sample["image"]))

                cap_bytes = sample["caption"].encode("utf-8")
                cap_info = tarfile.TarInfo(name=f"{key}.txt")
                cap_info.size = len(cap_bytes)
                tf.addfile(cap_info, io.BytesIO(cap_bytes))

    return shard_paths


def write_jsonl_with_tar_refs(samples: list[dict], out_dir: Path, num_shards: int) -> tuple[list[str], str]:
    """Returns (jsonl_paths, tar_base_dir)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_paths: list[str] = []
    per_shard = (len(samples) + num_shards - 1) // num_shards

    for shard_idx in range(num_shards):
        jsonl_path = out_dir / f"shard_{shard_idx:05d}.jsonl"
        tar_path = out_dir / f"images_{shard_idx:05d}.tar"
        jsonl_paths.append(str(jsonl_path))
        start = shard_idx * per_shard
        end = min(start + per_shard, len(samples))

        with tarfile.open(tar_path, "w") as tf:
            for i in range(start, end):
                key = f"{i:06d}"
                img_info = tarfile.TarInfo(name=f"{key}.jpg")
                img_info.size = len(samples[i]["image"])
                tf.addfile(img_info, io.BytesIO(samples[i]["image"]))

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for i in range(start, end):
                key = f"{i:06d}"
                row = {
                    "caption": samples[i]["caption"],
                    "image": f"tar://{tar_path}#{key}.jpg",
                }
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

    return jsonl_paths, str(out_dir)


def write_jsonl_inline(samples: list[dict], out_dir: Path, num_shards: int) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_paths: list[str] = []
    per_shard = (len(samples) + num_shards - 1) // num_shards

    for shard_idx in range(num_shards):
        jsonl_path = out_dir / f"shard_{shard_idx:05d}.jsonl"
        jsonl_paths.append(str(jsonl_path))
        start = shard_idx * per_shard
        end = min(start + per_shard, len(samples))

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for i in range(start, end):
                row = {
                    "caption": samples[i]["caption"],
                    "image_b64": base64.b64encode(samples[i]["image"]).decode("ascii"),
                }
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

    return jsonl_paths


def write_parquet_files(samples: list[dict], out_dir: Path, num_shards: int) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    per_shard = (len(samples) + num_shards - 1) // num_shards

    for shard_idx in range(num_shards):
        path = out_dir / f"shard_{shard_idx:05d}.parquet"
        paths.append(str(path))
        start = shard_idx * per_shard
        end = min(start + per_shard, len(samples))

        table = pa.table(
            {
                "image": pa.array(
                    [samples[i]["image"] for i in range(start, end)],
                    type=pa.binary(),
                ),
                "caption": pa.array(
                    [samples[i]["caption"] for i in range(start, end)],
                    type=pa.string(),
                ),
            }
        )
        pq.write_table(table, path, row_group_size=1000)

    return paths


def write_lance_datasets(samples: list[dict], out_dir: Path, num_shards: int) -> list[str]:
    import lance

    out_dir.mkdir(parents=True, exist_ok=True)
    uris: list[str] = []
    per_shard = (len(samples) + num_shards - 1) // num_shards

    for shard_idx in range(num_shards):
        uri = str(out_dir / f"shard_{shard_idx:05d}.lance")
        uris.append(uri)
        start = shard_idx * per_shard
        end = min(start + per_shard, len(samples))

        table = pa.table(
            {
                "image": pa.array(
                    [samples[i]["image"] for i in range(start, end)],
                    type=pa.binary(),
                ),
                "caption": pa.array(
                    [samples[i]["caption"] for i in range(start, end)],
                    type=pa.string(),
                ),
            }
        )
        lance.write_dataset(table, uri)

    return uris


# ---------------------------------------------------------------------------
# Dataset builders — return Dataset objects for each format
# ---------------------------------------------------------------------------


def build_tar_dataset(shard_paths: list[str], ctx: RuntimeContext) -> Dataset:
    return TarDataset.from_source(shard_paths, context=ctx)


def build_jsonl_ref_dataset(jsonl_paths: list[str], tar_dir: str, ctx: RuntimeContext) -> Dataset:
    return JsonlDataset.from_source(jsonl_paths, context=ctx, ref_fields=[("image", tar_dir)])


def build_jsonl_inline_dataset(jsonl_paths: list[str], ctx: RuntimeContext) -> Dataset:
    return JsonlDataset.from_source(jsonl_paths, context=ctx)


def build_parquet_dataset(shard_paths: list[str], ctx: RuntimeContext) -> Dataset:
    return ParquetDataset.from_source(shard_paths, context=ctx)


def build_lance_dataset(uris: list[str], ctx: RuntimeContext) -> Dataset:
    return LanceDataset.from_source(uris, context=ctx)


# ---------------------------------------------------------------------------
# Map functions used as pipeline stages
# ---------------------------------------------------------------------------


def decode_tar_sample(sample: dict) -> dict:
    """Simulate decoding: image bytes length + uppercase caption."""
    return {
        "image_size": len(sample["jpg"]),
        "caption": sample["txt"].decode("utf-8").upper(),
        "__key__": sample["__key__"],
    }


def decode_columnar_sample(sample: dict) -> dict:
    """Simulate decoding: image bytes length + uppercase caption."""
    image = sample.get("image", b"")
    return {
        "image_size": len(image) if isinstance(image, (bytes, memoryview)) else 0,
        "caption": sample["caption"].upper() if isinstance(sample["caption"], str) else str(sample["caption"]).upper(),
        "__key__": sample.get("__key__", ""),
    }


def decode_jsonl_inline_sample(sample: dict) -> dict:
    """Simulate decoding: base64 decode + uppercase caption."""
    raw = sample.get("image_b64", "")
    image = base64.b64decode(raw) if isinstance(raw, str) else b""
    return {
        "image_size": len(image),
        "caption": sample["caption"].upper() if isinstance(sample["caption"], str) else str(sample["caption"]).upper(),
        "__key__": sample.get("__key__", ""),
    }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

SHUFFLE_BUFFER = 1000


def time_iteration(ds: Dataset, warmup: int, repeats: int) -> list[float]:
    """Drain the dataset iterator and return per-repeat wall-clock times."""
    for _ in range(warmup):
        count = 0
        for _ in ds:
            count += 1

    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        count = 0
        for _ in ds:
            count += 1
        times.append(time.perf_counter() - t0)
    return times


PipelineKind = str  # "source", "map", "map+shuffle"


def build_pipelines(base_ds: Dataset, map_fn: callable) -> dict[PipelineKind, Dataset]:
    """Build the three pipeline variants from a base dataset."""
    ds_map = base_ds.map(map_fn)
    ds_map_shuffle = ds_map.shuffle(buffer_size=SHUFFLE_BUFFER)
    return {
        "source": base_ds,
        "map": ds_map,
        "map+shuffle": ds_map_shuffle,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_row(name: str, times: list[float], num_samples: int, image_kb: int) -> str:
    avg = sum(times) / len(times)
    best = min(times)
    throughput = num_samples / avg
    data_mb = num_samples * image_kb / 1024
    bandwidth = data_mb / avg
    return f"  {name:<34s}  avg {avg:6.2f}s  best {best:6.2f}s  {throughput:8.0f} samples/s  {bandwidth:7.1f} MB/s"


def report(
    all_results: dict[str, dict[PipelineKind, list[float]]],
    num_samples: int,
    image_kb: int,
) -> None:
    pipeline_kinds = ["source", "map", "map+shuffle"]

    for kind in pipeline_kinds:
        print()
        print(f"{'=' * 86}")
        print(f"  Pipeline: {kind:<16s}  ({num_samples} samples, ~{image_kb}KB image each)")
        print(f"{'=' * 86}")
        for source_name, results_by_kind in all_results.items():
            if kind in results_by_kind:
                print(_fmt_row(source_name, results_by_kind[kind], num_samples, image_kb))
        print(f"{'=' * 86}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark mvp-dataset source read speed")
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--image-kb", type=int, default=50)
    parser.add_argument("--num-shards", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--keep-data", action="store_true", help="Keep generated data after run")
    args = parser.parse_args()

    ctx = RuntimeContext(seed=42)

    print(f"Generating {args.num_samples} synthetic samples ({args.image_kb}KB image each) ...")
    samples = generate_samples(args.num_samples, args.image_kb)
    total_mb = args.num_samples * args.image_kb / 1024
    print(f"  Total payload: ~{total_mb:.0f} MB")

    tmpdir_obj = tempfile.mkdtemp(prefix="bench_sources_")
    tmpdir = Path(tmpdir_obj)
    print(f"  Working directory: {tmpdir}")

    try:
        # ---- write all formats ----
        print("\nWriting formats ...")

        t0 = time.perf_counter()
        tar_paths = write_tar_shards(samples, tmpdir / "tar", args.num_shards)
        print(f"  tar:              {time.perf_counter() - t0:.2f}s  ({len(tar_paths)} shards)")

        t0 = time.perf_counter()
        jsonl_ref_paths, jsonl_ref_tar_dir = write_jsonl_with_tar_refs(samples, tmpdir / "jsonl_ref", args.num_shards)
        print(f"  jsonl+tar-ref:    {time.perf_counter() - t0:.2f}s  ({len(jsonl_ref_paths)} shards)")

        t0 = time.perf_counter()
        jsonl_inline_paths = write_jsonl_inline(samples, tmpdir / "jsonl_inline", args.num_shards)
        print(f"  jsonl-inline:     {time.perf_counter() - t0:.2f}s  ({len(jsonl_inline_paths)} shards)")

        t0 = time.perf_counter()
        parquet_paths = write_parquet_files(samples, tmpdir / "parquet", args.num_shards)
        print(f"  parquet:          {time.perf_counter() - t0:.2f}s  ({len(parquet_paths)} shards)")

        t0 = time.perf_counter()
        lance_uris = write_lance_datasets(samples, tmpdir / "lance", args.num_shards)
        print(f"  lance:            {time.perf_counter() - t0:.2f}s  ({len(lance_uris)} datasets)")

        del samples

        # ---- build datasets ----
        sources: dict[str, tuple[Dataset, callable]] = {
            "tar": (
                build_tar_dataset(tar_paths, ctx),
                decode_tar_sample,
            ),
            "jsonl+tar-ref": (
                build_jsonl_ref_dataset(jsonl_ref_paths, jsonl_ref_tar_dir, ctx),
                decode_columnar_sample,
            ),
            "jsonl-inline (b64)": (
                build_jsonl_inline_dataset(jsonl_inline_paths, ctx),
                decode_jsonl_inline_sample,
            ),
            "parquet": (
                build_parquet_dataset(parquet_paths, ctx),
                decode_columnar_sample,
            ),
            "lance": (
                build_lance_dataset(lance_uris, ctx),
                decode_columnar_sample,
            ),
        }

        # ---- benchmark ----
        print(f"\nBenchmarking (warmup={args.warmup}, repeats={args.repeats}, shuffle_buffer={SHUFFLE_BUFFER}) ...")
        all_results: dict[str, dict[PipelineKind, list[float]]] = {}

        for source_name, (base_ds, map_fn) in sources.items():
            pipelines = build_pipelines(base_ds, map_fn)
            all_results[source_name] = {}

            for kind, ds in pipelines.items():
                label = f"{source_name} / {kind}"
                print(f"  {label} ...")
                all_results[source_name][kind] = time_iteration(ds, args.warmup, args.repeats)

        # ---- report ----
        report(all_results, args.num_samples, args.image_kb)

    finally:
        if not args.keep_data:
            shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            print(f"Data kept at: {tmpdir}")


if __name__ == "__main__":
    main()
