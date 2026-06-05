"""JSONL splitting and spill sharding."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

from ...core.types import PathLikeStr, Sample
from .reader import _parse_jsonl_line


def _wc_lines(path: str) -> int:
    """Count lines using wc -l."""
    result = subprocess.run(
        ["wc", "-l", path],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "LC_ALL": "C"},
    )
    return int(result.stdout.strip().split()[0])


def split_jsonl_files(paths: list[str], min_chunks: int) -> list[str]:
    """Split JSONL files into at least *min_chunks* pieces using ``split``.

    Args:
        paths: Input JSONL file paths.
        min_chunks: Minimum number of output chunks to produce.

    Returns:
        Materialized shard file paths."""
    if len(paths) >= min_chunks:
        return paths

    total_lines = sum(_wc_lines(p) for p in paths)
    if total_lines == 0 or min_chunks <= 0:
        return paths

    chunk_dir = os.path.join(os.path.dirname(paths[0]) or ".", ".chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    result_paths: list[str] = []
    for path in paths:
        n_lines = _wc_lines(path)
        n_splits = max(1, round(n_lines / total_lines * min_chunks))
        if n_splits <= 1:
            result_paths.append(path)
            continue

        stem = os.path.basename(path)
        existing_chunks = sorted(
            os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) if f.startswith(stem + "._chunk_")
        )
        if existing_chunks:
            result_paths.extend(existing_chunks)
            continue

        lines_per_chunk = max(1, (n_lines + n_splits - 1) // n_splits)
        prefix = os.path.join(chunk_dir, stem + "._chunk_")
        subprocess.run(
            ["split", "-l", str(lines_per_chunk), "-d", "-a", "5", path, prefix],
            check=True,
        )
        chunk_files = sorted(
            os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) if f.startswith(stem + "._chunk_")
        )
        result_paths.extend(chunk_files)

    return result_paths


def materialize_jsonl_shards(
    files: Sequence[str],
    *,
    group_key: str | None,
    num_shards: int | None,
    target_samples_per_shard: int | None,
    spill_buckets: int,
    output_dir: PathLikeStr | None,
) -> list[str]:
    """Spill raw JSONL rows into balanced local shard files.

    Args:
        files: Input JSONL files to materialize.
        group_key: Optional sample field used to keep related rows in the same shard.
        num_shards: Explicit number of output shards.
        target_samples_per_shard: Target number of samples per output shard.
        spill_buckets: Number of temporary hash buckets used while sharding.
        output_dir: Directory where materialized shards are written.

    Returns:
        Materialized shard file paths."""

    if spill_buckets <= 0:
        msg = f"[InvalidSpillBucketCount] spill_buckets must be > 0, got={spill_buckets}"
        raise ValueError(msg)
    if num_shards is not None and num_shards <= 0:
        msg = f"[InvalidShardCount] num_shards must be > 0, got={num_shards}"
        raise ValueError(msg)
    if target_samples_per_shard is not None and target_samples_per_shard <= 0:
        msg = f"[InvalidTargetSamplesPerShard] target_samples_per_shard must be > 0, got={target_samples_per_shard}"
        raise ValueError(msg)

    fingerprint = _jsonl_shard_plan_fingerprint(
        files=files,
        group_key=group_key,
        num_shards=num_shards,
        target_samples_per_shard=target_samples_per_shard,
        spill_buckets=spill_buckets,
    )
    root = Path(output_dir) if output_dir is not None else Path(".mvp_dataset_jsonl_shards")
    dataset_dir = root / fingerprint
    manifest_path = dataset_dir / "manifest.json"
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        shard_paths = payload.get("shards")
        if isinstance(shard_paths, list) and all(isinstance(path, str) for path in shard_paths):
            return [str(dataset_dir / path) for path in shard_paths]

    dataset_dir.mkdir(parents=True, exist_ok=True)
    bucket_dir = dataset_dir / "buckets"
    bucket_dir.mkdir(parents=True, exist_ok=True)

    bucket_handles: dict[int, object] = {}
    bucket_counts = [0] * spill_buckets
    total_rows = 0
    try:
        for file in files:
            with open(file, encoding="utf-8") as handle:
                for i, line in enumerate(handle):
                    sample = _parse_jsonl_line(file, i, line)
                    bucket_id = _bucket_id_for_sample(sample, group_key=group_key, spill_buckets=spill_buckets)
                    bucket_counts[bucket_id] += 1
                    total_rows += 1
                    bucket_handle = bucket_handles.get(bucket_id)
                    if bucket_handle is None:
                        bucket_path = bucket_dir / f"bucket_{bucket_id:05d}.jsonl"
                        bucket_handle = bucket_path.open("w", encoding="utf-8")
                        bucket_handles[bucket_id] = bucket_handle
                    bucket_handle.write(json.dumps(sample, ensure_ascii=True) + "\n")
    finally:
        for handle in bucket_handles.values():
            handle.close()

    if total_rows == 0:
        msg = "[EmptyJsonlSource] no rows found in input files"
        raise ValueError(msg)

    final_shard_count = _resolve_final_shard_count(
        total_rows=total_rows,
        num_shards=num_shards,
        target_samples_per_shard=target_samples_per_shard,
    )
    shard_targets = _balanced_shard_targets(total_rows=total_rows, shard_count=final_shard_count)

    shard_paths = [dataset_dir / f"shard_{index:05d}.jsonl" for index in range(final_shard_count)]
    shard_handles = [path.open("w", encoding="utf-8") for path in shard_paths]
    try:
        current_shard = 0
        rows_in_current_shard = 0
        for bucket_id in sorted(i for i, c in enumerate(bucket_counts) if c > 0):
            bucket_path = bucket_dir / f"bucket_{bucket_id:05d}.jsonl"
            with bucket_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if current_shard < final_shard_count - 1 and rows_in_current_shard >= shard_targets[current_shard]:
                        current_shard += 1
                        rows_in_current_shard = 0
                    shard_handles[current_shard].write(line)
                    rows_in_current_shard += 1
        manifest = {
            "files": list(files),
            "group_key": group_key,
            "num_shards": final_shard_count,
            "target_samples_per_shard": target_samples_per_shard,
            "spill_buckets": spill_buckets,
            "shards": [path.name for path in shard_paths],
        }
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=True, indent=2, sort_keys=True)
    finally:
        for handle in shard_handles:
            handle.close()

    return [str(path) for path in shard_paths]


def _bucket_id_for_sample(sample: Sample, *, group_key: str | None, spill_buckets: int) -> int:
    """Return the deterministic shard bucket id for a sample key."""
    if group_key is None:
        key = str(sample["__key__"])
    else:
        value = sample.get(group_key)
        if isinstance(value, str):
            key = value.split("#", 1)[0]
        elif isinstance(value, list):
            if not value:
                key = str(sample["__key__"])
            elif all(isinstance(item, str) for item in value):
                key = value[0].split("#", 1)[0]
            else:
                msg = f"[InvalidGroupKey] sample missing string refs for group_key={group_key!r}"
                raise ValueError(msg)
        else:
            msg = f"[InvalidGroupKey] sample missing string key for group_key={group_key!r}"
            raise ValueError(msg)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % spill_buckets


def _jsonl_shard_plan_fingerprint(
    *,
    files: Sequence[str],
    group_key: str | None,
    num_shards: int | None,
    target_samples_per_shard: int | None,
    spill_buckets: int,
) -> str:
    """Return a fingerprint for the JSONL shard assignment plan."""
    payload = {
        "files": [(file, s.st_mtime_ns, s.st_size) for file in files for s in (Path(file).stat(),)],
        "group_key": group_key,
        "num_shards": num_shards,
        "target_samples_per_shard": target_samples_per_shard,
        "spill_buckets": spill_buckets,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_final_shard_count(
    *,
    total_rows: int,
    num_shards: int | None,
    target_samples_per_shard: int | None,
) -> int:
    """Resolve the output shard count after distributed sharding."""
    if num_shards is not None:
        return num_shards
    if target_samples_per_shard is None:
        return 1
    return max(1, (total_rows + target_samples_per_shard - 1) // target_samples_per_shard)


def _balanced_shard_targets(*, total_rows: int, shard_count: int) -> list[int]:
    """Return per-shard row targets whose totals differ by at most one."""

    base = total_rows // shard_count
    remainder = total_rows % shard_count
    return [base + (1 if index < remainder else 0) for index in range(shard_count)]
