"""JSONL splitting through the unified persistent cache."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ...cache import CacheBuildResult, CacheManager, fingerprint_local_files
from .types import JsonlShard, JsonlSplitPlan

JSONL_SPLIT_FORMAT_VERSION = 1


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


def split_jsonl_files(paths: list[str], min_chunks: int) -> JsonlSplitPlan:
    """Split JSONL files into at least *min_chunks* pieces using ``split``.

    Args:
        paths: Input JSONL file paths.
        min_chunks: Minimum number of output chunks to produce.

    Returns:
        Source identity and physical shards with stable logical row ranges."""
    manager = CacheManager()
    source = fingerprint_local_files(paths, mode=manager.config.fingerprint_mode)
    logical_paths = tuple(str(item["path"]) for item in source.manifest["files"])

    if len(paths) >= min_chunks or min_chunks <= 0:
        return JsonlSplitPlan(
            source=source,
            shards=tuple(
                JsonlShard(
                    physical_path=path,
                    source_index=index,
                    logical_path=logical_paths[index],
                    line_start=0,
                    line_count=None,
                )
                for index, path in enumerate(paths)
            ),
        )

    line_counts = [_wc_lines(path) for path in paths]
    total_lines = sum(line_counts)
    if total_lines == 0:
        return JsonlSplitPlan(
            source=source,
            shards=tuple(
                JsonlShard(
                    physical_path=path,
                    source_index=index,
                    logical_path=logical_paths[index],
                    line_start=0,
                    line_count=line_counts[index],
                )
                for index, path in enumerate(paths)
            ),
        )

    parameters = {
        "line_counts": line_counts,
        "min_chunks": min_chunks,
    }

    def _build(temporary_dir: Path) -> CacheBuildResult:
        files: list[str] = []
        for index, (path, line_count) in enumerate(zip(paths, line_counts, strict=True)):
            split_count = max(1, round(line_count / total_lines * min_chunks))
            if split_count <= 1:
                continue
            lines_per_chunk = max(1, (line_count + split_count - 1) // split_count)
            prefix_name = f"{index:05d}-{Path(path).name}.chunk-"
            subprocess.run(
                ["split", "-l", str(lines_per_chunk), "-d", "-a", "5", path, str(temporary_dir / prefix_name)],
                check=True,
            )
            files.extend(chunk.name for chunk in sorted(temporary_dir.glob(f"{prefix_name}*")))
        return CacheBuildResult.from_files(files)

    entry = manager.ensure(
        source=source,
        kind="jsonl-split",
        format_version=JSONL_SPLIT_FORMAT_VERSION,
        parameters=parameters,
        build=_build,
    )

    result_shards: list[JsonlShard] = []
    for index, (path, line_count) in enumerate(zip(paths, line_counts, strict=True)):
        split_count = max(1, round(line_count / total_lines * min_chunks))
        if split_count <= 1:
            result_shards.append(
                JsonlShard(
                    physical_path=path,
                    source_index=index,
                    logical_path=logical_paths[index],
                    line_start=0,
                    line_count=line_count,
                )
            )
            continue
        lines_per_chunk = max(1, (line_count + split_count - 1) // split_count)
        prefix_name = f"{index:05d}-{Path(path).name}.chunk-"
        chunks = sorted(entry.path.glob(f"{prefix_name}*"))
        if not chunks:
            msg = f"[InvalidJsonlSplitCache] no cached chunks for input index {index}"
            raise RuntimeError(msg)
        for chunk_index, chunk in enumerate(chunks):
            line_start = chunk_index * lines_per_chunk
            result_shards.append(
                JsonlShard(
                    physical_path=str(chunk),
                    source_index=index,
                    logical_path=logical_paths[index],
                    line_start=line_start,
                    line_count=min(lines_per_chunk, line_count - line_start),
                )
            )
    return JsonlSplitPlan(source=source, shards=tuple(result_shards))
