"""Parquet chunk discovery."""

from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from typing import Final

import pyarrow.parquet as pq

from ...core.types import PathLikeStr

_MAX_METADATA_WORKERS: Final[int] = 16


@dataclass(frozen=True, slots=True)
class ParquetChunk:
    """One schedulable parquet row-group chunk."""

    path: str
    row_groups: tuple[int, ...]
    row_offset: int
    num_rows: int
    row_group_offsets: tuple[int, ...]
    row_group_num_rows: tuple[int, ...]


def list_parquet_chunks(
    shard_paths: Sequence[PathLikeStr],
    *,
    min_row_groups_per_chunk: int = 1,
    min_chunks: int = 0,
) -> list[ParquetChunk]:
    """Expand parquet files into schedulable row-group chunks.

    Args:
        shard_paths: Shard paths to read.
        min_row_groups_per_chunk: Minimum row groups combined into one Parquet chunk.
        min_chunks: Minimum number of chunks to produce.

    Returns:
        Parquet chunks collected from the input shards."""
    min_row_groups_per_chunk = _validate_min_row_groups_per_chunk(min_row_groups_per_chunk)
    chunks = _collect_parquet_chunks(shard_paths, min_row_groups_per_chunk)
    if len(chunks) < min_chunks:
        chunks = _collect_parquet_chunks(shard_paths, 1)
    return chunks


def _validate_min_row_groups_per_chunk(value: int) -> int:
    """Validate Parquet row-group chunk settings."""
    if value <= 0:
        msg = f"[InvalidParquetChunkConfig] min_row_groups_per_chunk must be >= 1, got {value}"
        raise ValueError(msg)
    return value


def _collect_parquet_chunks(
    shard_paths: Sequence[PathLikeStr],
    min_row_groups_per_chunk: int,
) -> list[ParquetChunk]:
    """Collect Parquet chunks for all shards."""
    shards = [str(shard_path) for shard_path in shard_paths]
    if not shards:
        return []

    chunks: list[ParquetChunk] = []
    num_workers = _metadata_num_workers(len(shards))
    if num_workers == 1:
        for shard in shards:
            chunks.extend(_collect_parquet_chunks_for_shard(shard, min_row_groups_per_chunk))
        return chunks

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for shard_chunks in executor.map(
            _collect_parquet_chunks_for_shard,
            shards,
            repeat(min_row_groups_per_chunk),
        ):
            chunks.extend(shard_chunks)
    return chunks


def _metadata_num_workers(num_shards: int) -> int:
    """Return the worker count used for Parquet metadata reads."""
    return min(num_shards, max(1, min(_MAX_METADATA_WORKERS, os.cpu_count() or 1)))


def _collect_parquet_chunks_for_shard(
    shard: str,
    min_row_groups_per_chunk: int,
) -> list[ParquetChunk]:
    """Collect Parquet chunks for one shard."""
    metadata = pq.read_metadata(shard)
    chunks: list[ParquetChunk] = []
    row_offset = 0
    pending_groups: list[int] = []
    pending_group_offsets: list[int] = []
    pending_group_num_rows: list[int] = []
    pending_rows = 0

    for row_group in range(metadata.num_row_groups):
        num_rows = metadata.row_group(row_group).num_rows
        pending_groups.append(row_group)
        pending_group_offsets.append(row_offset + pending_rows)
        pending_group_num_rows.append(num_rows)
        pending_rows += num_rows
        if len(pending_groups) >= min_row_groups_per_chunk:
            chunks.append(
                ParquetChunk(
                    path=shard,
                    row_groups=tuple(pending_groups),
                    row_offset=row_offset,
                    num_rows=pending_rows,
                    row_group_offsets=tuple(pending_group_offsets),
                    row_group_num_rows=tuple(pending_group_num_rows),
                )
            )
            row_offset += pending_rows
            pending_groups = []
            pending_group_offsets = []
            pending_group_num_rows = []
            pending_rows = 0

    if pending_groups:
        chunks.append(
            ParquetChunk(
                path=shard,
                row_groups=tuple(pending_groups),
                row_offset=row_offset,
                num_rows=pending_rows,
                row_group_offsets=tuple(pending_group_offsets),
                row_group_num_rows=tuple(pending_group_num_rows),
            )
        )

    return chunks
