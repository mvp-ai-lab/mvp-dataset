"""Parquet source implementation for row-group-parallel sample iteration."""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from typing import Final

import pyarrow.parquet as pq

from ...core.types import PathLikeStr, Sample

_DEFAULT_BATCH_SIZE: Final[int] = 65536
_FRAGMENT_SUFFIX: Final[str] = "@@rg="
_MAX_METADATA_WORKERS: Final[int] = 16
_BATCH_SIZE_ENV_VAR: Final[str] = "MVP_DATASET_PARQUET_BATCH_SIZE"


@dataclass(frozen=True, slots=True)
class ParquetFragment:
    """One schedulable parquet row-group fragment."""

    path: str
    row_groups: tuple[int, ...]
    row_offset: int
    num_rows: int

    @property
    def cache_key(self) -> str:
        """Stable shard identifier used by cache manifests and routing."""

        rg_spec = (
            str(self.row_groups[0]) if len(self.row_groups) == 1 else f"{self.row_groups[0]}-{self.row_groups[-1]}"
        )
        return f"{self.path}{_FRAGMENT_SUFFIX}{rg_spec}"


def list_parquet_fragments(
    shard_paths: Sequence[PathLikeStr],
    *,
    min_row_groups_per_fragment: int = 1,
    min_fragments: int = 0,
) -> list[ParquetFragment]:
    """Expand parquet files into schedulable row-group fragments.

    Consecutive row groups are merged until each fragment contains at least
    *min_row_groups_per_fragment* row groups. When the resulting fragment count
    is below *min_fragments*, the threshold is lowered to 1 so that every row
    group becomes its own fragment.
    """
    min_row_groups_per_fragment = _validate_min_row_groups_per_fragment(min_row_groups_per_fragment)
    fragments = _collect_parquet_fragments(shard_paths, min_row_groups_per_fragment)
    if len(fragments) < min_fragments:
        fragments = _collect_parquet_fragments(shard_paths, 1)
    return fragments


def _validate_min_row_groups_per_fragment(value: int) -> int:
    if value <= 0:
        msg = f"[InvalidParquetFragmentConfig] min_row_groups_per_fragment must be >= 1, got {value}"
        raise ValueError(msg)
    return value


def resolve_parquet_batch_size(batch_size: int | None = None) -> int:
    if batch_size is None:
        raw_value = os.environ.get(_BATCH_SIZE_ENV_VAR)
        if raw_value is None:
            batch_size = _DEFAULT_BATCH_SIZE
        else:
            try:
                batch_size = int(raw_value)
            except ValueError as exc:
                msg = f"[InvalidParquetBatchSize] {_BATCH_SIZE_ENV_VAR} must be an integer, got {raw_value!r}"
                raise ValueError(msg) from exc
    if batch_size <= 0:
        msg = f"[InvalidParquetBatchSize] {_BATCH_SIZE_ENV_VAR} must be >= 1, got {batch_size}"
        raise ValueError(msg)
    return batch_size


def _collect_parquet_fragments(
    shard_paths: Sequence[PathLikeStr],
    min_row_groups_per_fragment: int,
) -> list[ParquetFragment]:
    shards = [str(shard_path) for shard_path in shard_paths]
    if not shards:
        return []

    fragments: list[ParquetFragment] = []
    num_workers = _metadata_num_workers(len(shards))
    if num_workers == 1:
        for shard in shards:
            fragments.extend(_collect_parquet_fragments_for_shard(shard, min_row_groups_per_fragment))
        return fragments

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for shard_fragments in executor.map(
            _collect_parquet_fragments_for_shard,
            shards,
            repeat(min_row_groups_per_fragment),
        ):
            fragments.extend(shard_fragments)
    return fragments


def _metadata_num_workers(num_shards: int) -> int:
    return min(num_shards, max(1, min(_MAX_METADATA_WORKERS, os.cpu_count() or 1)))


def _collect_parquet_fragments_for_shard(
    shard: str,
    min_row_groups_per_fragment: int,
) -> list[ParquetFragment]:
    metadata = pq.read_metadata(shard)
    fragments: list[ParquetFragment] = []
    row_offset = 0
    pending_groups: list[int] = []
    pending_rows = 0

    for row_group in range(metadata.num_row_groups):
        num_rows = metadata.row_group(row_group).num_rows
        pending_groups.append(row_group)
        pending_rows += num_rows
        if len(pending_groups) >= min_row_groups_per_fragment:
            fragments.append(
                ParquetFragment(
                    path=shard,
                    row_groups=tuple(pending_groups),
                    row_offset=row_offset,
                    num_rows=pending_rows,
                )
            )
            row_offset += pending_rows
            pending_groups = []
            pending_rows = 0

    if pending_groups:
        fragments.append(
            ParquetFragment(
                path=shard,
                row_groups=tuple(pending_groups),
                row_offset=row_offset,
                num_rows=pending_rows,
            )
        )

    return fragments


def iter_parquet(
    fragment: ParquetFragment,
    *,
    columns: Sequence[str] | None = None,
    batch_size: int | None = None,
    use_threads: bool = True,
) -> Iterator[Sample]:
    """Iterate one parquet row-group fragment and yield one sample dict per row."""

    resolved_batch_size = resolve_parquet_batch_size(batch_size)
    parquet_file = pq.ParquetFile(fragment.path)
    index_in_file = fragment.row_offset

    for record_batch in parquet_file.iter_batches(
        batch_size=resolved_batch_size,
        row_groups=list(fragment.row_groups),
        columns=columns,
        use_threads=use_threads,
    ):
        column_names = record_batch.schema.names
        columns_data = [record_batch.column(i) for i in range(record_batch.num_columns)]
        for batch_row_index in range(record_batch.num_rows):
            sample: Sample = {
                name: columns_data[column_index][batch_row_index].as_py()
                for column_index, name in enumerate(column_names)
            }
            sample["__file__"] = fragment.path
            sample["__index_in_file__"] = index_in_file
            sample["__key__"] = f"{fragment.path}:{index_in_file}"
            yield sample
            index_in_file += 1


def iter_parquets(
    fragments: Iterator[ParquetFragment],
    *,
    columns: Sequence[str] | None = None,
    batch_size: int | None = None,
    use_threads: bool = True,
) -> Iterator[Sample]:
    """Iterate parquet row-group fragments in order and yield row samples."""

    for fragment in fragments:
        yield from iter_parquet(
            fragment,
            columns=columns,
            batch_size=batch_size,
            use_threads=use_threads,
        )
