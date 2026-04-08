"""Parquet source utilities."""

from __future__ import annotations

import importlib
from collections.abc import Iterator

from ..core.types import PathLikeStr, Sample

_DEFAULT_PARQUET_BATCH_SIZE = 1024


def _load_pyarrow_parquet():
    try:
        return importlib.import_module("pyarrow.parquet")
    except ModuleNotFoundError as exc:
        msg = "Parquet support requires `pyarrow`. Install it to use `Dataset.from_parquet(...)`."
        raise ModuleNotFoundError(msg) from exc


def iter_parquets(
    shard_paths: Iterator[PathLikeStr],
    *,
    batch_size: int = _DEFAULT_PARQUET_BATCH_SIZE,
) -> Iterator[Sample]:
    """Stream row samples from parquet shards.

    Each yielded sample is annotated with ``__file__``, ``__index_in_file__``,
    and ``__key__`` metadata when it is not already fully pre-annotated.
    """

    for shard_path in shard_paths:
        yield from _iter_one_parquet(str(shard_path), batch_size=batch_size)


def _iter_one_parquet(
    file: str,
    *,
    batch_size: int,
) -> Iterator[Sample]:
    parquet = _load_pyarrow_parquet()
    parquet_file = parquet.ParquetFile(file)

    row_index = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            yield _parse_parquet_row(
                file,
                row_index,
                row,
                allow_preannotated=True,
            )
            row_index += 1


def _parse_parquet_row(
    file: str,
    index_in_file: int,
    row: object,
    *,
    allow_preannotated: bool = False,
) -> Sample:
    if not isinstance(row, dict):
        msg = f"[InvalidParquetSample] file={file!r} row={index_in_file + 1} expected object row"
        raise ValueError(msg)

    sample: Sample = dict(row)
    if allow_preannotated and _has_parquet_metadata(sample):
        return sample

    sample["__index_in_file__"] = index_in_file
    sample["__file__"] = file
    sample["__key__"] = f"{file}:{index_in_file}"
    return sample


def _has_parquet_metadata(sample: Sample) -> bool:
    return (
        isinstance(sample.get("__index_in_file__"), int)
        and isinstance(sample.get("__file__"), str)
        and isinstance(sample.get("__key__"), str)
    )
