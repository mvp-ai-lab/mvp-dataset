"""Lance-backed cache store for warm/cold detection and read/write."""

from __future__ import annotations

import json
import os
import pickle
import shutil
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from itertools import islice
from pathlib import Path
from typing import Any

import lance as _lance
import pyarrow as pa
from lance import DatasetBasePath
from lance.fragment import FragmentMetadata

from ..core.types import Sample, StageSpec

# ---------------------------------------------------------------------------
# Type conversion helpers
# ---------------------------------------------------------------------------


def _build_pre_cache_stream(
    iter_source_stream: Callable[[Iterable[object]], Iterable[object]],
    source_items: list[object],
    pre_specs: tuple[StageSpec, ...],
) -> Iterator[object]:
    """Build the pre-cache iterator over a fixed list of source items."""
    stream: Iterable[object] = iter_source_stream(iter(source_items))
    for spec in pre_specs:
        stream = spec.apply(stream)
    return iter(stream)


def _split_cache_work(
    source_items: list[object],
    num_workers: int,
) -> list[list[object]]:
    """Split source items into contiguous chunks to preserve source order."""
    actual_workers = min(num_workers, len(source_items))
    if actual_workers <= 1:
        return [source_items]

    base, extra = divmod(len(source_items), actual_workers)
    chunks: list[list[object]] = []
    start = 0
    for worker_index in range(actual_workers):
        size = base + (1 if worker_index < extra else 0)
        end = start + size
        if start != end:
            chunks.append(source_items[start:end])
        start = end
    return chunks


def _infer_arrow_type(value: object) -> pa.DataType:
    """Infer a pyarrow type from a single Python value."""
    if isinstance(value, (bytes, bytearray)):
        return pa.large_binary()
    if isinstance(value, str):
        return pa.large_utf8()
    if isinstance(value, bool):
        return pa.bool_()
    if isinstance(value, int):
        return pa.int64()
    if isinstance(value, float):
        return pa.float64()
    # numpy arrays
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return pa.from_numpy_dtype(value.dtype)
    except ImportError:
        pass
    # fallback: pickle to binary
    return pa.large_binary()


def _escape_field_name(name: str) -> str:
    """Escape dots in field names so Lance doesn't interpret them as struct paths."""
    return name.replace(".", "__dot__")


def _unescape_field_name(name: str) -> str:
    """Reverse :func:`_escape_field_name`."""
    return name.replace("__dot__", ".")


def _escape_sample(sample: Sample) -> Sample:
    """Escape all field names in a sample dict."""
    return {_escape_field_name(k): v for k, v in sample.items()}


def _unescape_sample(sample: Sample) -> Sample:
    """Restore original field names in a sample dict."""
    return {_unescape_field_name(k): v for k, v in sample.items()}


def _convert_value(value: object, target_type: pa.DataType) -> object:
    """Convert a Python value for Arrow storage."""
    if target_type == pa.large_binary() and not isinstance(value, (bytes, bytearray)):
        return pickle.dumps(value)
    return value


def _schema_from_samples(samples: list[Sample]) -> pa.Schema:
    """Infer a stable Arrow schema from the first non-empty sample batch."""
    fields: list[tuple[str, pa.DataType]] = []
    seen: set[str] = set()
    for sample in samples:
        for key, value in sample.items():
            if key in seen:
                continue
            fields.append((key, _infer_arrow_type(value)))
            seen.add(key)
    return pa.schema(fields)


def _samples_to_record_batch(samples: list[Sample], schema: pa.Schema) -> pa.RecordBatch:
    """Convert a list of sample dicts to a RecordBatch under a fixed schema."""
    arrays: dict[str, list[Any]] = {field.name: [] for field in schema}
    schema_names = set(arrays)

    for sample in samples:
        extra_fields = [key for key in sample if key not in schema_names]
        if extra_fields:
            msg = f"[CacheSchemaError] encountered unexpected field(s) {extra_fields!r} after schema inference"
            raise ValueError(msg)
        for field in schema:
            arrays[field.name].append(_convert_value(sample.get(field.name), field.type))

    columns = [pa.array(arrays[field.name], type=field.type) for field in schema]
    return pa.RecordBatch.from_arrays(columns, schema=schema)


# ---------------------------------------------------------------------------
# Warm/cold detection
# ---------------------------------------------------------------------------


def _full_cache_uri(cache_dir: str, fingerprint: str) -> str:
    return os.path.join(cache_dir, f"{fingerprint}.lance")


def _node_partial_cache_uri(cache_dir: str, fingerprint: str, node_rank: int) -> str:
    return os.path.join(cache_dir, f".{fingerprint}.node-{node_rank:05d}.tmp.lance")


def is_lance_dataset_warm(uri: str) -> bool:
    """Check if a Lance dataset exists and contains at least one row."""
    if not os.path.isdir(uri):
        return False
    try:
        ds = _lance.dataset(uri)
        return ds.count_rows() > 0
    except Exception:
        return False


def is_full_cache_warm(cache_dir: str, fingerprint: str) -> bool:
    """Check if a full-mode cache dataset exists and is valid."""
    return is_lance_dataset_warm(_full_cache_uri(cache_dir, fingerprint))


def wait_for_lance_datasets(
    uris: Sequence[str],
    *,
    poll_interval: float = 0.5,
    timeout: float = 3600.0,
) -> None:
    """Block until all specified Lance datasets exist and are readable."""
    pending = {uri for uri in uris}
    deadline = time.monotonic() + timeout
    while pending:
        if time.monotonic() > deadline:
            msg = f"[CacheWaitTimeout] timed out after {timeout}s waiting for {len(pending)} cache dataset(s)"
            raise TimeoutError(msg)
        time.sleep(poll_interval)
        pending = {uri for uri in pending if not is_lance_dataset_warm(uri)}


def is_incremental_cache_warm(
    source_uri: str,
    fingerprint: str,
) -> bool:
    """Check if an incremental cache has already been written."""
    try:
        ds = _lance.dataset(source_uri)
    except Exception:
        return False

    meta = ds.schema.metadata
    if meta is None:
        return False

    stored_fp = meta.get(b"mvp_cache_fp")
    if stored_fp is None or stored_fp.decode() != fingerprint:
        return False

    return True


# ---------------------------------------------------------------------------
# Cold write — full mode
# ---------------------------------------------------------------------------

_DEFAULT_BATCH_SIZE = 8192


def _write_lance_dataset(
    stream: Iterator[Sample],
    uri: str,
    *,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    max_rows_per_group: int | None = None,
) -> str:
    """Write a Python sample stream to a Lance dataset via a streaming reader."""
    parent_dir = os.path.dirname(uri)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    first_chunk = [_escape_sample(s) for s in islice(stream, batch_size)]
    if not first_chunk:
        return uri

    schema = _schema_from_samples(first_chunk)

    def _iter_batches() -> Iterator[pa.RecordBatch]:
        chunk = first_chunk
        while chunk:
            yield _samples_to_record_batch(chunk, schema)
            chunk = [_escape_sample(s) for s in islice(stream, batch_size)]

    if os.path.isdir(uri):
        shutil.rmtree(uri, ignore_errors=True)

    reader = pa.RecordBatchReader.from_batches(schema, _iter_batches())
    _lance.write_dataset(
        reader,
        uri,
        schema=schema,
        max_rows_per_group=max_rows_per_group or batch_size,
    )
    return uri


def write_full_cache(
    stream: Iterator[Sample],
    cache_dir: str,
    fingerprint: str,
    *,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> str:
    """Consume *stream*, write to a standalone Lance cache dataset.

    Returns the URI of the written dataset.
    """
    uri = _full_cache_uri(cache_dir, fingerprint)
    return _write_lance_dataset(stream, uri, batch_size=batch_size)


def merge_full_cache_parts(
    part_uris: Sequence[str],
    cache_dir: str,
    fingerprint: str,
    *,
    batch_size: int = 65536,
) -> str:
    """Merge ordered partial full-cache datasets into the final Lance cache."""
    existing_parts = [uri for uri in part_uris if os.path.isdir(uri)]
    final_uri = _full_cache_uri(cache_dir, fingerprint)
    if not existing_parts:
        return final_uri

    datasets = [_lance.dataset(uri) for uri in existing_parts]
    schema = datasets[0].schema
    for dataset in datasets[1:]:
        if dataset.schema != schema:
            msg = "[CacheSchemaError] partial cache datasets produced different schemas"
            raise ValueError(msg)

    def _iter_batches() -> Iterator[pa.RecordBatch]:
        for dataset in datasets:
            yield from dataset.scanner(batch_size=batch_size).to_batches()

    if os.path.isdir(final_uri):
        shutil.rmtree(final_uri, ignore_errors=True)

    reader = pa.RecordBatchReader.from_batches(schema, _iter_batches())
    _lance.write_dataset(
        reader,
        final_uri,
        schema=schema,
        max_rows_per_group=batch_size,
    )
    return final_uri


# ---------------------------------------------------------------------------
# Cold write — incremental mode
# ---------------------------------------------------------------------------


def write_incremental_cache(
    source_uri: str,
    pre_cache_stream: Iterator[Sample],
    fingerprint: str,
) -> None:
    """Append cached outputs to an existing Lance dataset."""
    ds = _lance.dataset(source_uri)
    col_data: dict[str, list[Any]] = {}

    for sample in pre_cache_stream:
        keys = [k for k in sample if not (k.startswith("__") and k.endswith("__"))]
        for col in keys:
            col_data.setdefault(col, []).append(sample.get(col))

    if not col_data:
        return

    new_table = pa.table(col_data)
    ds.add_columns(new_table)

    # Stamp fingerprint
    ds = _lance.dataset(source_uri)
    ds.update_schema_metadata({"mvp_cache_fp": fingerprint}, replace=False)


# ---------------------------------------------------------------------------
# Warm read — full mode
# ---------------------------------------------------------------------------


def iter_full_cache(
    cache_dir: str,
    fingerprint: str,
    *,
    columns: Sequence[str] | None = None,
    batch_size: int = 65536,
) -> Iterator[Sample]:
    """Iterate a full-mode cache dataset and yield sample dicts."""
    uri = _full_cache_uri(cache_dir, fingerprint)
    ds = _lance.dataset(uri)
    scanner = ds.scanner(columns=columns, batch_size=batch_size)

    for batch in scanner.to_batches():
        col_names = batch.schema.names
        col_arrays = [batch.column(i) for i in range(batch.num_columns)]
        for row_idx in range(batch.num_rows):
            yield {
                _unescape_field_name(name): col_arrays[col_idx][row_idx].as_py()
                for col_idx, name in enumerate(col_names)
            }


def merge_lance(
    uris,
    out_uri,
    *,
    check_schema=True,
    do_compact=False,
):
    """
    Merge multiple Lance datasets into a single dataset using fragment-level concatenation.

    This implementation is highly efficient because it:
    - Avoids reading or materializing data into memory
    - Operates purely on fragment metadata
    - Achieves near O(1) merge time relative to data size

    Args:
        uris (List[str]):
            List of input Lance dataset URIs (directories ending with `.lance`).

        out_uri (str):
            Output URI where the merged dataset will be written.
            Must not already exist.

        check_schema (bool, optional):
            If True, validates that all input datasets have identical schemas.
            Raises ValueError if mismatch is detected.

        do_compact (bool, optional):
            If True, performs dataset compaction after merge to reduce fragment count
            and improve query performance.

    Returns:
        lance.dataset:
            The merged Lance dataset.

    Raises:
        ValueError:
            If `uris` is empty or schemas do not match.

        FileExistsError:
            If `out_uri` already exists.
    """

    if not uris:
        raise ValueError("`uris` must not be empty")

    datasets = [_lance.dataset(u) for u in uris]

    # ---------- 1. Schema validation ----------
    if check_schema:
        base_schema = datasets[0].schema
        for i, ds in enumerate(datasets[1:], 1):
            if ds.schema != base_schema:
                raise ValueError(f"Schema mismatch between uris[0] and uris[{i}]\n{base_schema}\n!=\n{ds.schema}")

    # ---------- 2. Collect fragments with base_id pointing to source data dirs ----------
    all_metadata: list[FragmentMetadata] = []
    bases: list[DatasetBasePath] = []

    for idx, (uri, ds) in enumerate(zip(uris, datasets, strict=True)):
        base_id = idx + 1  # 0 is reserved for the dataset's own data dir
        bases.append(
            DatasetBasePath(
                os.path.join(uri, "data") + os.sep,
                name=f"part-{idx}",
                id=base_id,
            )
        )
        for frag in ds.get_fragments():
            j = frag.metadata.to_json()
            for f in j["files"]:
                f["base_id"] = base_id
            all_metadata.append(FragmentMetadata.from_json(json.dumps(j)))

    if not all_metadata:
        raise ValueError("No fragments found in input datasets")

    # ---------- 3. Write merged dataset ----------
    out_path = Path(out_uri)
    if out_path.exists():
        raise FileExistsError(f"{out_uri} already exists")

    # Stitch fragments via metadata-only commit — no data is read or copied.
    schema = datasets[0].schema
    operation = _lance.LanceOperation.Overwrite(schema, all_metadata, initial_bases=bases)
    merged = _lance.LanceDataset.commit(out_uri, operation)

    # ---------- 4. Compact and clean up per-rank datasets ----------
    # Compact copies data files into the merged dataset's own data dir,
    # after which the per-rank source directories are no longer needed.
    if do_compact:
        merged.optimize.compact_files()
        merged.cleanup_old_versions()
        for uri in uris:
            shutil.rmtree(uri, ignore_errors=True)

    return merged
