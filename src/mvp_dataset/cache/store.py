"""Lance-backed cache store for warm/cold detection and read/write."""

from __future__ import annotations

import json
import os
import pickle
import shutil
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import Any

import lance as _lance
import pyarrow as pa
from lance import DatasetBasePath
from lance.fragment import FragmentMetadata

from ..core.types import Sample

# ---------------------------------------------------------------------------
# Type conversion helpers
# ---------------------------------------------------------------------------


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
# Cold write
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


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


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
