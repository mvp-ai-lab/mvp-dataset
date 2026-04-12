"""Lance-backed cache store for warm/cold detection and read/write."""

from __future__ import annotations

import json
import os
import pickle
import shutil
from collections.abc import Iterator, Mapping, Sequence
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


def _normalize_value_for_arrow(value: object) -> object:
    """Convert Arrow / NumPy containers into plain Python containers."""

    if isinstance(value, pa.ChunkedArray | pa.Array):
        return [_normalize_value_for_arrow(item) for item in value.to_pylist()]
    if isinstance(value, pa.Scalar):
        return _normalize_value_for_arrow(value.as_py())
    if isinstance(value, pa.RecordBatch | pa.Table):
        return [_normalize_value_for_arrow(item) for item in value.to_pylist()]
    if isinstance(value, Mapping):
        return {key: _normalize_value_for_arrow(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_normalize_value_for_arrow(item) for item in value]

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return [_normalize_value_for_arrow(item) for item in value.tolist()]
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass

    return value


def _infer_arrow_type(values: list[object]) -> pa.DataType:
    """Infer a pyarrow type from sample values for one field."""
    non_null_values = [value for value in values if value is not None]
    if not non_null_values:
        return pa.null()

    first_value = non_null_values[0]
    if isinstance(first_value, (bytes, bytearray)):
        return pa.large_binary()
    if isinstance(first_value, str):
        return pa.large_utf8()
    if isinstance(first_value, bool):
        return pa.bool_()
    if isinstance(first_value, int):
        return pa.int64()
    if isinstance(first_value, float):
        return pa.float64()
    try:
        inferred_type = pa.infer_type(non_null_values, from_pandas=False)
    except (TypeError, ValueError, pa.ArrowInvalid, pa.ArrowNotImplementedError):
        inferred_type = None
    if inferred_type is not None and inferred_type != pa.null():
        return _promote_to_large_offsets(inferred_type)
    # fallback: pickle to binary
    return pa.large_binary()


def _promote_to_large_offsets(data_type: pa.DataType) -> pa.DataType:
    """Recursively widen variable-width Arrow offsets to their large variants."""

    if pa.types.is_string(data_type):
        return pa.large_utf8()
    if pa.types.is_binary(data_type):
        return pa.large_binary()
    if pa.types.is_list(data_type):
        return pa.large_list(_promote_to_large_offsets(data_type.value_type))
    if pa.types.is_large_list(data_type):
        return pa.large_list(_promote_to_large_offsets(data_type.value_type))
    if pa.types.is_fixed_size_list(data_type):
        return pa.list_(_promote_to_large_offsets(data_type.value_type), data_type.list_size)
    if pa.types.is_struct(data_type):
        return pa.struct(
            [
                pa.field(
                    field.name,
                    _promote_to_large_offsets(field.type),
                    nullable=field.nullable,
                    metadata=field.metadata,
                )
                for field in data_type
            ]
        )
    if pa.types.is_map(data_type):
        return pa.map_(
            _promote_to_large_offsets(data_type.key_type),
            _promote_to_large_offsets(data_type.item_type),
            keys_sorted=data_type.keys_sorted,
        )
    if pa.types.is_dictionary(data_type):
        return pa.dictionary(
            data_type.index_type,
            _promote_to_large_offsets(data_type.value_type),
            ordered=data_type.ordered,
        )
    return data_type


def _escape_field_name(name: str) -> str:
    """Escape dots in field names so Lance doesn't interpret them as struct paths."""
    return name.replace(".", "__dot__")


def _unescape_field_name(name: str) -> str:
    """Reverse :func:`_escape_field_name`."""
    return name.replace("__dot__", ".")


def _escape_sample(sample: Sample) -> Sample:
    """Escape all field names in a sample dict."""
    return {_escape_field_name(k): _normalize_value_for_arrow(v) for k, v in sample.items()}


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
    field_names: list[str] = []
    seen: set[str] = set()
    for sample in samples:
        for key in sample:
            if key in seen:
                continue
            field_names.append(key)
            seen.add(key)
    fields = [(key, _infer_arrow_type([sample.get(key) for sample in samples])) for key in field_names]
    return pa.schema(fields)


def _summarize_value(value: object) -> str:
    """Return a compact type/value summary for cache write diagnostics."""

    if isinstance(value, pa.ChunkedArray):
        return f"ChunkedArray(type={value.type}, chunks={value.num_chunks}, len={len(value)})"
    if isinstance(value, pa.Array):
        return f"Array(type={value.type}, len={len(value)})"
    if isinstance(value, pa.Scalar):
        return f"Scalar(type={value.type}, value={value.as_py()!r})"
    if isinstance(value, pa.RecordBatch):
        return f"RecordBatch(rows={value.num_rows}, cols={value.num_columns}, schema={value.schema})"
    if isinstance(value, pa.Table):
        return f"Table(rows={value.num_rows}, cols={value.num_columns}, schema={value.schema})"

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return f"ndarray(shape={value.shape}, dtype={value.dtype})"
        if isinstance(value, np.generic):
            return f"{type(value).__name__}(dtype={value.dtype}, value={value.item()!r})"
    except ImportError:
        pass

    if isinstance(value, dict):
        keys = list(value)[:5]
        return f"dict(keys={keys!r}, len={len(value)})"
    if isinstance(value, list):
        return f"list(len={len(value)})"
    if isinstance(value, tuple):
        return f"tuple(len={len(value)})"
    return f"{type(value).__name__}({value!r})"


def _collect_non_python_paths(value: object, path: str = "") -> list[tuple[str, str]]:
    """Find residual Arrow/NumPy containers that should not reach from_pylist."""

    results: list[tuple[str, str]] = []
    if isinstance(value, pa.ChunkedArray | pa.Array | pa.Scalar | pa.RecordBatch | pa.Table):
        results.append((path or "<root>", _summarize_value(value)))
        return results

    try:
        import numpy as np

        if isinstance(value, np.ndarray | np.generic):
            results.append((path or "<root>", _summarize_value(value)))
            return results
    except ImportError:
        pass

    if isinstance(value, Mapping):
        for key, item in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            results.extend(_collect_non_python_paths(item, child_path))
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, item in enumerate(value):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            results.extend(_collect_non_python_paths(item, child_path))
            if len(results) >= 32:
                break
    return results


def _values_to_array(values: list[object], target_type: pa.DataType, *, field_name: str) -> pa.Array:
    """Build one Arrow array and surface field-local diagnostics on failure."""

    try:
        array = pa.array(values, type=target_type)
    except (TypeError, ValueError, pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
        diagnostics: list[str] = []
        for sample_index, value in enumerate(values[:4]):
            paths = _collect_non_python_paths(value)
            for value_path, summary in paths[:8]:
                prefix = f"sample[{sample_index}]"
                diagnostics.append(f"{prefix} {value_path}: {summary}")
            if not paths:
                diagnostics.append(
                    f"sample[{sample_index}] type={type(value).__name__} summary={_summarize_value(value)}"
                )
        msg = (
            f"[CacheArrowFieldError] failed to convert field={field_name!r} to type={target_type}. "
            f"diagnostics={' | '.join(diagnostics)}"
        )
        raise TypeError(msg) from exc

    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    if isinstance(array, pa.ChunkedArray):
        msg = f"[CacheArrowFieldError] field={field_name!r} unexpectedly produced ChunkedArray type={target_type}"
        raise TypeError(msg)
    return array


def _samples_to_record_batch(samples: list[Sample], schema: pa.Schema) -> pa.RecordBatch:
    """Convert a list of sample dicts to a RecordBatch under a fixed schema."""
    schema_names = {field.name for field in schema}
    columns_by_name: dict[str, list[Any]] = {field.name: [] for field in schema}

    for sample in samples:
        extra_fields = [key for key in sample if key not in schema_names]
        if extra_fields:
            msg = f"[CacheSchemaError] encountered unexpected field(s) {extra_fields!r} after schema inference"
            raise ValueError(msg)
        for field in schema:
            columns_by_name[field.name].append(_convert_value(sample.get(field.name), field.type))

    try:
        columns = [_values_to_array(columns_by_name[field.name], field.type, field_name=field.name) for field in schema]
        return pa.RecordBatch.from_arrays(columns, schema=schema)
    except (TypeError, pa.ArrowInvalid) as exc:
        diagnostics: list[str] = []
        for field in schema:
            values = columns_by_name[field.name]
            for sample_index, value in enumerate(values[:4]):
                paths = _collect_non_python_paths(value, field.name)
                for value_path, summary in paths[:8]:
                    diagnostics.append(f"sample[{sample_index}] {value_path}: {summary}")
                if len(diagnostics) >= 16:
                    break
            if len(diagnostics) >= 16:
                break
        if not diagnostics:
            for field in schema:
                values = columns_by_name[field.name][:4]
                diagnostics.append(
                    f"field {field.name}: schema={field.type}, value_types={[type(v).__name__ for v in values]!r}"
                )
        msg = (
            "[CacheArrowConversionError] failed to build RecordBatch from normalized samples. "
            f"schema={schema}. diagnostics={' | '.join(diagnostics)}"
        )
        raise TypeError(msg) from exc


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
