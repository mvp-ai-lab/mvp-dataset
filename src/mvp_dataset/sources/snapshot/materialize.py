"""Lance snapshot cache materialization."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import lance
import pyarrow as pa

from mvp_dataset.cache import (
    CacheBuildResult,
    CacheConfig,
    CacheEntry,
    CacheManager,
    SourceFingerprint,
)
from mvp_dataset.core.context import RuntimeContext
from mvp_dataset.core.dataset import Dataset
from mvp_dataset.core.iterator import DatasetIterator

from .codecs import encode_snapshot_value

SNAPSHOT_FORMAT_VERSION = 1
SNAPSHOT_CACHE_KIND = "dataset-snapshot"
SNAPSHOT_LANCE_DIRECTORY = "data.lance"
SNAPSHOT_EMPTY_COLUMN = "__mvp_snapshot_empty__"
SNAPSHOT_WRITE_BATCH_SIZE = 1024
SNAPSHOT_CACHE_PARAMETERS = {
    "format": "lance",
    "row_order": "slot-major",
    "source_metadata": "aliased",
    "value_codec": "torch-tensor-v2",
    "write_batch_size": SNAPSHOT_WRITE_BATCH_SIZE,
}
_SOURCE_METADATA_ALIASES = {
    "__file__": "__source_file__",
    "__key__": "__source_key__",
}
_REPLACED_SOURCE_METADATA = ("__local_index__", "__global_index__")


def ensure_snapshot(
    upstream: Dataset,
    *,
    source_fingerprint: SourceFingerprint,
    cache_config: CacheConfig,
    build_context: RuntimeContext,
) -> CacheEntry:
    """Return a valid snapshot cache entry, building it when missing."""
    manager = CacheManager(cache_config)
    assigned_part = _snapshot_part(build_context.slot)
    return manager.ensure(
        source=source_fingerprint,
        kind=SNAPSHOT_CACHE_KIND,
        format_version=SNAPSHOT_FORMAT_VERSION,
        parameters=SNAPSHOT_CACHE_PARAMETERS,
        parts=lambda: _snapshot_parts(build_context.total_slots),
        assigned_parts=(assigned_part,),
        build=lambda part, root: _build_snapshot_part(upstream, root, part=part, context=build_context),
    )


def snapshot_lance_paths(entry: CacheEntry) -> tuple[Path, ...]:
    """Return completed Lance part paths in stable slot order."""
    metadata = entry.read_manifest().get("metadata")
    if not isinstance(metadata, dict):
        msg = "[InvalidSnapshotManifest] cache metadata must be an object"
        raise RuntimeError(msg)
    raw_parts = metadata.get("parts")
    if not isinstance(raw_parts, list):
        msg = "[InvalidSnapshotManifest] cache metadata.parts must be a list"
        raise RuntimeError(msg)

    part_names: list[str] = []
    for part in raw_parts:
        if not isinstance(part, dict) or not isinstance(part.get("name"), str):
            msg = "[InvalidSnapshotManifest] cache part metadata is invalid"
            raise RuntimeError(msg)
        part_names.append(part["name"])
    return tuple(entry.path / part / SNAPSHOT_LANCE_DIRECTORY for part in sorted(part_names))


def validate_finite_pipeline(dataset: Dataset) -> None:
    """Reject known infinite pipelines before snapshot construction."""
    if dataset._resume_state is not None:
        msg = "[SnapshotResumeUnsupported] snapshot() cannot materialize a pipeline with pending resume state"
        raise ValueError(msg)
    if dataset._resample:
        msg = "[InfiniteSnapshotSource] snapshot() requires a finite pipeline"
        raise ValueError(msg)
    if dataset._source_kind != "mixed":
        return
    for source in dataset._source:
        validate_finite_pipeline(source.dataset)


def _snapshot_parts(total_slots: int) -> tuple[str, ...]:
    return tuple(_snapshot_part(slot) for slot in range(total_slots))


def _snapshot_part(slot: int) -> str:
    return f"slot-{slot:08d}"


def _build_snapshot_part(
    upstream: Dataset,
    root: Path,
    *,
    part: str,
    context: RuntimeContext,
) -> CacheBuildResult:
    expected_part = _snapshot_part(context.slot)
    if part != expected_part:
        msg = f"[InvalidSnapshotPart] expected={expected_part!r} got={part!r}"
        raise RuntimeError(msg)
    output_path = root / SNAPSHOT_LANCE_DIRECTORY
    stream = DatasetIterator(upstream, context=context)
    first_rows = _read_rows(stream, SNAPSHOT_WRITE_BATCH_SIZE)

    if not first_rows:
        table = pa.table({SNAPSHOT_EMPTY_COLUMN: pa.array([], type=pa.bool_())})
        dataset = lance.write_dataset(table, str(output_path), mode="create")
    else:
        field_names = _field_names(first_rows)
        first_batch = _record_batch(first_rows, field_names=field_names)
        reader = pa.RecordBatchReader.from_batches(
            first_batch.schema,
            _record_batches(stream, first_batch, field_names=field_names),
        )
        dataset = lance.write_dataset(reader, str(output_path), mode="create")

    files = tuple(sorted(path.relative_to(root).as_posix() for path in output_path.rglob("*") if path.is_file()))
    return CacheBuildResult.from_files(
        files,
        metadata={
            "num_rows": dataset.count_rows(),
            "schema": dataset.schema.to_string(),
            "slot": context.slot,
            "total_slots": context.total_slots,
        },
    )


def _record_batches(
    stream: Iterator[object],
    first_batch: pa.RecordBatch,
    *,
    field_names: tuple[str, ...],
) -> Iterator[pa.RecordBatch]:
    yield first_batch
    while rows := _read_rows(stream, SNAPSHOT_WRITE_BATCH_SIZE):
        yield _record_batch(rows, field_names=field_names, schema=first_batch.schema)


def _read_rows(stream: Iterator[object], count: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    while len(rows) < count:
        try:
            item = next(stream)
        except StopIteration:
            break
        if not isinstance(item, dict):
            msg = f"[InvalidSnapshotSample] expected dict sample, got {type(item).__name__}"
            raise TypeError(msg)
        if not all(isinstance(name, str) for name in item):
            msg = "[InvalidSnapshotField] field names must be strings"
            raise TypeError(msg)
        row = dict(item)
        for source_name, alias in _SOURCE_METADATA_ALIASES.items():
            if source_name not in row:
                continue
            if alias in row:
                msg = f"[SnapshotSourceMetadataConflict] sample contains both {source_name!r} and {alias!r}"
                raise ValueError(msg)
            row[alias] = row.pop(source_name)
        for name in _REPLACED_SOURCE_METADATA:
            row.pop(name, None)
        if not row:
            msg = "[InvalidSnapshotSample] snapshot samples must contain at least one field"
            raise ValueError(msg)
        encoded = encode_snapshot_value(row)
        if not isinstance(encoded, dict):
            msg = "[SnapshotSerializationError] encoded snapshot sample must be a dict"
            raise RuntimeError(msg)
        rows.append(encoded)
    return rows


def _field_names(rows: list[dict[str, object]]) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for name in row:
            if name not in seen:
                names.append(name)
                seen.add(name)
    return tuple(names)


def _record_batch(
    rows: list[dict[str, object]],
    *,
    field_names: tuple[str, ...],
    schema: pa.Schema | None = None,
) -> pa.RecordBatch:
    expected = set(field_names)
    for row in rows:
        unexpected = set(row) - expected
        if unexpected:
            msg = f"[SnapshotSchemaChanged] new fields appeared after schema inference: {sorted(unexpected)!r}"
            raise ValueError(msg)
    columns = {name: [row.get(name) for row in rows] for name in field_names}
    try:
        table = pa.Table.from_pydict(columns, schema=schema)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, TypeError) as error:
        msg = f"[SnapshotSerializationError] unable to convert snapshot rows to Arrow: {error}"
        raise TypeError(msg) from error
    batches = table.to_batches()
    if len(batches) != 1:
        msg = "[SnapshotSerializationError] expected one Arrow record batch"
        raise RuntimeError(msg)
    return batches[0]
