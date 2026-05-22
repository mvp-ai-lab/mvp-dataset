"""Lance reference-column indexing and resolution."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import lance
import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads

from mvp_dataset.core.context import RuntimeContext
from mvp_dataset.core.types import Sample

from .types import LanceRefSpec, LanceSourceSpec

REF_INDEX_BUILDER_VERSION = 1
REF_INDEX_MISSING_ROW = -1
REF_INDEX_MANIFEST = "metadata.json"
REF_INDEX_DIR = "_mvp_ref_index"
REF_INDEX_BUILD_BATCH_SIZE = 65536
REF_INDEX_LOCK_POLL_SECONDS = 0.25
REF_INDEX_WAIT_TIMEOUT_SECONDS = 30 * 60

LanceRefIndexScope = Literal["shared", "node_local", "process"]


def parse_lance_ref_columns(ref_columns: object) -> tuple[LanceRefSpec, ...]:
    """Parse the public ref_columns mapping into immutable ref specs."""

    if ref_columns is None:
        return ()
    if not isinstance(ref_columns, dict):
        msg = "[InvalidLanceRefColumns] ref_columns must be a mapping of column name to reference config"
        raise TypeError(msg)

    specs: list[LanceRefSpec] = []
    for column, config in ref_columns.items():
        if not isinstance(column, str) or not column:
            msg = "[InvalidLanceRefColumn] ref column names must be non-empty strings"
            raise ValueError(msg)
        if not isinstance(config, dict):
            msg = f"[InvalidLanceRefColumn] config for {column!r} must be a mapping"
            raise TypeError(msg)
        missing = [key for key in ("uri", "key_column", "value_column") if key not in config]
        if missing:
            msg = f"[InvalidLanceRefColumn] config for {column!r} is missing: {', '.join(missing)}"
            raise ValueError(msg)
        specs.append(
            LanceRefSpec(
                column=column,
                uri=str(config["uri"]),
                key_column=str(config["key_column"]),
                value_column=str(config["value_column"]),
            )
        )
    return tuple(specs)


def attach_lance_ref_columns(source: LanceSourceSpec, ref_columns: object) -> LanceSourceSpec:
    return LanceSourceSpec(datasets=source.datasets, ref_columns=parse_lance_ref_columns(ref_columns))


def _read_table_rows(
    dataset_handle: object,
    row_indices: Sequence[int],
    *,
    columns: Sequence[str] | None = None,
) -> list[Sample]:
    take_indices = pa.array(row_indices, type=pa.int64())
    if isinstance(dataset_handle, pa.Table):
        table = dataset_handle
        if columns is not None:
            table = table.select(columns)
        table = table.take(take_indices)
    elif isinstance(dataset_handle, pads.Dataset):
        table = dataset_handle.take(take_indices, columns=columns)
    else:
        table = dataset_handle.take(row_indices, columns=columns)
    return table.to_pylist()


def _iter_table_record_batches(
    dataset_handle: object,
    *,
    columns: Sequence[str],
    batch_size: int = REF_INDEX_BUILD_BATCH_SIZE,
) -> Iterable[pa.RecordBatch]:
    if isinstance(dataset_handle, pa.Table):
        table = dataset_handle.select(columns)
        yield from table.to_batches(max_chunksize=batch_size)
        return
    if isinstance(dataset_handle, pads.Dataset):
        yield from dataset_handle.to_batches(columns=columns, batch_size=batch_size)
        return

    to_batches = getattr(dataset_handle, "to_batches", None)
    if callable(to_batches):
        yield from to_batches(columns=columns, batch_size=batch_size)
        return

    scanner = dataset_handle.scanner(columns=columns, batch_size=batch_size, scan_in_order=True)
    yield from scanner.to_batches()


def _iter_ref_keys(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, np.ndarray):
        return tuple(item for item in value.tolist() if item is not None)
    if isinstance(value, (list, tuple)):
        return tuple(item for item in value if item is not None)
    return (value,)


def _dataset_manifest_fingerprint(uri: str) -> dict[str, Any]:
    dataset = lance.dataset(uri)
    version = getattr(dataset, "version", None)
    if callable(version):
        version = version()
    if version is not None:
        version = str(version)

    fingerprint: dict[str, Any] = {
        "uri": uri,
        "num_rows": dataset.count_rows(),
        "version": version,
    }

    versions_dir = Path(uri) / "_versions"
    if versions_dir.exists():
        manifests = sorted(versions_dir.glob("*.manifest"))
        if manifests:
            latest = manifests[-1]
            stat = latest.stat()
            fingerprint["manifest"] = {
                "name": latest.name,
                "mtime_ns": stat.st_mtime_ns,
                "size": stat.st_size,
            }
    return fingerprint


def _ref_index_is_valid(
    index_dir: Path,
    manifest: dict[str, Any],
    ref_files: dict[str, dict[str, Any]],
    active_refs: Sequence[LanceRefSpec],
) -> bool:
    manifest_path = index_dir / REF_INDEX_MANIFEST
    if not index_dir.exists() or not manifest_path.exists():
        return False
    try:
        stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return stored_manifest == manifest and all(
        (index_dir / ref_files[ref.column]["offsets_file"]).exists()
        and (index_dir / ref_files[ref.column]["entries_file"]).exists()
        for ref in active_refs
    )


def _resolve_ref_index_scope(scope: LanceRefIndexScope | None) -> LanceRefIndexScope:
    raw_scope = scope or os.environ.get("MVP_LANCE_REF_INDEX_SCOPE", "node_local")
    if raw_scope not in ("shared", "node_local", "process"):
        msg = f"[InvalidLanceRefIndexScope] expected shared, node_local, or process, got {raw_scope!r}"
        raise ValueError(msg)
    return raw_scope


def _is_ref_index_builder(context: RuntimeContext | None, scope: LanceRefIndexScope) -> bool:
    if scope == "process" or context is None:
        return True
    if scope == "shared":
        return context.rank == 0 and context.worker_id == 0
    if scope == "node_local":
        return context.local_rank == 0 and context.worker_id == 0
    msg = f"[InvalidLanceRefIndexScope] expected shared, node_local, or process, got {scope!r}"
    raise ValueError(msg)


def _wait_for_ref_index(
    index_dir: Path,
    manifest: dict[str, Any],
    ref_files: dict[str, dict[str, Any]],
    active_refs: Sequence[LanceRefSpec],
    *,
    timeout_seconds: float = REF_INDEX_WAIT_TIMEOUT_SECONDS,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    poll_seconds = REF_INDEX_LOCK_POLL_SECONDS
    while not _ref_index_is_valid(index_dir, manifest, ref_files, active_refs):
        if time.monotonic() >= deadline:
            msg = f"[LanceRefIndexTimeout] timed out waiting for ref index at {index_dir}"
            raise TimeoutError(msg)
        time.sleep(poll_seconds)
        poll_seconds = min(poll_seconds * 2, 5.0)


def _publish_ref_index(tmp_index_dir: Path, index_dir: Path) -> None:
    try:
        tmp_index_dir.replace(index_dir)
    except FileExistsError:
        return
    except OSError:
        if not index_dir.exists():
            raise


def _build_ref_index(
    index_dir: Path,
    manifest: dict[str, Any],
    ref_files: dict[str, dict[str, Any]],
    active_refs: Sequence[LanceRefSpec],
    source: LanceSourceSpec,
) -> None:
    manifest_path = index_dir / REF_INDEX_MANIFEST
    index_dir.mkdir(parents=True, exist_ok=False)

    offsets_by_column: dict[str, np.memmap] = {}
    requested_positions: dict[str, dict[Any, list[int]]] = {ref.column: {} for ref in active_refs}
    entry_counts = {ref.column: 0 for ref in active_refs}
    for ref in active_refs:
        offsets_by_column[ref.column] = np.memmap(
            index_dir / ref_files[ref.column]["offsets_file"],
            dtype=np.uint64,
            mode="w+",
            shape=(source.total_rows + 1,),
        )
        offsets_by_column[ref.column][0] = 0

    global_row_index = 0
    main_columns = [ref.column for ref in active_refs]
    for dataset in source.datasets:
        for batch in _iter_table_record_batches(lance.dataset(dataset.uri), columns=main_columns):
            for row in batch.to_pylist():
                global_row_index += 1
                for ref in active_refs:
                    for key in _iter_ref_keys(row[ref.column]):
                        requested_positions[ref.column].setdefault(key, []).append(entry_counts[ref.column])
                        entry_counts[ref.column] += 1
                    offsets_by_column[ref.column][global_row_index] = entry_counts[ref.column]

    if global_row_index != source.total_rows:
        msg = f"[InvalidLanceRefIndex] expected {source.total_rows} main rows, scanned {global_row_index}"
        raise RuntimeError(msg)

    entries_by_column: dict[str, np.memmap | np.ndarray] = {}
    for ref in active_refs:
        offsets_by_column[ref.column].flush()
        entries_path = index_dir / ref_files[ref.column]["entries_file"]
        if entry_counts[ref.column] == 0:
            entries_path.touch()
            entries = np.empty(0, dtype=np.int64)
        else:
            entries = np.memmap(
                entries_path,
                dtype=np.int64,
                mode="w+",
                shape=(entry_counts[ref.column],),
            )
            entries[:] = REF_INDEX_MISSING_ROW
        entries_by_column[ref.column] = entries

    for ref in active_refs:
        row_index = 0
        resolved_keys: set[Any] = set()
        for batch in _iter_table_record_batches(lance.dataset(ref.uri), columns=[ref.key_column]):
            for row in batch.to_pylist():
                key = row[ref.key_column]
                positions = requested_positions[ref.column].get(key)
                if positions is not None:
                    if key in resolved_keys:
                        msg = f"[DuplicateLanceRefKey] duplicate key {key!r} in {ref.uri}:{ref.key_column}"
                        raise ValueError(msg)
                    resolved_keys.add(key)
                    entries_by_column[ref.column][positions] = row_index
                row_index += 1
        flush = getattr(entries_by_column[ref.column], "flush", None)
        if callable(flush):
            flush()

    tmp_manifest_path = manifest_path.with_suffix(f"{manifest_path.suffix}.tmp")
    tmp_manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2, default=str), encoding="utf-8")
    tmp_manifest_path.replace(manifest_path)


def prepare_ref_indexes(
    source: LanceSourceSpec,
    *,
    columns: Sequence[str] | None = None,
    load_in_memory: bool = False,
    context: RuntimeContext | None = None,
    ref_index_scope: LanceRefIndexScope | None = None,
) -> LanceSourceSpec:
    if not source.ref_columns:
        return source

    active_refs = (
        source.ref_columns if columns is None else tuple(ref for ref in source.ref_columns if ref.column in columns)
    )
    if not active_refs:
        return LanceSourceSpec(datasets=source.datasets, ref_columns=())

    ref_files = {
        ref.column: {
            "kind": "csr_row_index",
            "offsets_file": f"ref-{ref_i}.offsets.u64",
            "entries_file": f"ref-{ref_i}.entries.i64",
            "missing_row": REF_INDEX_MISSING_ROW,
        }
        for ref_i, ref in enumerate(active_refs)
    }
    manifest = {
        "builder_version": REF_INDEX_BUILDER_VERSION,
        "main_datasets": [_dataset_manifest_fingerprint(dataset.uri) for dataset in source.datasets],
        "main_total_rows": source.total_rows,
        "refs": {
            ref.column: {
                **ref_files[ref.column],
                "ref_dataset": _dataset_manifest_fingerprint(ref.uri),
                "key_column": ref.key_column,
                "value_column": ref.value_column,
            }
            for ref in active_refs
        },
    }
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:32]
    index_root = Path(source.datasets[0].uri) / REF_INDEX_DIR
    index_dir = index_root / f"ref-index-{digest}"
    scope = _resolve_ref_index_scope(ref_index_scope)
    is_builder = _is_ref_index_builder(context, scope)

    if not _ref_index_is_valid(index_dir, manifest, ref_files, active_refs):
        if is_builder:
            tmp_root = index_root / ".tmp"
            tmp_index_dir = tmp_root / f"ref-index-{digest}-{os.getpid()}-{uuid.uuid4().hex}"
            if tmp_index_dir.exists():
                shutil.rmtree(tmp_index_dir, ignore_errors=True)
            try:
                _build_ref_index(tmp_index_dir, manifest, ref_files, active_refs, source)
                if not _ref_index_is_valid(index_dir, manifest, ref_files, active_refs):
                    _publish_ref_index(tmp_index_dir, index_dir)
            finally:
                shutil.rmtree(tmp_index_dir, ignore_errors=True)
        else:
            _wait_for_ref_index(index_dir, manifest, ref_files, active_refs)

    if not _ref_index_is_valid(index_dir, manifest, ref_files, active_refs):
        msg = f"[InvalidLanceRefIndex] ref index was not completed at {index_dir}"
        raise RuntimeError(msg)

    prepared_refs: list[LanceRefSpec] = []
    for ref in active_refs:
        offsets_path = index_dir / ref_files[ref.column]["offsets_file"]
        entries_path = index_dir / ref_files[ref.column]["entries_file"]
        offsets: object = np.memmap(offsets_path, dtype=np.uint64, mode="r", shape=(source.total_rows + 1,))
        entry_count = int(offsets[-1])
        entries: object = (
            np.empty(0, dtype=np.int64)
            if entry_count == 0
            else np.memmap(entries_path, dtype=np.int64, mode="r", shape=(entry_count,))
        )
        value_dataset: object = lance.dataset(ref.uri)
        if load_in_memory:
            offsets = np.asarray(offsets)
            entries = np.asarray(entries)
        prepared_refs.append(
            LanceRefSpec(
                column=ref.column,
                uri=ref.uri,
                key_column=ref.key_column,
                value_column=ref.value_column,
                index_uri=str(index_dir),
                index_offsets_path=str(offsets_path),
                index_entries_path=str(entries_path),
                index_handle={"offsets": offsets, "entries": entries, "value_dataset": value_dataset},
            )
        )

    return LanceSourceSpec(datasets=source.datasets, ref_columns=tuple(prepared_refs))


def _apply_ref_columns(
    source: LanceSourceSpec,
    batch: list[Sample],
    columns: Sequence[str] | None,
) -> None:
    """Resolve configured Lance reference columns in-place for one sample batch.

    ``prepare_ref_indexes`` builds a CSR-style lookup for each configured ref
    column. For every main-dataset global row, ``offsets[row]:offsets[row + 1]``
    points into ``entries``. The ``entries`` slice contains the row indexes in
    the reference Lance dataset that correspond to the key or keys stored in
    the main sample column.

    This function uses those prepared indexes to replace each requested ref
    column value in ``batch``:

    1. Map every batch sample to its global main-table row index.
    2. For each active ref column, recover the reference row indexes needed by
       each sample.
    3. Fetch each unique reference row once from the reference Lance dataset.
    4. Scatter fetched values back into the original sample positions while
       preserving scalar-vs-multi-value shape.

    The input ``batch`` is mutated in place. Samples are expected to contain the
    internal ``__global_index__`` metadata added by the Lance resolver assembler.
    """

    # Step 1: Restrict work to the requested/projection-visible ref columns.
    # If ``columns`` is None, all configured refs are active. Otherwise a ref is
    # resolved only when its source column is present in the current projection.
    active_refs = (
        source.ref_columns if columns is None else tuple(ref for ref in source.ref_columns if ref.column in columns)
    )
    if not active_refs:
        return

    # Step 2: Convert each batch sample into the global row index used by the
    # prepared CSR ref index. The index is global across all physical Lance
    # datasets in ``source``, not local to a single dataset file.
    global_indices = np.asarray([item["__global_index__"] for item in batch], dtype=np.int64)
    for ref in active_refs:
        # Step 3: Validate that ``prepare_ref_indexes`` has attached the CSR
        # arrays and the prepared reference dataset handle for this ref column.
        if not isinstance(ref.index_handle, dict):
            msg = f"[UnpreparedLanceRefIndex] ref index for {ref.column!r} was not prepared"
            raise RuntimeError(msg)

        # Step 4: Read each sample's CSR range. For sample global row ``i``,
        # ``offsets[i]:offsets[i + 1]`` gives the slice in ``entries`` holding
        # the reference dataset row indexes for that sample's key(s).
        offsets = ref.index_handle["offsets"]
        entries = ref.index_handle["entries"]
        starts = np.asarray(offsets[global_indices], dtype=np.int64)
        ends = np.asarray(offsets[global_indices + 1], dtype=np.int64)

        # Step 5: Build a scatter plan.
        #
        # ``per_sample_row_indices`` keeps each sample's ref row indexes in
        # original key order. ``positions_by_row_index`` groups all output slots
        # by reference dataset row index, so duplicate references across the
        # batch are fetched once and then scattered back to every consumer.
        positions_by_row_index: dict[int, list[tuple[int, int]]] = {}
        per_sample_row_indices: list[np.ndarray] = []
        for sample_position, (start, end) in enumerate(zip(starts, ends, strict=True)):
            ref_row_indices = np.asarray(entries[start:end], dtype=np.int64)
            per_sample_row_indices.append(ref_row_indices)
            for value_position, row_index in enumerate(ref_row_indices):
                if int(row_index) != REF_INDEX_MISSING_ROW:
                    positions_by_row_index.setdefault(int(row_index), []).append((sample_position, value_position))

        # Step 6: Allocate the output slots before reading values. The slot
        # counts mirror the source cardinality exactly, which preserves list
        # lengths and keeps scalar refs distinguishable from empty multi-refs.
        resolved_values: list[list[Any]] = [[] for _ in per_sample_row_indices]
        for sample_position, ref_row_indices in enumerate(per_sample_row_indices):
            resolved_values[sample_position] = [None] * len(ref_row_indices)

        # Step 7: Fetch each unique reference row once, then scatter the fetched
        # ``value_column`` into every sample/value slot recorded in the plan.
        if positions_by_row_index:
            ordered_row_indices = list(positions_by_row_index)
            ref_dataset = ref.index_handle.get("value_dataset")
            if ref_dataset is None:
                ref_dataset = lance.dataset(ref.uri)
            ref_rows = _read_table_rows(ref_dataset, ordered_row_indices, columns=[ref.value_column])
            for row_index, row in zip(ordered_row_indices, ref_rows, strict=True):
                for sample_position, value_position in positions_by_row_index[row_index]:
                    resolved_values[sample_position][value_position] = row[ref.value_column]

        # Step 8: Replace the original key column with resolved values. Empty
        # CSR ranges resolve to ``None`` for scalar refs and ``[]`` for
        # list/tuple/ndarray refs; non-empty ranges preserve scalar-vs-multi
        # shape based on the original value.
        for sample, ref_row_indices, values in zip(batch, per_sample_row_indices, resolved_values, strict=True):
            original_value = sample.get(ref.column)
            is_multi_value = isinstance(original_value, (list, tuple, np.ndarray))
            if len(ref_row_indices) == 0:
                sample[ref.column] = [] if is_multi_value else None
                continue
            sample[ref.column] = values if is_multi_value else values[0]


def validate_ref_names(source: LanceSourceSpec, ref_names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(ref_names, str):
        raw_ref_names = (ref_names,)
    else:
        raw_ref_names = tuple(ref_names)

    if not raw_ref_names:
        msg = "[InvalidLanceRefNames] at least one ref column name is required"
        raise ValueError(msg)
    if not all(isinstance(ref_name, str) and ref_name for ref_name in raw_ref_names):
        msg = "[InvalidLanceRefNames] ref column names must be non-empty strings"
        raise ValueError(msg)
    normalized = tuple(dict.fromkeys(raw_ref_names))

    available_refs = {ref.column for ref in source.ref_columns}
    missing_refs = [ref_name for ref_name in normalized if ref_name not in available_refs]
    if missing_refs:
        msg = f"[UnknownLanceRefColumn] requested ref column(s) were not configured: {', '.join(missing_refs)}"
        raise ValueError(msg)

    return normalized


def iter_lance_ref_resolver(
    source: LanceSourceSpec,
    sample_stream: Iterable[object],
    ref_names: Sequence[str],
    *,
    batch_size: int = 1024,
    load_in_memory: bool = False,
    context: RuntimeContext | None = None,
    ref_index_scope: LanceRefIndexScope | None = None,
):
    """Resolve configured Lance reference columns for already-read samples."""

    assembler = LanceRefResolverAssembler(
        source=source,
        ref_names=ref_names,
        batch_size=batch_size,
        load_in_memory=load_in_memory,
        context=context,
        ref_index_scope=ref_index_scope,
    )
    for sample in sample_stream:
        yield from assembler.push(sample)
    yield from assembler.finish()


@dataclass(frozen=True, slots=True)
class LanceResolveRefFactory:
    source: LanceSourceSpec
    ref_names: tuple[str, ...]
    batch_size: int = 1024
    load_in_memory: bool = False
    ref_index_scope: LanceRefIndexScope | None = None

    def __call__(self, context: RuntimeContext) -> LanceRefResolverAssembler:
        return LanceRefResolverAssembler(
            source=self.source,
            ref_names=self.ref_names,
            batch_size=self.batch_size,
            load_in_memory=self.load_in_memory,
            context=context,
            ref_index_scope=self.ref_index_scope,
        )


class LanceRefResolverAssembler:
    def __init__(
        self,
        *,
        source: LanceSourceSpec,
        ref_names: Sequence[str],
        batch_size: int = 1024,
        load_in_memory: bool = False,
        context: RuntimeContext | None = None,
        ref_index_scope: LanceRefIndexScope | None = None,
    ) -> None:
        if batch_size <= 0:
            msg = f"[InvalidLanceRefBatchSize] batch_size must be > 0, got {batch_size}"
            raise ValueError(msg)

        self.ref_names = validate_ref_names(source, ref_names)
        self.source = prepare_ref_indexes(
            source,
            columns=self.ref_names,
            load_in_memory=load_in_memory,
            context=context,
            ref_index_scope=ref_index_scope,
        )
        self.batch_size = batch_size
        self.batch: list[Sample] = []
        self.queue_size = 0

    def _flush(self) -> Iterable[object]:
        if not self.batch:
            return ()

        if isinstance(self.batch[0], dict):
            _apply_ref_columns(self.source, self.batch, self.ref_names)
        elif isinstance(self.batch[0], list):
            for sub_batch in self.batch:
                if isinstance(sub_batch, list) and sub_batch and isinstance(sub_batch[0], dict):
                    _apply_ref_columns(self.source, sub_batch, self.ref_names)
                else:
                    raise TypeError(f"[InvalidLanceBatch] expected dict samples, got {type(sub_batch).__name__}")
        else:
            raise TypeError(f"[InvalidLanceBatch] expected dict or list samples, got {type(self.batch[0]).__name__}")

        flushed_batch = self.batch
        self.batch = []
        self.queue_size = 0
        return flushed_batch

    def push(self, sample: object) -> Iterable[object]:
        assert isinstance(sample, (dict, list)), (
            f"[InvalidLanceSample] expected dict or list sample, got {type(sample).__name__}"
        )
        is_single_sample = isinstance(sample, dict)
        self.batch.append(sample)
        self.queue_size += 1 if is_single_sample else len(sample)

        if not sample or self.queue_size >= self.batch_size:
            return self._flush()

        return ()

    def finish(self, *, drop_last: bool = False) -> Iterable[object]:
        if drop_last:
            self.batch.clear()
            self.queue_size = 0
            return ()
        return self._flush()
