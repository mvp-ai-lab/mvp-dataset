"""Lance ref-column index preparation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import lance
import numpy as np
import pyarrow as pa

from mvp_dataset.core.context import RuntimeContext

from ..source import list_lance_sources
from ..types import (
    LanceDatasetSpec,
    LanceRefIndexConfig,
    LanceRefIndexScope,
    LanceRefSpec,
    LanceSource,
)

REF_INDEX_BUILDER_VERSION = 1
REF_INDEX_MISSING_ROW = -1
REF_INDEX_MANIFEST = "metadata.json"
REF_INDEX_DIR = "_mvp_ref_index"
REF_INDEX_CACHE_DIR_ENV = "MVP_DATASET_LANCE_REF_INDEX_CACHE_DIR"
REF_INDEX_BUILD_BATCH_SIZE = 65536
REF_INDEX_LOCK_POLL_SECONDS = 0.25
REF_INDEX_WAIT_TIMEOUT_SECONDS = 30 * 60
REF_INDEX_DEFAULT_BUILD_STRATEGY = "auto"
REF_INDEX_DEFAULT_BUCKET_COUNT = 4096
REF_INDEX_AUTO_BUCKETED_MIN_ROWS = 1_000_000
REF_INDEX_MAX_OPEN_BUCKET_FILES = 128


def _iter_ref_keys(value: Any) -> tuple[Any, ...]:
    """Yield reference keys from source rows."""
    if value is None:
        return ()
    if isinstance(value, np.ndarray):
        return tuple(item for item in value.tolist() if item is not None)
    if isinstance(value, (list, tuple)):
        return tuple(item for item in value if item is not None)
    return (value,)


def _key_token(key: Any) -> str:
    """Return a stable typed token for a reference key."""
    if isinstance(key, np.generic):
        key = key.item()
    if isinstance(key, bool):
        payload = {"type": "bool", "value": key}
    elif isinstance(key, int):
        payload = {"type": "int", "value": key}
    elif isinstance(key, float):
        payload = {"type": "float", "value": key}
    elif isinstance(key, str):
        payload = {"type": "str", "value": key}
    elif isinstance(key, (bytes, bytearray, memoryview)):
        payload = {"type": "bytes", "value": bytes(key).hex()}
    else:
        payload = {"type": f"{type(key).__module__}.{type(key).__qualname__}", "value": str(key)}
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _bucket_id_for_key_token(key_token: str, *, bucket_count: int) -> int:
    digest = hashlib.sha256(key_token.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % bucket_count


def _iter_table_record_batches(
    dataset_handle: object,
    *,
    columns: Sequence[str],
    batch_size: int = REF_INDEX_BUILD_BATCH_SIZE,
) -> Iterable[pa.RecordBatch]:
    to_batches = getattr(dataset_handle, "to_batches", None)
    if callable(to_batches):
        yield from to_batches(columns=columns, batch_size=batch_size)
        return

    scanner = dataset_handle.scanner(columns=columns, batch_size=batch_size, scan_in_order=True)
    yield from scanner.to_batches()


class _BucketWriter:
    """Write hash-bucketed records without keeping all files open."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._handles: dict[int, Any] = {}

    def write(self, bucket_id: int, line: str) -> None:
        handle = self._handles.get(bucket_id)
        if handle is None:
            if len(self._handles) >= REF_INDEX_MAX_OPEN_BUCKET_FILES:
                old_bucket_id = next(iter(self._handles))
                self._handles.pop(old_bucket_id).close()
            handle = (self.root / f"bucket-{bucket_id:05d}.tsv").open("a", encoding="utf-8")
            self._handles[bucket_id] = handle
        handle.write(line)

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __enter__(self) -> _BucketWriter:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _resolve_ref_index_root(source: LanceSource) -> Path:
    raw_cache_dir = os.environ.get(REF_INDEX_CACHE_DIR_ENV)
    if raw_cache_dir:
        return Path(raw_cache_dir).expanduser()
    return Path(source.datasets[0].uri) / REF_INDEX_DIR


def _dataset_manifest_fingerprint(uri: str) -> dict[str, Any]:
    """Return a fingerprint for a Lance dataset manifest."""
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


def _ref_uris(ref: LanceRefSpec) -> tuple[str, ...]:
    """Return referenced Lance URIs for one reference field."""
    return (ref.uri,) if isinstance(ref.uri, str) else ref.uri


def _ref_manifest_fingerprint(ref: LanceRefSpec) -> dict[str, Any]:
    """Return a fingerprint for reference source manifests."""
    uris = _ref_uris(ref)
    if len(uris) == 1:
        return _dataset_manifest_fingerprint(uris[0])
    datasets = [_dataset_manifest_fingerprint(uri) for uri in uris]
    return {
        "uris": list(uris),
        "datasets": datasets,
        "num_rows": sum(int(dataset["num_rows"]) for dataset in datasets),
    }


def _open_ref_value_source(ref: LanceRefSpec) -> LanceSource:
    """Open the value source for a Lance reference field."""
    source = list_lance_sources(_ref_uris(ref))[0]
    datasets: list[LanceDatasetSpec] = []
    for dataset in source.datasets:
        ds_handle: object = lance.dataset(dataset.uri)
        datasets.append(
            LanceDatasetSpec(
                uri=dataset.uri,
                num_rows=dataset.num_rows,
                row_offset=dataset.row_offset,
                handle=ds_handle,
            )
        )
    return LanceSource(datasets=tuple(datasets), ref_columns=source.ref_columns)


def _ref_index_is_valid(
    index_dir: Path,
    manifest: dict[str, Any],
    ref_files: dict[str, dict[str, Any]],
    active_refs: Sequence[LanceRefSpec],
) -> bool:
    """Return whether an existing reference index matches the manifest."""
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


def _is_ref_index_builder(context: RuntimeContext | None, scope: LanceRefIndexScope) -> bool:
    """Return whether this worker should build the reference index."""
    if context is None or scope == "process":
        return True
    if scope == "shared":
        return context.rank == 0 and context.worker_id == 0
    if scope == "node_local":
        return context.local_rank == 0 and context.worker_id == 0
    return False


def _wait_for_ref_index(
    index_dir: Path,
    manifest: dict[str, Any],
    ref_files: dict[str, dict[str, Any]],
    active_refs: Sequence[LanceRefSpec],
    *,
    timeout_seconds: float = REF_INDEX_WAIT_TIMEOUT_SECONDS,
) -> None:
    """Wait for a reference index built by another worker."""
    deadline = time.monotonic() + timeout_seconds
    poll_seconds = REF_INDEX_LOCK_POLL_SECONDS
    while not _ref_index_is_valid(index_dir, manifest, ref_files, active_refs):
        if time.monotonic() >= deadline:
            msg = f"[LanceRefIndexTimeout] timed out waiting for ref index at {index_dir}"
            raise TimeoutError(msg)
        time.sleep(poll_seconds)
        poll_seconds = min(poll_seconds * 2, 5.0)


def _publish_ref_index(tmp_index_dir: Path, index_dir: Path) -> None:
    """Publish reference index files atomically."""
    try:
        tmp_index_dir.replace(index_dir)
    except FileExistsError:
        return
    except OSError:
        if not index_dir.exists():
            raise


def _build_ref_index_in_memory(
    index_dir: Path,
    manifest: dict[str, Any],
    ref_files: dict[str, dict[str, Any]],
    active_refs: Sequence[LanceRefSpec],
    source: LanceSource,
) -> None:
    """Build lookup indexes with an in-memory key-to-entry map."""
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

    entries_by_column: dict[str, np.memmap] = {}
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
        for uri in _ref_uris(ref):
            for batch in _iter_table_record_batches(lance.dataset(uri), columns=[ref.key_column]):
                for row in batch.to_pylist():
                    key = row[ref.key_column]
                    positions = requested_positions[ref.column].get(key)
                    if positions is not None:
                        if key in resolved_keys:
                            msg = f"[DuplicateLanceRefKey] duplicate key {key!r} in {uri}:{ref.key_column}"
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


def _build_ref_index_bucketed(
    index_dir: Path,
    manifest: dict[str, Any],
    ref_files: dict[str, dict[str, Any]],
    active_refs: Sequence[LanceRefSpec],
    source: LanceSource,
    *,
    bucket_count: int,
) -> None:
    """Build lookup indexes with hash buckets on disk."""
    manifest_path = index_dir / REF_INDEX_MANIFEST
    bucket_root = index_dir / "_bucket_tmp"
    index_dir.mkdir(parents=True, exist_ok=False)

    offsets_by_column: dict[str, np.memmap] = {}
    entry_counts = {ref.column: 0 for ref in active_refs}
    for ref in active_refs:
        offsets_by_column[ref.column] = np.memmap(
            index_dir / ref_files[ref.column]["offsets_file"],
            dtype=np.uint64,
            mode="w+",
            shape=(source.total_rows + 1,),
        )
        offsets_by_column[ref.column][0] = 0

    main_writers = {ref.column: _BucketWriter(bucket_root / ref.column / "main") for ref in active_refs}
    try:
        global_row_index = 0
        main_columns = [ref.column for ref in active_refs]
        for dataset in source.datasets:
            for batch in _iter_table_record_batches(lance.dataset(dataset.uri), columns=main_columns):
                for row in batch.to_pylist():
                    global_row_index += 1
                    for ref in active_refs:
                        for key in _iter_ref_keys(row[ref.column]):
                            key_token = _key_token(key)
                            bucket_id = _bucket_id_for_key_token(key_token, bucket_count=bucket_count)
                            main_writers[ref.column].write(
                                bucket_id,
                                f"{key_token}\t{entry_counts[ref.column]}\n",
                            )
                            entry_counts[ref.column] += 1
                        offsets_by_column[ref.column][global_row_index] = entry_counts[ref.column]
    finally:
        for writer in main_writers.values():
            writer.close()

    if global_row_index != source.total_rows:
        msg = f"[InvalidLanceRefIndex] expected {source.total_rows} main rows, scanned {global_row_index}"
        raise RuntimeError(msg)

    entries_by_column: dict[str, np.memmap] = {}
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

    try:
        for ref in active_refs:
            _write_ref_buckets(ref, bucket_root / ref.column / "ref", bucket_count=bucket_count)
            _join_ref_buckets(
                ref,
                main_bucket_dir=bucket_root / ref.column / "main",
                ref_bucket_dir=bucket_root / ref.column / "ref",
                entries=entries_by_column[ref.column],
                bucket_count=bucket_count,
            )
            flush = getattr(entries_by_column[ref.column], "flush", None)
            if callable(flush):
                flush()
    finally:
        shutil.rmtree(bucket_root, ignore_errors=True)

    tmp_manifest_path = manifest_path.with_suffix(f"{manifest_path.suffix}.tmp")
    tmp_manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2, default=str), encoding="utf-8")
    tmp_manifest_path.replace(manifest_path)


def _write_ref_buckets(ref: LanceRefSpec, bucket_dir: Path, *, bucket_count: int) -> None:
    """Write reference keys to hash buckets."""
    row_index = 0
    with _BucketWriter(bucket_dir) as writer:
        for uri in _ref_uris(ref):
            for batch in _iter_table_record_batches(lance.dataset(uri), columns=[ref.key_column]):
                for row in batch.to_pylist():
                    key = row[ref.key_column]
                    key_token = _key_token(key)
                    bucket_id = _bucket_id_for_key_token(key_token, bucket_count=bucket_count)
                    writer.write(bucket_id, f"{key_token}\t{row_index}\t{key!r}\n")
                    row_index += 1


def _join_ref_buckets(
    ref: LanceRefSpec,
    *,
    main_bucket_dir: Path,
    ref_bucket_dir: Path,
    entries: np.memmap | np.ndarray,
    bucket_count: int,
) -> None:
    """Join one ref column's bucket files and fill the CSR entries array."""
    for bucket_id in range(bucket_count):
        main_bucket_path = main_bucket_dir / f"bucket-{bucket_id:05d}.tsv"
        ref_bucket_path = ref_bucket_dir / f"bucket-{bucket_id:05d}.tsv"
        if not main_bucket_path.exists() and not ref_bucket_path.exists():
            continue

        ref_rows: dict[str, int] = {}
        if ref_bucket_path.exists():
            with ref_bucket_path.open(encoding="utf-8") as handle:
                for line in handle:
                    key_token, row_index, display_key = line.rstrip("\n").split("\t", 2)
                    if key_token in ref_rows:
                        msg = f"[DuplicateLanceRefKey] duplicate key {display_key} in {ref.key_column}"
                        raise ValueError(msg)
                    ref_rows[key_token] = int(row_index)

        if not main_bucket_path.exists() or not ref_rows:
            continue
        with main_bucket_path.open(encoding="utf-8") as handle:
            for line in handle:
                key_token, entry_position = line.rstrip("\n").split("\t", 1)
                ref_row_index = ref_rows.get(key_token)
                if ref_row_index is not None:
                    entries[int(entry_position)] = ref_row_index


def prepare_ref_indexes(
    source: LanceSource,
    *,
    ref_names: Sequence[str],
    context: RuntimeContext | None = None,
    config: LanceRefIndexConfig,
) -> LanceSource:
    """Ensure all configured Lance reference indexes are available.

    Args:
        source: Lance source specification.
        ref_names: Reference column names to prepare.
        context: Runtime context used for sharding and deterministic randomness.
        config: Reference index configuration.

    Returns:
        A Lance source specification whose reference indexes are ready."""
    if not source.ref_columns:
        return source

    ref_name_set = set(ref_names)
    active_refs = tuple(ref for ref in source.ref_columns if ref.column in ref_name_set)
    if not active_refs:
        return LanceSource(datasets=source.datasets, ref_columns=())

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
                "ref_dataset": _ref_manifest_fingerprint(ref),
                "key_column": ref.key_column,
                "value_column": ref.value_column,
            }
            for ref in active_refs
        },
    }
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:32]
    index_root = _resolve_ref_index_root(source)
    index_dir = index_root / f"ref-index-{digest}"
    scope = config.scope or "shared"
    build_strategy = config.build_strategy or REF_INDEX_DEFAULT_BUILD_STRATEGY
    if build_strategy == "auto":
        build_strategy = "bucketed" if source.total_rows >= REF_INDEX_AUTO_BUCKETED_MIN_ROWS else "in_memory"
    bucket_count = config.bucket_count or REF_INDEX_DEFAULT_BUCKET_COUNT

    if not _ref_index_is_valid(index_dir, manifest, ref_files, active_refs):
        if _is_ref_index_builder(context, scope):
            tmp_index_dir = index_root / ".tmp" / f"ref-index-{digest}-{os.getpid()}-{uuid.uuid4().hex}"
            try:
                if build_strategy == "in_memory":
                    _build_ref_index_in_memory(tmp_index_dir, manifest, ref_files, active_refs, source)
                else:
                    _build_ref_index_bucketed(
                        tmp_index_dir,
                        manifest,
                        ref_files,
                        active_refs,
                        source,
                        bucket_count=bucket_count,
                    )
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
        value_source = _open_ref_value_source(ref)
        prepared_refs.append(
            LanceRefSpec(
                column=ref.column,
                uri=ref.uri,
                key_column=ref.key_column,
                value_column=ref.value_column,
                index_uri=str(index_dir),
                index_offsets_path=str(offsets_path),
                index_entries_path=str(entries_path),
                index_handle={
                    "offsets": offsets,
                    "entries": entries,
                    "value_source": value_source,
                },
            )
        )

    return LanceSource(datasets=source.datasets, ref_columns=tuple(prepared_refs))
