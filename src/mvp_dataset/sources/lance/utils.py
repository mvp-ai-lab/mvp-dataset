"""Lance source implementation for slot-aware shuffled sample iteration."""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import shutil
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lance
import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads

from ...core.context import RuntimeContext
from ...core.types import PathLikeStr, Sample

REF_INDEX_BUILDER_VERSION = 1
REF_INDEX_MISSING_ROW = -1
REF_INDEX_MANIFEST = "metadata.json"
REF_INDEX_CACHE_DIR = "_mvp_ref_index"
REF_INDEX_BUILD_BATCH_SIZE = 65536
REF_INDEX_LOCK_POLL_SECONDS = 0.25


@dataclass(frozen=True, slots=True)
class LanceDatasetSpec:
    """Resolved metadata for one Lance dataset URI."""

    uri: str
    num_rows: int
    row_offset: int
    fragment_ids: tuple[int, ...]
    handle: object | None = None


@dataclass(frozen=True, slots=True)
class LanceRefSpec:
    """Configuration for one Lance reference column."""

    column: str
    uri: str
    key_column: str
    value_column: str
    index_uri: str | None = None
    index_offsets_path: str | None = None
    index_entries_path: str | None = None
    index_handle: object | None = None


@dataclass(frozen=True, slots=True)
class LanceSourceSpec:
    """One schedulable Lance source configuration."""

    datasets: list[LanceDatasetSpec]
    ref_columns: tuple[LanceRefSpec, ...] = ()

    @property
    def total_rows(self) -> int:
        return sum(dataset.num_rows for dataset in self.datasets)

    @property
    def total_fragments(self) -> int:
        return sum(len(dataset.fragment_ids) for dataset in self.datasets)


@dataclass(frozen=True, slots=True)
class LanceIndexItem:
    dataset_i: int
    local_index: int
    global_index: int


def list_lance_sources(dataset_uris: Sequence[PathLikeStr]) -> list[LanceSourceSpec]:
    """Resolve Lance dataset URIs into one slot-aware source configuration."""

    # 1. Resolve URIs into metadata specs, accumulating row offsets for global indexing.
    datasets: list[LanceDatasetSpec] = []
    row_offset = 0
    for uri in dataset_uris:
        ds = lance.dataset(str(uri))
        fragment_ids = tuple(fragment.fragment_id for fragment in ds.get_fragments())
        num_rows = ds.count_rows()
        datasets.append(
            LanceDatasetSpec(
                uri=str(uri),
                num_rows=num_rows,
                row_offset=row_offset,
                fragment_ids=fragment_ids,
            )
        )
        row_offset += num_rows

    if not datasets:
        msg = "[EmptyLanceSource] at least one Lance dataset URI is required"
        raise ValueError(msg)

    return [LanceSourceSpec(datasets=list(datasets))]


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
        cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return cached_manifest == manifest and all(
        (index_dir / ref_files[ref.column]["offsets_file"]).exists()
        and (index_dir / ref_files[ref.column]["entries_file"]).exists()
        for ref in active_refs
    )


def _acquire_ref_index_build_lock(
    lock_dir: Path,
    index_dir: Path,
    manifest: dict[str, Any],
    ref_files: dict[str, dict[str, Any]],
    active_refs: Sequence[LanceRefSpec],
) -> bool:
    lock_dir.parent.mkdir(parents=True, exist_ok=True)
    while True:
        if _ref_index_is_valid(index_dir, manifest, ref_files, active_refs):
            return False
        try:
            lock_dir.mkdir()
            (lock_dir / "owner.json").write_text(
                json.dumps({"pid": os.getpid(), "created_at": time.time()}, sort_keys=True),
                encoding="utf-8",
            )
            return True
        except FileExistsError:
            time.sleep(REF_INDEX_LOCK_POLL_SECONDS)


def prepare_ref_indexes(
    source: LanceSourceSpec,
    *,
    columns: Sequence[str] | None = None,
    load_in_memory: bool = False,
) -> LanceSourceSpec:
    # 1. Fast-path sources that do not define reference columns at all.
    if not source.ref_columns:
        return source

    # 2. Keep only the reference columns that the caller is going to read. If
    #    no requested columns need reference resolution, return an equivalent
    #    source without prepared references.
    active_refs = (
        source.ref_columns if columns is None else tuple(ref for ref in source.ref_columns if ref.column in columns)
    )
    if not active_refs:
        return LanceSourceSpec(datasets=source.datasets, ref_columns=())

    # 3. Build the deterministic cache manifest. The manifest fingerprints both
    #    the main datasets and reference datasets, so the cache directory changes
    #    whenever the indexed inputs or builder format change.
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
    index_root = Path(source.datasets[0].uri) / REF_INDEX_CACHE_DIR
    index_dir = index_root / f"ref-index-{digest}"
    manifest_path = index_dir / REF_INDEX_MANIFEST
    lock_dir = index_root / f"ref-index-{digest}.lock"

    # 4. Only one process may rebuild this cache directory. DataLoader workers
    #    commonly call prepare_ref_indexes() at the same time; without a lock one
    #    worker can delete a partial cache while another is opening its memmaps.
    should_build = _acquire_ref_index_build_lock(lock_dir, index_dir, manifest, ref_files, active_refs)
    if should_build:
        # 5. Start a fresh cache directory and initialize per-column CSR state.
        #    offsets[row] stores the starting entry index for one main row, while
        #    requested_positions maps each ref key to the entry slots that need
        #    to be filled after the reference dataset is scanned.
        try:
            if not _ref_index_is_valid(index_dir, manifest, ref_files, active_refs):
                if index_dir.exists():
                    shutil.rmtree(index_dir, ignore_errors=True)
                index_dir.mkdir(parents=True, exist_ok=True)

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

                # 6. Scan all main datasets in global row order. For each reference key
                #    found in a main row, append one pending entry slot and advance that
                #    column's CSR offsets so later reads can recover the row's key range
                #    with offsets[global_index:global_index + 2].
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

                # 7. Allocate the entries arrays. Empty reference columns get an empty
                #    in-memory array and a touched file, while non-empty columns are
                #    prefilled with the missing-row sentinel until their keys resolve.
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

                # 8. Scan each reference dataset once. When a reference key is requested
                #    by the main dataset, write the reference row index into every pending
                #    entry slot for that key. Duplicate keys are rejected because they
                #    would make a main key resolve to more than one reference row.
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

                # 9. Write the manifest last. Its presence marks the cache as complete
                #    for future calls because all index files have already been created
                #    and flushed.
                manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2, default=str), encoding="utf-8")
        finally:
            shutil.rmtree(lock_dir, ignore_errors=True)

    if not _ref_index_is_valid(index_dir, manifest, ref_files, active_refs):
        msg = f"[InvalidLanceRefIndex] ref index cache was not completed at {index_dir}"
        raise RuntimeError(msg)

    # 10. Attach read-only index handles to the returned source. By default the
    #     arrays stay memory-mapped to avoid loading large indexes; callers that
    #     requested an in-memory source get numpy arrays instead. The reference
    #     value dataset is opened once here so every batch can reuse the handle.
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


def assign_items(
    source: Sequence[LanceSourceSpec],
    *,
    context: RuntimeContext,
    resample: bool,
    shuffle: bool = False,
) -> Iterable[tuple[LanceSourceSpec, int, int]]:
    assert len(source) == 1, "Multiple Lance sources are not supported in this implementation"
    source: LanceSourceSpec = source[0]

    # 1. Yield globally slot-assigned indexes for all rows across all datasets.
    round_index = 0
    global_index_list = np.arange(source.total_rows)
    shuffle_rng = np.random.default_rng(context.seed + round_index)

    slot = context.slot
    total_slots = context.total_slots

    while True:
        if shuffle:
            shuffle_rng.shuffle(global_index_list)

        for i, global_index in enumerate(global_index_list):
            if i % total_slots != slot:
                continue

            dataset_i = (
                bisect.bisect_right(
                    [dataset.row_offset for dataset in source.datasets],
                    global_index,
                )
                - 1
            )
            dataset = source.datasets[dataset_i]
            local_index = global_index - dataset.row_offset
            yield LanceIndexItem(
                dataset_i=dataset_i,
                local_index=local_index,
                global_index=global_index,
            )

        if not resample:
            break
        round_index += 1


def _read_batch(
    source: LanceSourceSpec,
    batch_indexes: Sequence[LanceIndexItem],
    *,
    columns: Sequence[str] | None = None,
):
    # 1. group batch indexes by dataset
    if not batch_indexes:
        return []

    per_dataset_indices: dict[int, list[int]] = {}
    for index_item in batch_indexes:
        per_dataset_indices.setdefault(index_item.dataset_i, []).append(index_item.local_index)

    # 2. read each dataset batch
    per_dataset_rows: dict[int, list[Sample]] = {}
    take_indices_type = pa.int64()
    for dataset_i, local_indices in per_dataset_indices.items():
        dataset = source.datasets[dataset_i]
        dataset_handle = dataset.handle if dataset.handle is not None else lance.dataset(dataset.uri)

        if isinstance(dataset_handle, pa.Table):
            table = dataset_handle
            if columns is not None:
                table = table.select(columns)
            table = table.take(pa.array(local_indices, type=take_indices_type))
        elif isinstance(dataset_handle, pads.Dataset):
            table = dataset_handle.take(
                pa.array(local_indices, type=take_indices_type),
                columns=columns,
            )
        else:
            table = dataset_handle.take(local_indices, columns=columns)

        per_dataset_rows[dataset_i] = table.to_pylist()

    # 3. restore global order within batch
    per_dataset_offsets = {dataset_i: 0 for dataset_i in per_dataset_indices}

    # 4. concatenate into one batch and return
    batch: list[Sample] = []
    for index_item in batch_indexes:
        dataset = source.datasets[index_item.dataset_i]
        dataset_offset = per_dataset_offsets[index_item.dataset_i]
        sample = dict(per_dataset_rows[index_item.dataset_i][dataset_offset])
        sample["__file__"] = dataset.uri
        sample["__index_in_file__"] = index_item.local_index
        sample["__key__"] = f"{dataset.uri}:{index_item.local_index}"
        batch.append(sample)
        per_dataset_offsets[index_item.dataset_i] += 1

    return batch


def _apply_ref_columns(
    source: LanceSourceSpec,
    batch: list[Sample],
    batch_indexes: Sequence[LanceIndexItem],
    columns: Sequence[str] | None,
) -> None:
    # 1. Resolve only the ref columns that are part of this read. When the
    #    caller requested a projection, reference columns outside that projection
    #    should be left untouched because they are not present in the batch.
    active_refs = (
        source.ref_columns if columns is None else tuple(ref for ref in source.ref_columns if ref.column in columns)
    )
    if not active_refs:
        return

    # 2. Translate the batch-local LanceIndexItem objects into global row
    #    positions. The prepared CSR index is keyed by global row position across
    #    all source datasets, not by the per-file local row index.
    global_indices = np.asarray([index_item.global_index for index_item in batch_indexes], dtype=np.int64)
    for ref in active_refs:
        # 3. _apply_ref_columns is the read-time resolver, so it expects
        #    prepare_ref_indexes() to have attached the already-built index
        #    arrays. Failing here makes accidental unprepared use explicit.
        if not isinstance(ref.index_handle, dict):
            msg = f"[UnpreparedLanceRefIndex] ref index for {ref.column!r} was not prepared"
            raise RuntimeError(msg)

        # 4. Read the CSR slices for every sample in the batch. offsets contains
        #    one extra item, so offsets[i]:offsets[i + 1] gives the entries range
        #    for main row i. entries stores the resolved reference row indices.
        offsets = ref.index_handle["offsets"]
        entries = ref.index_handle["entries"]
        starts = np.asarray(offsets[global_indices], dtype=np.int64)
        ends = np.asarray(offsets[global_indices + 1], dtype=np.int64)

        # 5. Keep each sample's reference row indices in batch order, while also
        #    building a reverse lookup from reference row index to every output
        #    slot that needs that row's value. Missing sentinel rows are skipped
        #    for now and validated before values are written back.
        positions_by_row_index: dict[int, list[tuple[int, int]]] = {}
        per_sample_row_indices: list[np.ndarray] = []
        for sample_position, (start, end) in enumerate(zip(starts, ends, strict=True)):
            ref_row_indices = np.asarray(entries[start:end], dtype=np.int64)
            per_sample_row_indices.append(ref_row_indices)
            for value_position, row_index in enumerate(ref_row_indices):
                if int(row_index) != REF_INDEX_MISSING_ROW:
                    positions_by_row_index.setdefault(int(row_index), []).append((sample_position, value_position))

        # 6. Preallocate the final per-sample values with the exact cardinality
        #    from the source column. This preserves list length/order and gives
        #    the scatter step below stable slots to fill.
        resolved_values: list[list[Any]] = [[] for _ in per_sample_row_indices]
        for sample_position, ref_row_indices in enumerate(per_sample_row_indices):
            resolved_values[sample_position] = [None] * len(ref_row_indices)

        # 7. Fetch each unique reference row once through the cached value
        #    dataset handle, then scatter the requested value column back into
        #    every sample/value slot that pointed at that row. ordered_row_indices
        #    preserves the same order as ref_rows. The fallback keeps tests or
        #    externally constructed handles from depending on the cache key.
        if positions_by_row_index:
            ordered_row_indices = list(positions_by_row_index)
            ref_dataset = ref.index_handle.get("value_dataset")
            if ref_dataset is None:
                ref_dataset = lance.dataset(ref.uri)
            ref_rows = _read_table_rows(ref_dataset, ordered_row_indices, columns=[ref.value_column])
            for row_index, row in zip(ordered_row_indices, ref_rows, strict=True):
                for sample_position, value_position in positions_by_row_index[row_index]:
                    resolved_values[sample_position][value_position] = row[ref.value_column]

        # 8. Replace the original key column with resolved values. Empty key
        #    ranges become [] for list-like source values and None for scalar
        #    values; unresolved sentinels become a KeyError with the original key
        #    value; successful rows keep the original scalar-vs-list shape.
        for sample, ref_row_indices, values in zip(batch, per_sample_row_indices, resolved_values, strict=True):
            original_value = sample.get(ref.column)
            if len(ref_row_indices) == 0:
                sample[ref.column] = [] if isinstance(original_value, list) else None
                continue
            if np.any(ref_row_indices == REF_INDEX_MISSING_ROW):
                msg = (
                    f"[MissingLanceRefKey] {ref.column!r}={original_value!r} "
                    f"was not found in {ref.uri}:{ref.key_column}"
                )
                raise KeyError(msg)
            sample[ref.column] = values if isinstance(original_value, list) else values[0]


def iter_lance(
    source: LanceSourceSpec,
    index_stream: Iterable[LanceIndexItem],
    *,
    columns: Sequence[str] | None = None,
    batch_size: int = 65536,
    load_in_memory: bool = False,
):
    source = prepare_ref_indexes(source, columns=columns, load_in_memory=load_in_memory)

    # 1. Load all dataset into memory if requested.
    for dataset_i, dataset in enumerate(source.datasets):
        ds_handle = lance.dataset(dataset.uri)
        if load_in_memory:
            ds_handle = pads.InMemoryDataset(ds_handle.to_table())
        source.datasets[dataset_i] = LanceDatasetSpec(
            uri=dataset.uri,
            num_rows=dataset.num_rows,
            row_offset=dataset.row_offset,
            fragment_ids=dataset.fragment_ids,
            handle=ds_handle,
        )

    batch_indexes: list[LanceIndexItem] = []
    for index_item in index_stream:
        batch_indexes.append(index_item)
        if len(batch_indexes) >= batch_size:
            batch = _read_batch(source, batch_indexes, columns=columns)
            _apply_ref_columns(source, batch, batch_indexes, columns)
            yield from batch
            batch_indexes.clear()
    if batch_indexes:
        batch = _read_batch(source, batch_indexes, columns=columns)
        _apply_ref_columns(source, batch, batch_indexes, columns)
        yield from batch
