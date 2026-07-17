"""Disk-backed Lance filter index."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from bisect import bisect_left
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path
from urllib.parse import urlsplit

import lance
import numpy as np

from mvp_dataset.core.context import RuntimeContext

from .types import (
    LanceFilterIndex,
    LanceFilterIndexConfig,
    LanceFilterIndexConfigInput,
    LanceSource,
)

FILTER_INDEX_BUILDER_VERSION = 1
FILTER_INDEX_DIR = "_mvp_filter_index"
FILTER_INDEX_CACHE_DIR_ENV = "MVP_DATASET_LANCE_FILTER_INDEX_CACHE_DIR"
FILTER_INDEX_BUILD_BATCH_SIZE = 65_536
FILTER_INDEX_MAX_PARTS = 64
FILTER_INDEX_POLL_SECONDS = 0.25
FILTER_INDEX_WAIT_TIMEOUT_SECONDS = 30 * 60
FILTER_INDEX_MANIFEST = "manifest.json"
_ROW_OFFSET_MASK = (1 << 32) - 1


@dataclass(frozen=True, slots=True)
class _Fragment:
    dataset_i: int
    fragment_id: int
    global_offset: int
    physical_rows: int
    dataset: object = field(repr=False, compare=False)
    handle: object = field(repr=False, compare=False)


def resolve_filter_index_config(index: LanceFilterIndexConfigInput) -> LanceFilterIndexConfig:
    """Return validated Lance filter-index settings."""
    if index is None:
        return LanceFilterIndexConfig()
    if not isinstance(index, dict):
        msg = "[InvalidLanceFilterIndexConfig] index must be a mapping"
        raise TypeError(msg)
    unknown_keys = sorted(set(index) - {"scope"})
    if unknown_keys:
        msg = f"[InvalidLanceFilterIndexConfig] unknown config key(s): {', '.join(unknown_keys)}"
        raise ValueError(msg)
    scope = index.get("scope", "shared")
    if scope not in ("shared", "node_local", "process"):
        msg = f"[InvalidLanceFilterIndexScope] expected shared, node_local, or process, got {scope!r}"
        raise ValueError(msg)
    return LanceFilterIndexConfig(scope=scope)


def _plan_parts(source: LanceSource) -> tuple[tuple[_Fragment, ...], ...]:
    fragments: list[_Fragment] = []
    for dataset_i, spec in enumerate(source.datasets):
        dataset = lance.dataset(spec.uri, version=spec.version)
        local_offset = 0
        for fragment in dataset.get_fragments():
            metadata = fragment.metadata
            if metadata.deletion_file is not None:
                msg = f"[UnsupportedLanceFilterDeletedRows] dataset has deleted rows: {spec.uri}"
                raise ValueError(msg)
            physical_rows = int(metadata.physical_rows)
            fragments.append(
                _Fragment(
                    dataset_i=dataset_i,
                    fragment_id=int(fragment.fragment_id),
                    global_offset=spec.row_offset + local_offset,
                    physical_rows=physical_rows,
                    dataset=dataset,
                    handle=fragment,
                )
            )
            local_offset += physical_rows
        if local_offset != spec.num_rows:
            msg = f"[InvalidLanceFilterSource] expected {spec.num_rows} rows in {spec.uri}, found {local_offset}"
            raise RuntimeError(msg)

    if not fragments:
        return ()

    part_count = min(FILTER_INDEX_MAX_PARTS, len(fragments))
    prefix_rows = [0]
    for fragment in fragments:
        prefix_rows.append(prefix_rows[-1] + fragment.physical_rows)

    boundaries = [0]
    for part_i in range(1, part_count):
        target = prefix_rows[-1] * part_i / part_count
        boundary = bisect_left(prefix_rows, target)
        boundary = max(boundaries[-1] + 1, min(boundary, len(fragments) - (part_count - part_i)))
        boundaries.append(boundary)
    boundaries.append(len(fragments))
    return tuple(tuple(fragments[start:stop]) for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True))


def _load_filter_index(index_dir: Path, identity: dict[str, object]) -> tuple[LanceFilterIndex | None, dict[str, int]]:
    manifest_path = index_dir / FILTER_INDEX_MANIFEST
    if not manifest_path.is_file():
        return None, {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        parts = [(str(part["file"]), int(part["count"])) for part in manifest["parts"]]
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None, {}
    if manifest.get("identity") != identity:
        return None, {}

    expected_counts = dict(parts)
    paths: list[str] = []
    offsets = [0]
    for file_name, count in parts:
        path = index_dir / file_name
        if count < 0 or not path.is_file() or path.stat().st_size != count * np.dtype(np.int64).itemsize:
            return None, expected_counts
        paths.append(str(path))
        offsets.append(offsets[-1] + count)
    if manifest.get("offsets") != offsets:
        return None, expected_counts
    return LanceFilterIndex(paths=tuple(paths), offsets=tuple(offsets), count=offsets[-1]), expected_counts


def _build_parts(
    entries: tuple[tuple[Path, tuple[_Fragment, ...], int | None], ...],
    predicate_groups: tuple[tuple[str, ...], ...],
) -> None:
    pending = []
    for path, fragments, expected_count in entries:
        if path.is_file():
            size = path.stat().st_size
            if expected_count is None and size % np.dtype(np.int64).itemsize == 0:
                continue
            if expected_count is not None and size == expected_count * np.dtype(np.int64).itemsize:
                continue
        pending.append((path, fragments))
    if not pending:
        return

    fragments = tuple(fragment for _, part in pending for fragment in part)
    span_start = fragments[0].global_offset
    span_rows = fragments[-1].global_offset + fragments[-1].physical_rows - span_start
    direct = len(pending) == 1 and len(predicate_groups) == 1 and len(predicate_groups[0]) == 1
    bitmap = None
    tmp_paths = {path: path.with_name(f"{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}") for path, _ in pending}
    output = tmp_paths[pending[0][0]].open("wb") if direct else None
    try:
        dataset_groups = tuple(tuple(grouped) for _, grouped in groupby(fragments, key=lambda item: item.dataset_i))
        for predicates in predicate_groups:
            group_bitmap = None if direct else np.zeros((span_rows + 7) // 8, dtype=np.uint8)
            for group in dataset_groups:
                part_fragment_ids = np.asarray([fragment.fragment_id for fragment in group], dtype=np.int64)
                part_offsets = np.asarray([fragment.global_offset for fragment in group], dtype=np.int64)
                sort_order = np.argsort(part_fragment_ids)
                part_fragment_ids = part_fragment_ids[sort_order]
                part_offsets = part_offsets[sort_order]
                for predicate in predicates:
                    scanner = group[0].dataset.scanner(
                        fragments=[fragment.handle for fragment in group],
                        columns=[],
                        filter=predicate,
                        with_row_address=True,
                        scan_in_order=True,
                        batch_size=FILTER_INDEX_BUILD_BATCH_SIZE,
                        use_scalar_index=True,
                    )
                    for batch in scanner.to_batches():
                        row_addresses = np.asarray(batch.column("_rowaddr"), dtype=np.uint64)
                        row_fragment_ids = np.right_shift(row_addresses, 32).astype(np.int64, copy=False)
                        row_offsets = np.bitwise_and(row_addresses, _ROW_OFFSET_MASK).astype(np.int64, copy=False)
                        global_indices = (
                            part_offsets[np.searchsorted(part_fragment_ids, row_fragment_ids)] + row_offsets
                        )
                        if group_bitmap is None:
                            global_indices.tofile(output)
                            continue
                        local_indices = global_indices - span_start
                        np.bitwise_or.at(
                            group_bitmap,
                            local_indices >> 3,
                            np.left_shift(np.uint8(1), (local_indices & 7).astype(np.uint8, copy=False)),
                        )
            if group_bitmap is not None:
                if bitmap is None:
                    bitmap = group_bitmap
                else:
                    np.bitwise_and(bitmap, group_bitmap, out=bitmap)

        if output is not None:
            output.close()
            output = None
        else:
            for path, part in pending:
                part_start = part[0].global_offset
                part_rows = sum(fragment.physical_rows for fragment in part)
                bit_start = part_start - span_start
                with tmp_paths[path].open("wb") as part_output:
                    for row_start in range(0, part_rows, FILTER_INDEX_BUILD_BATCH_SIZE):
                        row_stop = min(row_start + FILTER_INDEX_BUILD_BATCH_SIZE, part_rows)
                        absolute_start = bit_start + row_start
                        bit_offset = absolute_start & 7
                        bits = np.unpackbits(
                            bitmap[absolute_start // 8 : (absolute_start + row_stop - row_start + 7) // 8],
                            bitorder="little",
                        )[bit_offset : bit_offset + row_stop - row_start]
                        global_indices = np.flatnonzero(bits).astype(np.int64, copy=False)
                        global_indices += part_start + row_start
                        global_indices.tofile(part_output)

        for path, _ in pending:
            os.replace(tmp_paths[path], path)
    finally:
        if output is not None:
            output.close()
        for tmp_path in tmp_paths.values():
            tmp_path.unlink(missing_ok=True)


def _build_part(
    path: Path,
    fragments: tuple[_Fragment, ...],
    predicate_groups: tuple[tuple[str, ...], ...],
    expected_count: int | None = None,
) -> None:
    _build_parts(((path, fragments, expected_count),), predicate_groups)


def prepare_filter_index(
    source: LanceSource,
    *,
    predicate_groups: tuple[tuple[str, ...], ...],
    context: RuntimeContext | None,
    config: LanceFilterIndexConfig,
) -> LanceFilterIndex:
    """Build or open the disk-backed row mapping for Lance filter batches."""
    predicate_hash = hashlib.sha256(
        json.dumps(predicate_groups, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    identity: dict[str, object] = {
        "builder_version": FILTER_INDEX_BUILDER_VERSION,
        "predicate_hash": predicate_hash,
        "predicate_group_count": len(predicate_groups),
        "predicate_count": sum(len(group) for group in predicate_groups),
        "datasets": [
            {
                "uri": dataset.uri,
                "version": dataset.version,
                "num_rows": dataset.num_rows,
            }
            for dataset in source.datasets
        ],
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[
        :32
    ]
    raw_cache_dir = os.environ.get(FILTER_INDEX_CACHE_DIR_ENV)
    if raw_cache_dir:
        index_root = Path(raw_cache_dir).expanduser()
    else:
        source_uri = source.datasets[0].uri
        if urlsplit(source_uri).scheme:
            msg = f"[MissingLanceFilterIndexCacheDir] set {FILTER_INDEX_CACHE_DIR_ENV} for URI source {source_uri!r}"
            raise ValueError(msg)
        index_root = Path(source_uri) / FILTER_INDEX_DIR
    index_dir = index_root / f"filter-index-{digest}"
    index_dir.mkdir(parents=True, exist_ok=True)

    cached, expected_counts = _load_filter_index(index_dir, identity)
    if cached is not None:
        return cached

    parts = _plan_parts(source)
    part_names = [f"part-{part_i:03d}.i64" for part_i in range(len(parts))]
    if not parts:
        for dataset in source.datasets:
            handle = lance.dataset(dataset.uri, version=dataset.version)
            for group in predicate_groups:
                for predicate in group:
                    handle.count_rows(predicate)

    scope = config.scope
    if context is None or scope == "process":
        assignment: tuple[int, int] | None = (0, 1)
        leader = True
    elif scope == "shared":
        assignment = (
            context.rank * context.num_workers + context.worker_id,
            context.world_size * context.num_workers,
        )
        leader = context.rank == 0 and context.worker_id == 0
    else:
        assignment = (
            context.local_rank * context.num_workers + context.worker_id,
            context.local_world_size * context.num_workers,
        )
        leader = context.local_rank == 0 and context.worker_id == 0

    if leader and expected_counts:
        _build_parts(
            tuple(
                (index_dir / part_name, part, expected_counts[part_name])
                for part_name, part in zip(part_names, parts, strict=True)
                if part_name in expected_counts
            ),
            predicate_groups,
        )

    if assignment is not None:
        builder_id, builder_count = assignment
        if len(predicate_groups) == 1 and len(predicate_groups[0]) == 1:
            for part_i in range(builder_id, len(parts), builder_count):
                part_name = part_names[part_i]
                _build_part(index_dir / part_name, parts[part_i], predicate_groups, expected_counts.get(part_name))
        else:
            start = len(parts) * builder_id // builder_count
            stop = len(parts) * (builder_id + 1) // builder_count
            _build_parts(
                tuple(
                    (
                        index_dir / part_names[part_i],
                        parts[part_i],
                        expected_counts.get(part_names[part_i]),
                    )
                    for part_i in range(start, stop)
                ),
                predicate_groups,
            )

    deadline = time.monotonic() + FILTER_INDEX_WAIT_TIMEOUT_SECONDS
    while True:
        cached, _ = _load_filter_index(index_dir, identity)
        if cached is not None:
            return cached

        part_paths = [index_dir / part_name for part_name in part_names]
        if leader and all(path.is_file() and path.stat().st_size % 8 == 0 for path in part_paths):
            counts = [path.stat().st_size // 8 for path in part_paths]
            offsets = [0]
            for count in counts:
                offsets.append(offsets[-1] + count)
            manifest = {
                "identity": identity,
                "parts": [
                    {"file": part_name, "count": count} for part_name, count in zip(part_names, counts, strict=True)
                ],
                "offsets": offsets,
            }
            manifest_path = index_dir / FILTER_INDEX_MANIFEST
            tmp_path = manifest_path.with_name(f"{manifest_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
            try:
                tmp_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
                os.replace(tmp_path, manifest_path)
            finally:
                tmp_path.unlink(missing_ok=True)
            continue

        if time.monotonic() >= deadline:
            msg = f"[LanceFilterIndexTimeout] scope={scope!r} cache={str(index_dir)!r}"
            raise TimeoutError(msg)
        time.sleep(FILTER_INDEX_POLL_SECONDS)
