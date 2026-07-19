"""Disk-backed Lance filter index."""

from __future__ import annotations

import os
import uuid
from bisect import bisect_left
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path

import lance
import numpy as np

from mvp_dataset.cache import CacheBuildResult, CacheEntry, CacheManager

from .cache import fingerprint_lance_source
from .types import LanceFilterIndex, LanceSource

FILTER_INDEX_FORMAT_VERSION = 1
FILTER_INDEX_BUILD_BATCH_SIZE = 65_536
FILTER_INDEX_MAX_PARTS = 64
_ROW_OFFSET_MASK = (1 << 32) - 1


@dataclass(frozen=True, slots=True)
class _Fragment:
    dataset_i: int
    fragment_id: int
    global_offset: int
    physical_rows: int
    dataset: object = field(repr=False, compare=False)
    handle: object = field(repr=False, compare=False)


def validate_filter_index_config(index: dict[str, object] | None) -> None:
    """Reject removed filter-index options while preserving the common filter API shape."""
    if index is None:
        return
    if not isinstance(index, dict):
        msg = "[InvalidLanceFilterIndexConfig] index must be a mapping"
        raise TypeError(msg)
    unknown_keys = sorted(index)
    if unknown_keys:
        msg = f"[InvalidLanceFilterIndexConfig] unknown config key(s): {', '.join(unknown_keys)}"
        raise ValueError(msg)


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


def _open_filter_index(entry: CacheEntry) -> LanceFilterIndex:
    manifest = entry.read_manifest()
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        msg = f"[InvalidLanceFilterIndex] missing metadata at {entry.path}"
        raise RuntimeError(msg)
    raw_parts = metadata.get("parts")
    if not isinstance(raw_parts, list):
        msg = f"[InvalidLanceFilterIndex] invalid part metadata at {entry.path}"
        raise RuntimeError(msg)

    paths: list[str] = []
    offsets = [0]
    for part in raw_parts:
        if (
            not isinstance(part, dict)
            or not isinstance(part.get("files"), list)
            or len(part["files"]) != 1
            or not isinstance(part["files"][0], str)
            or not isinstance(part.get("metadata"), dict)
            or not isinstance(part["metadata"].get("count"), int)
        ):
            msg = f"[InvalidLanceFilterIndex] invalid part entry at {entry.path}"
            raise RuntimeError(msg)
        file_name = part["files"][0]
        count = part["metadata"]["count"]
        path = entry.path / file_name
        if count < 0 or not path.is_file() or path.stat().st_size != count * np.dtype(np.int64).itemsize:
            msg = f"[InvalidLanceFilterIndex] invalid part file at {path}"
            raise RuntimeError(msg)
        paths.append(str(path))
        offsets.append(offsets[-1] + count)
    return LanceFilterIndex(paths=tuple(paths), offsets=tuple(offsets), count=offsets[-1])


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


def prepare_filter_index(
    source: LanceSource,
    *,
    predicate_groups: tuple[tuple[str, ...], ...],
) -> LanceFilterIndex:
    """Build or open the disk-backed row mapping for Lance filter batches."""
    source_fingerprint = fingerprint_lance_source(source.datasets)
    parameters = {
        "predicate_groups": [list(group) for group in predicate_groups],
    }

    planned_parts: dict[str, tuple[_Fragment, ...]] = {}

    def _parts() -> tuple[str, ...]:
        fragment_parts = _plan_parts(source)
        if not fragment_parts:
            fragment_parts = ((),)
        planned_parts.update({f"part-{part_i:03d}": fragments for part_i, fragments in enumerate(fragment_parts)})
        return tuple(planned_parts)

    def _build_part(part: str, temporary_dir: Path) -> CacheBuildResult:
        fragments = planned_parts[part]
        output = temporary_dir / "rows.i64"
        if not fragments:
            for dataset in source.datasets:
                handle = lance.dataset(dataset.uri, version=dataset.version)
                for group in predicate_groups:
                    for predicate in group:
                        handle.count_rows(predicate)
            output.touch()
        else:
            _build_parts(((output, fragments, None),), predicate_groups)
        size = output.stat().st_size
        if size % np.dtype(np.int64).itemsize != 0:
            msg = f"[InvalidLanceFilterIndex] invalid part size for {part}"
            raise RuntimeError(msg)
        return CacheBuildResult.from_files([output.name], metadata={"count": size // 8})

    entry = CacheManager().ensure(
        source=source_fingerprint,
        kind="lance-filter-index",
        format_version=FILTER_INDEX_FORMAT_VERSION,
        parameters=parameters,
        parts=_parts,
        build=_build_part,
    )
    return _open_filter_index(entry)
