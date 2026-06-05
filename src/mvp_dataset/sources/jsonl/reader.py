"""JSONL row readers."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

from ...core.types import PathLikeStr, Sample, TarUriRefFieldSpec
from .refs import TarManager, resolve_ref_field_value


def iter_jsonls(
    shard_paths: Iterator[PathLikeStr],
    ref_fields: tuple[TarUriRefFieldSpec, ...],
) -> Iterator[Sample]:
    """Resolve tar data references while streaming JSONL shard files."""
    key_dot_level = int(os.environ.get("MVP_DATASET_TAR_KEY_DOT_LEVEL", "1"))
    max_open_tar_files = int(os.environ.get("MVP_DATASET_TAR_MAX_OPEN_FILES", "8"))

    def _resolve_one(sample: Sample, manager: TarManager) -> Sample:
        resolved = dict(sample)
        for field, base_dir in ref_fields:
            if field not in sample:
                continue
            resolved[field] = resolve_ref_field_value(
                sample[field],
                field=field,
                base_dir=base_dir,
                key_dot_level=key_dot_level,
                manager=manager,
            )
        return resolved

    with TarManager(max_open_files=max_open_tar_files) as manager:
        for shard_path in shard_paths:
            with open(shard_path, encoding="utf-8") as handle:
                for line_index, line in enumerate(handle):
                    sample = _parse_jsonl_line(str(shard_path), line_index, line, allow_preannotated=True)
                    yield _resolve_one(sample, manager)


def _parse_jsonl_line(
    file: str,
    index_in_file: int,
    line: str,
    *,
    allow_preannotated: bool = False,
) -> Sample:
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError as exc:
        msg = f"[InvalidJsonLine] file={file!r} line={index_in_file + 1} reason={exc.msg}"
        raise ValueError(msg) from exc
    if not isinstance(parsed, dict):
        msg = f"[InvalidJsonSample] file={file!r} line={index_in_file + 1} expected object row"
        raise ValueError(msg)

    sample: Sample = dict(parsed)
    if allow_preannotated and _has_jsonl_metadata(sample):
        return sample

    sample["__index_in_file__"] = index_in_file
    sample["__file__"] = file
    sample["__key__"] = f"{file}:{index_in_file}"
    return sample


def _has_jsonl_metadata(sample: Sample) -> bool:
    return (
        isinstance(sample.get("__index_in_file__"), int)
        and isinstance(sample.get("__file__"), str)
        and isinstance(sample.get("__key__"), str)
    )
