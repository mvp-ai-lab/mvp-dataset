"""Tar shard source implementation for streaming sample iteration."""

from __future__ import annotations

import tarfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

from ..core.types import Sample


def _is_meta_member(name: str) -> bool:
    """Return ``True`` if a tar member should be treated as metadata and skipped."""

    basename = PurePosixPath(name).name
    return basename.startswith("__") and basename.endswith("__")


def _split_key_and_field(name: str) -> tuple[str, str] | None:
    """Extract ``(key, field)`` from ``<key>.<field>`` file names.

    Returns ``None`` for files that do not follow this convention.
    """

    basename = PurePosixPath(name).name
    if "." not in basename:
        return None

    key, field = basename.rsplit(".", 1)
    if not key or not field:
        return None
    return key, field


def iter_tar_records(shard_path: str | Path) -> Iterator[Sample]:
    """Iterate a tar shard and yield grouped samples by key.

    Files are expected to follow ``<key>.<field>`` naming. Consecutive entries with
    the same key are grouped into one sample.

    Each yielded sample contains:
    - ``__key__``: sample key
    - ``__shard__``: shard path
    - ``__index_in_shard__``: zero-based row index within the shard
    - one entry per field with raw ``bytes`` payload
    """

    shard = str(shard_path)
    current_key: str | None = None
    current_sample: Sample | None = None
    index_in_shard = 0

    with tarfile.open(shard, mode="r|*") as stream:
        for member in stream:
            if not member.isfile():
                continue
            if _is_meta_member(member.name):
                continue

            key_field = _split_key_and_field(member.name)
            if key_field is None:
                continue
            key, field = key_field

            extracted = stream.extractfile(member)
            if extracted is None:
                msg = f"failed to extract tar member {member.name!r} from {shard!r}"
                raise tarfile.ExtractError(msg)
            payload = extracted.read()

            if key != current_key:
                if current_sample is not None:
                    yield current_sample
                    index_in_shard += 1
                current_key = key
                current_sample = {
                    "__key__": key,
                    "__shard__": shard,
                    "__index_in_shard__": index_in_shard,
                }

            assert current_sample is not None
            if field in current_sample:
                msg = f"duplicate field {field!r} for key {key!r} in shard {shard!r}"
                raise ValueError(msg)
            current_sample[field] = payload

    if current_sample is not None:
        yield current_sample
