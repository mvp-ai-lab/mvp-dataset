"""Testing helpers for creating tar shards."""

from __future__ import annotations

import io
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path


def create_tar_shard(
    path: Path,
    rows: Sequence[tuple[str, Mapping[str, bytes]]],
    *,
    meta_files: Mapping[str, bytes] | None = None,
) -> None:
    """Create a tar shard where each row is written as ``<key>.<field>`` files."""

    with tarfile.open(path, mode="w") as archive:
        if meta_files is not None:
            for name, payload in meta_files.items():
                _add_bytes_member(archive, name=name, payload=payload)

        for key, fields in rows:
            for field, payload in fields.items():
                _add_bytes_member(archive, name=f"{key}.{field}", payload=payload)


def _add_bytes_member(archive: tarfile.TarFile, *, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name=name)
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload))
