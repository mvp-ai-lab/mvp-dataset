"""Mount-independent cache fingerprint helpers."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import FingerprintMode

SOURCE_FINGERPRINT_SCHEMA_VERSION = 1
_CONTENT_HASH_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True, init=False)
class SourceFingerprint:
    """A stable source digest and the manifest used to produce it."""

    value: str
    _manifest_bytes: bytes = field(repr=False)

    def __init__(self, *, value: str, manifest: Mapping[str, Any]) -> None:
        """Store an immutable canonical manifest and verify its digest."""
        manifest_bytes = canonical_json_bytes(dict(manifest))
        expected = hashlib.sha256(manifest_bytes).hexdigest()
        if value != expected:
            msg = f"[InvalidSourceFingerprint] expected={expected} got={value}"
            raise ValueError(msg)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "_manifest_bytes", manifest_bytes)

    @property
    def manifest(self) -> dict[str, Any]:
        """Return a detached copy of the canonical source manifest."""
        manifest = json.loads(self._manifest_bytes)
        if not isinstance(manifest, dict):
            msg = "[InvalidSourceManifest] expected object"
            raise RuntimeError(msg)
        return manifest


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a JSON-compatible value deterministically."""
    return json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")


def fingerprint_payload(value: object) -> str:
    """Return a SHA-256 fingerprint for a JSON-compatible value."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def fingerprint_source_manifest(manifest: Mapping[str, Any]) -> SourceFingerprint:
    """Return a source fingerprint from a mount-independent manifest."""
    normalized = json.loads(canonical_json_bytes(dict(manifest)))
    return SourceFingerprint(value=fingerprint_payload(normalized), manifest=normalized)


def fingerprint_local_files(
    paths: Sequence[str | os.PathLike[str]],
    *,
    mode: FingerprintMode = "metadata",
) -> SourceFingerprint:
    """Fingerprint local files without including their absolute mount prefix."""
    if not paths:
        msg = "[EmptyCacheSource] at least one source file is required"
        raise ValueError(msg)
    if mode not in ("metadata", "content"):
        msg = f"[InvalidCacheFingerprintMode] mode={mode!r}"
        raise ValueError(msg)

    absolute_paths = [Path(os.path.abspath(Path(path).expanduser())) for path in paths]
    common_root = Path(os.path.commonpath([str(path.parent) for path in absolute_paths]))
    files: list[dict[str, object]] = []
    for path in absolute_paths:
        stat = path.stat()
        item: dict[str, object] = {
            "path": path.relative_to(common_root).as_posix(),
            "size": stat.st_size,
        }
        if mode == "metadata":
            item["mtime_ns"] = stat.st_mtime_ns
        else:
            item["sha256"] = _hash_file(path)
        files.append(item)

    return fingerprint_source_manifest(
        {
            "schema_version": SOURCE_FINGERPRINT_SCHEMA_VERSION,
            "kind": "local-files",
            "fingerprint_mode": mode,
            "files": files,
        }
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CONTENT_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()
