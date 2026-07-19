"""Lance source identities for persistent cache artifacts."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import unquote, urlsplit

from mvp_dataset.cache import SourceFingerprint, fingerprint_source_manifest

from .types import LanceDatasetSpec

_LANCE_MANIFEST_VERSION_BASE = (1 << 64) - 1


def fingerprint_lance_source(datasets: Sequence[LanceDatasetSpec]) -> SourceFingerprint:
    """Fingerprint resolved Lance dataset versions without local mount prefixes."""
    return fingerprint_source_manifest(lance_source_manifest(datasets))


def lance_source_manifest(datasets: Sequence[LanceDatasetSpec]) -> dict[str, object]:
    """Describe resolved Lance dataset versions without local mount prefixes."""
    logical_uris = _logical_uris([dataset.uri for dataset in datasets])
    return {
        "schema_version": 1,
        "kind": "lance",
        "datasets": [
            _dataset_manifest(dataset, logical_uri=logical_uri)
            for dataset, logical_uri in zip(datasets, logical_uris, strict=True)
        ],
        "num_rows": sum(dataset.num_rows for dataset in datasets),
    }


def _dataset_manifest(dataset: LanceDatasetSpec, *, logical_uri: str) -> dict[str, object]:
    manifest: dict[str, object] = {
        "uri": logical_uri,
        "num_rows": dataset.num_rows,
        "row_offset": dataset.row_offset,
        "version": dataset.version,
    }
    local_path = _local_path(dataset.uri)
    if local_path is None or dataset.version is None:
        return manifest

    version_manifest = _version_manifest_path(local_path, dataset.version)
    if version_manifest is None:
        return manifest
    stat = version_manifest.stat()
    manifest["manifest"] = {
        "name": version_manifest.name,
        "size": stat.st_size,
        "sha256": _hash_file(version_manifest),
    }
    return manifest


def _version_manifest_path(root: Path, version: int | str) -> Path | None:
    versions_dir = root / "_versions"
    candidates = [versions_dir / f"{version}.manifest"]
    try:
        numeric_version = int(version)
    except (TypeError, ValueError):
        numeric_version = -1
    if 0 <= numeric_version <= _LANCE_MANIFEST_VERSION_BASE:
        candidates.append(versions_dir / f"{_LANCE_MANIFEST_VERSION_BASE - numeric_version}.manifest")
    return next((path for path in candidates if path.is_file()), None)


def _logical_uris(uris: Sequence[str]) -> tuple[str, ...]:
    local_paths = [_local_path(uri) for uri in uris]
    concrete_paths = [path for path in local_paths if path is not None]
    common_root = Path(os.path.commonpath([str(path.parent) for path in concrete_paths])) if concrete_paths else None
    return tuple(
        uri if path is None or common_root is None else path.relative_to(common_root).as_posix()
        for uri, path in zip(uris, local_paths, strict=True)
    )


def _local_path(uri: str) -> Path | None:
    parsed = urlsplit(uri)
    if not parsed.scheme:
        return Path(os.path.abspath(Path(uri).expanduser()))
    if parsed.scheme == "file" and parsed.netloc in ("", "localhost"):
        return Path(os.path.abspath(Path(unquote(parsed.path)).expanduser()))
    return None


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
