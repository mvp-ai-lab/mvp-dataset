"""Crash-safe persistent cache management."""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import socket
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, cast, overload

from .config import CacheConfig
from .fingerprint import SourceFingerprint, canonical_json_bytes, fingerprint_payload

CACHE_MANIFEST_SCHEMA_VERSION = 1
CACHE_MANIFEST_NAME = "manifest.json"
CACHE_COMPLETE_NAME = "complete"
SOURCE_MANIFEST_NAME = "source.json"
_CACHE_KIND_PATTERN = re.compile(r"^(?P<kind>[a-z][a-z0-9-]*)-v(?P<version>[1-9][0-9]*)$")
_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PARTITION_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_PARTITION_SCHEMA_VERSION = 1
_PARTITION_ACTIVE_NAME = "active.json"
_PARTITION_JOIN_SECONDS = 0.1


@dataclass(frozen=True, slots=True)
class CacheBuildResult:
    """Files and result metadata produced by one cache build."""

    files: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_files(
        cls,
        files: Sequence[str],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> CacheBuildResult:
        """Build a normalized result from relative file paths."""
        return cls(files=tuple(files), metadata=dict(metadata or {}))


@dataclass(frozen=True, slots=True)
class CacheKey:
    """Complete identity for one cache artifact."""

    source_fingerprint: str
    kind: str
    format_version: int
    artifact_fingerprint: str
    parameters: dict[str, Any]

    @classmethod
    def create(
        cls,
        *,
        source_fingerprint: str,
        kind: str,
        format_version: int,
        parameters: Mapping[str, Any],
    ) -> CacheKey:
        """Create and validate a cache key."""
        if not _FINGERPRINT_PATTERN.fullmatch(source_fingerprint):
            msg = f"[InvalidSourceFingerprint] value={source_fingerprint!r}"
            raise ValueError(msg)
        if not re.fullmatch(r"[a-z][a-z0-9-]*", kind):
            msg = f"[InvalidCacheKind] kind={kind!r}"
            raise ValueError(msg)
        if isinstance(format_version, bool) or not isinstance(format_version, int) or format_version <= 0:
            msg = f"[InvalidCacheFormatVersion] expected a positive integer, got={format_version!r}"
            raise ValueError(msg)
        normalized_parameters = json.loads(canonical_json_bytes(dict(parameters)))
        artifact_fingerprint = fingerprint_payload(
            {
                "kind": kind,
                "format_version": format_version,
                "parameters": normalized_parameters,
            }
        )
        return cls(
            source_fingerprint=source_fingerprint,
            kind=kind,
            format_version=format_version,
            artifact_fingerprint=artifact_fingerprint,
            parameters=normalized_parameters,
        )

    @property
    def kind_directory(self) -> str:
        """Return the versioned cache-kind directory name."""
        return f"{self.kind}-v{self.format_version}"


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """Resolved paths for one cache artifact."""

    key: CacheKey
    path: Path
    lock_path: Path
    temporary_parent: Path

    @property
    def manifest_path(self) -> Path:
        """Return the artifact manifest path."""
        return self.path / CACHE_MANIFEST_NAME

    @property
    def complete_path(self) -> Path:
        """Return the artifact completion marker path."""
        return self.path / CACHE_COMPLETE_NAME

    def read_manifest(self) -> dict[str, Any]:
        """Read the completed artifact manifest."""
        with self.manifest_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            msg = f"[InvalidCacheManifest] expected object at {self.manifest_path}"
            raise RuntimeError(msg)
        return payload


@dataclass(frozen=True, slots=True)
class CacheEntryInfo:
    """Summary of one completed cache artifact."""

    source_fingerprint: str
    kind: str
    format_version: int
    artifact_fingerprint: str
    path: Path
    size_bytes: int


@dataclass(frozen=True, slots=True)
class _PartitionAttempt:
    generation: str
    created_ns: int
    parts: tuple[str, ...]
    root: Path

    @property
    def payload_path(self) -> Path:
        return self.root / "payload"

    @property
    def results_path(self) -> Path:
        return self.root / "results"

    @property
    def locks_path(self) -> Path:
        return self.root / "locks"

    @property
    def temporary_path(self) -> Path:
        return self.root / "tmp"


@dataclass(slots=True)
class _FileLease:
    """An advisory file lock owned through one open file description."""

    lease_id: str
    path: Path
    handle: BinaryIO = field(repr=False)
    released: bool = False

    @property
    def is_held(self) -> bool:
        """Return whether this process still owns the lease handle."""
        return not self.released and not self.handle.closed


class CacheManager:
    """Manage cache paths, validation, concurrent builds, and atomic publication."""

    def __init__(self, config: CacheConfig | None = None) -> None:
        """Initialize the manager from explicit or resolved configuration."""
        self.config = CacheConfig.resolve() if config is None else config
        self.root = self.config.root

    def entry(self, key: CacheKey) -> CacheEntry:
        """Resolve all paths for a cache key."""
        artifact_path = self.root / key.source_fingerprint / key.kind_directory / key.artifact_fingerprint
        lock_path = (
            self.root / "_locks" / key.source_fingerprint / key.kind_directory / (f"{key.artifact_fingerprint}.lock")
        )
        temporary_parent = self.root / "_tmp" / key.source_fingerprint / key.kind_directory / (key.artifact_fingerprint)
        return CacheEntry(key=key, path=artifact_path, lock_path=lock_path, temporary_parent=temporary_parent)

    @overload
    def ensure(
        self,
        *,
        source: SourceFingerprint,
        kind: str,
        format_version: int,
        parameters: Mapping[str, Any],
        build: Callable[[Path], CacheBuildResult],
        parts: None = None,
        assigned_parts: None = None,
    ) -> CacheEntry: ...

    @overload
    def ensure(
        self,
        *,
        source: SourceFingerprint,
        kind: str,
        format_version: int,
        parameters: Mapping[str, Any],
        build: Callable[[str, Path], CacheBuildResult],
        parts: Callable[[], Sequence[str]],
        assigned_parts: Sequence[str] | None = None,
    ) -> CacheEntry: ...

    def ensure(
        self,
        *,
        source: SourceFingerprint,
        kind: str,
        format_version: int,
        parameters: Mapping[str, Any],
        build: Callable[[Path], CacheBuildResult] | Callable[[str, Path], CacheBuildResult],
        parts: Callable[[], Sequence[str]] | None = None,
        assigned_parts: Sequence[str] | None = None,
    ) -> CacheEntry:
        """Return a valid entry using either one builder or distributed part builders."""
        key = CacheKey.create(
            source_fingerprint=source.value,
            kind=kind,
            format_version=format_version,
            parameters=parameters,
        )
        entry = self.entry(key)
        self._ensure_source_manifest(source)
        if parts is not None:
            return self._ensure_partitioned_entry(
                entry,
                parts=parts,
                build=cast(Callable[[str, Path], CacheBuildResult], build),
                assigned_parts=assigned_parts,
            )

        if assigned_parts is not None:
            msg = "[InvalidCacheAssignedParts] assigned_parts requires parts"
            raise ValueError(msg)

        artifact_build = cast(Callable[[Path], CacheBuildResult], build)

        deadline = time.monotonic() + self.config.wait_timeout_seconds

        while True:
            if self.is_valid(entry):
                return entry
            lease = self._acquire_lock(entry)
            if lease is not None:
                try:
                    if self.is_valid(entry):
                        return entry
                    return self._build_entry(entry, artifact_build, lease)
                finally:
                    self._release_lock(lease)
            if time.monotonic() >= deadline:
                msg = f"[CacheBuildTimeout] timed out waiting for {entry.path}"
                raise TimeoutError(msg)
            time.sleep(self.config.poll_interval_seconds)

    def _ensure_partitioned_entry(
        self,
        entry: CacheEntry,
        *,
        parts: Callable[[], Sequence[str]],
        build: Callable[[str, Path], CacheBuildResult],
        assigned_parts: Sequence[str] | None,
    ) -> CacheEntry:
        if self.is_valid(entry):
            return entry

        partition_names = self._normalize_partition_names(parts())
        build_part_names = (
            partition_names if assigned_parts is None else self._normalize_partition_names(assigned_parts)
        )
        if not build_part_names:
            msg = "[InvalidCacheAssignedParts] assigned_parts must be non-empty"
            raise ValueError(msg)
        unknown_parts = set(build_part_names) - set(partition_names)
        if unknown_parts:
            msg = f"[InvalidCacheAssignedParts] unknown parts={sorted(unknown_parts)!r}"
            raise ValueError(msg)
        deadline = time.monotonic() + self.config.wait_timeout_seconds

        while True:
            if self.is_valid(entry):
                return entry
            attempt = self._ensure_partition_attempt(entry, partition_names, deadline)
            if attempt is None:
                return entry
            completed = self._run_partition_attempt(
                entry,
                attempt,
                build_part_names,
                build,
                deadline,
            )
            if completed is not None:
                return completed

    def is_valid(self, entry: CacheEntry) -> bool:
        """Return whether an entry is complete and matches its expected key."""
        if not entry.complete_path.is_file() or not entry.manifest_path.is_file():
            return False
        try:
            manifest = entry.read_manifest()
        except (OSError, json.JSONDecodeError, RuntimeError):
            return False
        key = entry.key
        if manifest.get("schema_version") != CACHE_MANIFEST_SCHEMA_VERSION:
            return False
        if manifest.get("source_fingerprint") != key.source_fingerprint:
            return False
        if manifest.get("kind") != key.kind or manifest.get("format_version") != key.format_version:
            return False
        if manifest.get("artifact_fingerprint") != key.artifact_fingerprint:
            return False
        if manifest.get("parameters") != key.parameters:
            return False
        files = manifest.get("files")
        if not isinstance(files, list):
            return False
        return all(self._manifest_file_is_valid(entry.path, item) for item in files)

    def list_entries(self) -> tuple[CacheEntryInfo, ...]:
        """Return completed cache entries under the configured root."""
        if not self.root.is_dir():
            return ()
        entries: list[CacheEntryInfo] = []
        for source_dir in sorted(self.root.iterdir()):
            if not source_dir.is_dir() or source_dir.name.startswith("_"):
                continue
            for kind_dir in sorted(source_dir.iterdir()):
                match = _CACHE_KIND_PATTERN.fullmatch(kind_dir.name)
                if not kind_dir.is_dir() or match is None:
                    continue
                for artifact_dir in sorted(kind_dir.iterdir()):
                    if not artifact_dir.is_dir() or not _FINGERPRINT_PATTERN.fullmatch(artifact_dir.name):
                        continue
                    manifest_path = artifact_dir / CACHE_MANIFEST_NAME
                    complete_path = artifact_dir / CACHE_COMPLETE_NAME
                    if not manifest_path.is_file() or not complete_path.is_file():
                        continue
                    entries.append(
                        CacheEntryInfo(
                            source_fingerprint=source_dir.name,
                            kind=match.group("kind"),
                            format_version=int(match.group("version")),
                            artifact_fingerprint=artifact_dir.name,
                            path=artifact_dir,
                            size_bytes=sum(path.stat().st_size for path in artifact_dir.rglob("*") if path.is_file()),
                        )
                    )
        return tuple(entries)

    def clear(
        self,
        *,
        source_fingerprint: str | None = None,
        kind: str | None = None,
    ) -> int:
        """Remove matching completed cache entries and return the removal count."""
        removed = 0
        for info in self.list_entries():
            if source_fingerprint is not None and info.source_fingerprint != source_fingerprint:
                continue
            if kind is not None and info.kind != kind:
                continue
            self._remove_path(info.path)
            removed += 1
        return removed

    def _ensure_source_manifest(self, source: SourceFingerprint) -> None:
        source_manifest = source.manifest
        if fingerprint_payload(source_manifest) != source.value:
            msg = f"[InvalidSourceFingerprint] value={source.value}"
            raise ValueError(msg)
        source_dir = self.root / source.value
        source_dir.mkdir(parents=True, exist_ok=True)
        destination = source_dir / SOURCE_MANIFEST_NAME
        payload = {
            "schema_version": CACHE_MANIFEST_SCHEMA_VERSION,
            "source_fingerprint": source.value,
            "source": source_manifest,
        }
        if destination.is_file():
            try:
                existing = json.loads(destination.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = None
            if existing == payload:
                return
            if existing is not None:
                msg = f"[CacheSourceFingerprintCollision] conflicting source manifest at {destination}"
                raise RuntimeError(msg)
            destination.unlink(missing_ok=True)
        temporary = source_dir / f".{SOURCE_MANIFEST_NAME}.{uuid.uuid4().hex}.tmp"
        temporary.write_bytes(canonical_json_bytes(payload) + b"\n")
        try:
            temporary.replace(destination)
        except OSError:
            if not destination.is_file():
                raise
        finally:
            temporary.unlink(missing_ok=True)

    def _acquire_lock(self, entry: CacheEntry) -> _FileLease | None:
        return self._acquire_path_lock(entry.lock_path)

    @staticmethod
    def _acquire_path_lock(lock_path: Path) -> _FileLease | None:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+b")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return None
        lease_id = uuid.uuid4().hex
        try:
            owner = {
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "created_ns": time.time_ns(),
                "lease_id": lease_id,
            }
            handle.seek(0)
            handle.truncate()
            handle.write(canonical_json_bytes(owner) + b"\n")
            handle.flush()
        except Exception:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
            raise
        return _FileLease(lease_id=lease_id, path=lock_path, handle=handle)

    @staticmethod
    def _release_lock(lease: _FileLease) -> None:
        CacheManager._release_path_lock(lease)

    @staticmethod
    def _release_path_lock(lease: _FileLease) -> None:
        if not lease.is_held:
            return
        try:
            fcntl.flock(lease.handle.fileno(), fcntl.LOCK_UN)
        finally:
            lease.released = True
            lease.handle.close()

    @staticmethod
    def _lock_is_owned(entry: CacheEntry, lease: _FileLease) -> bool:
        return entry.lock_path == lease.path and lease.is_held

    @staticmethod
    def _path_lock_is_owned(lock_path: Path, lease: _FileLease) -> bool:
        return lock_path == lease.path and lease.is_held

    @staticmethod
    def _normalize_partition_names(raw_parts: Sequence[str]) -> tuple[str, ...]:
        parts = tuple(raw_parts)
        for part in parts:
            if (
                not isinstance(part, str)
                or not _PARTITION_NAME_PATTERN.fullmatch(part)
                or part in {CACHE_MANIFEST_NAME, CACHE_COMPLETE_NAME}
            ):
                msg = f"[InvalidCachePartName] part={part!r}"
                raise ValueError(msg)
        if len(set(parts)) != len(parts):
            msg = "[DuplicateCachePartName] cache part names must be unique"
            raise ValueError(msg)
        return parts

    def _ensure_partition_attempt(
        self,
        entry: CacheEntry,
        parts: tuple[str, ...],
        deadline: float,
    ) -> _PartitionAttempt | None:
        while True:
            if self.is_valid(entry):
                return None
            lease = self._acquire_lock(entry)
            if lease is not None:
                try:
                    if self.is_valid(entry):
                        return None
                    self._discard_invalid_entry(entry)
                    attempt = self._read_partition_attempt(entry, parts)
                    return attempt if attempt is not None else self._create_partition_attempt(entry, parts)
                finally:
                    self._release_lock(lease)
            if time.monotonic() >= deadline:
                msg = f"[CacheBuildTimeout] timed out initializing partitioned build for {entry.path}"
                raise TimeoutError(msg)
            time.sleep(min(self.config.poll_interval_seconds, 0.01))

    def _read_partition_attempt(
        self,
        entry: CacheEntry,
        expected_parts: tuple[str, ...],
    ) -> _PartitionAttempt | None:
        active_path = entry.temporary_parent / _PARTITION_ACTIVE_NAME
        try:
            active = json.loads(active_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(active, dict) or active.get("schema_version") != _PARTITION_SCHEMA_VERSION:
            return None
        generation = active.get("generation")
        created_ns = active.get("created_ns")
        raw_parts = active.get("parts")
        if not isinstance(generation, str) or not re.fullmatch(r"[0-9a-f]{32}", generation):
            return None
        if not isinstance(created_ns, int):
            return None
        if not isinstance(raw_parts, list) or not all(isinstance(part, str) for part in raw_parts):
            return None
        if tuple(raw_parts) != expected_parts:
            msg = f"[CachePartitionPlanMismatch] active plan differs for {entry.path}"
            raise RuntimeError(msg)
        attempt = _PartitionAttempt(
            generation=generation,
            created_ns=created_ns,
            parts=expected_parts,
            root=entry.temporary_parent / f"partition-{generation}",
        )
        required_paths = (
            attempt.payload_path,
            attempt.results_path,
            attempt.locks_path,
            attempt.temporary_path,
        )
        return attempt if all(path.is_dir() for path in required_paths) else None

    def _create_partition_attempt(
        self,
        entry: CacheEntry,
        parts: tuple[str, ...],
    ) -> _PartitionAttempt:
        entry.temporary_parent.mkdir(parents=True, exist_ok=True)
        generation = uuid.uuid4().hex
        created_ns = time.time_ns()
        attempt = _PartitionAttempt(
            generation=generation,
            created_ns=created_ns,
            parts=parts,
            root=entry.temporary_parent / f"partition-{generation}",
        )
        for path in (
            attempt.payload_path,
            attempt.results_path,
            attempt.locks_path,
            attempt.temporary_path,
        ):
            path.mkdir(parents=True, exist_ok=False)
        active = {
            "schema_version": _PARTITION_SCHEMA_VERSION,
            "generation": generation,
            "created_ns": created_ns,
            "parts": list(parts),
        }
        self._write_json_atomic(entry.temporary_parent / _PARTITION_ACTIVE_NAME, active)
        for path in entry.temporary_parent.glob("partition-*"):
            if path != attempt.root:
                self._remove_path(path)
        return attempt

    def _run_partition_attempt(
        self,
        entry: CacheEntry,
        attempt: _PartitionAttempt,
        ordered_parts: tuple[str, ...],
        build: Callable[[str, Path], CacheBuildResult],
        deadline: float,
    ) -> CacheEntry | None:
        join_remaining = attempt.created_ns / 1_000_000_000 + _PARTITION_JOIN_SECONDS - time.time()
        if join_remaining > 0:
            time.sleep(join_remaining)
        while True:
            if self.is_valid(entry):
                return entry
            if not self._partition_attempt_is_active(entry, attempt):
                return None

            results = tuple(self._read_partition_result(attempt, part) for part in attempt.parts)
            if all(result is not None for result in results):
                completed = self._publish_partition_attempt(entry, attempt)
                if completed is not None:
                    return completed

            made_progress = False
            for part in ordered_parts:
                if self._read_partition_result(attempt, part) is not None:
                    continue
                lock_path = attempt.locks_path / f"{part}.lock"
                lease = self._acquire_path_lock(lock_path)
                if lease is not None:
                    try:
                        if (
                            not self.is_valid(entry)
                            and self._partition_attempt_is_active(entry, attempt)
                            and self._read_partition_result(attempt, part) is None
                        ):
                            self._build_partition_part(entry, attempt, part, build, lease)
                    finally:
                        self._release_path_lock(lease)
                    made_progress = True
                    break
            if made_progress:
                continue
            if time.monotonic() >= deadline:
                msg = f"[CacheBuildTimeout] timed out waiting for partitioned build at {entry.path}"
                raise TimeoutError(msg)
            time.sleep(self.config.poll_interval_seconds)

    def _build_partition_part(
        self,
        entry: CacheEntry,
        attempt: _PartitionAttempt,
        part: str,
        build: Callable[[str, Path], CacheBuildResult],
        lease: _FileLease,
    ) -> None:
        result_path = attempt.results_path / f"{part}.json"
        destination = attempt.payload_path / part
        result_path.unlink(missing_ok=True)
        self._remove_path(destination)
        temporary_parent = attempt.temporary_path / part
        temporary_parent.mkdir(parents=True, exist_ok=True)
        for path in temporary_parent.iterdir():
            self._remove_path(path)
        temporary = temporary_parent / f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex}"
        temporary.mkdir()
        try:
            result = build(part, temporary)
            files = self._validate_build_result(temporary, result)
            result_payload = {
                "schema_version": _PARTITION_SCHEMA_VERSION,
                "part": part,
                "metadata": json.loads(canonical_json_bytes(result.metadata)),
                "files": files,
            }
            if not self._partition_attempt_is_active(entry, attempt):
                msg = f"[CachePartGenerationLost] active generation changed for {entry.path}"
                raise RuntimeError(msg)
            if not self._path_lock_is_owned(lease.path, lease):
                msg = f"[CachePartLeaseLost] build lease was released for part {part!r}"
                raise RuntimeError(msg)
            temporary.replace(destination)
            self._write_json_atomic(result_path, result_payload)
        finally:
            self._remove_path(temporary)

    def _read_partition_result(self, attempt: _PartitionAttempt, part: str) -> dict[str, Any] | None:
        result_path = attempt.results_path / f"{part}.json"
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(result, dict) or result.get("schema_version") != _PARTITION_SCHEMA_VERSION:
            return None
        if result.get("part") != part or not isinstance(result.get("metadata"), dict):
            return None
        files = result.get("files")
        if not isinstance(files, list):
            return None
        part_root = attempt.payload_path / part
        if not all(self._manifest_file_is_valid(part_root, item) for item in files):
            return None
        return result

    def _publish_partition_attempt(
        self,
        entry: CacheEntry,
        attempt: _PartitionAttempt,
    ) -> CacheEntry | None:
        lease = self._acquire_lock(entry)
        if lease is None:
            return None
        part_leases: list[_FileLease] = []
        remove_temporary_parent = False
        try:
            if self.is_valid(entry):
                return entry
            if not self._partition_attempt_is_active(entry, attempt):
                return None
            for part in attempt.parts:
                lock_path = attempt.locks_path / f"{part}.lock"
                part_lease = self._acquire_path_lock(lock_path)
                if part_lease is None:
                    return None
                part_leases.append(part_lease)
            results = tuple(self._read_partition_result(attempt, part) for part in attempt.parts)
            if any(result is None for result in results):
                return None

            files: list[dict[str, object]] = []
            part_metadata: list[dict[str, object]] = []
            for part, result in zip(attempt.parts, results, strict=True):
                assert result is not None
                part_files = []
                for item in result["files"]:
                    relative_path = f"{part}/{item['path']}"
                    files.append({"path": relative_path, "size": item["size"]})
                    part_files.append(relative_path)
                part_metadata.append(
                    {
                        "name": part,
                        "files": part_files,
                        "metadata": result["metadata"],
                    }
                )
            manifest = {
                "schema_version": CACHE_MANIFEST_SCHEMA_VERSION,
                "source_fingerprint": entry.key.source_fingerprint,
                "kind": entry.key.kind,
                "format_version": entry.key.format_version,
                "artifact_fingerprint": entry.key.artifact_fingerprint,
                "parameters": entry.key.parameters,
                "metadata": {"parts": part_metadata},
                "files": files,
            }
            (attempt.payload_path / CACHE_MANIFEST_NAME).write_bytes(canonical_json_bytes(manifest) + b"\n")
            (attempt.payload_path / CACHE_COMPLETE_NAME).write_text("complete\n", encoding="utf-8")
            if not self._partition_attempt_is_active(entry, attempt):
                return None
            if not self._lock_is_owned(entry, lease):
                return None
            self._discard_invalid_entry(entry)
            entry.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                attempt.payload_path.replace(entry.path)
            except OSError:
                if not self.is_valid(entry):
                    raise
            if not self.is_valid(entry):
                msg = f"[InvalidPublishedCache] cache publication failed validation at {entry.path}"
                raise RuntimeError(msg)
            if self._partition_attempt_is_active(entry, attempt):
                remove_temporary_parent = True
            return entry
        finally:
            for part_lease in part_leases:
                self._release_path_lock(part_lease)
            self._release_lock(lease)
            if remove_temporary_parent:
                self._remove_path(entry.temporary_parent)

    @staticmethod
    def _partition_attempt_is_active(entry: CacheEntry, attempt: _PartitionAttempt) -> bool:
        try:
            active = json.loads((entry.temporary_parent / _PARTITION_ACTIVE_NAME).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return isinstance(active, dict) and active.get("generation") == attempt.generation

    @staticmethod
    def _write_json_atomic(destination: Path, payload: Mapping[str, Any]) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_bytes(canonical_json_bytes(dict(payload)) + b"\n")
        try:
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    def _build_entry(
        self,
        entry: CacheEntry,
        build: Callable[[Path], CacheBuildResult],
        lease: _FileLease,
    ) -> CacheEntry:
        self._discard_invalid_entry(entry)
        self._remove_temporary_directories(entry)
        entry.temporary_parent.mkdir(parents=True, exist_ok=True)
        temporary = entry.temporary_parent / f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex}"
        temporary.mkdir()
        try:
            result = build(temporary)
            files = self._validate_build_result(temporary, result)
            manifest = {
                "schema_version": CACHE_MANIFEST_SCHEMA_VERSION,
                "source_fingerprint": entry.key.source_fingerprint,
                "kind": entry.key.kind,
                "format_version": entry.key.format_version,
                "artifact_fingerprint": entry.key.artifact_fingerprint,
                "parameters": entry.key.parameters,
                "metadata": result.metadata,
                "files": files,
            }
            (temporary / CACHE_MANIFEST_NAME).write_bytes(canonical_json_bytes(manifest) + b"\n")
            (temporary / CACHE_COMPLETE_NAME).write_text("complete\n", encoding="utf-8")
            if not self._lock_is_owned(entry, lease):
                if self.is_valid(entry):
                    return entry
                msg = f"[CacheBuildLeaseLost] build lease was released for {entry.path}"
                raise RuntimeError(msg)
            entry.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                temporary.replace(entry.path)
            except OSError:
                if not self.is_valid(entry):
                    raise
            if not self.is_valid(entry):
                msg = f"[InvalidPublishedCache] cache publication failed validation at {entry.path}"
                raise RuntimeError(msg)
            return entry
        finally:
            self._remove_path(temporary)

    def _discard_invalid_entry(self, entry: CacheEntry) -> None:
        if not entry.path.exists() or self.is_valid(entry):
            return
        entry.temporary_parent.mkdir(parents=True, exist_ok=True)
        invalid_path = entry.temporary_parent / f"invalid-{uuid.uuid4().hex}"
        try:
            entry.path.replace(invalid_path)
        except FileNotFoundError:
            return
        self._remove_path(invalid_path)

    def _remove_temporary_directories(self, entry: CacheEntry) -> None:
        if not entry.temporary_parent.is_dir():
            return
        for path in entry.temporary_parent.iterdir():
            self._remove_path(path)

    @staticmethod
    def _remove_path(path: Path) -> None:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)

    @staticmethod
    def _validate_build_result(root: Path, result: CacheBuildResult) -> list[dict[str, object]]:
        if not isinstance(result, CacheBuildResult):
            msg = "[InvalidCacheBuildResult] cache builders must return CacheBuildResult"
            raise TypeError(msg)
        files: list[dict[str, object]] = []
        for relative_path in result.files:
            relative = Path(relative_path)
            if relative.is_absolute() or ".." in relative.parts:
                msg = f"[InvalidCacheFilePath] path={relative_path!r}"
                raise ValueError(msg)
            path = root / relative
            if not path.is_file():
                msg = f"[MissingCacheFile] path={path}"
                raise RuntimeError(msg)
            files.append({"path": relative.as_posix(), "size": path.stat().st_size})
        canonical_json_bytes(result.metadata)
        return files

    @staticmethod
    def _manifest_file_is_valid(root: Path, item: object) -> bool:
        if not isinstance(item, dict):
            return False
        relative_path = item.get("path")
        expected_size = item.get("size")
        if not isinstance(relative_path, str) or not isinstance(expected_size, int):
            return False
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            return False
        path = root / relative
        try:
            return path.is_file() and path.stat().st_size == expected_size
        except OSError:
            return False
