from __future__ import annotations

import multiprocessing
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from mvp_dataset import CacheConfig, clear_cache, list_cache_entries
from mvp_dataset.cache import (
    CacheBuildResult,
    CacheKey,
    CacheManager,
    SourceFingerprint,
    fingerprint_local_files,
)
from mvp_dataset.cache.fingerprint import fingerprint_source_manifest
from mvp_dataset.sources.jsonl.sharding import split_jsonl_files
from mvp_dataset.sources.lance.cache import fingerprint_lance_source
from mvp_dataset.sources.lance.types import LanceDatasetSpec


def _cache_config(root: Path) -> CacheConfig:
    return CacheConfig(
        root=root,
        wait_timeout_seconds=2,
        poll_interval_seconds=0.01,
    )


def _source():
    return fingerprint_source_manifest(
        {
            "schema_version": 1,
            "kind": "test",
            "files": [{"path": "source.bin", "size": 1, "mtime_ns": 1}],
        }
    )


def _write_payload(root: Path, payload: bytes = b"value") -> CacheBuildResult:
    (root / "payload.bin").write_bytes(payload)
    return CacheBuildResult.from_files(["payload.bin"])


def _hold_cache_lock_until_killed(cache_root: str, ready) -> None:
    manager = CacheManager(_cache_config(Path(cache_root)))
    source = _source()
    key = CacheKey.create(
        source_fingerprint=source.value,
        kind="test-artifact",
        format_version=1,
        parameters={},
    )
    entry = manager.entry(key)
    lease = manager._acquire_lock(entry)
    if lease is None:
        ready.put(False)
        return
    partial = entry.temporary_parent / "crashed-build"
    partial.mkdir(parents=True)
    (partial / "partial.bin").write_bytes(b"partial")
    ready.put(True)
    time.sleep(60)


def test_source_fingerprint_ignores_mount_prefix(tmp_path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_file = source_root / "shard.jsonl"
    source_file.write_text("{}\n", encoding="utf-8")
    mount_a = tmp_path / "mount-a"
    mount_b = tmp_path / "mount-b"
    mount_a.symlink_to(source_root, target_is_directory=True)
    mount_b.symlink_to(source_root, target_is_directory=True)

    first = fingerprint_local_files([mount_a / source_file.name])
    second = fingerprint_local_files([mount_b / source_file.name])

    assert first.value == second.value
    assert first.manifest["files"] == second.manifest["files"]
    assert str(tmp_path) not in str(first.manifest)


def test_content_fingerprint_ignores_mtime_changes(tmp_path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    first = fingerprint_local_files([source], mode="content")
    changed_time = source.stat().st_mtime + 10
    os.utime(source, (changed_time, changed_time))

    second = fingerprint_local_files([source], mode="content")

    assert first == second


def test_source_fingerprint_manifest_is_immutable(tmp_path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    fingerprint = fingerprint_local_files([source], mode="content")

    manifest = fingerprint.manifest
    manifest["files"][0]["size"] = 999

    assert fingerprint.manifest["files"][0]["size"] == 3
    with pytest.raises(ValueError, match=r"\[InvalidSourceFingerprint\]"):
        SourceFingerprint(value="0" * 64, manifest=fingerprint.manifest)


def test_lance_source_fingerprint_ignores_mount_prefix(tmp_path) -> None:
    source_root = tmp_path / "source"
    version_dir = source_root / "dataset.lance" / "_versions"
    version_dir.mkdir(parents=True)
    (version_dir / f"{(1 << 64) - 1 - 7}.manifest").write_bytes(b"lance-manifest")
    mount_a = tmp_path / "mount-a"
    mount_b = tmp_path / "mount-b"
    mount_a.symlink_to(source_root, target_is_directory=True)
    mount_b.symlink_to(source_root, target_is_directory=True)

    first = fingerprint_lance_source(
        (LanceDatasetSpec(str(mount_a / "dataset.lance"), num_rows=5, row_offset=0, version=7),)
    )
    second = fingerprint_lance_source(
        (LanceDatasetSpec(f"file://{mount_b / 'dataset.lance'}", num_rows=5, row_offset=0, version=7),)
    )

    assert first == second
    assert str(tmp_path) not in str(first.manifest)
    assert first.manifest["datasets"][0]["manifest"]["name"].endswith(".manifest")


def test_cache_manager_reuses_completed_entry(tmp_path) -> None:
    manager = CacheManager(_cache_config(tmp_path / "cache"))
    calls = 0

    def build(root: Path) -> CacheBuildResult:
        nonlocal calls
        calls += 1
        return _write_payload(root)

    parameters = {"columns": ("id", "value")}
    first = manager.ensure(source=_source(), kind="test-artifact", format_version=1, parameters=parameters, build=build)
    second = manager.ensure(
        source=_source(), kind="test-artifact", format_version=1, parameters=parameters, build=build
    )

    assert first.path == second.path
    assert calls == 1
    assert first.path.parts[-3] == _source().value
    assert first.path.parts[-2] == "test-artifact-v1"
    assert first.key.parameters == {"columns": ["id", "value"]}
    assert first.complete_path.is_file()
    assert manager.is_valid(first)


@pytest.mark.parametrize("format_version", [True, 1.5, "1"])
def test_cache_key_requires_integer_format_version(format_version) -> None:
    with pytest.raises(ValueError, match=r"\[InvalidCacheFormatVersion\]"):
        CacheKey.create(
            source_fingerprint=_source().value,
            kind="test-artifact",
            format_version=format_version,
            parameters={},
        )


def test_cache_manager_does_not_publish_failed_build(tmp_path) -> None:
    manager = CacheManager(_cache_config(tmp_path / "cache"))

    def fail(root: Path) -> CacheBuildResult:
        (root / "partial.bin").write_bytes(b"partial")
        raise RuntimeError("build failed")

    with pytest.raises(RuntimeError, match="build failed"):
        manager.ensure(source=_source(), kind="test-artifact", format_version=1, parameters={}, build=fail)

    assert manager.list_entries() == ()
    entry = manager.ensure(
        source=_source(),
        kind="test-artifact",
        format_version=1,
        parameters={},
        build=_write_payload,
    )
    assert manager.is_valid(entry)
    assert not (entry.path / "partial.bin").exists()


def test_cache_manager_recovers_lock_and_partial_directory_after_sigkill(tmp_path) -> None:
    manager = CacheManager(_cache_config(tmp_path / "cache"))
    source = _source()
    key = CacheKey.create(
        source_fingerprint=source.value,
        kind="test-artifact",
        format_version=1,
        parameters={},
    )
    entry = manager.entry(key)
    context = multiprocessing.get_context("spawn")
    ready = context.Queue()
    process = context.Process(target=_hold_cache_lock_until_killed, args=(str(manager.root), ready))
    process.start()
    assert ready.get(timeout=10) is True
    process.kill()
    process.join(timeout=10)
    assert process.exitcode is not None and process.exitcode < 0

    recovered = manager.ensure(
        source=source,
        kind="test-artifact",
        format_version=1,
        parameters={},
        build=_write_payload,
    )

    assert manager.is_valid(recovered)
    assert not (entry.temporary_parent / "crashed-build").exists()
    assert entry.lock_path.is_file()


def test_released_lease_cannot_unlock_current_lease(tmp_path) -> None:
    manager = CacheManager(_cache_config(tmp_path / "cache"))
    key = CacheKey.create(
        source_fingerprint=_source().value,
        kind="test-artifact",
        format_version=1,
        parameters={},
    )
    entry = manager.entry(key)
    previous_lease = manager._acquire_lock(entry)
    assert previous_lease is not None
    assert manager._acquire_lock(entry) is None
    manager._release_lock(previous_lease)

    current_lease = manager._acquire_lock(entry)
    assert current_lease is not None
    manager._release_lock(previous_lease)

    assert manager._lock_is_owned(entry, current_lease)
    manager._release_lock(current_lease)
    assert entry.lock_path.is_file()


def test_cache_manager_serializes_concurrent_builds(tmp_path) -> None:
    manager = CacheManager(_cache_config(tmp_path / "cache"))
    calls = 0
    calls_lock = threading.Lock()

    def build(root: Path) -> CacheBuildResult:
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.1)
        return _write_payload(root)

    def ensure():
        return manager.ensure(
            source=_source(),
            kind="test-artifact",
            format_version=1,
            parameters={},
            build=build,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        entries = list(executor.map(lambda _: ensure(), range(2)))

    assert entries[0].path == entries[1].path
    assert calls == 1


def test_cache_manager_builds_parts_concurrently(tmp_path) -> None:
    manager = CacheManager(_cache_config(tmp_path / "cache"))
    part_names = tuple(f"part-{index}" for index in range(6))
    built_by: dict[str, int] = {}
    built_lock = threading.Lock()

    def ensure(worker: int):
        def build_part(part: str, root: Path) -> CacheBuildResult:
            with built_lock:
                assert part not in built_by
                built_by[part] = worker
            time.sleep(0.03)
            (root / "payload.bin").write_text(part, encoding="utf-8")
            return CacheBuildResult.from_files(["payload.bin"], metadata={"worker": worker})

        return manager.ensure(
            source=_source(),
            kind="partitioned-artifact",
            format_version=1,
            parameters={},
            parts=lambda: part_names,
            build=build_part,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        entries = list(executor.map(ensure, range(2)))

    assert entries[0].path == entries[1].path
    assert set(built_by) == set(part_names)
    assert set(built_by.values()) == {0, 1}
    assert all((entries[0].path / part / "payload.bin").is_file() for part in part_names)
    assert manager.is_valid(entries[0])
    assert not entries[0].temporary_parent.exists() or not any(entries[0].temporary_parent.iterdir())


def test_cache_manager_builds_only_assigned_parts(tmp_path) -> None:
    manager = CacheManager(_cache_config(tmp_path / "cache"))
    part_names = ("part-0", "part-1")
    built_by: dict[str, str] = {}
    built_lock = threading.Lock()

    def ensure(assigned_part: str):
        def build_part(part: str, root: Path) -> CacheBuildResult:
            with built_lock:
                built_by[part] = assigned_part
            (root / "payload.bin").write_text(part, encoding="utf-8")
            return CacheBuildResult.from_files(["payload.bin"])

        return manager.ensure(
            source=_source(),
            kind="assigned-partitioned-artifact",
            format_version=1,
            parameters={},
            parts=lambda: part_names,
            assigned_parts=(assigned_part,),
            build=build_part,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        entries = list(executor.map(ensure, part_names))

    assert entries[0].path == entries[1].path
    assert built_by == {"part-0": "part-0", "part-1": "part-1"}
    assert manager.is_valid(entries[0])


def test_cache_manager_resumes_failed_partitioned_build(tmp_path) -> None:
    manager = CacheManager(_cache_config(tmp_path / "cache"))
    calls = {part: 0 for part in ("part-0", "part-1", "part-2")}

    def build_part(part: str, root: Path) -> CacheBuildResult:
        calls[part] += 1
        (root / "payload.bin").write_text(part, encoding="utf-8")
        if part == "part-1" and calls[part] == 1:
            raise RuntimeError("part build failed")
        return CacheBuildResult.from_files(["payload.bin"])

    with pytest.raises(RuntimeError, match="part build failed"):
        manager.ensure(
            source=_source(),
            kind="partitioned-artifact",
            format_version=1,
            parameters={},
            parts=lambda: tuple(calls),
            build=build_part,
        )

    assert manager.list_entries() == ()
    entry = manager.ensure(
        source=_source(),
        kind="partitioned-artifact",
        format_version=1,
        parameters={},
        parts=lambda: tuple(calls),
        build=build_part,
    )

    assert calls == {"part-0": 1, "part-1": 2, "part-2": 1}
    assert manager.is_valid(entry)


def test_cache_manager_recovers_abandoned_partition_temporary_directory(tmp_path) -> None:
    manager = CacheManager(_cache_config(tmp_path / "cache"))
    source = _source()
    key = CacheKey.create(
        source_fingerprint=source.value,
        kind="partitioned-artifact",
        format_version=1,
        parameters={},
    )
    entry = manager.entry(key)
    manager._ensure_source_manifest(source)
    attempt = manager._ensure_partition_attempt(entry, ("part-0",), time.monotonic() + 1)
    assert attempt is not None
    lock_path = attempt.locks_path / "part-0.lock"
    lease = manager._acquire_path_lock(lock_path)
    assert lease is not None
    partial = attempt.temporary_path / "part-0" / "crashed"
    partial.mkdir(parents=True)
    (partial / "payload.bin").write_bytes(b"partial")
    manager._release_path_lock(lease)

    recovered = manager.ensure(
        source=source,
        kind="partitioned-artifact",
        format_version=1,
        parameters={},
        parts=lambda: ("part-0",),
        build=lambda _part, root: _write_payload(root),
    )

    assert manager.is_valid(recovered)
    assert not partial.exists()


@pytest.mark.parametrize("part", ["manifest.json", "complete"])
def test_cache_manager_rejects_reserved_partition_names(tmp_path, part: str) -> None:
    manager = CacheManager(_cache_config(tmp_path / "cache"))

    with pytest.raises(ValueError, match=r"\[InvalidCachePartName\]"):
        manager.ensure(
            source=_source(),
            kind="partitioned-artifact",
            format_version=1,
            parameters={},
            parts=lambda: (part,),
            build=lambda _part, root: _write_payload(root),
        )


def test_jsonl_splits_use_unified_cache(tmp_path, monkeypatch) -> None:
    source = tmp_path / "records.jsonl"
    source.write_text("".join(f'{{"id": {index}}}\n' for index in range(8)), encoding="utf-8")
    cache_root = tmp_path / "shared-cache"
    monkeypatch.setenv("MVP_DATASET_CACHE_DIR", str(cache_root))

    first = split_jsonl_files([str(source)], min_chunks=2)
    second = split_jsonl_files([str(source)], min_chunks=2)

    assert first == second
    assert len(first.shards) == 2
    assert all(Path(shard.physical_path).is_relative_to(cache_root) for shard in first.shards)
    assert [shard.logical_path for shard in first.shards] == ["records.jsonl", "records.jsonl"]
    assert [shard.line_start for shard in first.shards] == [0, 4]
    assert [shard.line_count for shard in first.shards] == [4, 4]
    assert not (tmp_path / ".chunks").exists()
    entries = list_cache_entries(cache_root)
    assert len(entries) == 1
    assert entries[0].kind == "jsonl-split"
    assert clear_cache(cache_root, kind="jsonl-split") == 1
    assert list_cache_entries(cache_root) == ()
