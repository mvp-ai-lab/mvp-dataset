"""Unified persistent cache management."""

from pathlib import Path

from .config import (
    CACHE_DIR_ENV,
    CACHE_FINGERPRINT_MODE_ENV,
    CACHE_WAIT_TIMEOUT_SECONDS_ENV,
    CacheConfig,
    FingerprintMode,
)
from .fingerprint import (
    SourceFingerprint,
    fingerprint_local_files,
    fingerprint_payload,
    fingerprint_source_manifest,
)
from .manager import (
    CacheBuildResult,
    CacheEntry,
    CacheEntryInfo,
    CacheKey,
    CacheManager,
)


def list_cache_entries(cache_dir: str | Path | None = None) -> tuple[CacheEntryInfo, ...]:
    """List completed cache entries under an explicit or resolved root."""
    return CacheManager(CacheConfig.resolve(cache_dir)).list_entries()


def clear_cache(
    cache_dir: str | Path | None = None,
    *,
    source_fingerprint: str | None = None,
    kind: str | None = None,
) -> int:
    """Remove completed cache entries matching the optional filters."""
    manager = CacheManager(CacheConfig.resolve(cache_dir))
    return manager.clear(source_fingerprint=source_fingerprint, kind=kind)


__all__ = [
    "CACHE_DIR_ENV",
    "CACHE_FINGERPRINT_MODE_ENV",
    "CACHE_WAIT_TIMEOUT_SECONDS_ENV",
    "CacheBuildResult",
    "CacheConfig",
    "CacheEntry",
    "CacheEntryInfo",
    "CacheKey",
    "CacheManager",
    "FingerprintMode",
    "SourceFingerprint",
    "fingerprint_local_files",
    "fingerprint_payload",
    "fingerprint_source_manifest",
    "clear_cache",
    "list_cache_entries",
]
