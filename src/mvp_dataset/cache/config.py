"""Unified cache configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

CACHE_DIR_ENV = "MVP_DATASET_CACHE_DIR"
CACHE_FINGERPRINT_MODE_ENV = "MVP_DATASET_CACHE_FINGERPRINT_MODE"
FingerprintMode = Literal["metadata", "content"]


@dataclass(frozen=True, slots=True)
class CacheConfig:
    """Configuration shared by all persistent cache artifacts."""

    root: Path
    fingerprint_mode: FingerprintMode = "metadata"
    wait_timeout_seconds: float = 30 * 60
    poll_interval_seconds: float = 0.25

    def __post_init__(self) -> None:
        """Validate and normalize the configuration."""
        object.__setattr__(self, "root", Path(os.path.abspath(Path(self.root).expanduser())))
        if self.fingerprint_mode not in ("metadata", "content"):
            msg = f"[InvalidCacheFingerprintMode] mode={self.fingerprint_mode!r}"
            raise ValueError(msg)
        if self.wait_timeout_seconds <= 0:
            msg = "[InvalidCacheWaitTimeout] wait_timeout_seconds must be > 0"
            raise ValueError(msg)
        if self.poll_interval_seconds <= 0:
            msg = "[InvalidCachePollInterval] poll_interval_seconds must be > 0"
            raise ValueError(msg)

    @classmethod
    def resolve(
        cls,
        cache_dir: str | Path | None = None,
        *,
        fingerprint_mode: FingerprintMode | None = None,
    ) -> CacheConfig:
        """Resolve an explicit, environment, or platform-default cache root."""
        if cache_dir is not None:
            root = Path(cache_dir)
        elif raw_root := os.environ.get(CACHE_DIR_ENV):
            root = Path(raw_root)
        elif xdg_cache_home := os.environ.get("XDG_CACHE_HOME"):
            root = Path(xdg_cache_home) / "mvp-dataset"
        else:
            root = Path.home() / ".cache" / "mvp-dataset"
        resolved_mode = fingerprint_mode or os.environ.get(CACHE_FINGERPRINT_MODE_ENV, "metadata")
        return cls(root=root, fingerprint_mode=resolved_mode)
