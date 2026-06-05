"""TAR sidecar helpers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from mvp_dataset.core.resume import callable_fingerprint
from mvp_dataset.core.types import SidecarSpec


def _sidecar_fingerprint(sidecars: tuple[SidecarSpec, ...], shards: Sequence[str]) -> list[dict[str, object]]:
    """Return a fingerprint for tar sidecar metadata files."""
    return [
        {
            "name": name,
            "resolver": callable_fingerprint(resolver),
            "shards": [
                {
                    "path": sidecar_path,
                    "mtime_ns": stat.st_mtime_ns,
                    "size": stat.st_size,
                }
                for shard in shards
                for sidecar_path in (str(resolver(shard)),)
                for stat in (Path(sidecar_path).stat(),)
            ],
        }
        for name, resolver in sidecars
    ]
