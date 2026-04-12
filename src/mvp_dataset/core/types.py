"""Core data types used by mvp-dataset."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal, Protocol

# ---------------------------------------------------------------------------
# Shared aliases
# ---------------------------------------------------------------------------

PathLikeStr = str | os.PathLike[str]
"""Path representation accepted by public APIs."""

Sample = dict[str, object]
"""In-memory sample mapping exchanged across pipeline stages."""

GroupedSample = list[Sample]
"""A group of samples sharing the same grouping key."""

ShardInput = PathLikeStr
"""Shard path input accepted by dataset constructors."""

PathResolver = Callable[[PathLikeStr], PathLikeStr]
"""Callable that maps a main shard path to a related shard path."""

SidecarSpec = tuple[str, PathResolver]
"""Sidecar join specification as ``(name, path_resolver)``."""

RefFieldSpec = tuple[str, PathLikeStr]
"""Reference resolver spec as ``(field_name, base_dir)``."""

Stage = Callable[[Iterable[object]], Iterable[object]]
"""One lazy transformation stage in the iterator pipeline."""

CacheLayout = Literal["single_dataset", "global_merge"]
"""Cache materialization layout for full-cache mode."""

CacheTracePolicy = Literal["traceable", "unsupported"]
"""Whether a stage participates in cache invalidation."""

StageKind = Literal["map", "select", "shuffle", "batch", "assemble", "unbatch"]
"""Recognized pipeline stage kinds."""


@dataclass(frozen=True, slots=True)
class StageSpec:
    """Metadata wrapper around one pipeline stage.

    Attributes:
        kind: Symbolic stage name (``"map"``, ``"shuffle"``, ``"batch"``, etc.).
        apply: Standard stage callable used during normal (non-cache) iteration.
        fn_fingerprint: Stable hash of the user-provided callable, used to
            detect when a stage's logic changes and the cache must be rebuilt.
            Empty string for unsupported stages.
        cache_trace_policy: Whether the stage participates in per-field
            signature tracking and cache invalidation.
        cache_stage: Optional cache-aware version of the stage that propagates
            ``__cache_meta__`` through the sample stream.  ``None`` for
            unsupported stages that use the regular ``apply`` callable during
            warm-up.
    """

    kind: StageKind
    apply: Stage
    fn_fingerprint: str | None = None
    fn_ref: Callable | None = None
    trace_policy: CacheTracePolicy | None = None


@dataclass(frozen=True, slots=True)
class CacheSpec:
    """Cache boundary descriptor attached to a :class:`~mvp_dataset.Dataset`.

    Attributes:
        boundary_index: Number of pre-cache :class:`StageSpec` entries.
        cache_dir: Root directory for standalone cache datasets.
        cache_num_workers: Number of worker threads used to build the full
            cache from independent source shards or fragments.
        plan_fingerprint: Stable hash of all pre-cache stage fingerprints
            combined with the source identity.
    """

    boundary_index: int
    cache_dir: str
    cache_num_workers: int
    plan_fingerprint: str


class Assembler[T, U](Protocol):
    """Stateful stream assembler that may emit outputs after consuming inputs."""

    def push(self, sample: T) -> Iterable[U]:
        """Consume one upstream sample and yield any completed outputs."""

    def finish(self, *, drop_last: bool = False) -> Iterable[U]:
        """Flush remaining state at end of stream."""


SourceKind = Literal["jsonl", "tars", "parquet", "lance"]
SourceStore = list[str] | list[object]
