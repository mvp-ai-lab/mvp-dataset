"""JSONL source types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mvp_dataset.cache import SourceFingerprint

JsonlShuffleMode = Literal["none", "shard_aware", "global"]


@dataclass(frozen=True, slots=True)
class JsonlShard:
    """Physical JSONL shard location with a mount-independent logical identity."""

    physical_path: str
    source_index: int
    logical_path: str
    line_start: int
    line_count: int | None


@dataclass(frozen=True, slots=True)
class JsonlSplitPlan:
    """Prepared JSONL source identity and its physical read shards."""

    source: SourceFingerprint
    shards: tuple[JsonlShard, ...]
