"""JSONL source types."""

from __future__ import annotations

from typing import Literal

JsonlShuffleMode = Literal["none", "shard_aware", "global"]
