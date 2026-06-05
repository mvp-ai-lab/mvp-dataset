"""TAR source types."""

from __future__ import annotations

from typing import Literal

TarShuffleMode = Literal["none", "shard_aware", "global"]
