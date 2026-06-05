"""Parquet source types."""

from __future__ import annotations

from typing import Literal

ParquetShuffleMode = Literal["none", "fragment_aware", "global"]
