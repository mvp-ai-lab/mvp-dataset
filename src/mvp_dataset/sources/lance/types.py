"""Shared Lance source data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

LanceShuffleMode = Literal["none", "global", "fragment_aware", "chunk_aware"]
LanceRefIndexScope = Literal["shared", "node_local", "process"]


@dataclass(frozen=True, slots=True)
class LanceDatasetSpec:
    """Resolved metadata for one Lance dataset URI."""

    uri: str
    num_rows: int
    row_offset: int
    fragment_ids: tuple[int, ...]
    fragment_row_counts: tuple[int, ...] = ()
    handle: object | None = None


@dataclass(frozen=True, slots=True)
class LanceRefSpec:
    """Configuration for one Lance reference column."""

    column: str
    uri: str | tuple[str, ...]
    key_column: str
    value_column: str
    index_uri: str | None = None
    index_offsets_path: str | None = None
    index_entries_path: str | None = None
    index_handle: object | None = None


@dataclass(frozen=True, slots=True)
class LanceSourceSpec:
    """One schedulable Lance source configuration."""

    datasets: list[LanceDatasetSpec]
    ref_columns: tuple[LanceRefSpec, ...] = ()

    @property
    def total_rows(self) -> int:
        """Return the total number of rows across Lance sources.

        Returns:
            Total rows across all Lance dataset specs."""
        return sum(dataset.num_rows for dataset in self.datasets)

    @property
    def total_fragments(self) -> int:
        """Return the total number of fragments across Lance sources.

        Returns:
            Total fragments across all Lance dataset specs."""
        return sum(len(dataset.fragment_ids) for dataset in self.datasets)


@dataclass(frozen=True, slots=True)
class LanceIndexItem:
    """Physical Lance row location used by the source iterator."""

    dataset_i: int
    local_index: int
    global_index: int
