"""Shared Lance source data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

LanceShuffleMode = Literal["none", "global", "chunk"]
LanceRefIndexBuildStrategy = Literal["auto", "in_memory", "bucketed"]
LanceRefIndexConfigInput = dict[str, object] | None


@dataclass(frozen=True, slots=True)
class LanceDatasetSpec:
    """Resolved metadata for one Lance dataset URI."""

    uri: str
    num_rows: int
    row_offset: int
    version: int | str | None = None
    handle: object | None = None


@dataclass(frozen=True, slots=True)
class LanceFilterIndex:
    """Disk-backed mapping from filtered positions to source row indexes."""

    paths: tuple[str, ...]
    offsets: tuple[int, ...]
    count: int


@dataclass(frozen=True, slots=True)
class LanceRefSpec:
    """Configuration for one Lance reference column.

    Attributes:
        column: Field the resolved values are written to. It need not be a column of the main table.
        uri: Reference dataset URI, or several read as one merged row space.
        key_column: Key column of the reference dataset.
        value_column: Value column of the reference dataset.
        ref_locator: Where the main table records its keys, as a dotted path whose first segment is
            the column, such as ``info.links.images.id``. A list on the way is walked, so one row may
            yield several keys, and the column itself is left intact so it can keep carrying whatever
            else it holds. Defaults to ``column``, taken whole: the case where a row names its
            references in the same field the resolved values replace.
    """

    column: str
    uri: str | tuple[str, ...]
    key_column: str
    value_column: str
    ref_locator: str | None = None
    datasets: tuple[LanceDatasetSpec, ...] = ()
    index_uri: str | None = None
    index_offsets_path: str | None = None
    index_entries_path: str | None = None
    index_handle: object | None = None


@dataclass(frozen=True, slots=True)
class LanceRefIndexConfig:
    """Reference index preparation configuration."""

    build_strategy: LanceRefIndexBuildStrategy | None = None
    bucket_count: int | None = None


@dataclass(frozen=True, slots=True)
class LanceRefResolverConfig:
    """Reference resolver runtime configuration."""

    resolve_batch_size: int = 1024
    index: LanceRefIndexConfig = field(default_factory=LanceRefIndexConfig)


@dataclass(frozen=True, slots=True)
class LanceSource:
    """Merged Lance source metadata over one global row space."""

    datasets: tuple[LanceDatasetSpec, ...]
    ref_columns: tuple[LanceRefSpec, ...] = ()
    row_offsets: tuple[int, ...] = field(init=False)
    total_rows: int = field(init=False)

    def __post_init__(self) -> None:
        datasets = tuple(self.datasets)
        object.__setattr__(self, "datasets", datasets)
        object.__setattr__(self, "ref_columns", tuple(self.ref_columns))
        object.__setattr__(self, "row_offsets", tuple(dataset.row_offset for dataset in datasets))
        total_rows = 0 if not datasets else datasets[-1].row_offset + datasets[-1].num_rows
        object.__setattr__(self, "total_rows", total_rows)


@dataclass(frozen=True, slots=True)
class LanceSelection:
    """Row-subset window used by ``split()`` and ``sample()``.

    Attributes:
        start: Window start in source logical positions.
        count: Number of rows in the subset (the effective row space size).
        total: Total rows in the underlying source.
        seed: Optional seed for sample membership permutation.
    """

    start: int
    count: int
    total: int
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class LanceIndexItem:
    """Physical Lance row location used by the source iterator."""

    dataset_i: int
    local_index: int
    global_index: int
