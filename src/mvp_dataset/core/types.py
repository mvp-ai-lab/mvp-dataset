"""Core data types used by mvp-dataset."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

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

StageKind = Literal["map", "select", "shuffle", "batch", "assemble", "unbatch"]
"""Recognized pipeline stage kinds."""


@dataclass(frozen=True, slots=True)
class StageSpec:
    """Metadata wrapper around one pipeline stage.

    Attributes:
        kind: Symbolic stage name (``"map"``, ``"shuffle"``, ``"batch"``, etc.).
        apply: Stage callable used during iteration.
    """

    kind: StageKind
    apply: Stage


class Assembler[T, U](Protocol):
    """Stateful stream assembler that may emit outputs after consuming inputs."""

    def push(self, sample: T) -> Iterable[U]:
        """Consume one upstream sample and yield any completed outputs."""

    def finish(self, *, drop_last: bool = False) -> Iterable[U]:
        """Flush remaining state at end of stream."""


@runtime_checkable
class StatefulAssembler(Protocol):
    """Assembler that can persist and restore its internal state."""

    def push(self, sample: object) -> Iterable[object]: ...

    def finish(self, *, drop_last: bool = False) -> Iterable[object]: ...

    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state: dict[str, object]) -> None: ...

    def fingerprint(self) -> str: ...


SourceKind = Literal["jsonl", "tars", "parquet", "lance"]
SourceStore = list[str] | list[object]
