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

TarUriRefFieldSpec = tuple[str, PathLikeStr]
"""JSONL tar-reference resolver spec as ``(field_name, base_dir)``."""

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
        """Consume one upstream sample and yield any completed outputs.

        Args:
            sample: Input sample consumed by the assembler.

        Returns:
            Completed outputs produced after consuming the sample."""

    def finish(self, *, drop_last: bool = False) -> Iterable[U]:
        """Flush remaining state at end of stream.

        Args:
            drop_last: Whether to discard the final incomplete batch.

        Returns:
            Remaining outputs produced during final flush."""


class Consumer(Protocol):
    """Terminal stream consumer that returns a final result."""

    def push(self, item: object) -> bool | None:
        """Consume one pipeline output.

        Args:
            item: Output item yielded by the dataset pipeline.

        Returns:
            False to stop consuming early. True or None to continue."""
        ...

    def finish(self) -> object:
        """Return the final consume result.

        Returns:
            User-defined result produced after consumption stops."""
        ...


@runtime_checkable
class StatefulAssembler(Protocol):
    """Assembler that can persist and restore its internal state."""

    def push(self, sample: object) -> Iterable[object]:
        """Consume one upstream sample and return completed outputs.

        Args:
            sample: Input sample consumed by the assembler.

        Returns:
            Completed outputs produced after consuming the sample."""
        ...

    def finish(self, *, drop_last: bool = False) -> Iterable[object]:
        """Flush pending assembler state at end of input.

        Args:
            drop_last: Whether to discard the final incomplete batch.

        Returns:
            Remaining outputs produced during final flush."""
        ...

    def state_dict(self) -> dict[str, object]:
        """Return the resumable assembler state.

        Returns:
            A dictionary that can be passed to load_state_dict()."""
        ...

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore the assembler from a resumable state dictionary.

        Args:
            state: Resume state dictionary to validate and load.

        Returns:
            None."""
        ...

    def fingerprint(self) -> str:
        """Return a stable fingerprint for resume compatibility checks.

        Returns:
            A stable fingerprint string."""
        ...


SourceKind = Literal["jsonl", "tar", "parquet", "lance"]
