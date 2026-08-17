"""Map stage."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

from ..resume import ResumeStateError, identity


@dataclass(frozen=True, slots=True)
class _MapStage:
    """Stage configuration that maps each upstream sample with a callable."""

    fn: Callable[[object], object]

    def __call__(self, data: Iterable[object]) -> Iterable[object]:
        """Apply this callable object."""
        return _MapStageIterator(upstream=data, fn=self.fn)

    def identity(self) -> dict[str, object]:
        """Return a process-stable identity for this stage."""
        return {"kind": "map", "fn": identity(self.fn)}


class _MapStageIterator:
    """Live map iterator with empty resumable state."""

    def __init__(self, *, upstream: Iterable[object], fn: Callable[[object], object]) -> None:
        self.upstream = iter(upstream)
        self.fn = fn

    def __iter__(self) -> Iterator[object]:
        return self

    def __next__(self) -> object:
        return self.fn(next(self.upstream))

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state != {}:
            raise ResumeStateError("[InvalidResumeState] map stage state must be empty")
