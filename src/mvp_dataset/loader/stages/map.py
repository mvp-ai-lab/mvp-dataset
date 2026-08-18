"""Loader-side map stage."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ...core.resume import identity
from ...core.stages.map import _MapStageIterator


@dataclass(frozen=True, slots=True)
class _LoaderMapStage:
    """Stage configuration that maps each loader output with a callable."""

    fn: Callable[[object], object]

    kind = "map"

    def __call__(self, data: Iterable[object]) -> Iterable[object]:
        """Apply this callable object."""
        return _MapStageIterator(upstream=data, fn=self.fn)

    def identity(self) -> dict[str, object]:
        """Return a process-stable identity for this stage."""
        return {"kind": self.kind, "fn": identity(self.fn)}
