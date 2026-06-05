"""Map stage."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ..resume import ResumeStateError, callable_fingerprint, stable_fingerprint


@dataclass(frozen=True, slots=True)
class _MapStage:
    """Stage configuration that maps each upstream sample with a callable."""

    fn: Callable[[object], object]

    def __call__(self, data: Iterable[object]) -> Iterable[object]:
        """Apply this callable object."""
        for sample in data:
            yield self.fn(sample)

    def state_dict(self) -> dict[str, object]:
        """Return the resumable state for this object."""
        return {}

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore this object from a resumable state dictionary."""
        if state != {}:
            msg = "[InvalidResumeState] map stage state must be empty"
            raise ResumeStateError(msg)

    def fingerprint(self) -> str:
        """Return a stable fingerprint for resume compatibility checks."""
        return stable_fingerprint({"kind": "map", "fn": callable_fingerprint(self.fn)})
