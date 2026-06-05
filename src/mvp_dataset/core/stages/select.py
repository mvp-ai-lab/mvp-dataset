"""Select stage."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ..resume import ResumeStateError, stable_fingerprint


@dataclass(frozen=True, slots=True)
class _SelectStage:
    """Stage configuration that projects dictionary samples to selected fields."""

    fields: tuple[str, ...]

    def __call__(self, data: Iterable[object]) -> Iterable[object]:
        """Apply this callable object."""
        selected = set(self.fields)
        for sample in data:
            if not isinstance(sample, dict):
                msg = f"select() expects dict samples, got {type(sample)!r}"
                raise TypeError(msg)
            yield {
                key: value
                for key, value in sample.items()
                if key in selected or (key.startswith("__") and key.endswith("__"))
            }

    def state_dict(self) -> dict[str, object]:
        """Return the resumable state for this object."""
        return {}

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore this object from a resumable state dictionary."""
        if state != {}:
            msg = "[InvalidResumeState] select stage state must be empty"
            raise ResumeStateError(msg)

    def fingerprint(self) -> str:
        """Return a stable fingerprint for resume compatibility checks."""
        return stable_fingerprint({"kind": "select", "fields": list(self.fields)})
