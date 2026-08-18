"""Select stage."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from ..resume import ResumeStateError


@dataclass(frozen=True, slots=True)
class _SelectStage:
    """Stage configuration that projects dictionary samples to selected fields."""

    fields: tuple[str, ...]

    def __call__(self, data: Iterable[object]) -> Iterable[object]:
        """Apply this callable object."""
        return _SelectStageIterator(upstream=data, fields=self.fields)

    def identity(self) -> dict[str, object]:
        """Return a process-stable identity for this stage."""
        return {"kind": "select", "fields": list(self.fields)}


class _SelectStageIterator:
    """Live select iterator with empty resumable state."""

    def __init__(self, *, upstream: Iterable[object], fields: tuple[str, ...]) -> None:
        self.upstream = iter(upstream)
        self.fields = fields
        self._selected = set(fields)

    def __iter__(self) -> Iterator[object]:
        return self

    def __next__(self) -> object:
        sample = next(self.upstream)
        if not isinstance(sample, dict):
            raise TypeError(f"select() expects dict samples, got {type(sample)!r}")
        return {
            key: value
            for key, value in sample.items()
            if key in self._selected or (key.startswith("__") and key.endswith("__"))
        }

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state != {}:
            raise ResumeStateError("[InvalidResumeState] select stage state must be empty")
