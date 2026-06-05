"""Unbatch stage."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from ..resume import ResumeStateError, stable_fingerprint


@dataclass(frozen=True, slots=True)
class _UnbatchStage:
    """Stage configuration that expands batch-like inputs into samples."""

    def __call__(self, data: Iterable[object]) -> Iterable[object]:
        """Apply this callable object."""
        return _UnbatchStageIterator(upstream=data)

    def fingerprint(self) -> str:
        """Return a stable fingerprint for resume compatibility checks."""
        return stable_fingerprint({"kind": "unbatch"})


class _UnbatchStageIterator:
    """Iterator that expands list, tuple, or dict batches into samples."""

    def __init__(self, *, upstream: Iterable[object]) -> None:
        """Initialize the object."""
        self.upstream = iter(upstream)
        self.pending: list[object] = []

    def __iter__(self) -> Iterator[object]:
        """Return the iterator object."""
        return self

    def __next__(self) -> object:
        """Return the next output item."""
        while not self.pending:
            self.pending.extend(self._expand_batch(next(self.upstream)))
        return self.pending.pop(0)

    def state_dict(self) -> dict[str, object]:
        """Return the resumable state for this object."""
        return {"pending": list(self.pending)}

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore this object from a resumable state dictionary."""
        pending = state.get("pending")
        if not isinstance(pending, list):
            msg = "[InvalidResumeState] unbatch stage pending must be a list"
            raise ResumeStateError(msg)
        self.pending = list(pending)

    def fingerprint(self) -> str:
        """Return a stable fingerprint for resume compatibility checks."""
        return stable_fingerprint({"kind": "unbatch"})

    def _expand_batch(self, batch: object) -> list[object]:
        """Expand one batch object into a list of samples."""
        if isinstance(batch, (list, tuple)):
            return list(batch)

        if isinstance(batch, dict):
            if not batch:
                return []
            if not all(isinstance(value, (list, tuple)) for value in batch.values()):
                msg = "dict batches must contain only list/tuple values"
                raise TypeError(msg)

            lengths = {len(value) for value in batch.values()}
            if len(lengths) != 1:
                msg = f"dict batch values must have equal lengths, got {sorted(lengths)}"
                raise ValueError(msg)

            return [{key: value[index] for key, value in batch.items()} for index in range(next(iter(lengths)))]

        msg = f"unsupported batch type: {type(batch)!r}"
        raise TypeError(msg)
