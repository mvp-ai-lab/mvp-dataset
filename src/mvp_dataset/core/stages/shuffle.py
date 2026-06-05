"""Shuffle stage."""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from ..context import RuntimeContext
from ..resume import ResumeStateError, stable_fingerprint


@dataclass(frozen=True, slots=True)
class _ShuffleStage:
    """Stage configuration for deterministic bounded-buffer sample shuffling."""

    context: RuntimeContext
    buffer_size: int
    initial: int | None = None

    def __call__(self, data: Iterable[object]) -> Iterable[object]:
        """Apply this callable object."""
        runtime_context = RuntimeContext.from_runtime(base=self.context)
        return _ShuffleStageIterator(
            upstream=data,
            buffer_size=self.buffer_size,
            initial=self.initial,
            rng=random.Random(runtime_context.sample_shuffle_seed),
        )

    def fingerprint(self) -> str:
        """Return a stable fingerprint for resume compatibility checks."""
        return stable_fingerprint(
            {
                "kind": "shuffle",
                "buffer_size": self.buffer_size,
                "initial": self.initial,
            }
        )


class _ShuffleStageIterator:
    """Iterator that implements resumable bounded-buffer sample shuffling."""

    def __init__(
        self,
        *,
        upstream: Iterable[object],
        buffer_size: int,
        initial: int | None,
        rng: random.Random,
    ) -> None:
        """Initialize the object."""
        if buffer_size <= 0:
            msg = f"buffer_size must be > 0, got {buffer_size}"
            raise ValueError(msg)
        if initial is not None and initial <= 0:
            msg = f"initial must be > 0, got {initial}"
            raise ValueError(msg)

        self.upstream = iter(upstream)
        self.buffer_size = buffer_size
        self.initial_config = initial
        self.initial = min(buffer_size, buffer_size if initial is None else initial)
        self.rng = rng
        self.buffer: list[object] = []
        self.upstream_exhausted = False

    def __iter__(self) -> Iterator[object]:
        """Return the iterator object."""
        return self

    def __next__(self) -> object:
        """Return the next output item."""
        while not self.upstream_exhausted:
            self._read_one()
            if len(self.buffer) < self.buffer_size:
                self._read_one()
            if len(self.buffer) >= self.initial:
                return self._pop_random()

        if self.buffer:
            return self._pop_random()

        raise StopIteration

    def state_dict(self) -> dict[str, object]:
        """Return the resumable state for this object."""
        return {
            "buffer": list(self.buffer),
            "rng_state": self.rng.getstate(),
            "upstream_exhausted": self.upstream_exhausted,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore this object from a resumable state dictionary."""
        buffer = state.get("buffer")
        if not isinstance(buffer, list):
            msg = "[InvalidResumeState] shuffle stage buffer must be a list"
            raise ResumeStateError(msg)
        if len(buffer) > self.buffer_size:
            msg = "[InvalidResumeState] shuffle stage buffer cannot exceed buffer_size"
            raise ResumeStateError(msg)
        upstream_exhausted = state.get("upstream_exhausted")
        if not isinstance(upstream_exhausted, bool):
            msg = "[InvalidResumeState] shuffle stage upstream_exhausted must be a bool"
            raise ResumeStateError(msg)

        try:
            self.rng.setstate(state.get("rng_state"))
        except (TypeError, ValueError) as error:
            msg = "[InvalidResumeState] shuffle stage rng_state is invalid"
            raise ResumeStateError(msg) from error
        self.buffer = list(buffer)
        self.upstream_exhausted = upstream_exhausted

    def fingerprint(self) -> str:
        """Return a stable fingerprint for resume compatibility checks."""
        return stable_fingerprint(
            {
                "kind": "shuffle",
                "buffer_size": self.buffer_size,
                "initial": self.initial_config,
            }
        )

    def _read_one(self) -> None:
        """Read one upstream item into the shuffle buffer if available."""
        try:
            self.buffer.append(next(self.upstream))
        except StopIteration:
            self.upstream_exhausted = True

    def _pop_random(self) -> object:
        """Remove and return one random item from the shuffle buffer."""
        index = self.rng.randrange(len(self.buffer))
        picked = self.buffer[index]
        self.buffer[index] = self.buffer[-1]
        self.buffer.pop()
        return picked
