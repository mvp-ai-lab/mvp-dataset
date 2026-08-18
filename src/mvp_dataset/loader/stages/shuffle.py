"""Loader-side shuffle stage."""

from __future__ import annotations

import random
from collections.abc import Iterable

from ...core.stages import _ShuffleStageIterator


class _LoaderShuffleStageIterator(_ShuffleStageIterator):
    """TorchLoader shuffle iterator."""

    def __init__(
        self,
        *,
        upstream: Iterable[object],
        buffer_size: int,
        initial: int | None,
        seed: int,
    ) -> None:
        """Initialize the object."""
        super().__init__(
            upstream=upstream,
            buffer_size=buffer_size,
            initial=initial,
            rng=random.Random(seed),
        )


class _LoaderShuffleStage:
    """TorchLoader stage configuration for output shuffling."""

    kind = "shuffle"

    def __init__(self, *, buffer_size: int, initial: int | None, seed: int) -> None:
        """Initialize the object."""
        self.buffer_size = buffer_size
        self.initial = initial
        self.seed = seed

    def __call__(self, data: Iterable[object]) -> Iterable[object]:
        """Apply this callable object."""
        return _LoaderShuffleStageIterator(
            upstream=data,
            buffer_size=self.buffer_size,
            initial=self.initial,
            seed=self.seed,
        )

    def identity(self) -> dict[str, object]:
        """Return a process-stable identity for this stage."""
        return {
            "kind": self.kind,
            "buffer_size": self.buffer_size,
            "initial": self.initial,
            "seed": self.seed,
        }
