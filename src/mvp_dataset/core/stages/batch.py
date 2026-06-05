"""Batch stage."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

from ..resume import ResumeStateError, callable_fingerprint, stable_fingerprint


@dataclass(frozen=True, slots=True)
class _BatchStage:
    batch_size: int
    drop_last: bool = False
    collate_fn: Callable[[list[object]], object] | None = None

    def __call__(self, data: Iterable[object]) -> Iterable[object]:
        return _BatchStageIterator(
            upstream=data,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            collate_fn=self.collate_fn,
        )

    def fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "kind": "batch",
                "batch_size": self.batch_size,
                "drop_last": self.drop_last,
                "collate_fn": callable_fingerprint(self.collate_fn),
            }
        )


class _BatchStageIterator:
    def __init__(
        self,
        *,
        upstream: Iterable[object],
        batch_size: int,
        drop_last: bool,
        collate_fn: Callable[[list[object]], object] | None,
    ) -> None:
        if batch_size <= 0:
            msg = f"batch_size must be > 0, got {batch_size}"
            raise ValueError(msg)
        self.upstream = iter(upstream)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.pending: list[object] = []
        self.emitted = 0

    def __iter__(self) -> Iterator[object]:
        return self

    def __next__(self) -> object:
        while len(self.pending) < self.batch_size:
            try:
                self.pending.append(next(self.upstream))
            except StopIteration:
                if not self.pending or self.drop_last:
                    self.pending.clear()
                    raise
                break

        batch = list(self.pending)
        output = self.collate_fn(batch) if self.collate_fn is not None else batch
        self.pending.clear()
        self.emitted += 1
        return output

    def state_dict(self) -> dict[str, object]:
        return {
            "pending": list(self.pending),
            "emitted": self.emitted,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        pending = state.get("pending")
        if not isinstance(pending, list):
            msg = "[InvalidResumeState] batch stage pending must be a list"
            raise ResumeStateError(msg)
        if len(pending) >= self.batch_size:
            msg = "[InvalidResumeState] batch stage pending must be smaller than batch_size"
            raise ResumeStateError(msg)
        emitted = state.get("emitted")
        if not isinstance(emitted, int) or emitted < 0:
            msg = "[InvalidResumeState] batch stage emitted must be a non-negative integer"
            raise ResumeStateError(msg)
        self.pending = list(pending)
        self.emitted = emitted

    def fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "kind": "batch",
                "batch_size": self.batch_size,
                "drop_last": self.drop_last,
                "collate_fn": callable_fingerprint(self.collate_fn),
            }
        )
