"""Loader-side batch stage."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from ...core.resume import identity
from ...core.stages import _BatchStageIterator


class _LoaderBatchStage:
    """TorchLoader stage configuration for output batching."""

    kind = "batch"

    def __init__(
        self,
        *,
        batch_size: int,
        drop_last: bool,
        collate_fn: Callable[[list[object]], object] | None,
    ) -> None:
        """Initialize the object."""
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.collate_fn = collate_fn

    def __call__(self, data: Iterable[object]) -> Iterable[object]:
        """Apply this callable object."""
        return _BatchStageIterator(
            upstream=data,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            collate_fn=self.collate_fn,
        )

    def identity(self) -> dict[str, object]:
        """Return a process-stable identity for this stage."""
        return {
            "kind": self.kind,
            "batch_size": self.batch_size,
            "drop_last": self.drop_last,
            "collate_fn": identity(self.collate_fn),
        }
