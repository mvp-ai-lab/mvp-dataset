"""Loader-side unbatch stage."""

from __future__ import annotations

from collections.abc import Iterable

from ...core.stages import _UnbatchStageIterator


class _LoaderUnbatchStage:
    """TorchLoader stage configuration for unbatching outputs."""

    kind = "unbatch"

    def __call__(self, data: Iterable[object]) -> Iterable[object]:
        """Apply this callable object."""
        return _UnbatchStageIterator(upstream=data)

    def identity(self) -> dict[str, object]:
        """Return a process-stable identity for this stage."""
        return {"kind": self.kind}
