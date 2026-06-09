"""Loader-side unbatch stage."""

from __future__ import annotations

from collections.abc import Iterable

from ...core.resume import stable_fingerprint
from ...core.stages import _UnbatchStageIterator


class _LoaderUnbatchStage:
    """TorchLoader stage configuration for unbatching outputs."""

    kind = "unbatch"

    def __call__(self, data: Iterable[object]) -> Iterable[object]:
        """Apply this callable object."""
        return _UnbatchStageIterator(upstream=data)

    def fingerprint(self) -> str:
        """Return a stable fingerprint for resume compatibility checks."""
        return stable_fingerprint({"kind": self.kind})
