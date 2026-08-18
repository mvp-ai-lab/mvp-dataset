"""Snapshot source iterator value decoding."""

from __future__ import annotations

from dataclasses import dataclass

from mvp_dataset.core.resume import Stateful

from .codecs import decode_snapshot_value


@dataclass(slots=True)
class _SnapshotSourceIterator:
    """Decode materialized values while preserving Lance iterator state."""

    upstream: Stateful

    def __iter__(self) -> _SnapshotSourceIterator:
        """Return this iterator."""
        return self

    def __next__(self) -> object:
        """Return the next decoded snapshot value."""
        return decode_snapshot_value(next(self.upstream))

    def state_dict(self) -> dict[str, object]:
        """Return the delegated Lance source state."""
        return self.upstream.state_dict()

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore the delegated Lance source state."""
        self.upstream.load_state_dict(state)
