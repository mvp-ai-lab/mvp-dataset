"""Lance source iterator."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from mvp_dataset.core.context import RuntimeContext
from mvp_dataset.core.resume import ResumeStateError

from .reader import _read_batch
from .shuffle import fragment_aware_index_order, lance_index_at, lance_round_size
from .types import LanceIndexItem, LanceShuffleMode, LanceSourceSpec


@dataclass(slots=True)
class _LanceSourceIterator:
    """Stateful iterator over Lance rows with deterministic shuffling."""

    source: LanceSourceSpec
    context: RuntimeContext
    resample: bool
    columns: Sequence[str] | None
    batch_size: int
    source_fingerprint: str
    shuffle_mode: LanceShuffleMode = "none"
    round_index: int = 0
    position_in_round: int = 0
    _pending_samples: list[object] = field(default_factory=list)
    _pending_positions: list[tuple[int, int]] = field(default_factory=list)
    _index_order: list[LanceIndexItem] = field(default_factory=list)
    _index_order_round: int | None = None

    def __post_init__(self) -> None:
        """Validate dataclass configuration after initialization."""
        if self.batch_size <= 0:
            msg = "[InvalidLanceBatchSize] batch_size must be a positive integer"
            raise ValueError(msg)

    def __iter__(self):
        """Return the iterator object."""
        return self

    def __next__(self) -> object:
        """Return the next output item."""
        if not self._pending_samples:
            self._fill_pending()
        if not self._pending_samples:
            raise StopIteration

        sample = self._pending_samples.pop(0)
        self.round_index, self.position_in_round = self._pending_positions.pop(0)
        return sample

    def state_dict(self) -> dict[str, object]:
        """Return the resumable state for this object."""
        return {
            "kind": "lance",
            "shuffle_mode": self.shuffle_mode,
            "round_index": self.round_index,
            "position_in_round": self.position_in_round,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore this object from a resumable state dictionary."""
        if state.get("kind") != "lance":
            msg = f"[InvalidResumeState] expected source kind='lance', got={state.get('kind')!r}"
            raise ResumeStateError(msg)
        if state.get("shuffle_mode") != self.shuffle_mode:
            msg = "[InvalidResumeState] shuffle_mode does not match"
            raise ResumeStateError(msg)

        round_index = state.get("round_index")
        if not isinstance(round_index, int) or round_index < 0:
            msg = "[InvalidResumeState] round_index must be a non-negative integer"
            raise ResumeStateError(msg)
        if round_index != 0 and not self.resample:
            msg = "[InvalidResumeState] round_index must be 0 when resample=False"
            raise ResumeStateError(msg)

        position_in_round = state.get("position_in_round")
        if not isinstance(position_in_round, int) or position_in_round < 0:
            msg = "[InvalidResumeState] position_in_round must be a non-negative integer"
            raise ResumeStateError(msg)
        if position_in_round > self._round_size(round_index):
            msg = "[InvalidResumeState] position_in_round is out of range"
            raise ResumeStateError(msg)

        self.round_index = round_index
        self.position_in_round = position_in_round
        self._pending_samples.clear()
        self._pending_positions.clear()

    def fingerprint(self) -> str:
        """Return a stable fingerprint for resume compatibility checks."""
        return self.source_fingerprint

    def _fill_pending(self) -> None:
        """Fill the pending sample buffer from Lance rows."""
        batch_indexes: list[LanceIndexItem] = []
        batch_positions: list[tuple[int, int]] = []
        round_index = self.round_index
        position_in_round = self.position_in_round
        while len(batch_indexes) < self.batch_size:
            round_size = self._round_size(round_index)
            if position_in_round >= round_size:
                if not self.resample:
                    break
                round_index += 1
                position_in_round = 0
                round_size = self._round_size(round_index)
                if round_size <= 0:
                    break

            batch_indexes.append(self._index_item_at(round_index, position_in_round))
            position_in_round += 1
            batch_positions.append((round_index, position_in_round))

        if not batch_indexes:
            return
        self._pending_samples.extend(_read_batch(self.source, batch_indexes, columns=self.columns))
        self._pending_positions.extend(batch_positions)

    def _round_size(self, round_index: int) -> int:
        """Return the number of source items in one Lance round."""
        fragment_order = self._index_order_for_round(round_index) if self.shuffle_mode == "fragment_aware" else []
        return lance_round_size(self.source, self.context, self.shuffle_mode, fragment_order)

    def _index_item_at(self, round_index: int, position_in_round: int) -> LanceIndexItem:
        """Return the physical Lance row location for a logical position."""
        fragment_order = self._index_order_for_round(round_index) if self.shuffle_mode == "fragment_aware" else []
        return lance_index_at(
            self.source,
            self.context,
            self.shuffle_mode,
            round_index,
            position_in_round,
            fragment_order,
        )

    def _index_order_for_round(self, round_index: int) -> list[LanceIndexItem]:
        """Return the deterministic fragment-aware order for one round."""
        if self._index_order_round == round_index:
            return self._index_order

        if self.shuffle_mode == "fragment_aware":
            index_order = fragment_aware_index_order(self.source, self.context, round_index)
        else:
            msg = (
                "[InvalidLanceShuffleMode] index order is only materialized for "
                f"fragment_aware, got {self.shuffle_mode!r}"
            )
            raise ValueError(msg)

        self._index_order = index_order
        self._index_order_round = round_index
        return self._index_order
