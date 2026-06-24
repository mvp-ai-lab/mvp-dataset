"""Lance source iterator."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field

from mvp_dataset.core.context import RuntimeContext
from mvp_dataset.core.resume import ResumeStateError

from .order import ChunkShuffleConfig, LanceIndexOrder, build_lance_index_order
from .reader import LanceBatchReader
from .types import LanceIndexItem, LanceSelection, LanceShuffleMode, LanceSource


@dataclass(slots=True)
class _LanceSourceIterator:
    """Runtime iterator over a Lance source."""

    source: LanceSource
    context: RuntimeContext
    resample: bool
    columns: Sequence[str] | None
    read_batch_size: int
    source_fingerprint: str
    shuffle_mode: LanceShuffleMode = "none"
    chunk_config: ChunkShuffleConfig | None = None
    selection: LanceSelection | None = None
    round_index: int = 0
    position_in_round: int = 0
    _pending_samples: deque[object] = field(default_factory=deque)
    _pending_positions: deque[tuple[int, int]] = field(default_factory=deque)
    index_order: LanceIndexOrder = field(init=False)
    batch_reader: LanceBatchReader = field(init=False)

    def __post_init__(self) -> None:
        """Validate configuration and build runtime helpers."""
        if self.read_batch_size <= 0:
            msg = "[InvalidLanceReadBatchSize] read_batch_size must be a positive integer"
            raise ValueError(msg)
        if self.shuffle_mode not in ("none", "global", "chunk"):
            msg = f"[InvalidLanceShuffleMode] expected none, global, or chunk, got {self.shuffle_mode!r}"
            raise ValueError(msg)
        if self.shuffle_mode == "chunk" and self.chunk_config is None:
            msg = "[InvalidLanceChunkShuffle] chunk_config is required for shuffle_mode='chunk'"
            raise ValueError(msg)
        self.index_order = build_lance_index_order(
            self.source,
            self.context,
            self.shuffle_mode,
            chunk_config=self.chunk_config,
            selection=self.selection,
        )
        self.batch_reader = LanceBatchReader(self.source)

    def __iter__(self):
        """Return the iterator object."""
        return self

    def __next__(self) -> object:
        """Return the next output item."""
        if not self._pending_samples:
            self._fill_pending()
        if not self._pending_samples:
            raise StopIteration

        sample = self._pending_samples.popleft()
        self.round_index, self.position_in_round = self._pending_positions.popleft()
        return sample

    def state_dict(self) -> dict[str, object]:
        """Return the resumable state for this source iterator."""
        return {
            "kind": "lance",
            "shuffle_mode": self.shuffle_mode,
            "chunk_shuffle_chunk_size": self._chunk_size,
            "chunk_shuffle_k": self._chunk_k,
            "chunk_row_order": self._chunk_row_order,
            "round_index": self.round_index,
            "position_in_round": self.position_in_round,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore this iterator from a resumable state dictionary."""
        if state.get("kind") != "lance":
            msg = f"[InvalidResumeState] expected source kind='lance', got={state.get('kind')!r}"
            raise ResumeStateError(msg)
        if state.get("shuffle_mode") != self.shuffle_mode:
            msg = "[InvalidResumeState] shuffle_mode does not match"
            raise ResumeStateError(msg)
        if self.shuffle_mode == "chunk":
            if state.get("chunk_shuffle_chunk_size") != self._chunk_size:
                msg = "[InvalidResumeState] chunk_shuffle_chunk_size does not match"
                raise ResumeStateError(msg)
            if state.get("chunk_shuffle_k") != self._chunk_k:
                msg = "[InvalidResumeState] chunk_shuffle_k does not match"
                raise ResumeStateError(msg)
            if state.get("chunk_row_order") != self._chunk_row_order:
                msg = "[InvalidResumeState] chunk_row_order does not match"
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
        if position_in_round > self.index_order.round_size(round_index):
            msg = "[InvalidResumeState] position_in_round is out of range"
            raise ResumeStateError(msg)

        self.round_index = round_index
        self.position_in_round = position_in_round
        self._pending_samples.clear()
        self._pending_positions.clear()

    def fingerprint(self) -> str:
        """Return a stable fingerprint for resume compatibility checks."""
        return self.source_fingerprint

    @property
    def _chunk_size(self) -> int | None:
        return None if self.chunk_config is None else self.chunk_config.chunk_size

    @property
    def _chunk_k(self) -> int | None:
        return None if self.chunk_config is None else self.chunk_config.k

    @property
    def _chunk_row_order(self) -> str | None:
        return None if self.chunk_config is None else self.chunk_config.row_order

    def _fill_pending(self) -> None:
        """Fill pending samples by aggregating row indexes into one Lance read."""
        indexes: list[LanceIndexItem] = []
        positions: list[tuple[int, int]] = []
        round_index = self.round_index
        position_in_round = self.position_in_round

        while len(indexes) < self.read_batch_size:
            round_size = self.index_order.round_size(round_index)
            if position_in_round >= round_size:
                if not self.resample:
                    break
                round_index += 1
                position_in_round = 0
                if self.index_order.round_size(round_index) <= 0:
                    break
                continue

            count = min(self.read_batch_size - len(indexes), round_size - position_in_round)
            indexes.extend(self.index_order.items(round_index, position_in_round, count))
            positions.extend((round_index, position_in_round + offset + 1) for offset in range(count))
            position_in_round += count

        if not indexes:
            return

        self._pending_samples.extend(self.batch_reader.read(indexes, columns=self.columns))
        self._pending_positions.extend(positions)
