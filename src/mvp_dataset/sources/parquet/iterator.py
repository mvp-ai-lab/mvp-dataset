"""Parquet source iterator."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from mvp_dataset.core.context import RuntimeContext
from mvp_dataset.core.resume import ResumeStateError

from .fragments import ParquetFragment
from .reader import iter_parquet
from .types import ParquetShuffleMode


@dataclass(slots=True)
class _ParquetSourceIterator:
    fragments: Sequence[ParquetFragment]
    context: RuntimeContext
    resample: bool
    columns: Sequence[str] | None
    use_threads: bool
    source_fingerprint: str
    shuffle_mode: ParquetShuffleMode = "fragment_aware"
    round_index: int = 0
    fragment_index: int = 0
    row_group_index: int = 0
    row_in_row_group: int = 0
    _row_group_iter: Iterator[dict[str, object]] | None = None
    _round_fragments_cache: list[ParquetFragment] = field(default_factory=list)
    _round_fragments_round: int | None = None

    def __post_init__(self) -> None:
        if not self.fragments:
            msg = f"[InsufficientItemsForSlot] items=0 total_slots={self.context.total_slots} slot={self.context.slot}"
            raise ValueError(msg)

    def __iter__(self):
        return self

    def __next__(self) -> object:
        while True:
            fragment = self._current_fragment()
            if fragment is None:
                raise StopIteration
            if self.row_group_index >= len(fragment.row_groups):
                self._advance_fragment()
                continue
            if self.row_in_row_group >= fragment.row_group_num_rows[self.row_group_index]:
                self._advance_row_group()
                continue
            if self._row_group_iter is None:
                self._row_group_iter = iter_parquet(
                    fragment,
                    columns=self.columns,
                    use_threads=self.use_threads,
                    row_group_index=self.row_group_index,
                )
                for _ in range(self.row_in_row_group):
                    next(self._row_group_iter)

            try:
                sample = next(self._row_group_iter)
            except StopIteration:
                self._advance_row_group()
                continue
            self.row_in_row_group += 1
            return sample

    def state_dict(self) -> dict[str, object]:
        return {
            "kind": "parquet",
            "shuffle_mode": self.shuffle_mode,
            "round_index": self.round_index,
            "fragment_index": self.fragment_index,
            "row_group_index": self.row_group_index,
            "row_in_row_group": self.row_in_row_group,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("kind") != "parquet":
            msg = f"[InvalidResumeState] expected source kind='parquet', got={state.get('kind')!r}"
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

        fragment_index = state.get("fragment_index")
        if not isinstance(fragment_index, int) or fragment_index < 0:
            msg = "[InvalidResumeState] fragment_index must be a non-negative integer"
            raise ResumeStateError(msg)
        round_fragments = self._round_fragments(round_index)
        if fragment_index > len(round_fragments):
            msg = "[InvalidResumeState] fragment_index is out of range"
            raise ResumeStateError(msg)

        row_group_index = state.get("row_group_index")
        if not isinstance(row_group_index, int) or row_group_index < 0:
            msg = "[InvalidResumeState] row_group_index must be a non-negative integer"
            raise ResumeStateError(msg)
        if fragment_index < len(round_fragments) and row_group_index > len(round_fragments[fragment_index].row_groups):
            msg = "[InvalidResumeState] row_group_index is out of range"
            raise ResumeStateError(msg)
        if fragment_index == len(round_fragments) and row_group_index != 0:
            msg = "[InvalidResumeState] row_group_index must be 0 at end of round"
            raise ResumeStateError(msg)

        row_in_row_group = state.get("row_in_row_group")
        if not isinstance(row_in_row_group, int) or row_in_row_group < 0:
            msg = "[InvalidResumeState] row_in_row_group must be a non-negative integer"
            raise ResumeStateError(msg)
        if fragment_index < len(round_fragments):
            fragment = round_fragments[fragment_index]
            if row_group_index < len(fragment.row_groups):
                if row_in_row_group > fragment.row_group_num_rows[row_group_index]:
                    msg = "[InvalidResumeState] row_in_row_group is out of range"
                    raise ResumeStateError(msg)
            elif row_in_row_group != 0:
                msg = "[InvalidResumeState] row_in_row_group must be 0 at end of fragment"
                raise ResumeStateError(msg)
        if fragment_index == len(round_fragments) and row_in_row_group != 0:
            msg = "[InvalidResumeState] row_in_row_group must be 0 at end of round"
            raise ResumeStateError(msg)

        self.round_index = round_index
        self.fragment_index = fragment_index
        self.row_group_index = row_group_index
        self.row_in_row_group = row_in_row_group
        self._row_group_iter = None

    def fingerprint(self) -> str:
        return self.source_fingerprint

    def _current_fragment(self) -> ParquetFragment | None:
        while True:
            round_fragments = self._round_fragments(self.round_index)
            if self.fragment_index < len(round_fragments):
                return round_fragments[self.fragment_index]
            if not self.resample:
                return None
            self.round_index += 1
            self.fragment_index = 0
            self.row_group_index = 0
            self.row_in_row_group = 0
            self._row_group_iter = None

    def _advance_fragment(self) -> None:
        self.fragment_index += 1
        self.row_group_index = 0
        self.row_in_row_group = 0
        self._row_group_iter = None

    def _advance_row_group(self) -> None:
        self.row_group_index += 1
        self.row_in_row_group = 0
        self._row_group_iter = None

    def _round_fragments(self, round_index: int) -> list[ParquetFragment]:
        if self._round_fragments_round == round_index:
            return self._round_fragments_cache

        ordered = list(self.fragments)
        if self.shuffle_mode == "fragment_aware":
            random.Random(self.context.seed + round_index).shuffle(ordered)
        elif self.shuffle_mode != "none":
            msg = f"[UnsupportedParquetShuffleMode] shuffle_mode={self.shuffle_mode!r}"
            raise ValueError(msg)
        base_offset = round_index * len(ordered)
        self._round_fragments_cache = [
            fragment
            for local_index, fragment in enumerate(ordered)
            if (base_offset + local_index) % self.context.total_slots == self.context.slot
        ]
        self._round_fragments_round = round_index
        return self._round_fragments_cache
