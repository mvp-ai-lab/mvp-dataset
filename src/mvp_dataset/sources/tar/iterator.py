"""TAR source iterator."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from mvp_dataset.core.context import RuntimeContext
from mvp_dataset.core.resume import ResumeStateError
from mvp_dataset.core.types import SidecarSpec

from .reader import iter_tar_shards
from .types import TarShuffleMode


@dataclass(slots=True)
class _TarSourceIterator:
    shards: Sequence[str]
    context: RuntimeContext
    resample: bool
    sidecars: tuple[SidecarSpec, ...] = ()
    shuffle_mode: TarShuffleMode = "shard_aware"
    source_fingerprint: str = ""
    round_index: int = 0
    shard_index: int = 0
    sample_index: int = 0
    _sample_iter: Iterator[object] | None = None
    _current_shard: str | None = None
    _round_shards_cache: list[str] = field(default_factory=list)
    _round_shards_round: int | None = None

    def __post_init__(self) -> None:
        if not self.shards:
            msg = f"[InsufficientItemsForSlot] items=0 total_slots={self.context.total_slots} slot={self.context.slot}"
            raise ValueError(msg)

    def __iter__(self):
        return self

    def __next__(self) -> object:
        while True:
            shard = self._current_shard_path()
            if shard is None:
                self._close_iterator()
                raise StopIteration
            if self._sample_iter is None or self._current_shard != shard:
                self._open_shard(shard)

            if self._sample_iter is None:
                msg = "[InvalidTarIteratorState] shard iterator is not open"
                raise RuntimeError(msg)
            try:
                sample = next(self._sample_iter)
            except StopIteration:
                self._advance_shard()
                continue
            self.sample_index += 1
            return sample

    def __del__(self) -> None:
        self._close_iterator()

    def state_dict(self) -> dict[str, object]:
        return {
            "kind": "tar",
            "shuffle_mode": self.shuffle_mode,
            "round_index": self.round_index,
            "shard_index": self.shard_index,
            "sample_index": self.sample_index,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("kind") != "tar":
            msg = f"[InvalidResumeState] expected source kind='tar', got={state.get('kind')!r}"
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

        shard_index = state.get("shard_index")
        if not isinstance(shard_index, int) or shard_index < 0:
            msg = "[InvalidResumeState] shard_index must be a non-negative integer"
            raise ResumeStateError(msg)
        round_shards = self._round_shards(round_index)
        if shard_index > len(round_shards):
            msg = "[InvalidResumeState] shard_index is out of range"
            raise ResumeStateError(msg)

        sample_index = state.get("sample_index")
        if not isinstance(sample_index, int) or sample_index < 0:
            msg = "[InvalidResumeState] sample_index must be a non-negative integer"
            raise ResumeStateError(msg)
        if shard_index == len(round_shards) and sample_index != 0:
            msg = "[InvalidResumeState] sample_index must be 0 at end of round"
            raise ResumeStateError(msg)

        self.round_index = round_index
        self.shard_index = shard_index
        self.sample_index = sample_index
        self._close_iterator()

    def fingerprint(self) -> str:
        return self.source_fingerprint

    def _current_shard_path(self) -> str | None:
        while True:
            round_shards = self._round_shards(self.round_index)
            if self.shard_index < len(round_shards):
                return round_shards[self.shard_index]
            if not self.resample:
                return None
            self.round_index += 1
            self.shard_index = 0
            self.sample_index = 0
            self._close_iterator()

    def _round_shards(self, round_index: int) -> list[str]:
        if self._round_shards_round == round_index:
            return self._round_shards_cache

        ordered = list(self.shards)
        if self.shuffle_mode == "shard_aware":
            random.Random(self.context.seed + round_index).shuffle(ordered)
        elif self.shuffle_mode != "none":
            msg = f"[UnsupportedTarShuffleMode] shuffle_mode={self.shuffle_mode!r}"
            raise ValueError(msg)
        base_offset = round_index * len(ordered)
        self._round_shards_cache = [
            shard
            for local_index, shard in enumerate(ordered)
            if (base_offset + local_index) % self.context.total_slots == self.context.slot
        ]
        self._round_shards_round = round_index
        return self._round_shards_cache

    def _open_shard(self, shard: str) -> None:
        self._close_iterator()
        self._sample_iter = iter_tar_shards(iter([shard]), sidecars=self.sidecars or None)
        for _ in range(self.sample_index):
            next(self._sample_iter)
        self._current_shard = shard

    def _advance_shard(self) -> None:
        self.shard_index += 1
        self.sample_index = 0
        self._close_iterator()

    def _close_iterator(self) -> None:
        if self._sample_iter is not None:
            close = getattr(self._sample_iter, "close", None)
            if close is not None:
                close()
            self._sample_iter = None
            self._current_shard = None
