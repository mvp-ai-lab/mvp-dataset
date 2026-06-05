"""JSONL source iterator."""

from __future__ import annotations

import os
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from mvp_dataset.core.context import RuntimeContext
from mvp_dataset.core.resume import ResumeStateError
from mvp_dataset.core.types import TarUriRefFieldSpec

from .reader import _parse_jsonl_line
from .refs import TarManager, resolve_ref_field_value
from .types import JsonlShuffleMode


@dataclass(slots=True)
class _JsonlSourceIterator:
    shards: Sequence[str]
    context: RuntimeContext
    resample: bool
    shuffle_mode: JsonlShuffleMode = "shard_aware"
    ref_fields: tuple[TarUriRefFieldSpec, ...] = ()
    source_fingerprint: str = ""
    round_index: int = 0
    shard_index: int = 0
    byte_offset: int = 0
    line_index: int = 0
    _handle: BinaryIO | None = None
    _current_shard: str | None = None
    _tar_manager: TarManager | None = None
    _round_shards_cache: list[str] = field(default_factory=list)
    _round_shards_round: int | None = None

    def __post_init__(self) -> None:
        if not self.shards:
            msg = f"[InsufficientItemsForSlot] items=0 total_slots={self.context.total_slots} slot={self.context.slot}"
            raise ValueError(msg)
        if self.ref_fields:
            max_open_tar_files = int(os.environ.get("MVP_DATASET_TAR_MAX_OPEN_FILES", "8"))
            self._tar_manager = TarManager(max_open_files=max_open_tar_files)

    def __iter__(self):
        return self

    def __next__(self) -> object:
        while True:
            shard = self._current_shard_path()
            if shard is None:
                self._close()
                raise StopIteration
            if self._handle is None or self._current_shard != shard:
                self._open_shard(shard)

            if self._handle is None:
                msg = "[InvalidJsonlIteratorState] shard handle is not open"
                raise RuntimeError(msg)
            line = self._handle.readline()
            if not line:
                self._advance_shard()
                continue

            sample = _parse_jsonl_line(shard, self.line_index, line.decode("utf-8"), allow_preannotated=True)
            sample = self._resolve_refs(sample)
            self.byte_offset = self._handle.tell()
            self.line_index += 1
            return sample

    def __del__(self) -> None:
        self._close()

    def state_dict(self) -> dict[str, object]:
        return {
            "kind": "jsonl",
            "shuffle_mode": self.shuffle_mode,
            "round_index": self.round_index,
            "shard_index": self.shard_index,
            "byte_offset": self.byte_offset,
            "line_index": self.line_index,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("kind") != "jsonl":
            msg = f"[InvalidResumeState] expected source kind='jsonl', got={state.get('kind')!r}"
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

        byte_offset = state.get("byte_offset")
        if not isinstance(byte_offset, int) or byte_offset < 0:
            msg = "[InvalidResumeState] byte_offset must be a non-negative integer"
            raise ResumeStateError(msg)
        if shard_index < len(round_shards) and byte_offset > Path(round_shards[shard_index]).stat().st_size:
            msg = "[InvalidResumeState] byte_offset is out of range"
            raise ResumeStateError(msg)
        if shard_index == len(round_shards) and byte_offset != 0:
            msg = "[InvalidResumeState] byte_offset must be 0 at end of round"
            raise ResumeStateError(msg)

        line_index = state.get("line_index")
        if not isinstance(line_index, int) or line_index < 0:
            msg = "[InvalidResumeState] line_index must be a non-negative integer"
            raise ResumeStateError(msg)
        if shard_index == len(round_shards) and line_index != 0:
            msg = "[InvalidResumeState] line_index must be 0 at end of round"
            raise ResumeStateError(msg)

        self.round_index = round_index
        self.shard_index = shard_index
        self.byte_offset = byte_offset
        self.line_index = line_index
        self._close_handle()

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
            self.byte_offset = 0
            self.line_index = 0
            self._close_handle()

    def _round_shards(self, round_index: int) -> list[str]:
        if self._round_shards_round == round_index:
            return self._round_shards_cache

        ordered = list(self.shards)
        if self.shuffle_mode == "shard_aware":
            random.Random(self.context.seed + round_index).shuffle(ordered)
        elif self.shuffle_mode != "none":
            msg = f"[UnsupportedJsonlShuffleMode] shuffle_mode={self.shuffle_mode!r}"
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
        self._close_handle()
        self._handle = open(shard, "rb")
        self._handle.seek(self.byte_offset)
        self._current_shard = shard

    def _advance_shard(self) -> None:
        self.shard_index += 1
        self.byte_offset = 0
        self.line_index = 0
        self._close_handle()

    def _resolve_refs(self, sample: dict[str, object]) -> dict[str, object]:
        if not self.ref_fields:
            return sample
        if self._tar_manager is None:
            msg = "[InvalidJsonlIteratorState] tar manager is not initialized"
            raise RuntimeError(msg)
        key_dot_level = int(os.environ.get("MVP_DATASET_TAR_KEY_DOT_LEVEL", "1"))
        resolved = dict(sample)
        for field_name, base_dir in self.ref_fields:
            if field_name in sample:
                resolved[field_name] = resolve_ref_field_value(
                    sample[field_name],
                    field=field_name,
                    base_dir=base_dir,
                    key_dot_level=key_dot_level,
                    manager=self._tar_manager,
                )
        return resolved

    def _close_handle(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._current_shard = None

    def _close(self) -> None:
        self._close_handle()
        if self._tar_manager is not None:
            self._tar_manager.close()
