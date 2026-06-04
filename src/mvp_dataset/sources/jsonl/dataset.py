import os
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Literal

from mvp_dataset.core.context import RuntimeContext
from mvp_dataset.core.dataset import Dataset
from mvp_dataset.core.resume import ResumeStateError, stable_fingerprint
from mvp_dataset.core.types import RefFieldSpec, ShardInput
from mvp_dataset.utils.url import normalize_paths

from .utils import (
    TarManager,
    _parse_jsonl_line,
    resolve_ref_field_value,
    split_jsonl_files,
)

JsonlShuffleMode = Literal["none", "shard_aware", "global"]


@dataclass(slots=True)
class _JsonlSourceIterator:
    shards: Sequence[str]
    context: RuntimeContext
    resample: bool
    shuffle_mode: JsonlShuffleMode = "shard_aware"
    ref_fields: tuple[RefFieldSpec, ...] = ()
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
            max_open_tars = int(os.environ.get("MVP_DATASET_TAR_MAX_OPEN_FILES", "8"))
            self._tar_manager = TarManager(max_open_files=max_open_tars)

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

            assert self._handle is not None
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
        assert self._tar_manager is not None
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


@dataclass(frozen=True, slots=True)
class JsonlDataset(Dataset):
    _ref_fields: tuple[RefFieldSpec, ...] = ()
    _shuffle_mode: JsonlShuffleMode = "shard_aware"

    @classmethod
    def from_source(
        cls,
        shards: ShardInput | Sequence[ShardInput],
        context: RuntimeContext | None = None,
        resample: bool = False,
        ref_fields: Sequence[RefFieldSpec] | None = None,
        shuffle_mode: JsonlShuffleMode = "shard_aware",
    ):
        """Build a dataset from local JSONL shard paths.

        Args:
            shards: One or more file paths, glob specs, or brace-expansion specs.
            context: Optional execution context. If omitted, inferred from runtime.
            resample: Whether to loop shards indefinitely across rounds.
            ref_fields: Optional sequence of ``(field_name, base_dir)`` pairs for
                resolving tar-referenced fields in JSONL rows.
            shuffle_mode: ``"shard_aware"`` shuffles shard order by round;
                ``"none"`` reads shards in original order. ``"global"`` is not
                supported for JSONL row access.

        Returns:
            A dataset whose source is the normalized JSONL shard path list.

        Raises:
            ValueError: If any input path does not end with ``.jsonl``.
        """
        runtime_context = RuntimeContext.from_runtime() if context is None else context
        if shuffle_mode == "global":
            msg = "[UnsupportedJsonlShuffleMode] shuffle_mode='global'"
            raise ValueError(msg)
        if shuffle_mode not in ("none", "shard_aware"):
            msg = f"[InvalidJsonlShuffleMode] expected none or shard_aware, got={shuffle_mode!r}"
            raise ValueError(msg)
        normalized_shards = normalize_paths(shards)
        if not all(path.endswith(".jsonl") for path in normalized_shards):
            msg = f"[InvalidSourceType] expected .jsonl inputs, got={normalized_shards!r}"
            raise ValueError(msg)

        source_items = split_jsonl_files(normalized_shards, runtime_context.total_slots)

        ref_fields_tuple = tuple(ref_fields) if ref_fields else ()

        return cls(
            context=runtime_context,
            _source=source_items,
            _resample=resample,
            _source_kind="jsonl",
            _stages=(),
            _iter_source_stream=None,
            _ref_fields=ref_fields_tuple,
            _shuffle_mode=shuffle_mode,
        )

    def _build_source_stream(self, *, context: RuntimeContext) -> Iterable[object]:
        return _JsonlSourceIterator(
            shards=self._source,
            context=context,
            resample=self._resample,
            shuffle_mode=self._shuffle_mode,
            ref_fields=self._ref_fields,
            source_fingerprint=stable_fingerprint(self._source_fingerprint()),
        )

    def _source_fingerprint(self) -> dict[str, object]:
        return {
            "kind": "jsonl",
            "resample": self._resample,
            "shuffle_mode": self._shuffle_mode,
            "ref_fields": [(field, str(base_dir)) for field, base_dir in self._ref_fields],
            "shards": [
                {
                    "path": shard,
                    "mtime_ns": stat.st_mtime_ns,
                    "size": stat.st_size,
                }
                for shard in self._source
                for stat in (Path(shard).stat(),)
            ],
        }
