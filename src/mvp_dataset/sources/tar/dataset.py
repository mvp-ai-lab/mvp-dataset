import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from mvp_dataset.core.context import RuntimeContext
from mvp_dataset.core.dataset import Dataset
from mvp_dataset.core.resume import (
    ResumeStateError,
    callable_fingerprint,
    stable_fingerprint,
)
from mvp_dataset.core.types import ShardInput, SidecarSpec
from mvp_dataset.utils.url import normalize_paths

from .utils import iter_tars

TarShuffleMode = Literal["none", "shard_aware", "global"]


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

            assert self._sample_iter is not None
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
            "kind": "tars",
            "shuffle_mode": self.shuffle_mode,
            "round_index": self.round_index,
            "shard_index": self.shard_index,
            "sample_index": self.sample_index,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("kind") != "tars":
            msg = f"[InvalidResumeState] expected source kind='tars', got={state.get('kind')!r}"
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
        self._sample_iter = iter_tars(iter([shard]), sidecars=self.sidecars or None)
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


def _sidecar_fingerprint(sidecars: tuple[SidecarSpec, ...], shards: Sequence[str]) -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "resolver": callable_fingerprint(resolver),
            "shards": [
                {
                    "path": sidecar_path,
                    "mtime_ns": stat.st_mtime_ns,
                    "size": stat.st_size,
                }
                for shard in shards
                for sidecar_path in (str(resolver(shard)),)
                for stat in (Path(sidecar_path).stat(),)
            ],
        }
        for name, resolver in sidecars
    ]


@dataclass(frozen=True, slots=True)
class TarDataset(Dataset):
    _sidecar_specs: tuple[SidecarSpec, ...] = ()
    _shuffle_mode: TarShuffleMode = "shard_aware"

    @classmethod
    def from_source(
        cls,
        shards: ShardInput | Sequence[ShardInput],
        context: RuntimeContext | None = None,
        resample: bool = False,
        sidecars: Sequence[SidecarSpec] | None = None,
        shuffle_mode: TarShuffleMode = "shard_aware",
    ):
        """Build a dataset from local tar shard paths.

        Args:
            shards: One or more file paths, glob specs, or brace-expansion specs.
            context: Optional execution context. If omitted, inferred from runtime.
            resample: Whether to loop shards indefinitely across rounds.
            sidecars: Optional sequence of ``(name, path_resolver)`` pairs for
                shard-level sidecar merges. Each ``path_resolver`` receives a main
                tar shard path and must return the matching sidecar tar shard path.
            shuffle_mode: ``"shard_aware"`` shuffles shard order by round;
                ``"none"`` reads shards in original order. ``"global"`` is not
                supported for tar sample access.

        Returns:
            A dataset whose source is the normalized tar shard path list.

        Raises:
            ValueError: If any input path does not end with ``.tar``.
        """
        runtime_context = RuntimeContext.from_runtime() if context is None else context
        if shuffle_mode == "global":
            msg = "[UnsupportedTarShuffleMode] shuffle_mode='global'"
            raise ValueError(msg)
        if shuffle_mode not in ("none", "shard_aware"):
            msg = f"[InvalidTarShuffleMode] expected none or shard_aware, got={shuffle_mode!r}"
            raise ValueError(msg)
        normalized_shards = normalize_paths(shards)
        if not all(path.endswith(".tar") for path in normalized_shards):
            msg = f"[InvalidSourceType] expected .tar inputs, got={normalized_shards!r}"
            raise ValueError(msg)
        if len(normalized_shards) < runtime_context.total_slots:
            msg = (
                f"[InsufficientTarShards] got {len(normalized_shards)} tar file(s) "
                f"but {runtime_context.total_slots} slot(s) — split your tar archives "
                f"into at least {runtime_context.total_slots} shards before loading"
            )
            raise ValueError(msg)

        sidecar_specs = tuple(sidecars) if sidecars else ()

        return cls(
            context=runtime_context,
            _source=normalized_shards,
            _resample=resample,
            _source_kind="tars",
            _stages=(),
            _source_stream_factory=None,
            _sidecar_specs=sidecar_specs,
            _shuffle_mode=shuffle_mode,
        )

    def _build_source_stream(self, *, context: RuntimeContext) -> Iterable[object]:
        return _TarSourceIterator(
            shards=self._source,
            context=context,
            resample=self._resample,
            sidecars=self._sidecar_specs,
            shuffle_mode=self._shuffle_mode,
            source_fingerprint=stable_fingerprint(self._source_fingerprint()),
        )

    def _source_fingerprint(self) -> dict[str, object]:
        return {
            "kind": "tars",
            "resample": self._resample,
            "shuffle_mode": self._shuffle_mode,
            "sidecars": _sidecar_fingerprint(self._sidecar_specs, self._source),
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
