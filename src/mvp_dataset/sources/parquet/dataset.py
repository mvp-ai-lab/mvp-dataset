import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Literal

from mvp_dataset.core.context import RuntimeContext
from mvp_dataset.core.dataset import Dataset
from mvp_dataset.core.resume import ResumeStateError, stable_fingerprint
from mvp_dataset.core.types import ShardInput
from mvp_dataset.utils.url import normalize_paths

from .utils import (
    ParquetFragment,
    iter_parquet,
    list_parquet_fragments,
    resolve_parquet_batch_size,
)

ParquetShuffleMode = Literal["none", "fragment_aware", "global"]


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


@dataclass(frozen=True, slots=True)
class ParquetDataset(Dataset):
    _columns: tuple[str, ...] | None = None
    _use_threads: bool = True
    _shuffle_mode: ParquetShuffleMode = "fragment_aware"

    @classmethod
    def from_source(
        cls,
        shards: ShardInput | Sequence[ShardInput],
        context: RuntimeContext | None = None,
        resample: bool = False,
        min_row_groups_per_fragment: int = 1,
        columns: Sequence[str] | None = None,
        use_threads: bool = True,
        shuffle_mode: ParquetShuffleMode = "fragment_aware",
    ):
        """Build a dataset from local Parquet file paths.

        Args:
            shards: One or more file paths, glob specs, or brace-expansion specs.
            context: Optional execution context. If omitted, inferred from runtime.
            resample: Whether to loop shards indefinitely across rounds.
            min_row_groups_per_fragment: Minimum number of consecutive parquet
                row groups to merge into one schedulable fragment. Defaults to
                ``1``, which creates one fragment per row group unless
                ``min_fragments`` fallback logic re-splits the source.
            columns: Optional list of column names to read.
            use_threads: Whether to use multi-threaded Arrow reads.
            shuffle_mode: ``"fragment_aware"`` shuffles fragments by round,
                ``"none"`` reads fragments in original order. ``"global"`` is
                not supported for Parquet row access.

        Arrow record batch size is controlled via the
        ``MVP_DATASET_PARQUET_BATCH_SIZE`` environment variable.

        Returns:
            A dataset whose source is the list of parquet fragments.

        Raises:
            ValueError: If any input path does not end with ``.parquet``.
        """
        runtime_context = RuntimeContext.from_runtime() if context is None else context
        if shuffle_mode == "global":
            msg = "[UnsupportedParquetShuffleMode] shuffle_mode='global'"
            raise ValueError(msg)
        if shuffle_mode not in ("none", "fragment_aware"):
            msg = f"[InvalidParquetShuffleMode] expected none or fragment_aware, got={shuffle_mode!r}"
            raise ValueError(msg)
        resolve_parquet_batch_size()
        normalized_shards = normalize_paths(shards)
        if not all(path.endswith(".parquet") for path in normalized_shards):
            msg = f"[InvalidSourceType] expected .parquet inputs, got={normalized_shards!r}"
            raise ValueError(msg)
        fragments = list_parquet_fragments(
            normalized_shards,
            min_row_groups_per_fragment=min_row_groups_per_fragment,
            min_fragments=runtime_context.total_slots,
        )

        return cls(
            context=runtime_context,
            _source=fragments,
            _resample=resample,
            _source_kind="parquet",
            _stages=(),
            _iter_source_stream=None,
            _columns=tuple(columns) if columns is not None else None,
            _use_threads=use_threads,
            _shuffle_mode=shuffle_mode,
        )

    def _build_source_stream(self, *, context: RuntimeContext) -> Iterable[object]:
        return _ParquetSourceIterator(
            fragments=self._source,
            context=context,
            resample=self._resample,
            columns=self._columns,
            use_threads=self._use_threads,
            source_fingerprint=stable_fingerprint(self._source_fingerprint()),
            shuffle_mode=self._shuffle_mode,
        )

    def _source_fingerprint(self) -> dict[str, object]:
        return {
            "kind": "parquet",
            "resample": self._resample,
            "shuffle_mode": self._shuffle_mode,
            "iter": {
                "columns": list(self._columns) if self._columns else None,
                "use_threads": self._use_threads,
            },
            "fragments": [
                {
                    "path": fragment.path,
                    "row_groups": list(fragment.row_groups),
                    "row_offset": fragment.row_offset,
                    "num_rows": fragment.num_rows,
                    "row_group_offsets": list(fragment.row_group_offsets),
                    "row_group_num_rows": list(fragment.row_group_num_rows),
                }
                for fragment in self._source
            ],
        }
