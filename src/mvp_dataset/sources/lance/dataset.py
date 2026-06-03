from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from mvp_dataset.core.context import RuntimeContext
from mvp_dataset.core.dataset import Dataset
from mvp_dataset.core.resume import ResumeStateError, stable_fingerprint
from mvp_dataset.core.stages import _AssembleStage
from mvp_dataset.core.types import ShardInput, StageSpec

from .utils import (
    LanceIndexItem,
    LanceRefIndexScope,
    LanceResolveRefFactory,
    LanceShuffleMode,
    LanceSourceSpec,
    _read_batch,
    attach_lance_ref_columns,
    list_lance_sources,
    resolve_lance_source_config,
    validate_ref_names,
)
from .utils.shuffle import fragment_aware_index_order, lance_index_at, lance_round_size


@dataclass(slots=True)
class _LanceSourceIterator:
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
        if self.batch_size <= 0:
            msg = "[InvalidLanceBatchSize] batch_size must be a positive integer"
            raise ValueError(msg)

    def __iter__(self):
        return self

    def __next__(self) -> object:
        if not self._pending_samples:
            self._fill_pending()
        if not self._pending_samples:
            raise StopIteration

        sample = self._pending_samples.pop(0)
        self.round_index, self.position_in_round = self._pending_positions.pop(0)
        return sample

    def state_dict(self) -> dict[str, object]:
        return {
            "kind": "lance",
            "shuffle_mode": self.shuffle_mode,
            "round_index": self.round_index,
            "position_in_round": self.position_in_round,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
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
        return self.source_fingerprint

    def _fill_pending(self) -> None:
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
        fragment_order = self._index_order_for_round(round_index) if self.shuffle_mode == "fragment_aware" else []
        return lance_round_size(self.source, self.context, self.shuffle_mode, fragment_order)

    def _index_item_at(self, round_index: int, position_in_round: int) -> LanceIndexItem:
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


@dataclass(frozen=True, slots=True)
class LanceDataset(Dataset):
    _shuffle_mode: LanceShuffleMode = "none"
    _columns: tuple[str, ...] | None = None
    _batch_size: int = 1024
    _ref_index_scope: LanceRefIndexScope | None = None

    @classmethod
    def from_source(
        cls,
        shards: ShardInput | Sequence[ShardInput],
        context: RuntimeContext | None = None,
        resample: bool = False,
        columns: Sequence[str] | None = None,
        batch_size: int = 1024,
        shuffle_mode: LanceShuffleMode = "none",
        ref_columns: dict[str, dict[str, str]] | None = None,
        ref_index_scope: LanceRefIndexScope | None = None,
    ):
        """Build a dataset from local Lance dataset paths.

        Args:
            shards: One or more Lance dataset URIs or directory paths. A single
                    JSON file may also be provided; it must contain ``main_uri``
                    (or ``shards``/``uri``) and optional ``ref_columns``.
            context: Optional execution context. If omitted, inferred from runtime.
            resample: Whether to loop shards indefinitely across rounds.
            columns: Optional list of column names to read.
            batch_size: Number of rows per Arrow batch during iteration.
            shuffle_mode: One of ``"none"``, ``"global"``, or ``"fragment_aware"``.
            ref_columns: Optional mapping of source column names to explicit Lance
                         reference configs containing uri, key_column, and value_column.
                         Use ``resolve_ref(...)`` later in the pipeline to read
                         and replace these referenced values.
            ref_index_scope: Reference-index build scope. ``node_local`` builds once
                             per node, ``shared`` builds once on global rank 0, and
                             ``process`` lets each process attempt to publish. Defaults
                             to ``MVP_LANCE_REF_INDEX_SCOPE`` or ``shared``.
        Returns:
            A lance dataset.
        """
        if batch_size <= 0:
            msg = "[InvalidLanceBatchSize] batch_size must be a positive integer"
            raise ValueError(msg)

        runtime_context = RuntimeContext.from_runtime() if context is None else context
        normalized_shards, resolved_ref_columns = resolve_lance_source_config(shards, ref_columns)
        # 1. Merge all shards into a single source spec, validating that they can be read together.
        sources = list_lance_sources(
            normalized_shards,
        )
        # 2. Attach reference column configs to the source spec, if any.
        source = attach_lance_ref_columns(sources[0], resolved_ref_columns)
        sources = [source]

        return cls(
            context=runtime_context,
            _source=sources,
            _resample=resample,
            _source_kind="lance",
            _stages=(),
            _iter_source_stream=None,
            _shuffle_mode=shuffle_mode,
            _columns=tuple(columns) if columns is not None else None,
            _batch_size=batch_size,
            _ref_index_scope=ref_index_scope,
        )

    def _build_source_stream(self, *, context: RuntimeContext) -> Iterable[object]:
        assert len(self._source) == 1, "Multiple Lance sources are not supported in this implementation"
        return _LanceSourceIterator(
            source=self._source[0],
            context=context,
            resample=self._resample,
            columns=self._columns,
            batch_size=self._batch_size,
            source_fingerprint=stable_fingerprint(self._source_fingerprint()),
            shuffle_mode=self._shuffle_mode,
        )

    def _source_fingerprint(self) -> dict[str, object]:
        source = self._source[0]
        return {
            "kind": "lance",
            "resample": self._resample,
            "shuffle_mode": self._shuffle_mode,
            "ref_index_scope": self._ref_index_scope,
            "iter": {
                "columns": list(self._columns) if self._columns else None,
                "batch_size": self._batch_size,
            },
            "datasets": [
                {
                    "uri": dataset.uri,
                    "num_rows": dataset.num_rows,
                    "row_offset": dataset.row_offset,
                    "fragment_ids": list(dataset.fragment_ids),
                    "fragment_row_counts": list(dataset.fragment_row_counts),
                }
                for dataset in source.datasets
            ],
            "ref_columns": [
                {
                    "column": ref.column,
                    "uri": ref.uri,
                    "key_column": ref.key_column,
                    "value_column": ref.value_column,
                    "index_uri": ref.index_uri,
                    "index_offsets_path": ref.index_offsets_path,
                    "index_entries_path": ref.index_entries_path,
                }
                for ref in source.ref_columns
            ],
        }

    def resolve_ref(
        self,
        ref_names: Sequence[str],
        *,
        batch_size: int = 1024,
        context: RuntimeContext | None = None,
        ref_index_scope: LanceRefIndexScope | None = None,
    ) -> Dataset:
        """Append a lazy stage that resolves configured Lance reference columns."""
        assert batch_size > 0, "batch_size must be a positive integer"

        ref_names = validate_ref_names(self._source[0], ref_names)
        source = self._source[0]

        factory = LanceResolveRefFactory(
            source=source,
            ref_names=ref_names,
            batch_size=batch_size,
            ref_index_scope=ref_index_scope if ref_index_scope is not None else self._ref_index_scope,
        )
        stage_context = self.context if context is None else context
        spec = StageSpec(
            kind="assemble",
            apply=_AssembleStage(factory=factory, context=stage_context),
        )
        return self._append_stage(spec)
