from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from mvp_dataset.core.context import RuntimeContext
from mvp_dataset.core.dataset import Dataset
from mvp_dataset.core.stages import _AssembleStage
from mvp_dataset.core.types import ShardInput, StageSpec

from .utils import (
    LanceRefIndexScope,
    LanceResolveRefFactory,
    LanceShuffleMode,
    LanceSourceSpec,
    assign_items,
    attach_lance_ref_columns,
    iter_lance,
    list_lance_sources,
    resolve_lance_source_config,
    validate_ref_names,
)


@dataclass(frozen=True, slots=True)
class _LanceSourceIter:
    source: LanceSourceSpec
    columns: Sequence[str] | None = None
    batch_size: int = 65536
    load_in_memory: bool = False

    handles_sharding = True

    def __call__(self, source_stream):
        return iter_lance(
            self.source,
            source_stream,
            columns=self.columns,
            batch_size=self.batch_size,
            load_in_memory=self.load_in_memory,
        )


@dataclass(frozen=True, slots=True)
class LanceDataset(Dataset):
    _shuffle_mode: LanceShuffleMode = "none"
    _load_in_memory: bool = False
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
        load_in_memory: bool = False,
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
            load_in_memory: Whether to load entire datasets into memory.
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
            _iter_source_stream=_LanceSourceIter(
                source=sources[0],
                columns=columns,
                batch_size=batch_size,
                load_in_memory=load_in_memory,
            ),
            _shuffle_mode=shuffle_mode,
            _load_in_memory=load_in_memory,
            _ref_index_scope=ref_index_scope,
        )

    def _build_source_stream(self, *, context: RuntimeContext) -> Iterable[object]:
        assert len(self._source) == 1, "Multiple Lance sources are not supported in this implementation"
        source_shard_stream = assign_items(
            self._source,
            context=context,
            resample=self._resample,
            shuffle_mode=self._shuffle_mode,
        )
        return self._iter_source_stream(source_shard_stream)

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
            load_in_memory=self._load_in_memory,
            ref_index_scope=ref_index_scope if ref_index_scope is not None else self._ref_index_scope,
        )
        stage_context = self.context if context is None else context
        spec = StageSpec(
            kind="assemble",
            apply=_AssembleStage(factory=factory, context=stage_context),
        )
        return self._append_stage(spec)
