"""Chainable iterator dataset API for mvp-dataset."""

from __future__ import annotations

import importlib
import os
import random
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from types import ModuleType
from typing import Literal, cast

from ..core.types import (
    Assembler,
    CacheSpec,
    RefFieldSpec,
    RuntimeContext,
    ShardInput,
    SidecarSpec,
    StageSpec,
)
from ..sources import (
    ParquetFragment,
    iter_jsonls,
    iter_parquets,
    iter_tars,
    list_parquet_fragments,
)
from ..sources.jsonl import materialize_jsonl_shards
from ..utils.sharding import iter_items
from ..utils.url import normalize_paths
from .ops import (
    assemble_samples,
    batch_samples,
    map_samples,
    shuffle_samples,
    unbatch_samples,
)

SourceKind = Literal["jsonl", "tars", "parquet"]
SourceStore = list[str] | list[ParquetFragment]
SourceShape = Literal["tar_paths", "jsonl_paths", "parquet_fragments"]

_UNSUPPORTED_CACHE_KINDS = frozenset({"shuffle", "batch", "unbatch"})


def torch_iterabledataset_class(
    import_module: Callable[[str], ModuleType] = importlib.import_module,
) -> type:
    """Resolve ``torch.utils.data.IterableDataset`` with a no-torch fallback."""

    try:
        torch_utils_data = import_module("torch.utils.data")
    except ModuleNotFoundError:

        class _IterableDatasetFallback:
            """Fallback IterableDataset shim when torch is unavailable."""

        return _IterableDatasetFallback

    return torch_utils_data.IterableDataset


@dataclass(frozen=True, slots=True)
class Dataset(torch_iterabledataset_class()):
    """Chainable iterable dataset built from local shard sources.

    A :class:`Dataset` is immutable: every transformation returns a new dataset
    instance while leaving the previous one unchanged. Source data is loaded
    lazily during iteration, then passed through the appended iterator stages in
    declaration order.

    The dataset stores one of three source backends:
    - ``"tars"``: a list of tar shard paths
    - ``"jsonl"``: a list of JSONL shard paths
    - ``"parquet"``: a list of parquet shard paths
    """

    context: RuntimeContext
    source_kind: SourceKind

    _source: SourceStore
    _source_shape: SourceShape
    _stages: tuple[StageSpec, ...]
    _resample: bool
    _sidecar_specs: tuple[SidecarSpec, ...]
    _ref_fields: tuple[RefFieldSpec, ...]
    _cache_spec: CacheSpec | None
    _parquet_columns: tuple[str, ...] | None
    _parquet_batch_size: int
    _parquet_use_threads: bool

    @classmethod
    def from_tars(
        cls,
        shards: ShardInput | Sequence[ShardInput],
        context: RuntimeContext | None = None,
        resample: bool = False,
    ) -> Dataset:
        """Build a dataset from local tar shard paths.

        Args:
            shards: One or more file paths, glob specs, or brace-expansion specs.
            context: Optional execution context. If omitted, inferred from runtime.
            resample: Whether to loop shards indefinitely across rounds.

        Returns:
            A dataset whose source is the normalized tar shard path list.

        Raises:
            ValueError: If any input path does not end with ``.tar``.
        """
        runtime_context = RuntimeContext.from_runtime() if context is None else context
        normalized_shards = normalize_paths(shards)
        if not all(path.endswith(".tar") for path in normalized_shards):
            msg = f"[InvalidSourceType] expected .tar inputs, got={normalized_shards!r}"
            raise ValueError(msg)
        return cls._build_dataset(
            source=normalized_shards,
            source_shape="tar_paths",
            context=runtime_context,
            source_kind="tars",
            resample=resample,
        )

    @classmethod
    def from_jsonl(
        cls,
        files: ShardInput | Sequence[ShardInput],
        context: RuntimeContext | None = None,
        resample: bool = False,
        *,
        group_key: str | None = None,
        num_shards: int | None = None,
        target_samples_per_shard: int | None = None,
        spill_buckets: int = 128,
        output_dir: str | os.PathLike[str] | None = None,
    ) -> Dataset:
        """Build a dataset from local JSONL files.

        When only ``files`` is provided, the dataset reads the JSONL files
        directly during iteration. When any sharding argument is supplied, the
        input files are first materialized into local JSONL shard files so that
        later reads can be scheduled shard-by-shard, similar to tar sources.

        Args:
            files: One or more JSONL file paths, globs, or brace-expansion specs.
            context: Optional execution context. If omitted, inferred from runtime.
            resample: Whether to loop JSONL shard inputs indefinitely across rounds.
            group_key: Optional string field used to improve locality while
                spilling rows into temporary buckets. For linked-tar
                references, the portion before ``#`` is used for grouping.
                If the field is a list of references, the first reference is
                used.
            num_shards: Optional exact number of final local JSONL shards to
                materialize before iteration.
            target_samples_per_shard: Optional target shard size used to derive
                the final shard count when ``num_shards`` is not provided.
            spill_buckets: Number of temporary bucket files used during spill
                sharding. Higher values reduce per-bucket size at the cost of
                more temporary files.
            output_dir: Optional directory used to store materialized JSONL
                shard files. When omitted, a cache directory under the current
                working directory is used.

        Returns:
            A dataset backed by normalized JSONL shard paths.

        Raises:
            ValueError: If any input path does not end with ``.jsonl`` or if the
                sharding parameters are invalid.
        """

        runtime_context = RuntimeContext.from_runtime() if context is None else context
        normalized_files = normalize_paths(files)
        if not all(path.endswith(".jsonl") for path in normalized_files):
            msg = f"[InvalidSourceType] expected .jsonl inputs, got={normalized_files!r}"
            raise ValueError(msg)

        shard_paths = normalized_files
        if group_key is not None or num_shards is not None or target_samples_per_shard is not None:
            shard_paths = materialize_jsonl_shards(
                normalized_files,
                group_key=group_key,
                num_shards=num_shards,
                target_samples_per_shard=target_samples_per_shard,
                spill_buckets=spill_buckets,
                output_dir=output_dir,
            )

        return cls._build_dataset(
            source=shard_paths,
            source_shape="jsonl_paths",
            context=runtime_context,
            source_kind="jsonl",
            resample=resample,
        )

    @classmethod
    def from_parquet(
        cls,
        files: ShardInput | Sequence[ShardInput],
        context: RuntimeContext | None = None,
        resample: bool = False,
        *,
        columns: Sequence[str] | None = None,
        batch_size: int = 65536,
        use_threads: bool = True,
    ) -> Dataset:
        """Build a dataset from local parquet files."""

        runtime_context = RuntimeContext.from_runtime() if context is None else context
        normalized_files = normalize_paths(files)
        if not all(path.endswith(".parquet") for path in normalized_files):
            msg = f"[InvalidSourceType] expected .parquet inputs, got={normalized_files!r}"
            raise ValueError(msg)
        if batch_size <= 0:
            msg = f"[InvalidParquetBatchSize] batch_size must be > 0, got={batch_size}"
            raise ValueError(msg)
        fragments = list_parquet_fragments(normalized_files)
        if not fragments:
            msg = f"[EmptyParquetSource] no row groups found in input files {normalized_files!r}"
            raise ValueError(msg)
        return cls._build_dataset(
            source=fragments,
            source_shape="parquet_fragments",
            context=runtime_context,
            source_kind="parquet",
            resample=resample,
            parquet_columns=None if columns is None else tuple(columns),
            parquet_batch_size=batch_size,
            parquet_use_threads=use_threads,
        )

    @classmethod
    def from_source(
        cls,
        shards: ShardInput | Sequence[ShardInput],
        context: RuntimeContext | None = None,
        resample: bool = False,
        *,
        group_key: str | None = None,
        num_shards: int | None = None,
        target_samples_per_shard: int | None = None,
        spill_buckets: int = 128,
        output_dir: str | os.PathLike[str] | None = None,
        columns: Sequence[str] | None = None,
        batch_size: int = 65536,
        use_threads: bool = True,
    ) -> Dataset:
        """Build a dataset from local tar, JSONL, or parquet inputs.

        This compatibility constructor dispatches to :meth:`from_tars` when all
        inputs end with ``.tar``, to :meth:`from_jsonl` when all inputs end
        with ``.jsonl``, and to :meth:`from_parquet` when all inputs end with
        ``.parquet``.

        Args:
            shards: One or more tar, JSONL, or parquet paths, globs, or brace-expansion specs.
            context: Optional execution context. If omitted, inferred from runtime.
            resample: Whether to loop the selected source shards indefinitely.
            group_key: JSONL-only spill sharding key forwarded to
                :meth:`from_jsonl`.
            num_shards: JSONL-only exact shard count forwarded to
                :meth:`from_jsonl`.
            target_samples_per_shard: JSONL-only target shard size forwarded to
                :meth:`from_jsonl`.
            spill_buckets: JSONL-only temporary spill bucket count forwarded to
                :meth:`from_jsonl`.
            output_dir: JSONL-only output directory for materialized shard files.
            columns: Parquet-only column selection forwarded to
                :meth:`from_parquet`.
            batch_size: Parquet-only record batch size forwarded to
                :meth:`from_parquet`.
            use_threads: Parquet-only threaded decode flag forwarded to
                :meth:`from_parquet`.

        Returns:
            A dataset backed by tar shard paths, JSONL shard paths, or parquet shard paths.

        Raises:
            ValueError: If inputs mix extensions, use an unsupported extension,
                or pass JSONL-only sharding arguments for tar or parquet sources.
        """

        runtime_context = RuntimeContext.from_runtime() if context is None else context
        normalized_shards = normalize_paths(shards)

        if all(path.endswith(".jsonl") for path in normalized_shards):
            if columns is not None or batch_size != 65536 or not use_threads:
                msg = "Parquet arguments are not supported for jsonl sources"
                raise ValueError(msg)
            return cls.from_jsonl(
                normalized_shards,
                context=runtime_context,
                resample=resample,
                group_key=group_key,
                num_shards=num_shards,
                target_samples_per_shard=target_samples_per_shard,
                spill_buckets=spill_buckets,
                output_dir=output_dir,
            )
        if all(path.endswith(".tar") for path in normalized_shards):
            if (
                group_key is not None
                or num_shards is not None
                or target_samples_per_shard is not None
                or spill_buckets != 128
                or output_dir is not None
                or columns is not None
                or batch_size != 65536
                or not use_threads
            ):
                msg = "JSONL/parquet arguments are not supported for tar sources"
                raise ValueError(msg)
            return cls.from_tars(normalized_shards, context=runtime_context, resample=resample)
        if all(path.endswith(".parquet") for path in normalized_shards):
            if (
                group_key is not None
                or num_shards is not None
                or target_samples_per_shard is not None
                or spill_buckets != 128
                or output_dir is not None
            ):
                msg = "JSONL sharding arguments are not supported for parquet sources"
                raise ValueError(msg)
            return cls.from_parquet(
                normalized_shards,
                context=runtime_context,
                resample=resample,
                columns=columns,
                batch_size=batch_size,
                use_threads=use_threads,
            )

        msg = f"[InvalidSourceType] all inputs must be .jsonl, all .parquet, or all .tar, got={normalized_shards!r}"
        raise ValueError(msg)

    @classmethod
    def _build_dataset(
        cls,
        *,
        source: SourceStore,
        source_shape: SourceShape,
        context: RuntimeContext,
        source_kind: SourceKind,
        resample: bool,
        parquet_columns: tuple[str, ...] | None = None,
        parquet_batch_size: int = 65536,
        parquet_use_threads: bool = True,
    ) -> Dataset:
        """Build one dataset instance from normalized source metadata.

        Args:
            source: Normalized source payload stored by the dataset.
            source_shape: Concrete shape of ``source``.
            context: Runtime execution context to attach to the dataset.
            source_kind: High-level source backend kind.
            resample: Whether the dataset should produce infinite shuffled rounds.

        Returns:
            A fully initialized :class:`Dataset`.

        Raises:
            ValueError: If ``source_shape`` and ``source_kind`` do not match.
        """

        if source_shape == "tar_paths" and source_kind != "tars":
            msg = f"[InvalidSourceShape] tar_paths requires source_kind='tars', got={source_kind!r}"
            raise ValueError(msg)
        if source_shape == "jsonl_paths" and source_kind != "jsonl":
            msg = f"[InvalidSourceShape] jsonl_paths requires source_kind='jsonl', got={source_kind!r}"
            raise ValueError(msg)
        if source_shape == "parquet_fragments" and source_kind != "parquet":
            msg = f"[InvalidSourceShape] parquet_fragments requires source_kind='parquet', got={source_kind!r}"
            raise ValueError(msg)

        return cls(
            _source=source,
            _source_shape=source_shape,
            _stages=(),
            context=context,
            source_kind=source_kind,
            _resample=resample,
            _sidecar_specs=(),
            _ref_fields=(),
            _cache_spec=None,
            _parquet_columns=parquet_columns,
            _parquet_batch_size=parquet_batch_size,
            _parquet_use_threads=parquet_use_threads,
        )

    def join(self, sidecars: Sequence[SidecarSpec]) -> Dataset:
        """Attach sidecar tar specs for shard-level field merges.

        Args:
            sidecars: Sequence of ``(name, path_resolver)`` pairs. Each
                ``path_resolver`` receives a main tar shard path and must return
                the matching sidecar tar shard path.

        Returns:
            A new dataset that merges sidecar fields into each main tar sample
            during iteration.

        Raises:
            ValueError: If the dataset source is not tar-based.
        """

        if self.source_kind != "tars":
            msg = "`join` currently supports only tar sources."
            raise ValueError(msg)
        return dataclass_replace(self, _sidecar_specs=self._sidecar_specs + tuple(sidecars))

    def resolve_refs(self, ref_fields: Sequence[RefFieldSpec]) -> Dataset:
        """Resolve linked tar URIs in selected JSONL fields during iteration.

        Args:
            ref_fields: Sequence of ``(field_name, base_dir)`` pairs that
                identify JSONL fields containing ``tar://...#...`` or
                ``...tar#...`` references. Field values may be a single string
                URI or a list of string URIs.

        Returns:
            A new dataset that replaces selected URI fields with raw bytes read
            from the referenced tar members at iteration time.

        Raises:
            ValueError: If the dataset source is not JSONL-based.
        """

        if self.source_kind != "jsonl":
            msg = "`resolve_refs` currently supports only jsonl sources."
            raise ValueError(msg)
        return dataclass_replace(self, _ref_fields=self._ref_fields + tuple(ref_fields))

    def _append_stage(self, spec: StageSpec) -> Dataset:
        """Return a new dataset with one extra lazy stage.

        Args:
            spec: Stage spec to append to the pipeline.

        Returns:
            A new :class:`Dataset` sharing the same source and context.
        """

        return dataclass_replace(self, _stages=self._stages + (spec,))

    def map(self, fn: Callable[[object], object]) -> Dataset:
        """Append a lazy map stage.

        Args:
            fn: Callable applied to each sample yielded by the upstream stage.

        Returns:
            A new dataset that applies ``fn`` lazily during iteration.
        """
        from ..cache.fingerprint import callable_fingerprint
        from ..cache.materialize import make_cache_aware_map

        fn_fp = callable_fingerprint(fn)

        def _apply(data: Iterable[object]) -> Iterable[object]:
            return map_samples(data, fn)

        spec = StageSpec(
            kind="map",
            apply=_apply,
            fn_fingerprint=fn_fp,
            cache_trace_policy="traceable",
            cache_stage=make_cache_aware_map(fn, fn_fp),
        )
        return self._append_stage(spec)

    def shuffle(self, buffer_size: int, initial: int | None = None) -> Dataset:
        """Append a deterministic sample-level shuffle stage.

        Args:
            buffer_size: Maximum number of samples to keep in the randomization
                buffer.
            initial: Minimum number of buffered samples before the stage starts
                yielding values. Defaults to ``buffer_size``.

        Returns:
            A new dataset with bounded-memory shuffling applied lazily.
        """

        def _apply(data: Iterable[object]) -> Iterable[object]:
            seed = self.context.sample_shuffle_seed
            rng = random.Random(seed)
            return shuffle_samples(data, buffer_size=buffer_size, initial=initial, rng=rng)

        spec = StageSpec(
            kind="shuffle",
            apply=_apply,
            fn_fingerprint="",
            cache_trace_policy="unsupported",
            cache_stage=None,
        )
        return self._append_stage(spec)

    def batch(
        self,
        batch_size: int,
        drop_last: bool = False,
        collate_fn: Callable[[list[object]], object] | None = None,
    ) -> Dataset:
        """Append a batching stage.

        Args:
            batch_size: Number of samples per yielded batch.
            drop_last: Whether to drop the final incomplete batch.
            collate_fn: Optional callable that transforms each list of samples
                into a user-defined batch object.

        Returns:
            A new dataset that yields batches instead of individual samples.
        """

        def _apply(data: Iterable[object]) -> Iterable[object]:
            return batch_samples(
                data,
                batch_size=batch_size,
                drop_last=drop_last,
                collate_fn=collate_fn,
            )

        spec = StageSpec(
            kind="batch",
            apply=_apply,
            fn_fingerprint="",
            cache_trace_policy="unsupported",
            cache_stage=None,
        )
        return self._append_stage(spec)

    def assemble(
        self,
        factory: Callable[[RuntimeContext], Assembler[object, object]],
        *,
        drop_last: bool = False,
    ) -> Dataset:
        """Append a stateful assembly stage.

        Args:
            factory: Callable that builds one fresh assembler for each iterator
                execution. The assembler may consume multiple upstream samples
                before yielding one or more downstream outputs.
            drop_last: Whether to discard unfinished tail state instead of
                delegating it to the assembler's final flush.

        Returns:
            A new dataset that assembles the upstream sample stream lazily.
        """
        from ..cache.fingerprint import callable_fingerprint
        from ..cache.materialize import make_cache_aware_assemble

        fn_fp = callable_fingerprint(factory)
        ctx = self.context

        def _apply(data: Iterable[object]) -> Iterable[object]:
            return assemble_samples(
                data,
                factory=lambda: factory(ctx),
                drop_last=drop_last,
            )

        spec = StageSpec(
            kind="assemble",
            apply=_apply,
            fn_fingerprint=fn_fp,
            cache_trace_policy="traceable",
            cache_stage=make_cache_aware_assemble(
                lambda: factory(ctx),
                fn_fp,
                drop_last,
            ),
        )
        return self._append_stage(spec)

    def unbatch(self) -> Dataset:
        """Append an unbatching stage.

        Returns:
            A new dataset that expands list, tuple, or dict-style batches back
            into individual samples during iteration.
        """

        def _apply(data: Iterable[object]) -> Iterable[object]:
            return unbatch_samples(data)

        spec = StageSpec(
            kind="unbatch",
            apply=_apply,
            fn_fingerprint="",
            cache_trace_policy="unsupported",
            cache_stage=None,
        )
        return self._append_stage(spec)

    def cache(
        self,
        groups: Sequence[Sequence[str]] | None = None,
        *,
        show_progress: bool = True,
    ) -> Dataset:
        """Insert a cache boundary after the current pipeline stages.

        On the first :meth:`__iter__` call the pipeline up to this boundary is
        materialized into tar files on disk.  Subsequent iterations read from
        those cached tars, skipping all upstream computation.

        Args:
            groups: Optional field grouping.  Each inner sequence specifies a
                set of fields that should be stored together in one tar.
                ``None`` (default) places all non-metadata fields in a single
                tar.  Fields not covered by any explicit group are stored as
                singleton groups.
            show_progress: Whether to print warm-up progress to ``stderr``.

        Returns:
            A new dataset with the cache boundary registered.

        Raises:
            ValueError: If a cache boundary has already been registered on this
                dataset.
        """
        if self._cache_spec is not None:
            msg = "[CacheError] only one .cache() boundary is allowed per dataset"
            raise ValueError(msg)

        from ..cache.fingerprint import hash_bytes

        normalized_groups: tuple[tuple[str, ...], ...] | None
        if groups is None:
            normalized_groups = None
        else:
            normalized_groups = tuple(tuple(g) for g in groups)

        # Build plan fingerprint from all traceable pre-cache stage fingerprints.
        pre_stage_fps = [spec.fn_fingerprint for spec in self._stages if spec.cache_trace_policy == "traceable"]
        groups_repr = repr(normalized_groups)
        plan_fp = hash_bytes(*pre_stage_fps, groups_repr)

        cache_spec = CacheSpec(
            boundary_index=len(self._stages),
            groups=normalized_groups,
            show_progress=show_progress,
            plan_fingerprint=plan_fp,
        )

        return dataclass_replace(self, _cache_spec=cache_spec)

    def __iter__(self) -> Iterator[object]:
        """Materialize and run the full lazy pipeline.

        Returns:
            An iterator over samples or batch objects produced by the dataset's
            source backend followed by all appended pipeline stages.
        """

        if self._cache_spec is not None:
            yield from self._iter_with_cache()
            return

        context = self.context.resolve_current_process()
        stream: Iterable[object]

        tar_key_dot_level = int(os.environ.get("LOADER_TAR_KEY_DOT_LEVEL", 1))
        if self.source_kind == "tars":
            tar_paths = self._as_tar_source()
            shard_stream = iter_items(
                tar_paths,
                context=context,
                resample=self._resample,
            )
            stream = iter_tars(
                shard_stream,
                key_dot_level=tar_key_dot_level,
                sidecars=self._sidecar_specs,
            )
        elif self.source_kind == "jsonl":
            tar_cache_size = int(os.environ.get("LOADER_JSONL_TAR_CACHE_SIZE", 8))
            shard_stream = iter_items(
                self._as_jsonl_source(),
                context=context,
                resample=self._resample,
            )
            stream = iter_jsonls(
                shard_stream,
                ref_fields=self._ref_fields,
                key_dot_level=tar_key_dot_level,
                tar_cache_size=tar_cache_size,
            )
        else:
            shard_stream = iter_items(
                self._as_parquet_source(),
                context=context,
                resample=self._resample,
            )
            stream = iter_parquets(
                shard_stream,
                columns=self._parquet_columns,
                batch_size=self._parquet_batch_size,
                use_threads=self._parquet_use_threads,
            )

        for spec in self._stages:
            stream = spec.apply(stream)

        yield from stream

    def _iter_with_cache(self) -> Iterator[object]:
        """Run the pipeline using the cache layer.

        On cache miss, builds a signature-annotated source stream, runs
        pre-cache stages through their cache-aware variants, and
        materializes the results into per-shard group tars.  On cache hit,
        reads directly from the cached tars.  In both cases, post-cache
        stages are applied on the output before yielding.

        Yields:
            Samples or batch objects produced by reading cached tars
            followed by all post-cache pipeline stages.
        """
        from ..cache.materialize import (
            iter_cache_shard,
            iter_jsonls_with_sigs,
            iter_parquets_with_sigs,
            iter_tars_with_sigs,
            read_manifest,
            wait_for_cache,
            warmup_cache,
        )

        cache_spec = self._cache_spec
        assert cache_spec is not None

        context = self.context.resolve_current_process()
        tar_key_dot_level = int(os.environ.get("LOADER_TAR_KEY_DOT_LEVEL", 1))
        tar_cache_size = int(os.environ.get("LOADER_JSONL_TAR_CACHE_SIZE", 8))

        assigned_parquet_fragments: list[ParquetFragment] | None = None
        if self.source_kind == "tars":
            all_shards = self._as_tar_source()
            assigned_shards = list(iter_items(all_shards, context=context, resample=False))
        elif self.source_kind == "jsonl":
            all_shards = self._as_jsonl_source()
            assigned_shards = list(iter_items(all_shards, context=context, resample=False))
        else:
            assigned_parquet_fragments = list(iter_items(self._as_parquet_source(), context=context, resample=False))
            assigned_shards = [fragment.cache_key for fragment in assigned_parquet_fragments]

        pre_specs = self._stages[: cache_spec.boundary_index]
        post_specs = self._stages[cache_spec.boundary_index :]

        # Check whether all assigned shards already have valid caches.
        all_cached = all(read_manifest(shard, cache_spec.plan_fingerprint) is not None for shard in assigned_shards)

        if not all_cached:
            if not context.is_cache_leader:
                # Non-leader TP co-member: skip warm-up, wait for the leader
                # to finish building the cache for our shared shards.
                wait_for_cache(
                    assigned_shards,
                    cache_spec.plan_fingerprint,
                    show_progress=cache_spec.show_progress,
                )
            else:
                # --- Warm-up: build the cache. ---
                # Build source stream with signature annotation.
                if self.source_kind == "tars":
                    shard_stream: Iterator[str] = iter(assigned_shards)
                    source_stream = iter_tars_with_sigs(
                        shard_stream,
                        key_dot_level=tar_key_dot_level,
                        sidecars=self._sidecar_specs,
                    )
                elif self.source_kind == "jsonl":
                    shard_stream = iter(assigned_shards)
                    source_stream = iter_jsonls_with_sigs(
                        shard_stream,
                        ref_fields=self._ref_fields,
                        key_dot_level=tar_key_dot_level,
                        tar_cache_size=tar_cache_size,
                    )
                else:
                    assert assigned_parquet_fragments is not None
                    source_stream = iter_parquets_with_sigs(
                        iter(assigned_parquet_fragments),
                        columns=self._parquet_columns,
                        batch_size=self._parquet_batch_size,
                        use_threads=self._parquet_use_threads,
                    )

                # Apply pre-cache stages with cache-aware versions where available.
                pre_stream: Iterable[object] = source_stream
                unsupported_kinds: list[str] = []
                for spec in pre_specs:
                    if spec.cache_stage is not None:
                        pre_stream = spec.cache_stage(pre_stream)
                    else:
                        if spec.kind in _UNSUPPORTED_CACHE_KINDS:
                            unsupported_kinds.append(spec.kind)
                        pre_stream = spec.apply(pre_stream)

                warmup_cache(
                    assigned_shards=assigned_shards,
                    pre_cache_stream=iter(pre_stream),
                    groups_spec=cache_spec.groups,
                    plan_fingerprint=cache_spec.plan_fingerprint,
                    show_progress=cache_spec.show_progress,
                    unsupported_stage_kinds=unsupported_kinds,
                )

        # --- Serve path: read from cache tars. ---
        def _cache_source() -> Iterator[object]:
            for shard in assigned_shards:
                yield from iter_cache_shard(shard, cache_spec.plan_fingerprint)

        stream: Iterable[object] = _cache_source()
        for spec in post_specs:
            stream = spec.apply(stream)

        yield from stream

    def _as_tar_source(self) -> list[str]:
        """Return tar shard paths with O(1) source-shape validation.

        Returns:
            The underlying tar shard path list.

        Raises:
            TypeError: If this dataset is not backed by tar shard paths.
        """

        if self._source_shape != "tar_paths":
            msg = "[InvalidSourceShape] expected tar_paths source shape"
            raise TypeError(msg)
        return cast(list[str], self._source)

    def _as_jsonl_source(self) -> list[str]:
        """Return JSONL shard paths with O(1) source-shape validation.

        Returns:
            The underlying JSONL shard path list.

        Raises:
            TypeError: If this dataset is not backed by JSONL shard paths.
        """

        if self._source_shape != "jsonl_paths":
            msg = "[InvalidSourceShape] expected jsonl_paths source shape"
            raise TypeError(msg)
        return cast(list[str], self._source)

    def _as_parquet_source(self) -> list[ParquetFragment]:
        """Return parquet row-group fragments with O(1) source-shape validation."""

        if self._source_shape != "parquet_fragments":
            msg = "[InvalidSourceShape] expected parquet_fragments source shape"
            raise TypeError(msg)
        return cast(list[ParquetFragment], self._source)
