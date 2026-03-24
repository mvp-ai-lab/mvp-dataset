"""Chainable iterator dataset API for mvp-dataset."""

from __future__ import annotations

import importlib
import os
import random
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Literal, cast

from ..core.types import (
    Assembler,
    RefFieldSpec,
    RuntimeContext,
    ShardInput,
    SidecarSpec,
    Stage,
    TarSelectPreprocessor,
)
from ..sources import iter_jsonls, iter_tars
from ..sources.jsonl import materialize_jsonl_shards
from ..sources.tar import cache_tar_path, normalize_select_output, sample_fields, write_select_cache
from ..utils.selection import normalize_selected_keys
from ..utils.sharding import iter_items
from ..utils.url import normalize_paths
from .ops import (
    assemble_samples,
    batch_samples,
    map_samples,
    shuffle_samples,
    unbatch_samples,
)

SourceKind = Literal["jsonl", "tars"]
SourceStore = list[str]
SourceShape = Literal["tar_paths", "jsonl_paths"]


def _cache_sidecar_spec(key: str) -> SidecarSpec:
    """Build the resolver used for cached select sidecars."""

    def resolver(shard_path: str, *, selected_key: str = key) -> str:
        return cache_tar_path(shard_path, selected_key)

    return key, resolver


def torch_iterabledataset_class(
    import_module: Callable[[str], ModuleType] = importlib.import_module,
) -> type:
    """Resolve ``torch.utils.data.IterableDataset`` with a no-torch fallback."""

    try:
        torch_utils_data = import_module("torch.utils.data")
    except Exception:  # noqa: BLE001

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

    The dataset stores one of two source backends:
    - ``"tars"``: a list of tar shard paths
    - ``"jsonl"``: a list of JSONL shard paths
    """

    context: RuntimeContext
    source_kind: SourceKind

    _source: SourceStore
    _source_shape: SourceShape
    _stages: tuple[Stage, ...]
    _resample: bool
    _sidecar_specs: tuple[SidecarSpec, ...]
    _ref_fields: tuple[RefFieldSpec, ...]
    _selected_keys: tuple[str, ...]

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
                spilling rows into temporary buckets. For ``tar://...#...``
                references, the portion before ``#`` is used for grouping.
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
    ) -> Dataset:
        """Build a dataset from local tar or JSONL inputs.

        This compatibility constructor dispatches to :meth:`from_tars` when all
        inputs end with ``.tar`` and to :meth:`from_jsonl` when all inputs end
        with ``.jsonl``.

        Args:
            shards: One or more tar or JSONL paths, globs, or brace-expansion specs.
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

        Returns:
            A dataset backed by tar shard paths or JSONL shard paths.

        Raises:
            ValueError: If inputs mix extensions, use an unsupported extension,
                or pass JSONL-only sharding arguments for tar sources.
        """

        runtime_context = RuntimeContext.from_runtime() if context is None else context
        normalized_shards = normalize_paths(shards)

        if all(path.endswith(".jsonl") for path in normalized_shards):
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
            ):
                msg = "JSONL sharding arguments are not supported for tar sources"
                raise ValueError(msg)
            return cls.from_tars(normalized_shards, context=runtime_context, resample=resample)

        msg = f"[InvalidSourceType] all inputs must be .jsonl or all must be .tar, got={normalized_shards!r}"
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

        return cls(
            _source=source,
            _source_shape=source_shape,
            _stages=(),
            context=context,
            source_kind=source_kind,
            _resample=resample,
            _sidecar_specs=(),
            _ref_fields=(),
            _selected_keys=(),
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
        return Dataset(
            _source=self._source,
            _source_shape=self._source_shape,
            _stages=self._stages,
            context=self.context,
            source_kind=self.source_kind,
            _resample=self._resample,
            _sidecar_specs=self._sidecar_specs + tuple(sidecars),
            _ref_fields=self._ref_fields,
            _selected_keys=self._selected_keys,
        )

    def resolve_refs(self, ref_fields: Sequence[RefFieldSpec]) -> Dataset:
        """Resolve ``tar://`` URIs in selected JSONL fields during iteration.

        Args:
            ref_fields: Sequence of ``(field_name, base_dir)`` pairs that
                identify JSONL fields containing ``tar://`` references.

        Returns:
            A new dataset that replaces selected URI fields with raw bytes read
            from the referenced tar members at iteration time.

        Raises:
            ValueError: If the dataset source is not JSONL-based.
        """

        if self.source_kind != "jsonl":
            msg = "`resolve_refs` currently supports only jsonl sources."
            raise ValueError(msg)
        return Dataset(
            _source=self._source,
            _source_shape=self._source_shape,
            _stages=self._stages,
            context=self.context,
            source_kind=self.source_kind,
            _resample=self._resample,
            _sidecar_specs=self._sidecar_specs,
            _ref_fields=self._ref_fields + tuple(ref_fields),
            _selected_keys=self._selected_keys,
        )

    def select(
        self,
        keys: Sequence[str],
        *,
        preprocessors: Mapping[str, TarSelectPreprocessor] | None = None,
    ) -> Dataset:
        """Select tar field groups and materialize missing groups into cached tars.

        Args:
            keys: Field-group prefixes to load. A key matches a field when the
                field name is exactly the key or starts with ``"{key}."``.
            preprocessors: Optional mapping used to create missing field groups.
                Each callable receives the merged sample produced by the current
                tar source (including any existing joins) and returns either raw
                bytes for ``key`` or a mapping like ``{"depth.png": payload}``.

        Returns:
            A new dataset that loads only the requested field groups, using
            cache sidecars named ``<shard_stem>_<key>.tar`` for groups that were
            materialized on demand.

        Raises:
            ValueError: If the dataset source is not tar-based, if requested key
                availability is inconsistent across shards, or if a missing key
                has no cache and no preprocessor.
        """

        if self.source_kind != "tars":
            msg = "`select` currently supports only tar sources."
            raise ValueError(msg)

        selected_keys = normalize_selected_keys(keys)
        preprocessor_map = {} if preprocessors is None else dict(preprocessors)
        key_dot_level = int(os.environ.get("LOADER_TAR_KEY_DOT_LEVEL", 1))
        tar_paths = self._as_tar_source()
        extra_sidecars: list[SidecarSpec] = []

        for key in selected_keys:
            availability = [
                self._shard_has_selected_key(shard_path, key=key, key_dot_level=key_dot_level)
                for shard_path in tar_paths
            ]
            if all(availability):
                continue
            if any(availability):
                msg = (
                    f"[InconsistentSelectKey] key={key!r} is present in only a subset of shards; "
                    "materialize the dataset into consistent shards before using select"
                )
                raise ValueError(msg)

            cached_ready = [self._cache_has_selected_key(shard_path, key=key, key_dot_level=key_dot_level) for shard_path in tar_paths]
            if not all(cached_ready):
                preprocessor = preprocessor_map.get(key)
                if preprocessor is None:
                    missing_shards = [str(Path(path).name) for path, ready in zip(tar_paths, cached_ready, strict=True) if not ready]
                    msg = (
                        f"[MissingSelectPreprocessor] key={key!r} is not present in the source shards and "
                        f"cache shards are missing for {missing_shards!r}"
                    )
                    raise ValueError(msg)
                for shard_path, ready in zip(tar_paths, cached_ready, strict=True):
                    if ready:
                        continue
                    self._materialize_select_cache(
                        shard_path,
                        key=key,
                        key_dot_level=key_dot_level,
                        preprocessor=preprocessor,
                    )

            extra_sidecars.append(_cache_sidecar_spec(key))

        return Dataset(
            _source=self._source,
            _source_shape=self._source_shape,
            _stages=self._stages,
            context=self.context,
            source_kind=self.source_kind,
            _resample=self._resample,
            _sidecar_specs=self._sidecar_specs + tuple(extra_sidecars),
            _ref_fields=self._ref_fields,
            _selected_keys=selected_keys,
        )

    def _append_stage(self, stage: Stage) -> Dataset:
        """Return a new dataset with one extra lazy stage.

        Args:
            stage: Iterator transformation applied after source loading.

        Returns:
            A dataset that shares the same source and context but appends
            ``stage`` to the pipeline.
        """

        return Dataset(
            _source=self._source,
            _source_shape=self._source_shape,
            _stages=self._stages + (stage,),
            context=self.context,
            source_kind=self.source_kind,
            _resample=self._resample,
            _sidecar_specs=self._sidecar_specs,
            _ref_fields=self._ref_fields,
            _selected_keys=self._selected_keys,
        )

    def map(self, fn: Callable[[object], object]) -> Dataset:
        """Append a lazy map stage.

        Args:
            fn: Callable applied to each sample yielded by the upstream stage.

        Returns:
            A new dataset that applies ``fn`` lazily during iteration.
        """

        def stage(data: Iterable[object]) -> Iterable[object]:
            return map_samples(data, fn)

        return self._append_stage(stage)

    def shuffle(self, buffer_size: int, initial: int | None = None) -> Dataset:
        """Ignore shuffle requests and return the dataset unchanged."""

        del buffer_size, initial
        return self

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

        def stage(data: Iterable[object]) -> Iterable[object]:
            return batch_samples(
                data,
                batch_size=batch_size,
                drop_last=drop_last,
                collate_fn=collate_fn,
            )

        return self._append_stage(stage)

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

        def stage(data: Iterable[object]) -> Iterable[object]:
            return assemble_samples(
                data,
                factory=lambda: factory(self.context),
                drop_last=drop_last,
            )

        return self._append_stage(stage)

    def unbatch(self) -> Dataset:
        """Append an unbatching stage.

        Returns:
            A new dataset that expands list, tuple, or dict-style batches back
            into individual samples during iteration.
        """

        def stage(data: Iterable[object]) -> Iterable[object]:
            return unbatch_samples(data)

        return self._append_stage(stage)

    def __iter__(self) -> Iterator[object]:
        """Materialize and run the full lazy pipeline.

        Returns:
            An iterator over samples or batch objects produced by the dataset's
            source backend followed by all appended pipeline stages.
        """

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
                field_prefixes=self._selected_keys if self._selected_keys else None,
            )
        else:
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

        for stage in self._stages:
            stream = stage(stream)

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

    def _shard_has_selected_key(
        self,
        shard_path: str,
        *,
        key: str,
        key_dot_level: int,
    ) -> bool:
        """Return whether one shard already exposes the selected field group."""

        for sample in iter_tars(
            iter((shard_path,)),
            key_dot_level=key_dot_level,
            sidecars=self._sidecar_specs,
            field_prefixes=(key,),
        ):
            if len(sample_fields(cast(dict[str, object], sample))) > 0:
                return True
        return False

    def _cache_has_selected_key(
        self,
        shard_path: str,
        *,
        key: str,
        key_dot_level: int,
    ) -> bool:
        """Return whether the cached sidecar exists and contains the selected key."""

        cached_path = cache_tar_path(shard_path, key)
        if not Path(cached_path).is_file():
            return False
        for sample in iter_tars(
            iter((cached_path,)),
            key_dot_level=key_dot_level,
            field_prefixes=(key,),
        ):
            if len(sample_fields(cast(dict[str, object], sample))) > 0:
                return True
        return False

    def _materialize_select_cache(
        self,
        shard_path: str,
        *,
        key: str,
        key_dot_level: int,
        preprocessor: TarSelectPreprocessor,
    ) -> None:
        """Create one cached select sidecar for a missing field group."""

        source_samples = iter_tars(
            iter((shard_path,)),
            key_dot_level=key_dot_level,
            sidecars=self._sidecar_specs,
        )

        def cached_samples() -> Iterator[dict[str, object]]:
            produced_any = False
            for sample in source_samples:
                normalized_fields = normalize_select_output(key, preprocessor(cast(dict[str, object], sample)))
                cached_sample: dict[str, object] = {"__key__": cast(dict[str, object], sample)["__key__"]}
                cached_sample.update(normalized_fields)
                produced_any = True
                yield cached_sample
            if not produced_any:
                msg = f"[EmptySelectSource] key={key!r} shard={shard_path!r} contained no samples"
                raise ValueError(msg)

        write_select_cache(shard_path, key=key, samples=cached_samples())
