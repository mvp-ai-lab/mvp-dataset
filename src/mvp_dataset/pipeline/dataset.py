"""Chainable iterator dataset API for mvp-dataset."""

from __future__ import annotations

import importlib
import os
import random
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Literal, cast

from ..core.types import RefFieldSpec, RuntimeContext, ShardInput, SidecarSpec, Stage
from ..sources import iter_jsonls, iter_tars
from ..sources.jsonl import materialize_jsonl_shards
from ..utils.sharding import iter_items
from ..utils.url import normalize_paths
from .ops import batch_samples, map_samples, shuffle_samples, unbatch_samples

SourceKind = Literal["jsonl", "tars"]
SourceStore = list[str]
SourceShape = Literal["tar_paths", "jsonl_paths"]


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
        """Append a deterministic sample-level shuffle stage.

        Args:
            buffer_size: Maximum number of samples to keep in the randomization
                buffer.
            initial: Minimum number of buffered samples before the stage starts
                yielding values. Defaults to ``buffer_size``.

        Returns:
            A new dataset with bounded-memory shuffling applied lazily.
        """

        def stage(data: Iterable[object]) -> Iterable[object]:
            seed = self.context.sample_shuffle_seed
            rng = random.Random(seed)
            return shuffle_samples(data, buffer_size=buffer_size, initial=initial, rng=rng)

        return self._append_stage(stage)

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
