"""Chainable iterator dataset API for mvp-dataset."""

from __future__ import annotations

import importlib
import queue
import random
import shutil
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from pathlib import Path
from types import ModuleType

from mvp_dataset.cache.store import merge_lance
from mvp_dataset.utils.barrier import FileBarrier

from ..log import get_logger
from ..pipeline.ops import (
    assemble_samples,
    batch_samples,
    map_samples,
    select_samples,
    shuffle_samples,
    unbatch_samples,
)
from ..utils.sharding import iter_items
from .context import RuntimeContext
from .types import Assembler, CacheSpec, SourceKind, SourceStore, StageSpec


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
class _MapStage:
    fn: Callable[[object], object]

    def __call__(self, data: Iterable[object]) -> Iterable[object]:
        return map_samples(data, self.fn)


@dataclass(frozen=True, slots=True)
class _ShuffleStage:
    context: RuntimeContext
    buffer_size: int
    initial: int | None = None

    def __call__(self, data: Iterable[object]) -> Iterable[object]:
        runtime_context = RuntimeContext.from_runtime(base=self.context)
        seed = runtime_context.sample_shuffle_seed
        rng = random.Random(seed)
        return shuffle_samples(data, buffer_size=self.buffer_size, initial=self.initial, rng=rng)


@dataclass(frozen=True, slots=True)
class _SelectStage:
    fields: tuple[str, ...]

    def __call__(self, data: Iterable[object]) -> Iterable[object]:
        return select_samples(data, self.fields)


@dataclass(frozen=True, slots=True)
class _BatchStage:
    batch_size: int
    drop_last: bool = False
    collate_fn: Callable[[list[object]], object] | None = None

    def __call__(self, data: Iterable[object]) -> Iterable[object]:
        return batch_samples(
            data,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            collate_fn=self.collate_fn,
        )


@dataclass(frozen=True, slots=True)
class _AssemblerFactoryAdapter:
    factory: Callable[[RuntimeContext], Assembler[object, object]]
    context: RuntimeContext

    def __call__(self) -> Assembler[object, object]:
        return self.factory(self.context)


@dataclass(frozen=True, slots=True)
class _AssembleStage:
    factory: Callable[[RuntimeContext], Assembler[object, object]]
    context: RuntimeContext
    drop_last: bool = False

    def __call__(self, data: Iterable[object]) -> Iterable[object]:
        return assemble_samples(
            data,
            factory=_AssemblerFactoryAdapter(self.factory, self.context),
            drop_last=self.drop_last,
        )


@dataclass(frozen=True, slots=True)
class _UnbatchStage:
    def __call__(self, data: Iterable[object]) -> Iterable[object]:
        return unbatch_samples(data)


@dataclass(frozen=True, slots=True)
class Dataset(torch_iterabledataset_class()):
    """Chainable iterable dataset built from local shard sources.

    A :class:`Dataset` is immutable: every transformation returns a new dataset
    instance while leaving the previous one unchanged. Source data is loaded
    lazily during iteration, then passed through the appended iterator stages in
    declaration order.
    """

    context: RuntimeContext

    _source_kind: SourceKind
    _source: SourceStore
    _stages: tuple[StageSpec, ...]
    _resample: bool
    _iter_source_stream: Callable
    _cache_spec: CacheSpec | None = None

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

        fn_fp = callable_fingerprint(fn)
        spec = StageSpec(kind="map", apply=_MapStage(fn), fn_fingerprint=fn_fp, fn_ref=fn, trace_policy="traceable")
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

        spec = StageSpec(
            kind="shuffle",
            apply=_ShuffleStage(context=self.context, buffer_size=buffer_size, initial=initial),
            trace_policy="unsupported",
        )
        return self._append_stage(spec)

    def select(self, fields: list[str] | tuple[str, ...]) -> Dataset:
        """Append a lazy field projection stage.

        The stage keeps requested user fields and preserves metadata keys such
        as ``__key__`` needed by downstream cache and source logic.
        """
        from ..cache.fingerprint import hash_bytes

        selected_fields = tuple(fields)
        select_fp = hash_bytes("<select>", *selected_fields)

        spec = StageSpec(
            kind="select",
            apply=_SelectStage(selected_fields),
            fn_fingerprint=select_fp,
            trace_policy="traceable",
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

        spec = StageSpec(
            kind="batch",
            apply=_BatchStage(
                batch_size=batch_size,
                drop_last=drop_last,
                collate_fn=collate_fn,
            ),
            trace_policy="unsupported",
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

        fn_fp = callable_fingerprint(factory)
        spec = StageSpec(
            kind="assemble",
            apply=_AssembleStage(factory=factory, context=self.context, drop_last=drop_last),
            fn_fingerprint=fn_fp,
            fn_ref=factory,
            trace_policy="traceable",
        )
        return self._append_stage(spec)

    def unbatch(self) -> Dataset:
        """Append an unbatching stage.

        Returns:
            A new dataset that expands list, tuple, or dict-style batches back
            into individual samples during iteration.
        """

        spec = StageSpec(
            kind="unbatch",
            apply=_UnbatchStage(),
            trace_policy="unsupported",
        )
        return self._append_stage(spec)

    def cache(
        self,
        *,
        cache_dir: str = ".cache",
        cache_num_workers: int = 1,
        max_rows_per_fragment: int = 5000,
    ) -> Dataset:
        """Insert a cache boundary at the current position in the pipeline.

        Args:
            cache_dir: Root directory for standalone cache datasets.
            cache_num_workers: Number of worker threads used to build a full
                cache from independent source shards or fragments.

        Returns:
            A new dataset with the cache boundary recorded.

        Raises:
            ValueError: On double cache, unsupported stages before cache,
                lambda/closure usage, or incremental mode with non-Lance source.
        """
        from ..cache.fingerprint import (
            hash_bytes,
            is_stable_function,
            source_fingerprint,
        )

        if self._cache_spec is not None:
            msg = "[CacheError] only one .cache() boundary allowed"
            raise ValueError(msg)

        if cache_num_workers <= 0:
            msg = f"[CacheError] cache_num_workers must be > 0, got {cache_num_workers}"
            raise ValueError(msg)

        for spec in self._stages:
            if spec.kind in ("batch", "unbatch"):
                msg = f"[CacheError] '{spec.kind}' cannot appear before .cache()"
                raise ValueError(msg)

        for i, spec in enumerate(self._stages):
            if spec.fn_ref is not None and not is_stable_function(spec.fn_ref):
                msg = (
                    f"[CacheError] stage {i} ({spec.kind}) uses a lambda or local "
                    f"closure. Use a named module-level function for cache compatibility."
                )
                raise ValueError(msg)

        stage_fps = [s.fn_fingerprint for s in self._stages if s.fn_fingerprint and s.trace_policy == "traceable"]
        source_id = source_fingerprint(self._source)

        plan_fp = hash_bytes(source_id, *stage_fps)

        cache_spec = CacheSpec(
            boundary_index=len(self._stages),
            cache_dir=str(cache_dir),
            cache_num_workers=cache_num_workers,
            plan_fingerprint=plan_fp,
        )

        # Build cache in this stage
        import threading
        import time as _time

        from ..cache.store import _write_lance_dataset
        from ..sources.lance.dataset import LanceDataset

        logger = get_logger()
        rank = self.context.rank

        cache_root = Path(cache_spec.cache_dir).absolute()
        final_cache_uri = cache_root / f"{cache_spec.plan_fingerprint}.lance"
        local_final_cache_uri = cache_root / f"{cache_spec.plan_fingerprint}-{rank}.lance"

        if not final_cache_uri.exists():
            # remove any stale local cache from previous runs that didn't complete
            if local_final_cache_uri.exists():
                if local_final_cache_uri.is_file():
                    local_final_cache_uri.unlink()
                else:
                    shutil.rmtree(local_final_cache_uri, ignore_errors=True)

            pre_specs = self._stages
            n_source = len(self._source)

            logger.info(
                "<MVP Dataset - rank %d> cache: building %s (%d source items, %d workers)",
                rank,
                local_final_cache_uri.name,
                n_source,
                cache_spec.cache_num_workers,
            )

            def _preprocess_func(item: object) -> object:
                stream = iter([item])
                for spec in pre_specs:
                    stream = spec.apply(stream)
                return list(stream).pop()

            source_shard_stream = iter_items(
                self._source,
                context=self.context,
                resample=False,
            )
            source_stream = self._iter_source_stream(source_shard_stream)

            max_inflight = cache_spec.cache_num_workers * 2

            result_queue: queue.Queue[object] = queue.Queue(maxsize=max_inflight)
            _SENTINEL = object()
            _produced = 0
            _t0 = _time.monotonic()

            def _producer() -> None:
                """Fan-out source items to the pool, push results into the queue."""
                nonlocal _produced
                try:
                    with ThreadPoolExecutor(max_workers=cache_spec.cache_num_workers) as pool:
                        futures: deque = deque()
                        for item in source_stream:
                            futures.append(pool.submit(_preprocess_func, item))
                            while len(futures) >= max_inflight:
                                result_queue.put(futures.popleft().result())
                                _produced += 1
                        while futures:
                            result_queue.put(futures.popleft().result())
                            _produced += 1
                finally:
                    result_queue.put(_SENTINEL)

            producer_thread = threading.Thread(target=_producer, daemon=True)
            producer_thread.start()

            _consumed = 0
            _last_log_time = _t0

            def _streaming_results() -> Iterator[object]:
                nonlocal _consumed, _last_log_time
                while True:
                    item = result_queue.get()
                    if item is _SENTINEL:
                        break
                    _consumed += 1
                    now = _time.monotonic()
                    if now - _last_log_time >= 10.0:
                        elapsed = now - _t0
                        rate = _consumed / elapsed if elapsed > 0 else 0
                        logger.info(
                            "<MVP Dataset - rank %d> cache: %d/%d samples written (%.1f samples/s)",
                            rank,
                            _consumed,
                            n_source,
                            rate,
                        )
                        _last_log_time = now
                    yield item

            _write_lance_dataset(
                _streaming_results(),
                str(local_final_cache_uri),
                max_rows_per_group=max_rows_per_fragment,
            )
            producer_thread.join()

            elapsed = _time.monotonic() - _t0
            rate = _consumed / elapsed if elapsed > 0 else 0
            logger.info(
                "<MVP Dataset - rank %d> cache: done — %d samples written in %.1fs (%.1f samples/s)",
                rank,
                _consumed,
                elapsed,
                rate,
            )

            # barrier 1: wait for all ranks to finish writing per-rank lance
            FileBarrier(
                shared_path=str(cache_root / f"{cache_spec.plan_fingerprint}-write-barrier"),
                world_size=self.context.world_size,
                rank=self.context.rank,
            ).wait()

            # merge from rank 0 only
            if rank == 0:
                logger.info("<MVP Dataset - rank %d> cache: merging per-rank datasets", rank)
                merge_lance(
                    uris=[str(p) for p in sorted(cache_root.glob(f"{cache_spec.plan_fingerprint}-*.lance"))],
                    out_uri=str(final_cache_uri),
                )
                logger.info("<MVP Dataset - rank %d> cache: merge done", rank)

            # barrier 2: wait for merge to complete before any rank reads
            FileBarrier(
                shared_path=str(cache_root / f"{cache_spec.plan_fingerprint}-merge-barrier"),
                world_size=self.context.world_size,
                rank=self.context.rank,
            ).wait()

        else:
            logger.info("<MVP Dataset - rank %d> cache: loading existing %s", rank, final_cache_uri.name)

        # Replace source with cache dataset
        cache_database_draft = LanceDataset.from_source(final_cache_uri, context=self.context, resample=self._resample)

        return dataclass_replace(
            self,
            _source_kind="lance",
            _source=cache_database_draft._source,
            _iter_source_stream=cache_database_draft._iter_source_stream,
            _stages=(),
            _cache_spec=cache_spec,
        )

    def __iter__(self) -> Iterator[object]:
        """Materialize and run the full lazy pipeline.

        Returns:
            An iterator over samples or batch objects produced by the dataset's
            source backend followed by all appended pipeline stages.
        """
        context = RuntimeContext.from_runtime(base=self.context)
        stream: Iterable[object]

        source_shard_stream = iter_items(
            self._source,
            context=context,
            resample=self._resample,
        )

        stream = self._iter_source_stream(source_shard_stream)

        for spec in self._stages:
            stream = spec.apply(stream)

        yield from stream

    @classmethod
    def from_source(cls, source_kind: SourceKind, *args, **kwargs) -> Dataset:
        """Construct a dataset from a supported source type.

        See the relevant source-specific classmethod constructors for details.
        """
        if source_kind == "tars":
            from ..sources.tar.dataset import TarDataset

            return TarDataset.from_source(*args, **kwargs)
        if source_kind == "jsonl":
            from ..sources.jsonl.dataset import JsonlDataset

            return JsonlDataset.from_source(*args, **kwargs)
        if source_kind == "parquet":
            from ..sources.parquet.dataset import ParquetDataset

            return ParquetDataset.from_source(*args, **kwargs)
        if source_kind == "lance":
            from ..sources.lance.dataset import LanceDataset

            return LanceDataset.from_source(*args, **kwargs)
        msg = f"[UnsupportedSourceKind] source_kind={source_kind!r}"
        raise ValueError(msg)
