"""Chainable iterator dataset API for mvp-dataset."""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from pathlib import Path

from mvp_dataset.cache.store import merge_lance
from mvp_dataset.utils.barrier import FileBarrier

from ..log import get_logger
from ..utils.sharding import iter_items
from .context import RuntimeContext
from .stages import (
    PARALLELIZABLE_STAGE_KINDS,
    _AssembleStage,
    _BatchStage,
    _MapStage,
    _SelectStage,
    _ShuffleStage,
    _UnbatchStage,
    iter_stage_group,
    torch_iterabledataset_class,
)
from .types import Assembler, CacheSpec, SourceKind, SourceStore, StageSpec


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
        from ..cache.fingerprint import callable_fingerprint, hash_bytes

        fn_fp = hash_bytes("<assemble>", callable_fingerprint(factory), drop_last)
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
        cache_write_batch_size: int = 1024,
        max_rows_per_fragment: int = 5000,
    ) -> Dataset:
        """Insert a cache boundary at the current position in the pipeline.

        Args:
            cache_dir: Root directory for standalone cache datasets.
            cache_num_workers: Number of worker threads used to build a full
                cache from independent source shards or fragments.
            cache_write_batch_size: Number of samples buffered before each
                flush to the cache writer. Lower this for large samples to
                reduce peak memory during cache construction.

        Returns:
            A new dataset with the cache boundary recorded.

        Raises:
            ValueError: On double cache, unsupported stages before cache,
                or lambda/closure usage.
        """
        from ..cache.fingerprint import (
            hash_bytes,
            is_stable_function,
            source_fingerprint,
        )
        from ..cache.store import _write_lance_dataset
        from ..sources.lance.dataset import LanceDataset

        if self._cache_spec is not None:
            msg = "[CacheError] only one .cache() boundary allowed"
            raise ValueError(msg)

        if cache_num_workers <= 0:
            msg = f"[CacheError] cache_num_workers must be > 0, got {cache_num_workers}"
            raise ValueError(msg)

        if cache_write_batch_size <= 0:
            msg = f"[CacheError] cache_write_batch_size must be > 0, got {cache_write_batch_size}"
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
        plan_fp = hash_bytes(source_fingerprint(self._source), *stage_fps)

        cache_spec = CacheSpec(
            boundary_index=len(self._stages),
            cache_dir=str(cache_dir),
            cache_num_workers=cache_num_workers,
            cache_write_batch_size=cache_write_batch_size,
            plan_fingerprint=plan_fp,
        )

        logger = get_logger()
        rank = self.context.rank
        dp_rank = self.context.mesh.dp_rank if self.context.mesh is not None else rank
        is_dp_leader = self.context.mesh.is_dp_leader if self.context.mesh is not None else True
        cache_root = Path(cache_dir).absolute()
        final_cache_uri = cache_root / f"{plan_fp}.lance"
        rank_cache_uri = cache_root / f"{plan_fp}-dp{dp_rank}.lance"
        unsupported_pre_cache_stage_kinds = sorted(
            {
                spec.kind
                for spec in self._stages
                if spec.trace_policy == "unsupported" and spec.kind not in {"batch", "unbatch"}
            }
        )

        if unsupported_pre_cache_stage_kinds:
            logger.warning(
                "<MVP Dataset - rank %d> cache: stages %s appear before .cache() but are not "
                "included in cache fingerprinting; changing their parameters will not invalidate "
                "the existing cache automatically",
                rank,
                unsupported_pre_cache_stage_kinds,
            )

        if not final_cache_uri.exists():
            # Remove any stale per-rank cache from a previous incomplete run.
            if is_dp_leader and rank_cache_uri.exists():
                shutil.rmtree(rank_cache_uri, ignore_errors=True)

            if is_dp_leader:
                n_source = len(self._source)
                logger.info(
                    "<MVP Dataset - rank %d> cache: building %s (%d source items, %d workers, write batch=%d)",
                    rank,
                    rank_cache_uri.name,
                    n_source,
                    cache_num_workers,
                    cache_write_batch_size,
                )

                # Split stages: parallelizable prefix runs in a thread pool; the rest run serially.
                split = next(
                    (i for i, s in enumerate(self._stages) if s.kind not in PARALLELIZABLE_STAGE_KINDS),
                    len(self._stages),
                )
                parallel_specs = self._stages[:split]
                serial_specs = self._stages[split:]

                source_shard_stream = iter_items(self._source, context=self.context, resample=False)
                stream: Iterable[object] = iter_stage_group(
                    self._iter_source_stream(source_shard_stream), parallel_specs, cache_num_workers
                )
                for spec in serial_specs:
                    stream = spec.apply(stream)

                # Wrap with a progress-logging passthrough before handing off to the lance writer.
                t0 = time.monotonic()
                count = 0
                last_log = t0

                def _log_progress(data: Iterable[object]) -> Iterator[object]:
                    nonlocal count, last_log
                    for item in data:
                        count += 1
                        now = time.monotonic()
                        if now - last_log >= 30.0:
                            elapsed = now - t0
                            logger.info(
                                "<MVP Dataset - rank %d> cache: %d samples written (%.1f samples/s)",
                                rank,
                                count,
                                count / elapsed,
                            )
                            last_log = now
                        yield item

                _write_lance_dataset(
                    _log_progress(stream),
                    str(rank_cache_uri),
                    batch_size=cache_write_batch_size,
                    max_rows_per_group=max_rows_per_fragment,
                )

                elapsed = time.monotonic() - t0
                logger.info(
                    "<MVP Dataset - rank %d> cache: done — %d samples in %.1fs (%.1f samples/s)",
                    rank,
                    count,
                    elapsed,
                    count / elapsed if elapsed > 0 else 0,
                )
            else:
                logger.info(
                    "<MVP Dataset - rank %d> cache: skipping local build because this rank is not the DP leader",
                    rank,
                )

            # Barrier 1: wait for all ranks to finish writing their per-rank lance.
            FileBarrier(
                shared_path=str(cache_root / f"{plan_fp}-write-barrier"),
                world_size=self.context.world_size,
                rank=rank,
            ).wait()

            # Rank 0 merges all per-rank lance datasets into the final shared one.
            if rank == 0:
                logger.info("<MVP Dataset - rank %d> cache: merging per-rank datasets", rank)
                merge_lance(
                    uris=[str(p) for p in sorted(cache_root.glob(f"{plan_fp}-dp*.lance"))],
                    out_uri=str(final_cache_uri),
                )
                logger.info("<MVP Dataset - rank %d> cache: merge done", rank)

            # Barrier 2: all ranks wait for merge to complete before reading.
            FileBarrier(
                shared_path=str(cache_root / f"{plan_fp}-merge-barrier"),
                world_size=self.context.world_size,
                rank=rank,
            ).wait()

        else:
            logger.info("<MVP Dataset - rank %d> cache: loading existing %s", rank, final_cache_uri.name)

        cache_dataset = LanceDataset.from_source(final_cache_uri, context=self.context, resample=self._resample)
        return dataclass_replace(
            self,
            _source_kind="lance",
            _source=cache_dataset._source,
            _iter_source_stream=cache_dataset._iter_source_stream,
            _stages=(),
            _cache_spec=cache_spec,
        )

    def __iter__(self) -> Iterator[object]:
        """Materialize and run the full lazy pipeline."""
        context = RuntimeContext.from_runtime(base=self.context)
        source_shard_stream = iter_items(self._source, context=context, resample=self._resample)
        stream: Iterable[object] = self._iter_source_stream(source_shard_stream)
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
