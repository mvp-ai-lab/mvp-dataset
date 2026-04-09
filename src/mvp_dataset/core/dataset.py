"""Chainable iterator dataset API for mvp-dataset."""

from __future__ import annotations

import importlib
import random
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from types import ModuleType

from ..pipeline.ops import (
    assemble_samples,
    batch_samples,
    map_samples,
    shuffle_samples,
    unbatch_samples,
)
from ..utils.sharding import iter_items
from .context import RuntimeContext
from .types import Assembler, SourceKind, SourceStore, StageSpec


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
    """

    context: RuntimeContext

    _source_kind: SourceKind
    _source: SourceStore
    _stages: tuple[StageSpec, ...]
    _resample: bool
    _iter_source_stream: Callable

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

        def _apply(data: Iterable[object]) -> Iterable[object]:
            return map_samples(data, fn)

        spec = StageSpec(
            kind="map",
            apply=_apply,
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
        )
        return self._append_stage(spec)

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
        msg = f"[UnsupportedSourceKind] source_kind={source_kind!r}"
        raise ValueError(msg)
