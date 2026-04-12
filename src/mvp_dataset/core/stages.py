"""Pipeline stage classes and group-execution utilities."""

from __future__ import annotations

import importlib
import random
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import ModuleType

from ..pipeline.ops import (
    assemble_samples,
    batch_samples,
    map_samples,
    select_samples,
    shuffle_samples,
    unbatch_samples,
)
from .context import RuntimeContext
from .types import Assembler, StageSpec


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


# ---------------------------------------------------------------------------
# Stage group execution (used by cache pipeline)
# ---------------------------------------------------------------------------

# Stage kinds that are safe to run in parallel (1-in → 1-out, no cross-sample state).
PARALLELIZABLE_STAGE_KINDS = frozenset({"map"})


def _apply_stage_group(item: object, specs: tuple[StageSpec, ...]) -> object:
    """Apply a group of 1-to-1 stages to a single item."""
    stream = iter((item,))
    for spec in specs:
        stream = spec.apply(stream)
    outputs = list(stream)
    if len(outputs) != 1:
        msg = "[CacheError] pre-cache map stages must preserve one input -> one output sample"
        raise ValueError(msg)
    return outputs[0]


def iter_stage_group(
    data: Iterable[object],
    specs: tuple[StageSpec, ...],
    num_workers: int,
) -> Iterator[object]:
    """Iterate ``data`` through ``specs``, running in a thread pool when ``num_workers > 1``."""
    if not specs:
        yield from data
        return
    if num_workers == 1:
        for item in data:
            yield _apply_stage_group(item, specs)
        return
    max_inflight = num_workers * 2
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        pending: deque = deque()
        for item in data:
            pending.append(executor.submit(_apply_stage_group, item, specs))
            if len(pending) >= max_inflight:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()
