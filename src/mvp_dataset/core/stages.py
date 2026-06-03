"""Pipeline stage classes."""

from __future__ import annotations

import importlib
import random
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from types import ModuleType

from ..pipeline.ops import (
    assemble_samples,
    map_samples,
    select_samples,
    shuffle_samples,
    unbatch_samples,
)
from .context import RuntimeContext
from .resume import ResumeStateError, stable_fingerprint
from .types import Assembler


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

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state != {}:
            msg = "[InvalidResumeState] map stage state must be empty"
            raise ResumeStateError(msg)

    def fingerprint(self) -> str:
        fn_type = type(self.fn)
        return stable_fingerprint(
            {
                "kind": "map",
                "fn_class": f"{fn_type.__module__}.{fn_type.__qualname__}",
                "fn_config": repr(self.fn),
            }
        )


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

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state != {}:
            msg = "[InvalidResumeState] select stage state must be empty"
            raise ResumeStateError(msg)

    def fingerprint(self) -> str:
        return stable_fingerprint({"kind": "select", "fields": list(self.fields)})


@dataclass(frozen=True, slots=True)
class _BatchStage:
    batch_size: int
    drop_last: bool = False
    collate_fn: Callable[[list[object]], object] | None = None

    def __call__(self, data: Iterable[object]) -> Iterable[object]:
        return _BatchStageIterator(
            upstream=data,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            collate_fn=self.collate_fn,
        )


class _BatchStageIterator:
    def __init__(
        self,
        *,
        upstream: Iterable[object],
        batch_size: int,
        drop_last: bool,
        collate_fn: Callable[[list[object]], object] | None,
    ) -> None:
        if batch_size <= 0:
            msg = f"batch_size must be > 0, got {batch_size}"
            raise ValueError(msg)
        self.upstream = iter(upstream)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.pending: list[object] = []
        self.emitted = 0

    def __iter__(self) -> Iterator[object]:
        return self

    def __next__(self) -> object:
        while len(self.pending) < self.batch_size:
            try:
                self.pending.append(next(self.upstream))
            except StopIteration:
                if not self.pending or self.drop_last:
                    self.pending.clear()
                    raise
                break

        batch = list(self.pending)
        output = self.collate_fn(batch) if self.collate_fn is not None else batch
        self.pending.clear()
        self.emitted += 1
        return output

    def state_dict(self) -> dict[str, object]:
        return {
            "pending": list(self.pending),
            "emitted": self.emitted,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        pending = state.get("pending")
        if not isinstance(pending, list):
            msg = "[InvalidResumeState] batch stage pending must be a list"
            raise ResumeStateError(msg)
        if len(pending) >= self.batch_size:
            msg = "[InvalidResumeState] batch stage pending must be smaller than batch_size"
            raise ResumeStateError(msg)
        emitted = state.get("emitted")
        if not isinstance(emitted, int) or emitted < 0:
            msg = "[InvalidResumeState] batch stage emitted must be a non-negative integer"
            raise ResumeStateError(msg)
        self.pending = list(pending)
        self.emitted = emitted

    def fingerprint(self) -> str:
        collate_type = type(self.collate_fn) if self.collate_fn is not None else None
        return stable_fingerprint(
            {
                "kind": "batch",
                "batch_size": self.batch_size,
                "drop_last": self.drop_last,
                "collate_fn_class": None
                if collate_type is None
                else f"{collate_type.__module__}.{collate_type.__qualname__}",
                "collate_fn_config": None if self.collate_fn is None else repr(self.collate_fn),
            }
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
        runtime_context = RuntimeContext.from_runtime(base=self.context)
        return assemble_samples(
            data,
            factory=_AssemblerFactoryAdapter(self.factory, runtime_context),
            drop_last=self.drop_last,
        )


@dataclass(frozen=True, slots=True)
class _UnbatchStage:
    def __call__(self, data: Iterable[object]) -> Iterable[object]:
        return unbatch_samples(data)
