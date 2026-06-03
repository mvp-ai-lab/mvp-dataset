"""Pipeline stage classes."""

from __future__ import annotations

import importlib
import random
from collections.abc import Callable, Iterable
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
