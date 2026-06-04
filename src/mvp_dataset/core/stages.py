"""Pipeline stage classes."""

from __future__ import annotations

import importlib
import random
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from types import ModuleType

from ..pipeline.ops import map_samples, select_samples, unbatch_samples
from .context import RuntimeContext
from .resume import ResumeStateError, UnsupportedResume, stable_fingerprint
from .types import Assembler, StatefulAssembler


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
        return _ShuffleStageIterator(
            upstream=data,
            buffer_size=self.buffer_size,
            initial=self.initial,
            rng=random.Random(runtime_context.sample_shuffle_seed),
        )


class _ShuffleStageIterator:
    def __init__(
        self,
        *,
        upstream: Iterable[object],
        buffer_size: int,
        initial: int | None,
        rng: random.Random,
    ) -> None:
        if buffer_size <= 0:
            msg = f"buffer_size must be > 0, got {buffer_size}"
            raise ValueError(msg)
        if initial is not None and initial <= 0:
            msg = f"initial must be > 0, got {initial}"
            raise ValueError(msg)

        self.upstream = iter(upstream)
        self.buffer_size = buffer_size
        self.initial_config = initial
        self.initial = min(buffer_size, buffer_size if initial is None else initial)
        self.rng = rng
        self.buffer: list[object] = []
        self.upstream_exhausted = False

    def __iter__(self) -> Iterator[object]:
        return self

    def __next__(self) -> object:
        while not self.upstream_exhausted:
            self._read_one()
            if len(self.buffer) < self.buffer_size:
                self._read_one()
            if len(self.buffer) >= self.initial:
                return self._pop_random()

        if self.buffer:
            return self._pop_random()

        raise StopIteration

    def state_dict(self) -> dict[str, object]:
        return {
            "buffer": list(self.buffer),
            "rng_state": self.rng.getstate(),
            "upstream_exhausted": self.upstream_exhausted,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        buffer = state.get("buffer")
        if not isinstance(buffer, list):
            msg = "[InvalidResumeState] shuffle stage buffer must be a list"
            raise ResumeStateError(msg)
        if len(buffer) > self.buffer_size:
            msg = "[InvalidResumeState] shuffle stage buffer cannot exceed buffer_size"
            raise ResumeStateError(msg)
        upstream_exhausted = state.get("upstream_exhausted")
        if not isinstance(upstream_exhausted, bool):
            msg = "[InvalidResumeState] shuffle stage upstream_exhausted must be a bool"
            raise ResumeStateError(msg)

        try:
            self.rng.setstate(state.get("rng_state"))
        except (TypeError, ValueError) as error:
            msg = "[InvalidResumeState] shuffle stage rng_state is invalid"
            raise ResumeStateError(msg) from error
        self.buffer = list(buffer)
        self.upstream_exhausted = upstream_exhausted

    def fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "kind": "shuffle",
                "buffer_size": self.buffer_size,
                "initial": self.initial_config,
            }
        )

    def _read_one(self) -> None:
        try:
            self.buffer.append(next(self.upstream))
        except StopIteration:
            self.upstream_exhausted = True

    def _pop_random(self) -> object:
        index = self.rng.randrange(len(self.buffer))
        picked = self.buffer[index]
        self.buffer[index] = self.buffer[-1]
        self.buffer.pop()
        return picked


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
class _AssembleStage:
    factory: Callable[[RuntimeContext], Assembler[object, object]]
    context: RuntimeContext
    drop_last: bool = False

    def __call__(self, data: Iterable[object]) -> Iterable[object]:
        runtime_context = RuntimeContext.from_runtime(base=self.context)
        assembler = self.factory(runtime_context)
        if not isinstance(assembler, StatefulAssembler):
            msg = "[UnsupportedResume] stage kind='assemble' requires a stateful assembler"
            raise UnsupportedResume(msg)
        return _AssembleStageIterator(
            upstream=data,
            assembler=assembler,
            factory=self.factory,
            drop_last=self.drop_last,
        )


class _AssembleStageIterator:
    def __init__(
        self,
        *,
        upstream: Iterable[object],
        assembler: StatefulAssembler,
        factory: Callable[[RuntimeContext], Assembler[object, object]],
        drop_last: bool,
    ) -> None:
        self.upstream = iter(upstream)
        self.assembler = assembler
        self.factory = factory
        self.drop_last = drop_last
        self.pending_outputs: list[object] = []
        self.finished = False

    def __iter__(self) -> Iterator[object]:
        return self

    def __next__(self) -> object:
        while not self.pending_outputs:
            if self.finished:
                raise StopIteration
            try:
                self.pending_outputs.extend(self.assembler.push(next(self.upstream)))
            except StopIteration:
                self.finished = True
                self.pending_outputs.extend(self.assembler.finish(drop_last=self.drop_last))

        return self.pending_outputs.pop(0)

    def state_dict(self) -> dict[str, object]:
        return {
            "assembler_state": self.assembler.state_dict(),
            "pending_outputs": list(self.pending_outputs),
            "finished": self.finished,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        assembler_state = state.get("assembler_state")
        if not isinstance(assembler_state, dict):
            msg = "[InvalidResumeState] assemble stage assembler_state must be a dict"
            raise ResumeStateError(msg)
        pending_outputs = state.get("pending_outputs")
        if not isinstance(pending_outputs, list):
            msg = "[InvalidResumeState] assemble stage pending_outputs must be a list"
            raise ResumeStateError(msg)
        finished = state.get("finished", False)
        if not isinstance(finished, bool):
            msg = "[InvalidResumeState] assemble stage finished must be a bool"
            raise ResumeStateError(msg)

        self.assembler.load_state_dict(assembler_state)
        self.pending_outputs = list(pending_outputs)
        self.finished = finished

    def fingerprint(self) -> str:
        factory_type = type(self.factory)
        return stable_fingerprint(
            {
                "kind": "assemble",
                "drop_last": self.drop_last,
                "factory_class": f"{factory_type.__module__}.{factory_type.__qualname__}",
                "factory_config": repr(self.factory),
                "assembler": self.assembler.fingerprint(),
            }
        )


@dataclass(frozen=True, slots=True)
class _UnbatchStage:
    def __call__(self, data: Iterable[object]) -> Iterable[object]:
        return unbatch_samples(data)
