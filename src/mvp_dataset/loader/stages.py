"""Loader-side transform stages."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable

from ..core import RuntimeContext
from ..core.resume import UnsupportedResume, callable_fingerprint, stable_fingerprint
from ..core.stages import (
    _AssembleStageIterator,
    _BatchStageIterator,
    _ShuffleStageIterator,
    _UnbatchStageIterator,
)
from ..core.types import Assembler, StatefulAssembler


class _LoaderUnbatchStage:
    kind = "unbatch"

    def __call__(self, data: Iterable[object]) -> Iterable[object]:
        return _UnbatchStageIterator(upstream=data)

    def fingerprint(self) -> str:
        return stable_fingerprint({"kind": self.kind})


class _LoaderShuffleStageIterator(_ShuffleStageIterator):
    def __init__(
        self,
        *,
        upstream: Iterable[object],
        buffer_size: int,
        initial: int | None,
        seed: int,
    ) -> None:
        self.seed = seed
        super().__init__(
            upstream=upstream,
            buffer_size=buffer_size,
            initial=initial,
            rng=random.Random(seed),
        )

    def fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "kind": "shuffle",
                "buffer_size": self.buffer_size,
                "initial": self.initial_config,
                "seed": self.seed,
            }
        )


class _LoaderShuffleStage:
    kind = "shuffle"

    def __init__(self, *, buffer_size: int, initial: int | None, seed: int) -> None:
        self.buffer_size = buffer_size
        self.initial = initial
        self.seed = seed

    def __call__(self, data: Iterable[object]) -> Iterable[object]:
        return _LoaderShuffleStageIterator(
            upstream=data,
            buffer_size=self.buffer_size,
            initial=self.initial,
            seed=self.seed,
        )

    def fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "kind": self.kind,
                "buffer_size": self.buffer_size,
                "initial": self.initial,
                "seed": self.seed,
            }
        )


class _LoaderAssembleStage:
    kind = "assemble"

    def __init__(
        self,
        *,
        factory: Callable[[RuntimeContext], Assembler[object, object]],
        base_context: RuntimeContext | None,
        drop_last: bool,
    ) -> None:
        self.factory = factory
        self.base_context = base_context
        self.drop_last = drop_last

    def __call__(self, data: Iterable[object]) -> Iterable[object]:
        assembler = self._build_assembler()
        return _AssembleStageIterator(
            upstream=data,
            assembler=assembler,
            factory=self.factory,
            drop_last=self.drop_last,
        )

    def fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "kind": self.kind,
                "drop_last": self.drop_last,
                "factory": callable_fingerprint(self.factory),
                "assembler": self._build_assembler().fingerprint(),
            }
        )

    def _build_assembler(self) -> StatefulAssembler:
        runtime_context = RuntimeContext.from_runtime(base=self.base_context)
        assembler = self.factory(runtime_context)
        if not isinstance(assembler, StatefulAssembler):
            msg = "[UnsupportedResume] loader stage kind='assemble' requires a stateful assembler"
            raise UnsupportedResume(msg)
        return assembler


class _LoaderBatchStage:
    kind = "batch"

    def __init__(
        self,
        *,
        batch_size: int,
        drop_last: bool,
        collate_fn: Callable[[list[object]], object] | None,
    ) -> None:
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.collate_fn = collate_fn

    def __call__(self, data: Iterable[object]) -> Iterable[object]:
        return _BatchStageIterator(
            upstream=data,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            collate_fn=self.collate_fn,
        )

    def fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "kind": self.kind,
                "batch_size": self.batch_size,
                "drop_last": self.drop_last,
                "collate_fn": callable_fingerprint(self.collate_fn),
            }
        )
