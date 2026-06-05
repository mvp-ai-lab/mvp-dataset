"""Assemble stage."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

from ..context import RuntimeContext
from ..resume import (
    ResumeStateError,
    UnsupportedResume,
    callable_fingerprint,
    stable_fingerprint,
)
from ..types import Assembler, StatefulAssembler


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

    def fingerprint(self) -> str:
        runtime_context = RuntimeContext.from_runtime(base=self.context)
        assembler = self.factory(runtime_context)
        if not isinstance(assembler, StatefulAssembler):
            msg = "[UnsupportedResume] stage kind='assemble' requires a stateful assembler"
            raise UnsupportedResume(msg)
        return stable_fingerprint(
            {
                "kind": "assemble",
                "drop_last": self.drop_last,
                "factory": callable_fingerprint(self.factory),
                "assembler": assembler.fingerprint(),
            }
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
        return stable_fingerprint(
            {
                "kind": "assemble",
                "drop_last": self.drop_last,
                "factory": callable_fingerprint(self.factory),
                "assembler": self.assembler.fingerprint(),
            }
        )
