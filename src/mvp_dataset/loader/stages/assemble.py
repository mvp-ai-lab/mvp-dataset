"""Loader-side assemble stage."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from ...core import RuntimeContext
from ...core.resume import UnsupportedResume, callable_fingerprint, stable_fingerprint
from ...core.stages import _AssembleStageIterator
from ...core.types import Assembler, StatefulAssembler


class _LoaderAssembleStage:
    """TorchLoader stage configuration for stateful output assembly."""

    kind = "assemble"

    def __init__(
        self,
        *,
        factory: Callable[[RuntimeContext], Assembler[object, object]],
        base_context: RuntimeContext | None,
        drop_last: bool,
    ) -> None:
        """Initialize the object."""
        self.factory = factory
        self.base_context = base_context
        self.drop_last = drop_last

    def __call__(self, data: Iterable[object]) -> Iterable[object]:
        """Apply this callable object."""
        assembler = self._build_assembler()
        return _AssembleStageIterator(
            upstream=data,
            assembler=assembler,
            factory=self.factory,
            drop_last=self.drop_last,
        )

    def fingerprint(self) -> str:
        """Return a stable fingerprint for resume compatibility checks."""
        return stable_fingerprint(
            {
                "kind": self.kind,
                "drop_last": self.drop_last,
                "factory": callable_fingerprint(self.factory),
                "assembler": self._build_assembler().fingerprint(),
            }
        )

    def _build_assembler(self) -> StatefulAssembler:
        """Build a stateful assembler for the loader stage."""
        runtime_context = RuntimeContext.from_runtime(base=self.base_context)
        assembler = self.factory(runtime_context)
        if not isinstance(assembler, StatefulAssembler):
            msg = "[UnsupportedResume] loader stage kind='assemble' requires a stateful assembler"
            raise UnsupportedResume(msg)
        return assembler
