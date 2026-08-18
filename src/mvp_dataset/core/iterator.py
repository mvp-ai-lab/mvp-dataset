"""Runtime iterator for Dataset pipelines."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from .context import RuntimeContext
from .resume import (
    ResumeStateError,
    Stateful,
    UnsupportedResume,
    check_identity,
    checkpoint,
    parse_checkpoint,
)

if TYPE_CHECKING:
    from .dataset import Dataset


class DatasetIterator:
    """Materialized iterator for one Dataset pipeline execution."""

    def __init__(self, dataset: Dataset, *, context: RuntimeContext | None = None):
        """Initialize the object."""
        self.dataset = dataset
        self.context = RuntimeContext.from_runtime(base=dataset.context) if context is None else context
        self.num_yielded = 0
        self._exhausted = False

        source = dataset._build_source_stream(context=self.context)
        if not isinstance(source, Stateful):
            raise UnsupportedResume(f"[UnsupportedResume] source kind={dataset._source_kind!r}")
        self.source = source

        stream: Iterable[object] = self.source
        self.stages: list[object] = []
        for spec in dataset._stages:
            stream = spec.apply(stream)
            stage = stream if isinstance(stream, Stateful) else spec.apply
            self.stages.append(stage)
        self.stream = iter(stream)

        pending = dataset._peek_pending_state()
        if pending is not None:
            self.load_state_dict(pending)
            dataset._take_pending_state()

    def __iter__(self) -> DatasetIterator:
        """Return the iterator object."""
        return self

    def __next__(self) -> object:
        """Return the next output item."""
        try:
            item = next(self.stream)
        except StopIteration:
            self._exhausted = True
            raise
        self.num_yielded += 1
        return item

    def state_dict(self) -> dict[str, object]:
        """Return a full resume envelope for this live iterator."""
        return checkpoint(self.dataset.identity(), None if self._exhausted else self.live_state())

    def live_state(self) -> dict[str, object]:
        """Return inner live state without the checkpoint envelope."""
        stage_states: list[object] = []
        for spec, stage in zip(self.dataset._stages, self.stages, strict=True):
            if not isinstance(stage, Stateful):
                raise UnsupportedResume(f"[UnsupportedResume] stage kind={spec.kind!r}")
            stage_states.append(stage.state_dict())
        return {
            "num_yielded": self.num_yielded,
            "source": self.source.state_dict(),
            "stages": stage_states,
        }

    def load_state_dict(self, state: object) -> None:
        """Restore live source and stage state.

        Accepts either the inner live state or a full checkpoint envelope.
        """
        if isinstance(state, dict) and "identity" in state and "state" in state and "version" in state:
            expected, inner = parse_checkpoint(state)
            check_identity(expected, self.dataset.identity())
            if inner is None:
                raise ResumeStateError("[InvalidResumeState] live iterator cannot load state=None")
            state = inner
        if not isinstance(state, dict):
            raise ResumeStateError("[InvalidResumeState] live state must be a dict")
        if "identity" in state:
            raise ResumeStateError("[InvalidResumeState] live state must not contain identity")
        num_yielded = state.get("num_yielded")
        if not isinstance(num_yielded, int) or num_yielded < 0:
            raise ResumeStateError("[InvalidResumeState] num_yielded must be a non-negative integer")
        source_state = state.get("source")
        if not isinstance(source_state, dict):
            raise ResumeStateError("[InvalidResumeState] source must be a dict")
        stages = state.get("stages")
        if not isinstance(stages, list):
            raise ResumeStateError("[InvalidResumeState] stages must be a list")
        if len(stages) != len(self.dataset._stages):
            raise ResumeStateError("[InvalidResumeState] stage count does not match")

        self.source.load_state_dict(source_state)
        self.num_yielded = num_yielded
        self._exhausted = False
        for spec, stage, stage_state in zip(self.dataset._stages, self.stages, stages, strict=True):
            if not isinstance(stage, Stateful):
                raise UnsupportedResume(f"[UnsupportedResume] stage kind={spec.kind!r}")
            if not isinstance(stage_state, dict):
                raise ResumeStateError("[InvalidResumeState] stage state must be a dict")
            stage.load_state_dict(stage_state)
