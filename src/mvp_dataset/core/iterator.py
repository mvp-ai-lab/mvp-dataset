"""Runtime iterator for Dataset pipelines."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from .context import RuntimeContext
from .resume import (
    RESUME_STATE_VERSION,
    ResumeStateError,
    StatefulIterator,
    StatefulStage,
    UnsupportedResume,
)

if TYPE_CHECKING:
    from .dataset import Dataset


class DatasetIterator:
    """Materialized iterator for one Dataset pipeline execution."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.context = RuntimeContext.from_runtime(base=dataset.context)
        self.num_yielded = 0

        source = dataset._build_source_stream(context=self.context)
        if not isinstance(source, StatefulIterator):
            msg = f"[UnsupportedResume] source kind={dataset._source_kind!r}"
            raise UnsupportedResume(msg)
        self.source = source

        stage_resume_states = self._load_resume_state(dataset._resume_state)

        stream: Iterable[object] = self.source
        self.stages: list[object] = []
        for spec in dataset._stages:
            stream = spec.apply(stream)
            stage = stream if isinstance(stream, StatefulStage) else spec.apply
            self.stages.append(stage)
        if stage_resume_states is not None:
            self._load_stage_resume_state(stage_resume_states)
        self.stream = iter(stream)

    def __iter__(self) -> DatasetIterator:
        return self

    def __next__(self) -> object:
        item = next(self.stream)
        self.num_yielded += 1
        return item

    def state_dict(self) -> dict[str, object]:
        stage_states: list[dict[str, object]] = []
        for index, (spec, stage) in enumerate(zip(self.dataset._stages, self.stages, strict=True)):
            if not isinstance(stage, StatefulStage):
                msg = f"[UnsupportedResume] stage kind={spec.kind!r} index={index}"
                raise UnsupportedResume(msg)
            stage_states.append(
                {
                    "kind": spec.kind,
                    "fingerprint": stage.fingerprint(),
                    "state": stage.state_dict(),
                }
            )

        return {
            "version": RESUME_STATE_VERSION,
            "runtime_fingerprint": self.dataset.context.fingerprint(),
            "pipeline_fingerprint": self.dataset._pipeline_fingerprint(),
            "num_yielded": self.num_yielded,
            "source": {
                "kind": self.dataset._source_kind,
                "fingerprint": self.source.fingerprint(),
                "state": self.source.state_dict(),
            },
            "stages": stage_states,
        }

    def _load_resume_state(self, state: dict[str, object] | None) -> list[object] | None:
        if state is None:
            return None

        num_yielded = state.get("num_yielded")
        if not isinstance(num_yielded, int) or num_yielded < 0:
            msg = "[InvalidResumeState] num_yielded must be a non-negative integer"
            raise ResumeStateError(msg)
        stages = state.get("stages")
        if not isinstance(stages, list):
            msg = "[InvalidResumeState] stages must be a list"
            raise ResumeStateError(msg)
        if len(stages) != len(self.dataset._stages):
            msg = "[ResumeStageMismatch] stage count does not match"
            raise ResumeStateError(msg)

        source_state = state.get("source")
        if not isinstance(source_state, dict):
            msg = "[InvalidResumeState] source must be a dict"
            raise ResumeStateError(msg)
        if source_state.get("fingerprint") != self.source.fingerprint():
            msg = "[ResumeSourceMismatch] source fingerprint does not match"
            raise ResumeStateError(msg)
        raw_source_state = source_state.get("state")
        if not isinstance(raw_source_state, dict):
            msg = "[InvalidResumeState] source.state must be a dict"
            raise ResumeStateError(msg)
        self.source.load_state_dict(raw_source_state)
        self.num_yielded = num_yielded
        return stages

    def _load_stage_resume_state(self, stages: list[object]) -> None:
        for index, (spec, stage, stage_state) in enumerate(zip(self.dataset._stages, self.stages, stages, strict=True)):
            if not isinstance(stage_state, dict):
                msg = "[InvalidResumeState] stage must be a dict"
                raise ResumeStateError(msg)
            if stage_state.get("kind") != spec.kind:
                msg = f"[ResumeStageMismatch] stage kind does not match index={index}"
                raise ResumeStateError(msg)
            if not isinstance(stage, StatefulStage):
                msg = f"[UnsupportedResume] stage kind={spec.kind!r} index={index}"
                raise UnsupportedResume(msg)
            if stage_state.get("fingerprint") != stage.fingerprint():
                msg = f"[ResumeStageMismatch] stage fingerprint does not match index={index}"
                raise ResumeStateError(msg)
            raw_stage_state = stage_state.get("state")
            if not isinstance(raw_stage_state, dict):
                msg = "[InvalidResumeState] stage.state must be a dict"
                raise ResumeStateError(msg)
            stage.load_state_dict(raw_stage_state)
