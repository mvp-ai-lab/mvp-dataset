"""TorchLoader iterator and resume merge logic."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import TYPE_CHECKING

from ..core.resume import (
    RESUME_STATE_VERSION,
    ResumeStateError,
    StatefulStage,
    UnsupportedResume,
)
from ._worker import _ResumeTrackingDataset, _WorkerDone, _WorkerItem, _WorkerState

if TYPE_CHECKING:
    from .torch_loader import TorchLoader


class _ResumeMergeIterator:
    """Iterator that merges resumed worker streams in deterministic order."""

    def __init__(
        self,
        stream: Iterable[object],
        *,
        num_workers: int,
        worker_states: dict[str, dict[str, object]],
        next_worker: int,
        pending_outputs: dict[str, list[object]],
        snapshot_event: object,
    ) -> None:
        """Initialize the object."""
        self.stream = iter(stream)
        self.num_workers = num_workers
        self.worker_states = dict(worker_states)
        self.next_worker = next_worker
        self.snapshot_event = snapshot_event
        self.buffers: dict[int, deque[object]] = {
            worker_id: deque(pending_outputs.get(str(worker_id), [])) for worker_id in range(num_workers)
        }
        self.done: set[int] = set()

    def __iter__(self) -> _ResumeMergeIterator:
        """Return the iterator object."""
        return self

    def __next__(self) -> object:
        """Return the next output item."""
        while True:
            worker_id = self._next_worker_with_buffer()
            if worker_id is not None:
                item = self.buffers[worker_id].popleft()
                self.next_worker = self._advance_worker(worker_id)
                return item

            if len(self.done) == self.num_workers and all(not buffer for buffer in self.buffers.values()):
                raise StopIteration

            item = next(self.stream)
            if isinstance(item, _WorkerItem):
                self.buffers[item.worker_id].append(item.item)
            elif isinstance(item, _WorkerState):
                self.worker_states[str(item.worker_id)] = item.state
            elif isinstance(item, _WorkerDone):
                self.worker_states[str(item.worker_id)] = item.state
                self.done.add(item.worker_id)
                if item.worker_id == self.next_worker:
                    self.next_worker = self._advance_worker(item.worker_id)
            else:
                msg = f"[InvalidTorchLoaderResumeItem] expected resume envelope, got={type(item).__name__}"
                raise ResumeStateError(msg)

    def state_dict(self) -> dict[str, object]:
        """Return the resumable state for this object."""
        self.snapshot_event.set()
        try:
            waiting_for = {worker_id for worker_id in range(self.num_workers) if worker_id not in self.done}
            while waiting_for:
                item = next(self.stream)
                if isinstance(item, _WorkerItem):
                    self.buffers[item.worker_id].append(item.item)
                elif isinstance(item, _WorkerState):
                    self.worker_states[str(item.worker_id)] = item.state
                    waiting_for.discard(item.worker_id)
                elif isinstance(item, _WorkerDone):
                    self.worker_states[str(item.worker_id)] = item.state
                    self.done.add(item.worker_id)
                    waiting_for.discard(item.worker_id)
                else:
                    msg = f"[InvalidTorchLoaderResumeItem] expected resume control item, got={type(item).__name__}"
                    raise ResumeStateError(msg)
        finally:
            self.snapshot_event.clear()
        return {
            "next_worker": self.next_worker,
            "workers": dict(self.worker_states),
            "pending_outputs": {str(worker_id): list(buffer) for worker_id, buffer in self.buffers.items() if buffer},
        }

    def _next_worker_with_buffer(self) -> int | None:
        """Return the next worker id that has buffered outputs."""
        for offset in range(self.num_workers):
            worker_id = (self.next_worker + offset) % self.num_workers
            if self.buffers[worker_id]:
                return worker_id
            if worker_id in self.done:
                continue
            return None
        return None

    def _advance_worker(self, worker_id: int) -> int:
        """Read one control item from a worker stream."""
        for offset in range(1, self.num_workers + 1):
            candidate = (worker_id + offset) % self.num_workers
            if self.buffers[candidate] or candidate not in self.done:
                return candidate
        return worker_id


class _TorchLoaderIterator:
    """Materialized iterator for one TorchLoader execution."""

    def __init__(self, loader: TorchLoader) -> None:
        """Initialize the object."""
        self.loader = loader
        self.num_yielded = 0
        worker_states, next_worker, pending_outputs, stage_states = self._load_resume_state(loader._resume_state)
        self.snapshot_event = loader._multiprocessing_event()
        self.merge_stream = _ResumeMergeIterator(
            loader._build_torch_dataloader(
                dataset=_ResumeTrackingDataset(
                    loader._dataset,
                    worker_states,
                    self.snapshot_event,
                    loader._batch_size,
                    loader._drop_last,
                    loader._collate_fn,
                ),
                resume_tracking=True,
            ),
            num_workers=loader._state_worker_count(),
            worker_states=worker_states,
            next_worker=next_worker,
            pending_outputs=pending_outputs,
            snapshot_event=self.snapshot_event,
        )

        stream: Iterable[object] = self.merge_stream
        self.stages: list[object] = []
        for stage in loader._stages:
            stream = stage(stream)
            self.stages.append(stream if isinstance(stream, StatefulStage) else stage)
        if stage_states is not None:
            self._load_stage_resume_state(stage_states)
        self.stream = iter(stream)

    def __iter__(self) -> _TorchLoaderIterator:
        """Return the iterator object."""
        return self

    def __next__(self) -> object:
        """Return the next output item."""
        item = next(self.stream)
        self.num_yielded += 1
        return item

    def state_dict(self) -> dict[str, object]:
        """Return the resumable state for this object."""
        merge_state = self.merge_stream.state_dict()
        return {
            "version": RESUME_STATE_VERSION,
            "loader_fingerprint": self.loader._loader_fingerprint(),
            "num_yielded": self.num_yielded,
            "stages": self._stage_state_dicts(),
            **merge_state,
        }

    def _load_resume_state(
        self,
        state: dict[str, object] | None,
    ) -> tuple[dict[str, dict[str, object]], int, dict[str, list[object]], list[object] | None]:
        """Validate and load pending resume state."""
        if state is None:
            return {}, 0, {}, None
        self.loader._validate_resume_state(state)

        num_yielded = state.get("num_yielded")
        if not isinstance(num_yielded, int) or num_yielded < 0:
            msg = "[InvalidResumeState] num_yielded must be a non-negative integer"
            raise ResumeStateError(msg)
        next_worker = state.get("next_worker")
        if not isinstance(next_worker, int) or next_worker < 0 or next_worker >= self.loader._state_worker_count():
            msg = "[InvalidResumeState] next_worker is out of range"
            raise ResumeStateError(msg)
        workers = state.get("workers")
        if not isinstance(workers, dict):
            msg = "[InvalidResumeState] workers must be a dict"
            raise ResumeStateError(msg)
        pending_outputs = state.get("pending_outputs", {})
        if not isinstance(pending_outputs, dict):
            msg = "[InvalidResumeState] pending_outputs must be a dict"
            raise ResumeStateError(msg)
        stages = state.get("stages")
        if not isinstance(stages, list):
            msg = "[InvalidResumeState] stages must be a list"
            raise ResumeStateError(msg)
        if len(stages) != len(self.loader._stages):
            msg = "[ResumeStageMismatch] loader stage count does not match"
            raise ResumeStateError(msg)

        worker_states: dict[str, dict[str, object]] = {}
        for worker_id, worker_state in workers.items():
            if not isinstance(worker_id, str) or not worker_id.isdigit():
                msg = "[InvalidResumeState] worker id must be a numeric string"
                raise ResumeStateError(msg)
            numeric_worker_id = int(worker_id)
            if numeric_worker_id < 0 or numeric_worker_id >= self.loader._state_worker_count():
                msg = "[InvalidResumeState] worker id is out of range"
                raise ResumeStateError(msg)
            if not isinstance(worker_state, dict):
                msg = "[InvalidResumeState] worker state must be a dict"
                raise ResumeStateError(msg)
            worker_states[worker_id] = worker_state
        parsed_pending: dict[str, list[object]] = {}
        for worker_id, items in pending_outputs.items():
            if not isinstance(worker_id, str) or not worker_id.isdigit():
                msg = "[InvalidResumeState] pending worker id must be a numeric string"
                raise ResumeStateError(msg)
            numeric_worker_id = int(worker_id)
            if numeric_worker_id < 0 or numeric_worker_id >= self.loader._state_worker_count():
                msg = "[InvalidResumeState] pending worker id is out of range"
                raise ResumeStateError(msg)
            if not isinstance(items, list):
                msg = "[InvalidResumeState] pending_outputs entry must be a list"
                raise ResumeStateError(msg)
            parsed_pending[worker_id] = items
        self.num_yielded = num_yielded
        return worker_states, next_worker, parsed_pending, stages

    def _stage_state_dicts(self) -> list[dict[str, object]]:
        """Collect resumable state from loader-side stages."""
        stage_states: list[dict[str, object]] = []
        for index, (stage_factory, stage) in enumerate(zip(self.loader._stages, self.stages, strict=True)):
            if not isinstance(stage, StatefulStage):
                msg = f"[UnsupportedResume] loader stage kind={getattr(stage_factory, 'kind', None)!r} index={index}"
                raise UnsupportedResume(msg)
            stage_states.append(
                {
                    "kind": stage_factory.kind,
                    "fingerprint": stage.fingerprint(),
                    "state": stage.state_dict(),
                }
            )
        return stage_states

    def _load_stage_resume_state(self, stages: list[object]) -> None:
        """Load resumable state into materialized stages."""
        for index, (stage_factory, stage, stage_state) in enumerate(
            zip(self.loader._stages, self.stages, stages, strict=True)
        ):
            if not isinstance(stage_state, dict):
                msg = "[InvalidResumeState] loader stage must be a dict"
                raise ResumeStateError(msg)
            if stage_state.get("kind") != stage_factory.kind:
                msg = f"[ResumeStageMismatch] loader stage kind does not match index={index}"
                raise ResumeStateError(msg)
            if not isinstance(stage, StatefulStage):
                msg = f"[UnsupportedResume] loader stage kind={stage_factory.kind!r} index={index}"
                raise UnsupportedResume(msg)
            if stage_state.get("fingerprint") != stage.fingerprint():
                msg = f"[ResumeStageMismatch] loader stage fingerprint does not match index={index}"
                raise ResumeStateError(msg)
            raw_stage_state = stage_state.get("state")
            if not isinstance(raw_stage_state, dict):
                msg = "[InvalidResumeState] loader stage.state must be a dict"
                raise ResumeStateError(msg)
            stage.load_state_dict(raw_stage_state)
