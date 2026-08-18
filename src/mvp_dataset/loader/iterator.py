"""TorchLoader iterator and resume merge logic."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import TYPE_CHECKING

from ..core.resume import ResumeStateError, Stateful, UnsupportedResume, checkpoint
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
                raise ResumeStateError(
                    f"[InvalidTorchLoaderResumeItem] expected resume envelope, got={type(item).__name__}"
                )

    def state_dict(self) -> dict[str, object]:
        """Return the resumable loader merge state."""
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
                    raise ResumeStateError(
                        f"[InvalidTorchLoaderResumeItem] expected resume control item, got={type(item).__name__}"
                    )
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
        """Advance to the next worker that may still produce output."""
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
        self._exhausted = False
        pending = loader._peek_pending_state()
        worker_states, next_worker, pending_outputs, stage_states = self._parse_live_state(pending)
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
            self.stages.append(stream if isinstance(stream, Stateful) else stage)
        if stage_states is not None:
            self._load_stage_state(stage_states)
        self.stream = iter(stream)
        if pending is not None:
            self.loader._take_pending_state()

    def __iter__(self) -> _TorchLoaderIterator:
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
        return checkpoint(self.loader.identity(), None if self._exhausted else self.live_state())

    def live_state(self) -> dict[str, object]:
        return {
            "num_yielded": self.num_yielded,
            "stages": self._stage_states(),
            "loader": self.merge_stream.state_dict(),
        }

    def load_state_dict(self, state: object) -> None:
        """Restore loader-side stage state on this live iterator.

        Worker streams are established at construction; restore them via
        ``TorchLoader.load_state_dict`` + ``iter(loader)``.
        """
        if isinstance(state, dict) and "identity" in state and "state" in state and "version" in state:
            from ..core.resume import check_identity, parse_checkpoint

            expected, inner = parse_checkpoint(state)
            check_identity(expected, self.loader.identity())
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
        stages = state.get("stages")
        if not isinstance(stages, list) or len(stages) != len(self.loader._stages):
            raise ResumeStateError("[InvalidResumeState] loader stage count does not match")
        self.num_yielded = num_yielded
        self._exhausted = False
        self._load_stage_state(stages)

    def _parse_live_state(
        self,
        state: object | None,
    ) -> tuple[dict[str, dict[str, object]], int, dict[str, list[object]], list[object] | None]:
        if state is None:
            return {}, 0, {}, None
        if not isinstance(state, dict):
            raise ResumeStateError("[InvalidResumeState] live state must be a dict")
        if "identity" in state:
            raise ResumeStateError("[InvalidResumeState] live state must not contain identity")
        num_yielded = state.get("num_yielded")
        if not isinstance(num_yielded, int) or num_yielded < 0:
            raise ResumeStateError("[InvalidResumeState] num_yielded must be a non-negative integer")
        loader_state = state.get("loader")
        if not isinstance(loader_state, dict):
            raise ResumeStateError("[InvalidResumeState] loader state must be a dict")
        next_worker = loader_state.get("next_worker")
        if not isinstance(next_worker, int) or next_worker < 0 or next_worker >= self.loader._state_worker_count():
            raise ResumeStateError("[InvalidResumeState] next_worker is out of range")
        workers = loader_state.get("workers")
        if not isinstance(workers, dict):
            raise ResumeStateError("[InvalidResumeState] workers must be a dict")
        pending_outputs = loader_state.get("pending_outputs", {})
        if not isinstance(pending_outputs, dict):
            raise ResumeStateError("[InvalidResumeState] pending_outputs must be a dict")
        stages = state.get("stages")
        if not isinstance(stages, list):
            raise ResumeStateError("[InvalidResumeState] stages must be a list")
        if len(stages) != len(self.loader._stages):
            raise ResumeStateError("[InvalidResumeState] loader stage count does not match")

        worker_states: dict[str, dict[str, object]] = {}
        for worker_id, worker_state in workers.items():
            if not isinstance(worker_id, str) or not worker_id.isdigit():
                raise ResumeStateError("[InvalidResumeState] worker id must be a numeric string")
            numeric_worker_id = int(worker_id)
            if numeric_worker_id < 0 or numeric_worker_id >= self.loader._state_worker_count():
                raise ResumeStateError("[InvalidResumeState] worker id is out of range")
            if not isinstance(worker_state, dict):
                raise ResumeStateError("[InvalidResumeState] worker state must be a dict")
            if "identity" in worker_state:
                raise ResumeStateError("[InvalidResumeState] live state must not contain identity")
            worker_states[worker_id] = worker_state
        parsed_pending: dict[str, list[object]] = {}
        for worker_id, items in pending_outputs.items():
            if not isinstance(worker_id, str) or not worker_id.isdigit():
                raise ResumeStateError("[InvalidResumeState] pending worker id must be a numeric string")
            numeric_worker_id = int(worker_id)
            if numeric_worker_id < 0 or numeric_worker_id >= self.loader._state_worker_count():
                raise ResumeStateError("[InvalidResumeState] pending worker id is out of range")
            if not isinstance(items, list):
                raise ResumeStateError("[InvalidResumeState] pending_outputs entry must be a list")
            parsed_pending[worker_id] = items
        self.num_yielded = num_yielded
        return worker_states, next_worker, parsed_pending, stages

    def _stage_states(self) -> list[object]:
        stage_states: list[object] = []
        for index, (stage_factory, stage) in enumerate(zip(self.loader._stages, self.stages, strict=True)):
            if not isinstance(stage, Stateful):
                raise UnsupportedResume(
                    f"[UnsupportedResume] loader stage kind={getattr(stage_factory, 'kind', None)!r} index={index}"
                )
            stage_states.append(stage.state_dict())
        return stage_states

    def _load_stage_state(self, stages: list[object]) -> None:
        for stage_factory, stage, stage_state in zip(self.loader._stages, self.stages, stages, strict=True):
            if not isinstance(stage, Stateful):
                raise UnsupportedResume(f"[UnsupportedResume] loader stage kind={stage_factory.kind!r}")
            if not isinstance(stage_state, dict):
                raise ResumeStateError("[InvalidResumeState] loader stage state must be a dict")
            stage.load_state_dict(stage_state)
