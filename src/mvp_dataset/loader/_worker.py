"""Worker-side resume tracking for TorchLoader."""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

from ..core.resume import UnsupportedResume
from ..core.torch_compat import (
    TorchIterableDataset,
    default_collate,
    get_worker_info,
    pin_memory_item,
)


@dataclass(slots=True)
class _WorkerItem:
    """Worker message that carries a user-visible output."""

    worker_id: int
    item: object

    def pin_memory(self) -> _WorkerItem:
        """Return a copy whose user payload is pinned in memory."""
        return _WorkerItem(worker_id=self.worker_id, item=pin_memory_item(self.item))


@dataclass(slots=True)
class _WorkerState:
    """Worker message that carries a requested resume snapshot."""

    worker_id: int
    state: dict[str, object]

    def pin_memory(self) -> _WorkerState:
        """Return a copy whose user payload is pinned in memory."""
        return self


@dataclass(slots=True)
class _WorkerDone:
    """Worker message that marks a worker stream as exhausted."""

    worker_id: int
    state: dict[str, object]

    def pin_memory(self) -> _WorkerDone:
        """Return a copy whose user payload is pinned in memory."""
        return self


class _ResumeTrackingDataset(TorchIterableDataset):
    """IterableDataset wrapper that exposes worker outputs and snapshots."""

    def __init__(
        self,
        dataset: Iterable[object],
        worker_states: dict[str, dict[str, object]],
        snapshot_event: object,
        batch_size: int | None,
        drop_last: bool,
        collate_fn: Callable[[list[object]], object] | None,
    ) -> None:
        """Initialize the object."""
        self.dataset = dataset
        self.worker_states = worker_states
        self.snapshot_event = snapshot_event
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.collate_fn = collate_fn

    def __iter__(self) -> Iterator[_WorkerItem | _WorkerState | _WorkerDone]:
        """Return the iterator object."""
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else int(worker_info.id)

        dataset = self.dataset
        worker_state = self.worker_states.get(str(worker_id))
        if worker_state is not None:
            load_state_dict = getattr(dataset, "load_state_dict", None)
            if load_state_dict is None:
                msg = "[UnsupportedResume] TorchLoader dataset does not support load_state_dict"
                raise UnsupportedResume(msg)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                dataset = load_state_dict(worker_state)

        iterator = iter(dataset)
        state_dict = getattr(iterator, "state_dict", None)

        for item in self._iter_outputs(iterator):
            yield _WorkerItem(worker_id=worker_id, item=item)
            if self.snapshot_event.is_set():
                if state_dict is None:
                    msg = "[UnsupportedResume] TorchLoader dataset iterator does not support state_dict"
                    raise UnsupportedResume(msg)
                yield _WorkerState(worker_id=worker_id, state=state_dict())
                while self.snapshot_event.is_set():
                    time.sleep(0.001)
        yield _WorkerDone(worker_id=worker_id, state={} if state_dict is None else state_dict())

    def _iter_outputs(self, iterator: Iterator[object]) -> Iterator[object]:
        """Yield user-visible outputs and snapshot messages for one worker."""
        if self.batch_size is None:
            yield from iterator
            return

        batch: list[object] = []
        for item in iterator:
            batch.append(item)
            if len(batch) == self.batch_size:
                yield self._collate(batch)
                batch = []
        if batch and not self.drop_last:
            yield self._collate(batch)

    def _collate(self, batch: list[object]) -> object:
        """Collate one raw batch according to loader configuration."""
        if self.collate_fn is not None:
            return self.collate_fn(batch)
        return default_collate(batch)
