"""Worker-side resume tracking for TorchLoader."""

from __future__ import annotations

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
    snapshot_id: int

    def pin_memory(self) -> _WorkerState:
        """Return a copy whose user payload is pinned in memory."""
        return self


@dataclass(slots=True)
class _WorkerPaused:
    """Complete a prefetch task without advancing a frozen worker's data cursor."""

    worker_id: int

    def pin_memory(self) -> _WorkerPaused:
        """Return the control message without allocating pinned payloads."""
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
        snapshot_request: object,
        batch_size: int | None,
        drop_last: bool,
        collate_fn: Callable[[list[object]], object] | None,
    ) -> None:
        """Initialize the object."""
        self.dataset = dataset
        # Own the mapping so releasing a bootstrap never mutates the caller's
        # checkpoint (including one still being serialized asynchronously).
        self.worker_states = dict(worker_states)
        self.snapshot_request = snapshot_request
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.collate_fn = collate_fn

    def __iter__(self) -> Iterator[_WorkerItem | _WorkerState | _WorkerPaused | _WorkerDone]:
        """Return the iterator object."""
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else int(worker_info.id)

        dataset = self.dataset
        worker_state = self.worker_states.get(str(worker_id))
        if worker_state is not None:
            load_live_state = getattr(dataset, "load_live_state", None)
            if load_live_state is None:
                raise UnsupportedResume("[UnsupportedResume] TorchLoader dataset does not support load_live_state")
            load_live_state(worker_state)

        iterator = iter(dataset)
        # load_live_state may defer consumption until iter(dataset). Do not keep
        # the consumed input graph in this suspended generator or retain states
        # for unrelated workers in each worker process.
        self.release_worker_states()
        del worker_state
        live_state = getattr(iterator, "live_state", None)

        for item in self._iter_outputs(iterator):
            yield _WorkerItem(worker_id=worker_id, item=item)
            captured = 0
            while snapshot_id := self.snapshot_request.value:
                if snapshot_id != captured:
                    if live_state is None:
                        raise UnsupportedResume(
                            "[UnsupportedResume] TorchLoader dataset iterator does not support live_state"
                        )
                    captured = snapshot_id
                    yield _WorkerState(worker_id=worker_id, state=live_state(), snapshot_id=snapshot_id)
                else:
                    # Blocking here deadlocks an ordered DataLoader when this
                    # worker's next task precedes another worker's state message.
                    yield _WorkerPaused(worker_id=worker_id)
        yield _WorkerDone(worker_id=worker_id, state={} if live_state is None else live_state())

    def release_worker_states(self) -> None:
        """Release this process's bootstrap references after a safe handoff."""
        self.worker_states.clear()

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
