"""Checkpoint progress and exact continuation under unequal worker prefetch."""

from __future__ import annotations

import time
from threading import Timer

import pytest
import torch.multiprocessing as mp
from torch.utils.data import get_worker_info

from mvp_dataset import TorchLoader


class _Input:
    def __init__(self, gate, progress, *, count=47, uneven=False):
        self.gate = gate
        self.progress = progress
        self.count = count
        self.uneven = uneven
        self.position = 0

    def identity(self):
        return {"test_input": 1, "count": self.count, "uneven": self.uneven}

    def load_live_state(self, state):
        self.position = state["position"]

    def __iter__(self):
        return _InputIterator(self)


class _InputIterator:
    def __init__(self, dataset):
        self.dataset = dataset
        self.position = dataset.position
        info = get_worker_info()
        self.worker = info.id if info else 0

    def __iter__(self):
        return self

    def __next__(self):
        dataset = self.dataset
        limit = dataset.count - self.worker if dataset.uneven else dataset.count
        if self.position >= limit:
            raise StopIteration
        if self.worker == 1 and self.position == 38:
            if not dataset.gate.wait(30):
                raise RuntimeError("test did not release worker 1")
        item = [self.worker, self.position]
        self.position += 1
        dataset.progress[self.worker] = self.position
        return item

    def live_state(self):
        return {"position": self.position}


def _shutdown(iterator):
    stop = getattr(iterator.merge_stream.stream, "_shutdown_workers", None)
    if stop is not None:
        stop()


@pytest.mark.parametrize("prefetch_factor", [2, 4])
@pytest.mark.parametrize("persistent_workers", [False, True])
def test_checkpoint_with_unequal_prefetch_and_immediate_reentry(prefetch_factor, persistent_workers):
    context = mp.get_context("spawn")
    gate = context.Event()
    progress = context.Array("i", [0] * 4)

    def build():
        return TorchLoader(
            _Input(gate, progress),
            num_workers=4,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            multiprocessing_context="spawn",
            timeout=15,
        )

    loader = build()
    iterator = iter(loader)
    expected = [[worker, position] for position in range(47) for worker in range(4)]
    try:
        consumed = [next(iterator) for _ in range(153)]
        assert consumed == expected[:153]
        # Worker 1 is blocked before item 39; other workers have prefetched item 40.
        deadline = time.monotonic() + 15
        while any(progress[worker] < 40 for worker in (0, 2, 3)):
            assert time.monotonic() < deadline, list(progress)
            time.sleep(0.001)
        timer = Timer(0.05, gate.set)
        timer.start()
        try:
            states = [iterator.state_dict() for _ in range(3)]
        finally:
            timer.join()
        assert list(iterator) == expected[153:]
    finally:
        gate.set()
        _shutdown(iterator)

    for state in states:
        restored = build()
        restored.load_state_dict(state)
        iterator = iter(restored)
        try:
            assert list(iterator) == expected[153:]
        finally:
            _shutdown(iterator)


@pytest.mark.parametrize("num_workers", [0, 1, 2, 4])
def test_checkpoints_before_first_item_and_during_worker_exhaustion(num_workers):
    context = mp.get_context("spawn")
    gate = context.Event()
    gate.set()
    progress = context.Array("i", [0] * max(1, num_workers))

    def build():
        return TorchLoader(
            _Input(gate, progress, count=5, uneven=True),
            num_workers=num_workers,
            multiprocessing_context="spawn",
            prefetch_factor=4,
            timeout=15 if num_workers else 0,
        )

    loader = build()
    iterator = iter(loader)
    expected = [
        [worker, position] for position in range(5) for worker in range(max(1, num_workers)) if position < 5 - worker
    ]
    try:
        initial = iterator.state_dict()
        assert [next(iterator) for _ in range(len(expected) - 1)] == expected[:-1]
        last = iterator.state_dict()
        assert list(iterator) == expected[-1:]
    finally:
        _shutdown(iterator)
    for state, remaining in [(initial, expected), (last, expected[-1:])]:
        restored = build()
        restored.load_state_dict(state)
        iterator = iter(restored)
        try:
            assert list(iterator) == remaining
        finally:
            _shutdown(iterator)
