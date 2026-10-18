"""Snapshot ownership, failure recovery and exact multi-worker continuation."""

from __future__ import annotations

import gc
import pickle
import weakref
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest
from torch.utils.data import get_worker_info

from mvp_dataset import TorchLoader
from mvp_dataset.loader._worker import _ResumeTrackingDataset, _WorkerDone, _WorkerItem, _WorkerState
from mvp_dataset.loader.iterator import _ResumeMergeIterator

from .test_loader_checkpoint import _shutdown


class _Graph:
    pass


class _Messages:
    def __init__(self, items):
        self.items = deque(items)

    def __iter__(self):
        return self

    def __next__(self):
        item = self.items.popleft()
        if isinstance(item, BaseException):
            raise item
        return item


def _merge(items, workers=2):
    return _ResumeMergeIterator(
        _Messages(items),
        num_workers=workers,
        next_worker=0,
        pending_outputs={},
        snapshot_request=SimpleNamespace(value=0),
    )


def test_active_snapshot_owned_by_caller_only_and_terminal_cursor_retained():
    active, terminal = _Graph(), _Graph()
    active_ref, terminal_ref = weakref.ref(active), weakref.ref(terminal)
    merge = _merge(
        [
            _WorkerDone(0, {"cursor": 10, "graph": terminal}),
            _WorkerState(1, {"cursor": 4, "graph": active}, 1),
        ]
    )
    del active, terminal
    first = merge.state_dict()
    assert first["workers"]["0"]["cursor"] == 10
    assert merge.snapshot_request.value == 0
    assert active_ref() is not None  # Async serializer may still own this payload.
    del first
    gc.collect()
    assert active_ref() is None
    assert terminal_ref() is not None  # Exhausted worker cannot answer another request.
    merge.stream.items.append(_WorkerState(1, {"cursor": 5}, 2))
    second = merge.state_dict()
    assert second["workers"]["0"]["cursor"] == 10
    assert second["workers"]["1"]["cursor"] == 5


def test_stale_prefetched_ack_is_not_retained():
    graph = _Graph()
    reference = weakref.ref(graph)
    merge = _merge([_WorkerState(0, {"graph": graph}, 1), _WorkerItem(0, "next")], workers=1)
    del graph
    assert next(merge) == "next"
    gc.collect()
    assert reference() is None


def test_partial_snapshot_failure_releases_request_and_retries_all_live_workers():
    merge = _merge(
        [
            _WorkerDone(0, {"cursor": 7}),
            _WorkerState(1, {"cursor": 3}, 1),
            RuntimeError("capture failed"),
        ],
        workers=3,
    )
    with pytest.raises(RuntimeError, match="capture failed"):
        merge.state_dict()
    assert merge.snapshot_request.value == 0
    merge.stream.items.extend(
        [
            _WorkerState(1, {"cursor": 1}, 1),  # Old generation must not satisfy retry.
            _WorkerItem(1, "pending"),
            _WorkerState(1, {"cursor": 9}, 2),
            _WorkerState(2, {"cursor": 11}, 2),
        ]
    )
    state = merge.state_dict()
    assert state == {
        "next_worker": 0,
        "workers": {"0": {"cursor": 7}, "1": {"cursor": 9}, "2": {"cursor": 11}},
        "pending_outputs": {"1": ["pending"]},
    }
    assert next(merge) == "pending"


class _SequenceDataset:
    """Defer restore consumption until iter(), as a real Dataset does."""

    def __init__(self, count=17):
        self.count = count
        self.pending = None
        self.fail_restore = False
        self.fail_iter = False

    def identity(self):
        return {"sequence": 1, "count": self.count}

    def load_live_state(self, state):
        if self.fail_restore:
            raise RuntimeError("restore failed")
        self.pending = state

    def __iter__(self):
        if self.fail_iter:
            raise RuntimeError("iterator failed")
        iterator = _SequenceIterator(self.count, self.pending)
        self.pending = None
        return iterator


class _SequenceIterator:
    def __init__(self, count, state):
        self.worker = info.id if (info := get_worker_info()) else 0
        self.limit = count - self.worker
        self.position = 0 if state is None else state["position"]

    def __iter__(self):
        return self

    def __next__(self):
        if self.position == self.limit:
            raise StopIteration
        value = [self.worker, self.position]
        self.position += 1
        return value

    def live_state(self):
        return {"position": self.position}


def test_zero_worker_bootstrap_released_after_lazy_restore_without_mutating_checkpoint():
    graph = _Graph()
    reference = weakref.ref(graph)
    checkpoint = {"0": {"position": 4, "graph": graph}, "1": {"position": 8}}
    wrapper = _ResumeTrackingDataset(_SequenceDataset(), checkpoint, SimpleNamespace(value=0), None, False, None)
    iterator = iter(wrapper)
    assert wrapper.worker_states  # Generator has not consumed the restore yet.
    assert next(iterator).item == [0, 4]
    assert wrapper.worker_states == {}
    assert checkpoint["0"]["graph"] is graph
    del checkpoint, graph
    gc.collect()
    assert reference() is None  # Neither generator local nor wrapper keeps old graph.


@pytest.mark.parametrize("failure", ["fail_restore", "fail_iter"])
def test_worker_restore_failure_keeps_bootstrap_for_retry(failure):
    dataset = _SequenceDataset()
    setattr(dataset, failure, True)
    wrapper = _ResumeTrackingDataset(dataset, {"0": {"position": 4}}, SimpleNamespace(value=0), None, False, None)
    with pytest.raises(RuntimeError, match="failed"):
        next(iter(wrapper))
    assert wrapper.worker_states == {"0": {"position": 4}}
    setattr(dataset, failure, False)
    assert next(iter(wrapper)).item == [0, 4]
    assert wrapper.worker_states == {}


@pytest.mark.parametrize("workers,context", [(0, None), (2, "spawn"), (2, "fork")])
@pytest.mark.parametrize("persistent", [False, True])
def test_delayed_async_serialization_and_resume_exactly(workers, context, persistent):
    def build():
        return TorchLoader(
            _SequenceDataset(),
            num_workers=workers,
            multiprocessing_context=context,
            prefetch_factor=4,
            persistent_workers=persistent if workers else False,
            timeout=30 if workers else 0,
        )

    loader = build()
    iterator = iter(loader)
    expected = [[w, p] for p in range(17) for w in range(max(1, workers)) if p < 17 - w]
    release = Event()

    def serialize_later(state):
        assert release.wait(30)
        return pickle.dumps(state)

    try:
        assert [next(iterator) for _ in range(5)] == expected[:5]
        first = loader.state_dict()
        assert first["version"] == 3
        frozen = pickle.dumps(first)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(serialize_later, first)
            try:
                second = loader.state_dict()  # Immediate reentry, no release observed by worker required.
                assert [next(iterator) for _ in range(3)] == expected[5:8]
                third = loader.state_dict()
                assert list(iterator) == expected[8:]
            finally:
                release.set()
            assert future.result() == frozen
        assert pickle.dumps(first) == frozen
    finally:
        _shutdown(iterator)

    for state, remaining in [(first, expected[5:]), (second, expected[5:]), (third, expected[8:])]:
        restored = build()
        restored.load_state_dict(state)
        iterator = iter(restored)
        try:
            if workers:
                assert iterator.merge_stream.stream._dataset.worker_states == {}
            # Save again before the first restored next(): pending outputs and
            # worker cursor must both survive this additional handoff.
            before_next = restored.state_dict()
            assert list(iterator) == remaining
        finally:
            _shutdown(iterator)
        again = build()
        again.load_state_dict(before_next)
        iterator = iter(again)
        try:
            assert list(iterator) == remaining
            assert again.state_dict()["state"] is None
        finally:
            _shutdown(iterator)


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("shuffle_mode", ["global", "chunk"])
def test_lance_loader_full_pipeline_resume(tmp_path, workers, shuffle_mode):
    from .test_lance_full_pipeline_resume import _build_full_pipeline
    from .test_resume import _consume, _remaining

    build_dataset = _build_full_pipeline(tmp_path, seed=13, shuffle_mode=shuffle_mode)

    def build():
        return (
            TorchLoader(
                build_dataset(), num_workers=workers, multiprocessing_context="spawn", timeout=30 if workers else 0
            )
            .shuffle(buffer_size=3, initial=2)
            .batch(2)
            .unbatch()
        )

    loader = build()
    iterator = iter(loader)
    try:
        consumed = _consume(iterator, 2)
        state = loader.state_dict()
        frozen = pickle.dumps(state)
        continued = _remaining(iterator)
        assert pickle.dumps(state) == frozen
    finally:
        _shutdown(iterator)
    restored = build()
    restored.load_state_dict(pickle.loads(frozen))
    iterator = iter(restored)
    try:
        assert _remaining(iterator) == continued
    finally:
        _shutdown(iterator)
    fresh = iter(build())
    try:
        assert consumed + continued == _remaining(fresh)
    finally:
        _shutdown(fresh)
