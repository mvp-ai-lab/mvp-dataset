"""Ordering, speculation bounds, cursor rollback and checkpoint atomicity."""

import threading
import time
from collections import deque
from types import SimpleNamespace

import pytest

from mvp_dataset.core.stages.assemble import _AssembleStageIterator
from mvp_dataset.core.stages.prefetch import _PrefetchAssembleIterator


class Source:
    def __init__(self):
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == 4:
            raise StopIteration
        result = self.index
        self.index += 1
        return result

    def state_dict(self):
        return {"index": self.index}

    def load_state_dict(self, state):
        self.index = state["index"]


class Assembler:
    def push(self, row):
        return [row * 10 + i for i in range(5)]

    def state_dict(self):
        return {}

    def finish(self, *, drop_last=False):
        return []

    def load_state_dict(self, state):
        assert state == {}


def iterator(source, *, max_samples=1, max_bytes=100, max_input_bytes=100, assembler=None, input_bytes=None):
    base = _AssembleStageIterator(upstream=source, assembler=assembler or Assembler(), drop_last=False)
    options = SimpleNamespace(max_samples=max_samples, max_bytes=max_bytes, max_input_bytes=max_input_bytes,
                              input_bytes=input_bytes or (lambda row: 2),
                              output_bytes=lambda values: len(values))

    class BoundedReady(deque):
        def append(self, entry):
            assert len(self) < max_samples
            assert sum(size for _, size in self) + entry[1] <= max_bytes
            super().append(entry)

    stream = _PrefetchAssembleIterator(base, options)
    stream.ready = BoundedReady()
    return stream


def test_finishes_next_sample_before_current_outputs_are_consumed():
    source = Source()
    stream = iterator(source)
    try:
        assert next(stream) == 0
        deadline = time.monotonic() + 3
        while not stream.ready and time.monotonic() < deadline:
            time.sleep(.01)
        assert len(stream.ready) == 1
        assert stream.ready[0][0] == list(range(10, 15))
        assert stream.base.pending_outputs == list(range(1, 5))
        assert stream.cached_bytes == 5
        assert source.index == 2
        assert list(stream) == list(range(1, 5)) + list(range(10, 15)) + list(range(20, 25)) + list(range(30, 35))
    finally:
        stream.close()


@pytest.mark.parametrize("limits", [{"max_bytes": 1}, {"max_input_bytes": 1}])
def test_budget_fallback_does_not_drop_repeat_or_reorder_samples(limits):
    source = Source()
    stream = iterator(source, **limits)
    try:
        assert list(stream) == [row * 10 + i for row in range(4) for i in range(5)]
        assert stream.cached_bytes == 0
    finally:
        stream.close()


def test_pause_waits_for_inflight_sample_and_checkpoint_resumes_with_or_without_lookahead():
    started = threading.Event()
    release = threading.Event()

    class BlockedAssembler(Assembler):
        def push(self, row):
            if row == 1:
                started.set()
                assert release.wait(3)
            return super().push(row)

    source = Source()
    stream = iterator(source, assembler=BlockedAssembler())
    try:
        assert next(stream) == 0
        assert started.wait(3)
        frozen = threading.Event()
        thread = threading.Thread(target=lambda: (stream.pause_for_snapshot(), frozen.set()))
        thread.start()
        assert not frozen.wait(.03)
        release.set()
        assert frozen.wait(3)
        thread.join()
        upstream = source.state_dict()
        state = stream.state_dict()
        stream.resume_after_snapshot()
        expected = list(stream)
        for enabled in (True, False):
            restored_source = Source()
            restored_source.load_state_dict(upstream)
            if enabled:
                restored = iterator(restored_source)
            else:
                restored = _AssembleStageIterator(upstream=restored_source, assembler=Assembler(), drop_last=False)
            restored.load_state_dict(state)
            try:
                assert list(restored) == expected
            finally:
                if enabled:
                    restored.close()
    finally:
        stream.close()


def test_budget_rollback_restores_stateful_assembler_and_finish():
    class StatefulAssembler(Assembler):
        def __init__(self):
            self.total = 0

        def push(self, row):
            self.total += row
            return [self.total] * 5

        def finish(self, *, drop_last=False):
            return [] if drop_last else [self.total + 100]

        def state_dict(self):
            return {"total": self.total}

        def load_state_dict(self, state):
            self.total = state["total"]

    stream = iterator(Source(), max_bytes=1, assembler=StatefulAssembler())
    try:
        assert list(stream) == [0] * 5 + [1] * 5 + [3] * 5 + [6] * 5 + [106]
    finally:
        stream.close()


def test_producer_failure_is_raised_by_consumer_and_checkpoint():
    class FailedAssembler(Assembler):
        def push(self, row):
            if row == 1:
                raise ValueError("failed future sample")
            return super().push(row)

    stream = iterator(Source(), assembler=FailedAssembler())
    try:
        assert [next(stream) for _ in range(5)] == list(range(5))
        with pytest.raises(ValueError, match="failed future sample"):
            next(stream)
        stream.pause_for_snapshot()
        with pytest.raises(ValueError, match="failed future sample"):
            stream.state_dict()
    finally:
        stream.close()


def test_close_stops_producer_and_ends_iteration():
    stream = iterator(Source())
    assert next(stream) == 0
    stream.close()
    assert not stream.thread.is_alive()
    with pytest.raises(StopIteration):
        next(stream)


def test_restore_drains_checkpointed_outputs_before_admitting_another_sample():
    source = Source()
    stream = iterator(source)
    try:
        assert next(stream) == 0
        deadline = time.monotonic() + 3
        while not stream.ready and time.monotonic() < deadline:
            time.sleep(.01)
        assert len(stream.ready) == 1
        stream.pause_for_snapshot()
        cursor = source.state_dict()
        state = stream.state_dict()
        assert len(state["pending_outputs"]) == 9
    finally:
        stream.close()
    restored_source = Source()
    restored_source.load_state_dict(cursor)
    restored = iterator(restored_source)
    restored.load_state_dict(state)
    try:
        assert [next(restored) for _ in range(9)] == list(range(1, 5)) + list(range(10, 15))
        assert restored_source.index == 2
        assert restored.thread is None
        assert list(restored) == list(range(20, 25)) + list(range(30, 35))
    finally:
        restored.close()


@pytest.mark.parametrize("saved_count", [0, 1, 2])
@pytest.mark.parametrize("restored_count", [0, 1, 2])
def test_resume_preserves_all_outputs_when_prefetch_count_changes(saved_count, restored_count):
    def build(source, count):
        if count:
            return iterator(source, max_samples=count)
        return _AssembleStageIterator(upstream=source, assembler=Assembler(), drop_last=False)

    source = Source()
    stream = build(source, saved_count)
    try:
        assert next(stream) == 0
        if saved_count:
            deadline = time.monotonic() + 3
            with stream.cv:
                assert stream.cv.wait_for(lambda: len(stream.ready) == saved_count,
                                          timeout=max(0, deadline - time.monotonic()))
            stream.pause_for_snapshot()
            assert len(stream.ready) == saved_count
            assert stream.cached_bytes == saved_count * 5
        cursor = source.state_dict()
        state = stream.state_dict()
        assert len(state["pending_outputs"]) == 4 + saved_count * 5
    finally:
        if saved_count:
            stream.close()
    restored_source = Source()
    restored_source.load_state_dict(cursor)
    restored = build(restored_source, restored_count)
    restored.load_state_dict(state)
    try:
        pending = len(state["pending_outputs"])
        observed = [next(restored) for _ in range(pending)]
        # A larger saved cache must drain before the new count admits raw rows.
        assert restored_source.state_dict() == cursor
        if restored_count:
            assert restored.thread is None
        observed.extend(restored)
        assert observed == [row * 10 + i for row in range(4) for i in range(5)][1:]
    finally:
        if restored_count:
            restored.close()


def test_multiple_sample_cache_respects_combined_byte_budget():
    release = threading.Event()

    def gated_input_bytes(row):
        if row == 1:
            assert release.wait(3)
        return 2

    stream = iterator(Source(), max_samples=3, max_bytes=6, input_bytes=gated_input_bytes)
    try:
        assert next(stream) == 0
        release.set()
        with stream.cv:
            assert stream.cv.wait_for(lambda: stream.deferred, timeout=3)
        assert len(stream.ready) == 1
        assert stream.cached_bytes == 5
        assert list(stream) == [row * 10 + i for row in range(4) for i in range(5)][1:]
        assert stream.cached_bytes == 0
    finally:
        release.set()
        stream.close()
