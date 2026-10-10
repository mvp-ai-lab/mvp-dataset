"""Bounded background assembly using the ordinary assemble resume format."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass


def _prefix_state(upstream):
    states = []
    node = upstream
    while node is not None:
        states.append((node, deepcopy(node.state_dict())))
        node = getattr(node, "upstream", None)
    return states


def _restore_prefix(states):
    for node, state in reversed(states):
        node.load_state_dict(state)


@dataclass(frozen=True, slots=True)
class AssemblyPrefetch:
    """Limit speculative inputs and retained assembled outputs per iterator.

    Size callbacks must return bytes. These limits cover retained future outputs
    and admitted raw inputs; native assembly may allocate transient memory. The
    assembler and upstream must support resume and exclusive thread ownership.
    Performance options are excluded from stage identity and checkpoint format.
    """

    max_samples: int
    max_bytes: int
    input_bytes: Callable[[object], int]
    output_bytes: Callable[[list[object]], int]
    max_input_bytes: int

    def __post_init__(self) -> None:
        """Reject unbounded or unusable prefetch configurations."""
        if self.max_samples < 1 or self.max_bytes < 1 or self.max_input_bytes < 1:
            raise ValueError("Assembly prefetch limits must be positive")
        if not callable(self.input_bytes) or not callable(self.output_bytes):
            raise TypeError("Assembly prefetch requires input/output byte counters")


class _PrefetchAssembleIterator:
    """Assemble complete future inputs while downstream consumes current outputs."""

    def __init__(self, base, config):
        if config.max_samples < 1 or config.max_bytes < 1 or config.max_input_bytes < 1:
            raise ValueError("Lookahead cache limits must be positive")
        self.base = base
        self.upstream = base.upstream
        self.config = config
        self.ready = deque()
        self.cached_bytes = 0
        self.cv = threading.Condition()
        self.thread = None
        self.busy = False
        self.paused = False
        self.closed = False
        self.deferred = False
        self.finished = False
        self.finish_done = False
        self.error = None
        self.restored_pending = False
        node = self.upstream
        while node is not None:
            if hasattr(node, "pause_for_snapshot"):
                raise ValueError("Only one background assembly stage is supported per source pipeline")
            node = getattr(node, "upstream", None)

    def __iter__(self):
        return self

    def start(self):
        if self.thread is None and not self.finished:
            self.thread = threading.Thread(target=self.produce, name="sample-lookahead", daemon=True)
            self.thread.start()

    def produce(self):
        while True:
            with self.cv:
                while not self.closed and (self.paused or self.deferred or len(self.ready) >= self.config.max_samples):
                    self.cv.wait()
                if self.closed:
                    return
                self.busy = True
            saved = None
            try:
                saved = _prefix_state(self.upstream)
                assembly_state = deepcopy(self.base.assembler.state_dict())
                row = next(self.upstream)
                raw_bytes = self.config.input_bytes(row)
                if not isinstance(raw_bytes, int) or raw_bytes < 0:
                    raise ValueError("Assembly input byte counter must return a non-negative integer")
                with self.cv:
                    remaining_bytes = self.config.max_bytes - self.cached_bytes
                if raw_bytes > self.config.max_input_bytes or remaining_bytes <= 0:
                    outputs = None
                    size = 0
                else:
                    outputs = list(self.base.assembler.push(row))
                    size = self.config.output_bytes(outputs)
                    if not isinstance(size, int) or size < 0:
                        raise ValueError("Assembly output byte counter must return a non-negative integer")
                if outputs is None or size > remaining_bytes:
                    # An oversized document belongs to the normal synchronous
                    # path. Restore its raw cursor and retain no speculative row
                    # or token arrays, so disabling this stage can still resume.
                    _restore_prefix(saved)
                    self.base.assembler.load_state_dict(assembly_state)
                    del row, outputs, saved
                    with self.cv:
                        self.deferred = True
                        self.busy = False
                        self.cv.notify_all()
                    continue
                del row, saved
                with self.cv:
                    self.ready.append((outputs, size))
                    self.cached_bytes += size
                    self.busy = False
                    self.cv.notify_all()
            except StopIteration:
                with self.cv:
                    self.finished = True
                    self.busy = False
                    self.cv.notify_all()
                return
            except BaseException as exc:
                with self.cv:
                    self.error = exc
                    self.busy = False
                    self.cv.notify_all()
                return

    def __next__(self):
        if self.closed:
            raise StopIteration
        if self.restored_pending:
            # The legacy checkpoint flattens current and speculative chunks.
            # Drain them before admitting another input, avoiding an extra
            # unaccounted future document after restoring that opaque buffer.
            if self.base.pending_outputs:
                return self.base.pending_outputs.pop(0)
            self.restored_pending = False
        self.start()
        while not self.base.pending_outputs:
            fallback = False
            with self.cv:
                while (not self.ready and not self.deferred and not self.finished
                       and self.error is None and not self.closed):
                    self.cv.wait()
                if self.closed:
                    raise StopIteration
                if self.ready:
                    chunks, size = self.ready.popleft()
                    self.cached_bytes -= size
                    self.base.pending_outputs = chunks
                    self.cv.notify_all()
                elif self.error is not None:
                    raise self.error
                elif self.deferred:
                    fallback = True
                else:
                    if self.finish_done:
                        raise StopIteration
                    self.base.pending_outputs = list(self.base.assembler.finish(drop_last=self.base.drop_last))
                    self.finish_done = True
            if fallback:
                row = next(self.upstream)
                self.base.pending_outputs = list(self.base.assembler.push(row))
                with self.cv:
                    self.deferred = False
                    self.cv.notify_all()
        return self.base.pending_outputs.pop(0)

    def pause_for_snapshot(self):
        with self.cv:
            self.paused = True
            while self.busy:
                self.cv.wait()

    def resume_after_snapshot(self):
        with self.cv:
            self.paused = False
            self.cv.notify_all()

    def state_dict(self):
        with self.cv:
            if self.error is not None:
                raise self.error
            if self.busy:
                raise RuntimeError("Pause sample lookahead before taking a pipeline checkpoint")
            pending = list(self.base.pending_outputs)
            for chunks, _ in self.ready:
                pending.extend(chunks)
            return {
                "assembler_state": self.base.assembler.state_dict(),
                "pending_outputs": pending,
                "finished": self.finished and self.finish_done,
            }

    def load_state_dict(self, state):
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError("Stop background assembly before restoring its upstream cursor")
        self.base.load_state_dict(state)
        self.restored_pending = bool(self.base.pending_outputs)
        self.ready.clear()
        self.cached_bytes = 0
        self.thread = None
        self.busy = False
        self.paused = False
        self.closed = False
        self.deferred = False
        self.error = None
        self.finished = self.base.finished
        self.finish_done = self.base.finished

    def stop_for_restore(self):
        self.close()

    def close(self):
        with self.cv:
            self.closed = True
            self.cv.notify_all()
        if self.thread is not None:
            self.thread.join()
