"""PyTorch-backed parallel loader for mvp-dataset datasets."""

from __future__ import annotations

import multiprocessing as mp
import random
import time
import warnings
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from ..core import RuntimeContext
from ..core.resume import (
    RESUME_STATE_VERSION,
    ResumeStateError,
    StatefulStage,
    UnsupportedResume,
    stable_fingerprint,
)
from ..core.stages import (
    _AssembleStageIterator,
    _BatchStageIterator,
    _ShuffleStageIterator,
    _UnbatchStageIterator,
)
from ..core.torch_compat import (
    TORCH_AVAILABLE,
    TorchDataLoader,
    TorchIterableDataset,
    default_collate,
    get_worker_info,
    pin_memory_item,
)
from ..core.types import Assembler, StatefulAssembler

LoaderStage = Callable[[Iterable[object]], Iterable[object]]
"""One post-DataLoader transformation stage."""


@dataclass(slots=True)
class _WorkerItem:
    worker_id: int
    item: object

    def pin_memory(self) -> _WorkerItem:
        return _WorkerItem(worker_id=self.worker_id, item=pin_memory_item(self.item))


@dataclass(slots=True)
class _WorkerState:
    worker_id: int
    state: dict[str, object]

    def pin_memory(self) -> _WorkerState:
        return self


@dataclass(slots=True)
class _WorkerDone:
    worker_id: int
    state: dict[str, object]

    def pin_memory(self) -> _WorkerDone:
        return self


class _LoaderUnbatchStage:
    kind = "unbatch"

    def __call__(self, data: Iterable[object]) -> Iterable[object]:
        return _UnbatchStageIterator(upstream=data)

    def fingerprint(self) -> str:
        return stable_fingerprint({"kind": self.kind})


class _LoaderShuffleStageIterator(_ShuffleStageIterator):
    def __init__(
        self,
        *,
        upstream: Iterable[object],
        buffer_size: int,
        initial: int | None,
        seed: int,
    ) -> None:
        self.seed = seed
        super().__init__(
            upstream=upstream,
            buffer_size=buffer_size,
            initial=initial,
            rng=random.Random(seed),
        )

    def fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "kind": "shuffle",
                "buffer_size": self.buffer_size,
                "initial": self.initial_config,
                "seed": self.seed,
            }
        )


class _LoaderShuffleStage:
    kind = "shuffle"

    def __init__(self, *, buffer_size: int, initial: int | None, seed: int) -> None:
        self.buffer_size = buffer_size
        self.initial = initial
        self.seed = seed

    def __call__(self, data: Iterable[object]) -> Iterable[object]:
        return _LoaderShuffleStageIterator(
            upstream=data,
            buffer_size=self.buffer_size,
            initial=self.initial,
            seed=self.seed,
        )

    def fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "kind": self.kind,
                "buffer_size": self.buffer_size,
                "initial": self.initial,
                "seed": self.seed,
            }
        )


class _LoaderAssembleStage:
    kind = "assemble"

    def __init__(
        self,
        *,
        factory: Callable[[RuntimeContext], Assembler[object, object]],
        base_context: RuntimeContext | None,
        drop_last: bool,
    ) -> None:
        self.factory = factory
        self.base_context = base_context
        self.drop_last = drop_last

    def __call__(self, data: Iterable[object]) -> Iterable[object]:
        assembler = self._build_assembler()
        return _AssembleStageIterator(
            upstream=data,
            assembler=assembler,
            factory=self.factory,
            drop_last=self.drop_last,
        )

    def fingerprint(self) -> str:
        factory_type = type(self.factory)
        return stable_fingerprint(
            {
                "kind": self.kind,
                "drop_last": self.drop_last,
                "factory_class": f"{factory_type.__module__}.{factory_type.__qualname__}",
                "factory_config": repr(self.factory),
                "assembler": self._build_assembler().fingerprint(),
            }
        )

    def _build_assembler(self) -> StatefulAssembler:
        runtime_context = RuntimeContext.from_runtime(base=self.base_context)
        assembler = self.factory(runtime_context)
        if not isinstance(assembler, StatefulAssembler):
            msg = "[UnsupportedResume] loader stage kind='assemble' requires a stateful assembler"
            raise UnsupportedResume(msg)
        return assembler


class _LoaderBatchStage:
    kind = "batch"

    def __init__(
        self,
        *,
        batch_size: int,
        drop_last: bool,
        collate_fn: Callable[[list[object]], object] | None,
    ) -> None:
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.collate_fn = collate_fn

    def __call__(self, data: Iterable[object]) -> Iterable[object]:
        return _BatchStageIterator(
            upstream=data,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            collate_fn=self.collate_fn,
        )

    def fingerprint(self) -> str:
        collate_type = type(self.collate_fn) if self.collate_fn is not None else None
        return stable_fingerprint(
            {
                "kind": self.kind,
                "batch_size": self.batch_size,
                "drop_last": self.drop_last,
                "collate_fn_class": None
                if collate_type is None
                else f"{collate_type.__module__}.{collate_type.__qualname__}",
                "collate_fn_config": None if self.collate_fn is None else repr(self.collate_fn),
            }
        )


class _ResumeTrackingDataset(TorchIterableDataset):
    def __init__(
        self,
        dataset: Iterable[object],
        worker_states: dict[str, dict[str, object]],
        snapshot_event: object,
        batch_size: int | None,
        drop_last: bool,
        collate_fn: Callable[[list[object]], object] | None,
    ) -> None:
        self.dataset = dataset
        self.worker_states = worker_states
        self.snapshot_event = snapshot_event
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.collate_fn = collate_fn

    def __iter__(self) -> Iterator[_WorkerItem | _WorkerState | _WorkerDone]:
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
        if self.collate_fn is not None:
            return self.collate_fn(batch)
        return default_collate(batch)


class _ResumeMergeIterator:
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
        return self

    def __next__(self) -> object:
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
        for offset in range(self.num_workers):
            worker_id = (self.next_worker + offset) % self.num_workers
            if self.buffers[worker_id]:
                return worker_id
            if worker_id in self.done:
                continue
            return None
        return None

    def _advance_worker(self, worker_id: int) -> int:
        for offset in range(1, self.num_workers + 1):
            candidate = (worker_id + offset) % self.num_workers
            if self.buffers[candidate] or candidate not in self.done:
                return candidate
        return worker_id


class _TorchLoaderIterator:
    def __init__(self, loader: TorchLoader) -> None:
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
        return self

    def __next__(self) -> object:
        item = next(self.stream)
        self.num_yielded += 1
        return item

    def state_dict(self) -> dict[str, object]:
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


class TorchLoader:
    """Parallel loader that delegates worker processes to PyTorch DataLoader.

    Throughput-oriented usage for sample-level global shuffle:
    1. let workers emit micro-batches via ``batch_size`` + ``collate_fn=lambda x: x``;
    2. call ``.unbatch()`` to restore a sample stream in the main process;
    3. call ``.shuffle(...)`` for global sample-level shuffle;
    4. optionally call ``.assemble(...)`` for post-merge packing or grouping;
    5. call ``.batch(...)`` to build training batches.
    """

    def __init__(
        self,
        dataset: Iterable[object],
        *,
        num_workers: int = 0,
        batch_size: int | None = None,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int = 2,
        multiprocessing_context: str | None = None,
        collate_fn: Callable[[list[object]], object] | None = None,
        drop_last: bool = False,
        _stages: tuple[LoaderStage, ...] | None = None,
        _resume_state: dict[str, object] | None = None,
        **loader_kwargs: Any,
    ) -> None:
        """Initialize a PyTorch-backed loader wrapper.

        Args:
            dataset: Upstream iterable dataset or iterator source.
            num_workers: Number of PyTorch worker processes.
            batch_size: Optional worker-side batch size passed to DataLoader.
            pin_memory: Forwarded to DataLoader.
            persistent_workers: Forwarded to DataLoader when workers are used.
            prefetch_factor: Forwarded to DataLoader when workers are used.
            multiprocessing_context: Optional multiprocessing context name passed
                to DataLoader.
            collate_fn: Optional worker-side collate function.
            drop_last: Whether to drop the final incomplete worker batch when
                ``batch_size`` is set.
            _stages: Internal tuple of post-merge loader stages.
            **loader_kwargs: Additional keyword arguments forwarded to
                ``torch.utils.data.DataLoader``.

        Raises:
            RuntimeError: If PyTorch is unavailable.
            ValueError: If worker-related arguments are invalid.
        """

        if not TORCH_AVAILABLE:
            msg = "[TorchUnavailable] install torch to use TorchLoader"
            raise RuntimeError(msg)
        if num_workers < 0:
            msg = f"num_workers must be >= 0, got {num_workers}"
            raise ValueError(msg)
        if prefetch_factor <= 0:
            msg = f"prefetch_factor must be > 0, got {prefetch_factor}"
            raise ValueError(msg)
        if num_workers == 0 and persistent_workers:
            msg = "persistent_workers=True requires num_workers > 0"
            raise ValueError(msg)
        if batch_size is not None and batch_size <= 0:
            msg = f"batch_size must be > 0, got {batch_size}"
            raise ValueError(msg)

        self._dataset = dataset
        self._num_workers = num_workers
        self._batch_size = batch_size
        self._pin_memory = pin_memory
        self._persistent_workers = persistent_workers
        self._prefetch_factor = prefetch_factor
        self._multiprocessing_context = multiprocessing_context
        self._collate_fn = collate_fn
        self._drop_last = drop_last
        self._loader_kwargs: dict[str, object] = dict(loader_kwargs)
        self._stages = tuple() if _stages is None else _stages
        self._resume_state = _resume_state

    def _append_stage(self, stage: LoaderStage) -> TorchLoader:
        """Return a new loader with one additional post-merge stage.

        Args:
            stage: Post-merge iterator transformation to append.

        Returns:
            A new :class:`TorchLoader` sharing the same DataLoader settings and
            upstream dataset.
        """

        return TorchLoader(
            self._dataset,
            num_workers=self._num_workers,
            batch_size=self._batch_size,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
            prefetch_factor=self._prefetch_factor,
            multiprocessing_context=self._multiprocessing_context,
            collate_fn=self._collate_fn,
            drop_last=self._drop_last,
            _stages=self._stages + (stage,),
            _resume_state=None,
            **self._loader_kwargs,
        )

    def _build_torch_dataloader(
        self,
        *,
        dataset: Iterable[object] | None = None,
        resume_tracking: bool = False,
    ) -> Iterable[object]:
        """Build the underlying ``torch.utils.data.DataLoader`` instance.

        Returns:
            A configured DataLoader ready to iterate over the upstream dataset.
        """

        DataLoader = TorchDataLoader
        kwargs: dict[str, object] = {
            "dataset": self._dataset if dataset is None else dataset,
            "num_workers": self._num_workers,
            "batch_size": None if resume_tracking else self._batch_size,
            "pin_memory": self._pin_memory,
        }
        if self._collate_fn is not None and not resume_tracking:
            kwargs["collate_fn"] = self._collate_fn
        if self._batch_size is not None and not resume_tracking:
            kwargs["drop_last"] = self._drop_last
        if self._num_workers > 0:
            kwargs["persistent_workers"] = self._persistent_workers
            kwargs["prefetch_factor"] = self._prefetch_factor
            if self._multiprocessing_context is not None:
                kwargs["multiprocessing_context"] = self._multiprocessing_context
        kwargs.update(self._loader_kwargs)
        return DataLoader(**kwargs)

    def _state_worker_count(self) -> int:
        return max(1, self._num_workers)

    def _multiprocessing_event(self) -> object:
        if self._multiprocessing_context is None:
            return mp.Event()
        return mp.get_context(self._multiprocessing_context).Event()

    def _check_resume_supported(self) -> None:
        for index, stage in enumerate(self._stages):
            if not hasattr(stage, "kind") or not hasattr(stage, "fingerprint"):
                msg = f"[UnsupportedResume] loader stage index={index} is not resumable"
                raise UnsupportedResume(msg)

    def _stages_fingerprint(self) -> list[str]:
        self._check_resume_supported()
        return [stage.fingerprint() for stage in self._stages]

    def _loader_fingerprint(self) -> str:
        collate_type = type(self._collate_fn) if self._collate_fn is not None else None
        payload = {
            "num_workers": self._num_workers,
            "batch_size": self._batch_size,
            "pin_memory": self._pin_memory,
            "persistent_workers": self._persistent_workers,
            "prefetch_factor": self._prefetch_factor,
            "multiprocessing_context": self._multiprocessing_context,
            "collate_fn_class": None
            if collate_type is None
            else f"{collate_type.__module__}.{collate_type.__qualname__}",
            "collate_fn_config": repr(self._collate_fn),
            "drop_last": self._drop_last,
            "loader_kwargs": repr(sorted(self._loader_kwargs.items())),
            "stages": self._stages_fingerprint(),
        }
        return stable_fingerprint(payload)

    def state_dict(self) -> dict[str, object]:
        """Return the initial resumable state without starting workers."""
        warnings.warn(
            "TorchLoader.state_dict() creates a fresh initial loader state. "
            "Use iterator.state_dict() to checkpoint an in-progress iteration.",
            UserWarning,
            stacklevel=2,
        )
        return {
            "version": RESUME_STATE_VERSION,
            "loader_fingerprint": self._loader_fingerprint(),
            "num_yielded": 0,
            "next_worker": 0,
            "workers": {},
            "pending_outputs": {},
            "stages": self._initial_stage_states(),
        }

    def _initial_stage_states(self) -> list[dict[str, object]]:
        stream: Iterable[object] = iter(())
        stage_states: list[dict[str, object]] = []
        for index, stage_factory in enumerate(self._stages):
            stream = stage_factory(stream)
            stage = stream if isinstance(stream, StatefulStage) else stage_factory
            if not isinstance(stage, StatefulStage):
                msg = f"[UnsupportedResume] loader stage kind={stage_factory.kind!r} index={index}"
                raise UnsupportedResume(msg)
            stage_states.append(
                {
                    "kind": stage_factory.kind,
                    "fingerprint": stage.fingerprint(),
                    "state": stage.state_dict(),
                }
            )
        return stage_states

    def _validate_resume_state(self, state: dict[str, object]) -> None:
        self._check_resume_supported()
        if not isinstance(state, dict):
            msg = "[InvalidResumeState] state must be a dict"
            raise ResumeStateError(msg)
        version = state.get("version")
        if version != RESUME_STATE_VERSION:
            msg = f"[InvalidResumeStateVersion] expected={RESUME_STATE_VERSION} got={version!r}"
            raise ResumeStateError(msg)
        loader_fingerprint = state.get("loader_fingerprint")
        if not isinstance(loader_fingerprint, str):
            msg = "[InvalidResumeState] loader_fingerprint must be a string"
            raise ResumeStateError(msg)
        if loader_fingerprint != self._loader_fingerprint():
            msg = "[ResumeLoaderMismatch] loader fingerprint does not match"
            raise ResumeStateError(msg)
        stages = state.get("stages")
        if not isinstance(stages, list):
            msg = "[InvalidResumeState] stages must be a list"
            raise ResumeStateError(msg)
        if len(stages) != len(self._stages):
            msg = "[ResumeStageMismatch] loader stage count does not match"
            raise ResumeStateError(msg)

    def load_state_dict(self, state: dict[str, object]) -> TorchLoader:
        """Return a loader with validated resume state attached."""
        self._validate_resume_state(state)
        warnings.warn(
            "TorchLoader.load_state_dict() stores pending resume state. "
            "The state is applied when the returned loader is iterated.",
            UserWarning,
            stacklevel=2,
        )
        return TorchLoader(
            self._dataset,
            num_workers=self._num_workers,
            batch_size=self._batch_size,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
            prefetch_factor=self._prefetch_factor,
            multiprocessing_context=self._multiprocessing_context,
            collate_fn=self._collate_fn,
            drop_last=self._drop_last,
            _stages=self._stages,
            _resume_state=state,
            **self._loader_kwargs,
        )

    def _default_shuffle_seed(self) -> int:
        """Derive a deterministic shuffle seed from dataset runtime context.

        Returns:
            ``context.sample_shuffle_seed`` when the upstream dataset exposes a
            :class:`RuntimeContext`; otherwise ``0``.
        """

        context = getattr(self._dataset, "context", None)
        if isinstance(context, RuntimeContext):
            return context.sample_shuffle_seed
        return 0

    def unbatch(self) -> TorchLoader:
        """Append an unbatch stage at loader side (after worker merge).

        This stage is typically used with worker micro-batches to reduce IPC overhead
        while keeping downstream sample-level transforms unchanged.

        Returns:
            A new loader that expands merged batches back into samples before
            any later loader-side stages run.
        """
        return self._append_stage(_LoaderUnbatchStage())

    def shuffle(
        self,
        buffer_size: int,
        initial: int | None = None,
        seed: int | None = None,
    ) -> TorchLoader:
        """Append a global (post-merge) sample-level shuffle stage.

        Args:
            buffer_size: Maximum number of samples held in the shuffle buffer.
            initial: Minimum buffered sample count before yielding values.
            seed: Optional explicit shuffle seed. When omitted, a deterministic
                seed is derived from the upstream dataset context.

        Returns:
            A new loader that performs bounded-memory global shuffling after
            worker output has been merged.
        """
        resolved_seed = self._default_shuffle_seed() if seed is None else seed
        return self._append_stage(
            _LoaderShuffleStage(
                buffer_size=buffer_size,
                initial=initial,
                seed=resolved_seed,
            )
        )

    def assemble(
        self,
        factory: Callable[[RuntimeContext], Assembler[object, object]],
        *,
        drop_last: bool = False,
    ) -> TorchLoader:
        """Append a global (post-merge) assembly stage.

        Args:
            factory: Callable that builds one fresh assembler for each loader
                iteration using the runtime-resolved context.
            drop_last: Whether to discard unfinished tail state instead of
                delegating it to the assembler's final flush.

        Returns:
            A new loader that assembles the merged sample stream in the main
            process.
        """
        base_context = getattr(self._dataset, "context", None)
        return self._append_stage(
            _LoaderAssembleStage(
                factory=factory,
                base_context=base_context if isinstance(base_context, RuntimeContext) else None,
                drop_last=drop_last,
            )
        )

    def batch(
        self,
        batch_size: int,
        drop_last: bool = False,
        collate_fn: Callable[[list[object]], object] | None = None,
    ) -> TorchLoader:
        """Append a batch stage at loader side.

        Args:
            batch_size: Number of post-merge samples per yielded batch.
            drop_last: Whether to drop the final incomplete batch.
            collate_fn: Optional callable that converts each list of samples into
                a user-defined batch object.

        Returns:
            A new loader that batches the merged sample stream in the main
            process.
        """
        return self._append_stage(
            _LoaderBatchStage(
                batch_size=batch_size,
                drop_last=drop_last,
                collate_fn=collate_fn,
            )
        )

    def __iter__(self) -> Iterator[object]:
        """Iterate the merged stream after applying all loader-side stages.

        Returns:
            An iterator over the DataLoader output after all appended loader-side
            stages have been applied in order.
        """

        return _TorchLoaderIterator(self)
