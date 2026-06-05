"""PyTorch-backed parallel loader for mvp-dataset datasets."""

from __future__ import annotations

import multiprocessing as mp
import warnings
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from ..core import RuntimeContext
from ..core.resume import (
    RESUME_STATE_VERSION,
    ResumeStateError,
    StatefulStage,
    UnsupportedResume,
    callable_fingerprint,
    stable_fingerprint,
)
from ..core.torch_compat import TORCH_AVAILABLE, TorchDataLoader
from ..core.types import Assembler
from .iterator import _TorchLoaderIterator
from .stages import (
    _LoaderAssembleStage,
    _LoaderBatchStage,
    _LoaderShuffleStage,
    _LoaderUnbatchStage,
)

LoaderStage = Callable[[Iterable[object]], Iterable[object]]
"""One post-DataLoader transformation stage."""


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
        payload = {
            "num_workers": self._num_workers,
            "batch_size": self._batch_size,
            "pin_memory": self._pin_memory,
            "persistent_workers": self._persistent_workers,
            "prefetch_factor": self._prefetch_factor,
            "multiprocessing_context": self._multiprocessing_context,
            "collate_fn": callable_fingerprint(self._collate_fn),
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
