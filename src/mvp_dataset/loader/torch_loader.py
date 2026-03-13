"""PyTorch-backed parallel loader for mvp-dataset datasets."""

from __future__ import annotations

import importlib
import random
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from ..core import RuntimeContext
from ..pipeline.ops import batch_samples, shuffle_samples, unbatch_samples

LoaderStage = Callable[[Iterable[object]], Iterable[object]]
"""One post-DataLoader transformation stage."""


def _torch_dataloader_class() -> type:
    """Return ``torch.utils.data.DataLoader`` or raise a clear runtime error.

    Returns:
        The imported ``torch.utils.data.DataLoader`` class.

    Raises:
        RuntimeError: If PyTorch is not installed in the current environment.
    """

    try:
        torch_utils_data = importlib.import_module("torch.utils.data")
    except (ModuleNotFoundError, ImportError) as exc:
        msg = (
            "[TorchUnavailable] failed to import 'torch.utils.data'; "
            "ensure PyTorch is installed and importable to use TorchLoader"
        )
        raise RuntimeError(msg) from exc
    return torch_utils_data.DataLoader


class TorchLoader:
    """Parallel loader that delegates worker processes to PyTorch DataLoader.

    Throughput-oriented usage for sample-level global shuffle:
    1. let workers emit micro-batches via ``batch_size`` + ``collate_fn=lambda x: x``;
    2. call ``.unbatch()`` to restore a sample stream in the main process;
    3. call ``.shuffle(...)`` for global sample-level shuffle;
    4. call ``.batch(...)`` to build training batches.
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

        _torch_dataloader_class()
        if num_workers < 0:
            msg = f"num_workers must be >= 0, got {num_workers}"
            raise ValueError(msg)
        if prefetch_factor <= 0:
            msg = f"prefetch_factor must be > 0, got {prefetch_factor}"
            raise ValueError(msg)
        if num_workers == 0 and persistent_workers:
            msg = "persistent_workers=True requires num_workers > 0"
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
            **self._loader_kwargs,
        )

    def _build_torch_dataloader(self) -> Iterable[object]:
        """Build the underlying ``torch.utils.data.DataLoader`` instance.

        Returns:
            A configured DataLoader ready to iterate over the upstream dataset.
        """

        DataLoader = _torch_dataloader_class()
        kwargs: dict[str, object] = {
            "dataset": self._dataset,
            "num_workers": self._num_workers,
            "batch_size": self._batch_size,
            "pin_memory": self._pin_memory,
        }
        if self._collate_fn is not None:
            kwargs["collate_fn"] = self._collate_fn
        if self._batch_size is not None:
            kwargs["drop_last"] = self._drop_last
        if self._num_workers > 0:
            kwargs["persistent_workers"] = self._persistent_workers
            kwargs["prefetch_factor"] = self._prefetch_factor
            if self._multiprocessing_context is not None:
                kwargs["multiprocessing_context"] = self._multiprocessing_context
        kwargs.update(self._loader_kwargs)
        return DataLoader(**kwargs)

    def _default_shuffle_seed(self) -> int:
        """Derive a deterministic shuffle seed from dataset runtime context.

        Returns:
            ``context.seed + context.rank`` when the upstream dataset exposes a
            :class:`RuntimeContext`; otherwise ``0``.
        """

        context = getattr(self._dataset, "context", None)
        if isinstance(context, RuntimeContext):
            return context.seed + context.rank
        return 0

    def unbatch(self) -> TorchLoader:
        """Append an unbatch stage at loader side (after worker merge).

        This stage is typically used with worker micro-batches to reduce IPC overhead
        while keeping downstream sample-level transforms unchanged.

        Returns:
            A new loader that expands merged batches back into samples before
            any later loader-side stages run.
        """

        def stage(data: Iterable[object]) -> Iterable[object]:
            return unbatch_samples(data)

        return self._append_stage(stage)

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

        def stage(data: Iterable[object]) -> Iterable[object]:
            rng_seed = self._default_shuffle_seed() if seed is None else seed
            rng = random.Random(rng_seed)
            return shuffle_samples(data, buffer_size=buffer_size, initial=initial, rng=rng)

        return self._append_stage(stage)

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

        def stage(data: Iterable[object]) -> Iterable[object]:
            return batch_samples(
                data,
                batch_size=batch_size,
                drop_last=drop_last,
                collate_fn=collate_fn,
            )

        return self._append_stage(stage)

    def __iter__(self) -> Iterator[object]:
        """Iterate the merged stream after applying all loader-side stages.

        Returns:
            An iterator over the DataLoader output after all appended loader-side
            stages have been applied in order.
        """

        stream: Iterable[object] = self._build_torch_dataloader()
        for stage in self._stages:
            stream = stage(stream)
        yield from stream
