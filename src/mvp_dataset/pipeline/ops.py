"""Iterator transformation operators used by the dataset pipeline."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Iterator

from ..core.types import Assembler


def map_samples[T, U](data: Iterable[T], fn: Callable[[T], U]) -> Iterator[U]:
    """Lazily apply ``fn`` to each sample in ``data``."""

    for sample in data:
        yield fn(sample)


def select_samples(
    data: Iterable[object],
    fields: tuple[str, ...],
) -> Iterator[dict[str, object]]:
    """Project dict samples to the requested fields plus metadata keys."""

    selected = set(fields)
    for sample in data:
        if not isinstance(sample, dict):
            msg = f"select() expects dict samples, got {type(sample)!r}"
            raise TypeError(msg)
        yield {
            key: value
            for key, value in sample.items()
            if key in selected or (key.startswith("__") and key.endswith("__"))
        }


def _pop_random[T](buffer: list[T], rng: random.Random) -> T:
    # O(1) removal: swap picked slot with the last slot, then pop tail.
    # This keeps shuffle semantics while avoiding O(n) middle-pop cost.
    index = rng.randrange(len(buffer))
    picked = buffer[index]
    buffer[index] = buffer[-1]
    buffer.pop()
    return picked


def shuffle_samples[T](
    data: Iterable[T],
    *,
    buffer_size: int,
    initial: int | None = None,
    rng: random.Random | None = None,
) -> Iterator[T]:
    """Shuffle a stream using a bounded in-memory buffer.

    This operation is deterministic when ``rng`` is initialized with a fixed seed.
    """

    if buffer_size <= 0:
        msg = f"buffer_size must be > 0, got {buffer_size}"
        raise ValueError(msg)

    if initial is None:
        initial = buffer_size
    if initial <= 0:
        msg = f"initial must be > 0, got {initial}"
        raise ValueError(msg)
    initial = min(initial, buffer_size)

    random_gen = random.Random() if rng is None else rng
    buffer: list[T] = []
    iterator = iter(data)

    # Consume the stream progressively, keeping a bounded randomization buffer.
    for sample in iterator:
        buffer.append(sample)
        if len(buffer) < buffer_size:
            try:
                buffer.append(next(iterator))
            except StopIteration:
                pass
        if len(buffer) >= initial:
            yield _pop_random(buffer, random_gen)

    while buffer:
        yield _pop_random(buffer, random_gen)


def batch_samples[T, U](
    data: Iterable[T],
    *,
    batch_size: int,
    drop_last: bool = False,
    collate_fn: Callable[[list[T]], U] | None = None,
) -> Iterator[list[T] | U]:
    """Group samples into batches."""

    if batch_size <= 0:
        msg = f"batch_size must be > 0, got {batch_size}"
        raise ValueError(msg)

    batch: list[T] = []
    for sample in data:
        batch.append(sample)
        if len(batch) == batch_size:
            yield collate_fn(batch) if collate_fn is not None else list(batch)
            batch.clear()

    if batch and not drop_last:
        yield collate_fn(batch) if collate_fn is not None else list(batch)


def assemble_samples[T, U](
    data: Iterable[T],
    *,
    factory: Callable[[], Assembler[T, U]],
    drop_last: bool = False,
) -> Iterator[U]:
    """Assemble a stream with stateful many-to-one or many-to-many logic."""

    assembler = factory()
    for sample in data:
        yield from assembler.push(sample)
    yield from assembler.finish(drop_last=drop_last)


def unbatch_samples(data: Iterable[object]) -> Iterator[object]:
    """Expand a stream of batches back into individual samples.

    Supported batch representations:
    - ``list`` or ``tuple`` of samples
    - ``dict[str, list | tuple]`` where every value has the same length
    """

    for batch in data:
        if isinstance(batch, (list, tuple)):
            yield from batch
            continue

        if isinstance(batch, dict):
            if not batch:
                continue
            if not all(isinstance(v, (list, tuple)) for v in batch.values()):
                msg = "dict batches must contain only list/tuple values"
                raise TypeError(msg)

            lengths = {len(v) for v in batch.values()}
            if len(lengths) != 1:
                msg = f"dict batch values must have equal lengths, got {sorted(lengths)}"
                raise ValueError(msg)

            batch_len = next(iter(lengths))
            for index in range(batch_len):
                yield {key: value[index] for key, value in batch.items()}
            continue

        msg = f"unsupported batch type: {type(batch)!r}"
        raise TypeError(msg)
