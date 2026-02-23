"""Tests for pipeline operations."""

from __future__ import annotations

import random

from mm_loader.pipeline import batch_samples, map_samples, shuffle_samples, unbatch_samples


def test_map_samples_is_lazy() -> None:
    events: list[str] = []

    def source() -> object:
        for index in range(3):
            events.append(f"source-{index}")
            yield index

    def transform(value: object) -> object:
        events.append(f"map-{value}")
        return int(value) + 1

    mapped = map_samples(source(), transform)
    assert events == []

    first = next(mapped)
    assert first == 1
    assert events == ["source-0", "map-0"]


def test_shuffle_samples_is_deterministic_with_seed() -> None:
    data = list(range(100))

    first = list(shuffle_samples(data, buffer_size=17, initial=8, rng=random.Random(123)))
    second = list(shuffle_samples(data, buffer_size=17, initial=8, rng=random.Random(123)))
    third = list(shuffle_samples(data, buffer_size=17, initial=8, rng=random.Random(124)))

    assert first == second
    assert first != third
    assert sorted(first) == data


def test_batch_and_unbatch_are_reversible_for_default_batches() -> None:
    samples = [{"value": index} for index in range(10)]

    batched = list(batch_samples(samples, batch_size=4, drop_last=False))
    restored = list(unbatch_samples(batched))

    assert [len(batch) for batch in batched] == [4, 4, 2]
    assert restored == samples
