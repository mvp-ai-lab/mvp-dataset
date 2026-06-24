"""Deterministic subset selection shared by ``split()`` and ``sample()``."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import replace as dataclass_replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dataset import Dataset


def split_offsets(total: int, fractions: Sequence[float]) -> list[int]:
    """Return integer boundaries that partition ``[0, total)``."""
    normalized = _normalize_fractions(fractions)
    cuts = [0]
    accumulated = 0.0
    for fraction in normalized[:-1]:
        accumulated += fraction
        cut = round(accumulated * total)
        cuts.append(max(cuts[-1], min(cut, total)))
    cuts.append(total)
    return cuts


def split_units(dataset: Dataset, weights: Sequence[float], fractions: Sequence[float]) -> tuple[Dataset, ...]:
    """Partition a unit-based source into disjoint datasets covering all units.

    Each split keeps a contiguous subset of ``dataset._source`` in source order.

    Args:
        dataset: Source dataset whose ``_source`` is an indexable unit sequence.
        weights: Per-unit weight used to balance the partition (e.g. row counts).
        fractions: Split fractions, normalized internally.
    Returns:
        One new dataset per fraction, in input order."""
    normalized = _normalize_fractions(fractions)
    total_weight = sum(weights)
    if total_weight <= 0:
        msg = f"[EmptySubsetSource] source_kind={dataset._source_kind!r} has no units to split"
        raise ValueError(msg)

    bounds = []
    accumulated = 0.0
    for fraction in normalized[:-1]:
        accumulated += fraction
        bounds.append(accumulated * total_weight)

    groups: list[list[int]] = [[] for _ in normalized]
    cumulative = 0.0
    group_index = 0
    for unit_index, weight in enumerate(weights):
        while group_index < len(bounds) and cumulative >= bounds[group_index]:
            group_index += 1
        groups[group_index].append(unit_index)
        cumulative += weight

    return tuple(_subset_dataset(dataset, group) for group in groups)


def sample_units(dataset: Dataset, weights: Sequence[float], fraction: float, seed: int) -> Dataset:
    """Return a dataset over a seeded random subset of a unit-based source.

    Args:
        dataset: Source dataset whose ``_source`` is an indexable unit sequence.
        weights: Per-unit weight used to size the subset (e.g. row counts).
        fraction: Fraction of total weight to keep, in ``(0, 1]``.
        seed: Seed controlling which units are kept.

    Returns:
        A new dataset reading only the sampled units."""
    fraction = float(fraction)
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        msg = f"[InvalidSampleFraction] fraction must be in (0, 1], got={fraction!r}"
        raise ValueError(msg)

    order = sorted(
        range(len(weights)),
        key=lambda index: hashlib.sha256(f"{seed}:{index}".encode()).digest(),
    )
    total_weight = sum(weights)
    if total_weight <= 0:
        msg = f"[EmptySubsetSource] source_kind={dataset._source_kind!r} has no units to sample"
        raise ValueError(msg)

    target = fraction * total_weight
    chosen: list[int] = []
    cumulative = 0.0
    for unit_index in order:
        if cumulative >= target:
            break
        chosen.append(unit_index)
        cumulative += weights[unit_index]

    return _subset_dataset(dataset, chosen)


def _normalize_fractions(fractions: Sequence[float]) -> list[float]:
    values = [float(fraction) for fraction in fractions]
    if not values:
        msg = "[InvalidSplitFractions] at least one fraction is required"
        raise ValueError(msg)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        msg = f"[InvalidSplitFractions] all fractions must be positive finite values, got={values!r}"
        raise ValueError(msg)
    total = sum(values)
    return [value / total for value in values]


def _subset_dataset(dataset: Dataset, unit_indices: Sequence[int]) -> Dataset:
    """Build a dataset over the given units, preserving original unit order."""
    min_units = dataset.context.total_slots
    if len(unit_indices) < min_units:
        msg = (
            f"[InsufficientSubsetUnits] split/sample produced {len(unit_indices)} unit(s) "
            f"for source_kind={dataset._source_kind!r} but {min_units} slot(s) are required - "
            f"use a larger fraction or split the source into more shards"
        )
        raise ValueError(msg)
    subset = tuple(dataset._source[index] for index in sorted(unit_indices))
    return dataclass_replace(dataset, _source=subset, _resume_state=None)
