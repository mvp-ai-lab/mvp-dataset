"""Mixed source types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mvp_dataset.core.dataset import Dataset

MixedStrategy = Literal["concat", "round_robin", "weighted_round_robin", "random", "weighted_random"]


@dataclass(frozen=True, slots=True)
class MixedSourceSpec:
    """One named dataset participating in a mixed source."""

    name: str
    dataset: Dataset
    weight: int = 1
