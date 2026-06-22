"""Mixed dataset source configuration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from mvp_dataset.core.context import RuntimeContext
from mvp_dataset.core.dataset import Dataset
from mvp_dataset.core.resume import stable_fingerprint

from .iterator import _MixedSourceIterator
from .types import MixedSourceSpec, MixedStrategy

_MIXED_STRATEGIES = {"concat", "round_robin", "weighted_round_robin", "random", "weighted_random"}


@dataclass(frozen=True, slots=True)
class MixedDataset(Dataset):
    """Dataset configuration for multiple mixed dataset sources."""

    _strategy: MixedStrategy = "weighted_round_robin"

    @classmethod
    def from_source(
        cls,
        sources: Mapping[str, Dataset],
        context: RuntimeContext | None = None,
        strategy: MixedStrategy = "weighted_round_robin",
        weights: Mapping[str, int] | None = None,
    ):
        """Build a dataset that mixes multiple child datasets.

        Args:
            sources: Named child datasets to mix.
            context: Runtime context used for mixed iteration.
            strategy: Source selection strategy.
            weights: Optional per-source weights for weighted round robin.

        Returns:
            A dataset configured for mixed-source iteration."""
        if strategy not in _MIXED_STRATEGIES:
            msg = f"[UnsupportedMixedStrategy] strategy={strategy!r}"
            raise ValueError(msg)

        source_specs = _normalize_sources(sources, weights)
        runtime_context = source_specs[0].dataset.context if context is None else context
        _validate_contexts(source_specs, runtime_context)

        return cls(
            context=runtime_context,
            _source=source_specs,
            _resample=False,
            _source_kind="mixed",
            _stages=(),
            _strategy=strategy,
        )

    def _build_source_stream(self, *, context: RuntimeContext) -> Iterable[object]:
        """Build the source iterator for a runtime context."""
        return _MixedSourceIterator(
            sources=self._source,
            context=context,
            strategy=self._strategy,
            source_fingerprint=stable_fingerprint(self._source_fingerprint()),
        )

    def _source_fingerprint(self) -> dict[str, object]:
        """Return the source portion of the pipeline fingerprint."""
        return {
            "kind": "mixed",
            "strategy": self._strategy,
            "sources": [
                {
                    "name": source.name,
                    "weight": source.weight,
                    "runtime": source.dataset.context.fingerprint(),
                    "pipeline": source.dataset._pipeline_fingerprint(),
                }
                for source in self._source
            ],
        }


def _normalize_sources(
    sources: Mapping[str, Dataset],
    weights: Mapping[str, int] | None,
) -> tuple[MixedSourceSpec, ...]:
    if not sources:
        msg = "[EmptyMixedSource] at least one source is required"
        raise ValueError(msg)

    result: list[MixedSourceSpec] = []
    seen_names: set[str] = set()
    for name, dataset in sources.items():
        weight = 1
        if weights is not None and name in weights:
            weight = weights[name]
        if not isinstance(name, str) or not name:
            msg = f"[InvalidMixedSourceName] name={name!r}"
            raise ValueError(msg)
        if name in seen_names:
            msg = f"[DuplicateMixedSourceName] name={name!r}"
            raise ValueError(msg)
        if not isinstance(dataset, Dataset):
            msg = f"[InvalidMixedSourceDataset] source={name!r} must be a Dataset"
            raise TypeError(msg)
        if not isinstance(weight, int) or weight <= 0:
            msg = f"[InvalidMixedSourceWeight] source={name!r} weight must be a positive integer"
            raise ValueError(msg)

        seen_names.add(name)
        result.append(MixedSourceSpec(name=name, dataset=dataset, weight=weight))

    return tuple(result)


def _validate_contexts(sources: tuple[MixedSourceSpec, ...], context: RuntimeContext) -> None:
    context_fingerprint = context.fingerprint()
    for source in sources:
        if source.dataset.context.fingerprint() != context_fingerprint:
            msg = f"[MixedSourceContextMismatch] source={source.name!r} context does not match mixed context"
            raise ValueError(msg)
