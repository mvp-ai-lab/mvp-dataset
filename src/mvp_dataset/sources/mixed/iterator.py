"""Mixed source iterator."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from mvp_dataset.core.context import RuntimeContext
from mvp_dataset.core.iterator import DatasetIterator
from mvp_dataset.core.resume import ResumeStateError

from .types import MixedSourceSpec, MixedStrategy

_MIXED_STRATEGIES = {"concat", "round_robin", "weighted_round_robin", "random", "weighted_random"}


@dataclass(slots=True)
class _WeightedSourceState:
    """Runtime state for one weighted child source."""

    spec: MixedSourceSpec
    iterator: DatasetIterator
    current: int = 0
    exhausted: bool = False


@dataclass(slots=True)
class _MixedSourceIterator:
    """Stateful iterator that mixes multiple dataset streams."""

    sources: tuple[MixedSourceSpec, ...]
    context: RuntimeContext
    strategy: MixedStrategy
    _source_states: list[_WeightedSourceState] = field(init=False)
    _cursor: int = 0
    _rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        """Initialize child dataset iterators."""
        if self.strategy not in _MIXED_STRATEGIES:
            msg = f"[UnsupportedMixedStrategy] strategy={self.strategy!r}"
            raise ValueError(msg)
        self._rng = random.Random(self.context.sample_shuffle_seed)
        self._source_states = [
            _WeightedSourceState(spec=source, iterator=DatasetIterator(source.dataset, context=self.context))
            for source in self.sources
        ]

    def __iter__(self):
        """Return the iterator object."""
        return self

    def __next__(self) -> object:
        """Return the next mixed output item."""
        while True:
            state = self._pick_source()
            if state is None:
                raise StopIteration

            try:
                sample = next(state.iterator)
            except StopIteration:
                state.exhausted = True
                continue

            return self._annotate_sample(sample, state.spec.name)

    def state_dict(self) -> dict[str, object]:
        """Return the resumable state for this object."""
        return {
            "kind": "mixed",
            "cursor": self._cursor,
            "rng_state": self._rng.getstate(),
            "sources": [
                {
                    "current": state.current,
                    "exhausted": state.exhausted,
                    "state": state.iterator.live_state(),
                }
                for state in self._source_states
            ],
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore this object from a resumable state dictionary."""
        if state.get("kind") != "mixed":
            raise ResumeStateError(f"[InvalidResumeState] expected source kind='mixed', got={state.get('kind')!r}")

        cursor = state.get("cursor")
        if not isinstance(cursor, int) or cursor < 0:
            raise ResumeStateError("[InvalidResumeState] mixed cursor must be a non-negative integer")
        try:
            self._rng.setstate(state.get("rng_state"))
        except (TypeError, ValueError) as error:
            raise ResumeStateError("[InvalidResumeState] mixed rng_state is invalid") from error

        raw_sources = state.get("sources")
        if not isinstance(raw_sources, list) or len(raw_sources) != len(self._source_states):
            raise ResumeStateError("[InvalidResumeState] mixed sources must match configured sources")

        for existing, raw_source in zip(self._source_states, raw_sources, strict=True):
            if not isinstance(raw_source, dict):
                raise ResumeStateError("[InvalidResumeState] mixed source state must be a dict")
            current = raw_source.get("current")
            if not isinstance(current, int):
                raise ResumeStateError("[InvalidResumeState] mixed source current must be an integer")
            exhausted = raw_source.get("exhausted")
            if not isinstance(exhausted, bool):
                raise ResumeStateError("[InvalidResumeState] mixed source exhausted must be a boolean")
            child_state = raw_source.get("state")
            if not isinstance(child_state, dict):
                raise ResumeStateError("[InvalidResumeState] mixed source child state must be a dict")
            if "identity" in child_state:
                raise ResumeStateError("[InvalidResumeState] live state must not contain identity")
            existing.iterator.load_state_dict(child_state)
            existing.current = current
            existing.exhausted = exhausted

        self._cursor = cursor

    def _pick_source(self) -> _WeightedSourceState | None:
        """Pick the next source for the configured strategy."""
        if self.strategy == "concat":
            return self._pick_concat_source()
        if self.strategy == "round_robin":
            return self._pick_round_robin_source()
        if self.strategy == "weighted_round_robin":
            return self._pick_weighted_source()
        if self.strategy == "random":
            return self._pick_random_source(weighted=False)
        if self.strategy == "weighted_random":
            return self._pick_random_source(weighted=True)
        msg = f"[UnsupportedMixedStrategy] strategy={self.strategy!r}"
        raise ValueError(msg)

    def _pick_concat_source(self) -> _WeightedSourceState | None:
        """Pick the current source until it is exhausted, then advance."""
        while self._cursor < len(self._source_states):
            state = self._source_states[self._cursor]
            if not state.exhausted:
                return state
            self._cursor += 1
        return None

    def _pick_round_robin_source(self) -> _WeightedSourceState | None:
        """Pick the next non-exhausted source in cyclic order."""
        if not self._source_states:
            return None
        for _ in self._source_states:
            index = self._cursor % len(self._source_states)
            self._cursor = index + 1
            state = self._source_states[index]
            if not state.exhausted:
                return state
        return None

    def _pick_weighted_source(self) -> _WeightedSourceState | None:
        """Pick the next source using smooth weighted round robin."""
        active = [state for state in self._source_states if not state.exhausted]
        if not active:
            return None

        total_weight = sum(state.spec.weight for state in active)
        for state in active:
            state.current += state.spec.weight

        selected = max(active, key=lambda state: state.current)
        selected.current -= total_weight
        return selected

    def _pick_random_source(self, *, weighted: bool) -> _WeightedSourceState | None:
        """Pick a random non-exhausted source."""
        active = [state for state in self._source_states if not state.exhausted]
        if not active:
            return None
        if not weighted:
            return self._rng.choice(active)

        pick = self._rng.randrange(sum(state.spec.weight for state in active))
        offset = 0
        for state in active:
            offset += state.spec.weight
            if pick < offset:
                return state
        return active[-1]

    def _annotate_sample(self, sample: object, source_name: str) -> object:
        """Add mixed-source metadata to dictionary samples."""
        if not isinstance(sample, dict):
            return sample
        return {**sample, "__source__": source_name}
