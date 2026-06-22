"""Mixed source iterator."""

from __future__ import annotations

import random
import warnings
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
    source_fingerprint: str
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
            "strategy": self.strategy,
            "cursor": self._cursor,
            "rng_state": self._rng.getstate(),
            "sources": [
                {
                    "name": state.spec.name,
                    "weight": state.spec.weight,
                    "current": state.current,
                    "exhausted": state.exhausted,
                    "state": state.iterator.state_dict(),
                }
                for state in self._source_states
            ],
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore this object from a resumable state dictionary."""
        if state.get("kind") != "mixed":
            msg = f"[InvalidResumeState] expected source kind='mixed', got={state.get('kind')!r}"
            raise ResumeStateError(msg)
        if state.get("strategy") != self.strategy:
            msg = "[InvalidResumeState] mixed strategy does not match"
            raise ResumeStateError(msg)

        cursor = state.get("cursor")
        if not isinstance(cursor, int) or cursor < 0:
            msg = "[InvalidResumeState] mixed cursor must be a non-negative integer"
            raise ResumeStateError(msg)
        try:
            self._rng.setstate(state.get("rng_state"))
        except (TypeError, ValueError) as error:
            msg = "[InvalidResumeState] mixed rng_state is invalid"
            raise ResumeStateError(msg) from error

        raw_sources = state.get("sources")
        if not isinstance(raw_sources, list) or len(raw_sources) != len(self._source_states):
            msg = "[InvalidResumeState] mixed sources must match configured sources"
            raise ResumeStateError(msg)

        restored: list[_WeightedSourceState] = []
        for configured, raw_source in zip(self.sources, raw_sources, strict=True):
            if not isinstance(raw_source, dict):
                msg = "[InvalidResumeState] mixed source state must be a dict"
                raise ResumeStateError(msg)
            if raw_source.get("name") != configured.name or raw_source.get("weight") != configured.weight:
                msg = "[ResumeSourceMismatch] mixed source name or weight does not match"
                raise ResumeStateError(msg)

            current = raw_source.get("current")
            if not isinstance(current, int):
                msg = "[InvalidResumeState] mixed source current must be an integer"
                raise ResumeStateError(msg)
            exhausted = raw_source.get("exhausted")
            if not isinstance(exhausted, bool):
                msg = "[InvalidResumeState] mixed source exhausted must be a boolean"
                raise ResumeStateError(msg)
            child_state = raw_source.get("state")
            if not isinstance(child_state, dict):
                msg = "[InvalidResumeState] mixed source child state must be a dict"
                raise ResumeStateError(msg)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                dataset = configured.dataset.load_state_dict(child_state)
            restored.append(
                _WeightedSourceState(
                    spec=configured,
                    iterator=DatasetIterator(dataset, context=self.context),
                    current=current,
                    exhausted=exhausted,
                )
            )

        self._source_states = restored
        self._cursor = cursor

    def fingerprint(self) -> str:
        """Return a stable fingerprint for resume compatibility checks."""
        return self.source_fingerprint

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
