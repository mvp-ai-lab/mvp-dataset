"""Snapshot-backed dataset source."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace

from mvp_dataset.cache import (
    CacheConfig,
    SourceFingerprint,
    fingerprint_source_manifest,
)
from mvp_dataset.core.context import RuntimeContext
from mvp_dataset.core.dataset import Dataset
from mvp_dataset.core.resume import stable_fingerprint
from mvp_dataset.core.subset import split_offsets
from mvp_dataset.core.types import FingerprintProvider
from mvp_dataset.sources.lance.iterator import _LanceSourceIterator
from mvp_dataset.sources.lance.source import list_lance_sources
from mvp_dataset.sources.lance.types import LanceSelection

from .iterator import _SnapshotSourceIterator
from .materialize import (
    SNAPSHOT_CACHE_PARAMETERS,
    SNAPSHOT_FORMAT_VERSION,
    ensure_snapshot,
    snapshot_lance_paths,
    validate_finite_pipeline,
)

_SNAPSHOT_SOURCE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class _SnapshotSplitSelection:
    fractions: tuple[float, ...]
    index: int


@dataclass(frozen=True, slots=True)
class _SnapshotSampleSelection:
    fraction: float
    seed: int


@dataclass(frozen=True, slots=True)
class SnapshotDataset(Dataset):
    """Dataset source backed by a lazily materialized Lance snapshot."""

    _upstream: Dataset | None = None
    _snapshot_fingerprint: SourceFingerprint | None = None
    _cache_config: CacheConfig | None = None
    _snapshot_selection: _SnapshotSplitSelection | _SnapshotSampleSelection | None = None

    @classmethod
    def from_upstream(
        cls,
        upstream: Dataset,
        *,
        fingerprint_provider: FingerprintProvider | None = None,
    ) -> SnapshotDataset:
        """Create a lazy snapshot source from an upstream pipeline."""
        validate_finite_pipeline(upstream)
        identity_kind = "pipeline"
        if fingerprint_provider is None:
            identity = upstream._pipeline_fingerprint()
        else:
            if not callable(fingerprint_provider):
                msg = "[InvalidFingerprintProvider] fingerprint_provider must be callable"
                raise TypeError(msg)
            identity = fingerprint_provider()
            identity_kind = "provider"
            if not isinstance(identity, str) or not identity:
                msg = "[InvalidSnapshotFingerprint] fingerprint provider must return a non-empty string"
                raise ValueError(msg)

        source_fingerprint = fingerprint_source_manifest(
            {
                "schema_version": _SNAPSHOT_SOURCE_SCHEMA_VERSION,
                "kind": "dataset-snapshot",
                "identity_kind": identity_kind,
                "identity": identity,
                "seed": upstream.context.seed,
                "epoch": upstream.context.epoch,
            }
        )
        return cls(
            context=upstream.context,
            _source=(source_fingerprint.value,),
            _resample=False,
            _source_kind="snapshot",
            _stages=(),
            _upstream=upstream,
            _snapshot_fingerprint=source_fingerprint,
            _cache_config=CacheConfig.resolve(),
        )

    def _build_source_stream(self, *, context: RuntimeContext) -> Iterable[object]:
        """Build the materialized Lance source iterator for this runtime context."""
        if self._upstream is None or self._snapshot_fingerprint is None or self._cache_config is None:
            msg = "[InvalidSnapshotSource] snapshot source configuration is incomplete"
            raise RuntimeError(msg)
        entry = ensure_snapshot(
            self._upstream,
            source_fingerprint=self._snapshot_fingerprint,
            cache_config=self._cache_config,
            build_context=context,
        )
        source = list_lance_sources(snapshot_lance_paths(entry))[0]
        return _SnapshotSourceIterator(
            _LanceSourceIterator(
                source=source,
                context=context,
                resample=False,
                columns=None,
                read_batch_size=1024,
                source_fingerprint=stable_fingerprint(self._source_fingerprint()),
                shuffle_mode="none",
                selection=self._resolve_selection(source.total_rows),
            )
        )

    def _source_fingerprint(self) -> dict[str, object]:
        """Return the source portion of the pipeline fingerprint."""
        if self._snapshot_fingerprint is None:
            msg = "[InvalidSnapshotSource] snapshot fingerprint is missing"
            raise RuntimeError(msg)
        return {
            "kind": "snapshot",
            "source_fingerprint": self._snapshot_fingerprint.value,
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "parameters": SNAPSHOT_CACHE_PARAMETERS,
            "selection": self._selection_fingerprint(),
        }

    def split(self, fractions: Sequence[float]) -> tuple[Dataset, ...]:
        """Partition the snapshot row space into disjoint contiguous subsets."""
        if self._snapshot_selection is not None:
            msg = "[UnsupportedNestedSnapshotSubset] apply split()/sample() on the base snapshot"
            raise ValueError(msg)
        values = tuple(float(fraction) for fraction in fractions)
        split_offsets(0, values)
        total = sum(values)
        normalized = tuple(value / total for value in values)
        return tuple(
            dataclass_replace(
                self,
                _snapshot_selection=_SnapshotSplitSelection(fractions=normalized, index=index),
                _resume_state=None,
            )
            for index in range(len(normalized))
        )

    def sample(self, fraction: float, *, seed: int = 0) -> Dataset:
        """Return a seeded random row subset of the snapshot."""
        if self._snapshot_selection is not None:
            msg = "[UnsupportedNestedSnapshotSubset] apply split()/sample() on the base snapshot"
            raise ValueError(msg)
        fraction = float(fraction)
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            msg = f"[InvalidSampleFraction] fraction must be in (0, 1], got={fraction!r}"
            raise ValueError(msg)
        return dataclass_replace(
            self,
            _snapshot_selection=_SnapshotSampleSelection(fraction=fraction, seed=seed),
            _resume_state=None,
        )

    def _resolve_selection(self, total_rows: int) -> LanceSelection | None:
        selection = self._snapshot_selection
        if selection is None:
            return None
        if isinstance(selection, _SnapshotSplitSelection):
            offsets = split_offsets(total_rows, selection.fractions)
            return LanceSelection(
                start=offsets[selection.index],
                count=offsets[selection.index + 1] - offsets[selection.index],
                total=total_rows,
            )
        return LanceSelection(
            start=0,
            count=round(selection.fraction * total_rows),
            total=total_rows,
            seed=selection.seed,
        )

    def _selection_fingerprint(self) -> dict[str, object] | None:
        selection = self._snapshot_selection
        if selection is None:
            return None
        if isinstance(selection, _SnapshotSplitSelection):
            return {
                "kind": "split",
                "fractions": list(selection.fractions),
                "index": selection.index,
            }
        return {
            "kind": "sample",
            "fraction": selection.fraction,
            "seed": selection.seed,
        }
