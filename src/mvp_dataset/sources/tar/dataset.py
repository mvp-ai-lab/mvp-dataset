"""Tar dataset source configuration."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from mvp_dataset.core.context import RuntimeContext
from mvp_dataset.core.dataset import Dataset
from mvp_dataset.core.resume import stable_fingerprint
from mvp_dataset.core.types import ShardInput, SidecarSpec
from mvp_dataset.utils.url import normalize_paths

from .iterator import _TarSourceIterator
from .refs import _sidecar_fingerprint
from .types import TarShuffleMode


@dataclass(frozen=True, slots=True)
class TarDataset(Dataset):
    """Dataset configuration for tar shards."""

    _sidecar_specs: tuple[SidecarSpec, ...] = ()
    _shuffle_mode: TarShuffleMode = "shard_aware"

    @classmethod
    def from_source(
        cls,
        shards: ShardInput | Sequence[ShardInput],
        context: RuntimeContext | None = None,
        resample: bool = False,
        sidecars: Sequence[SidecarSpec] | None = None,
        shuffle_mode: TarShuffleMode = "shard_aware",
    ):
        """Build a dataset from local tar shard paths.

        Args:
            shards: Input shard path or paths.
            context: Runtime context used for sharding and deterministic randomness.
            resample: Whether to repeat the source indefinitely across rounds.
            sidecars: Sidecar tar specifications joined to main samples.
            shuffle_mode: Source-level shuffle mode.

        Returns:
            A dataset configured for the requested source."""
        runtime_context = RuntimeContext.from_runtime() if context is None else context
        if shuffle_mode == "global":
            msg = "[UnsupportedTarShuffleMode] shuffle_mode='global'"
            raise ValueError(msg)
        if shuffle_mode not in ("none", "shard_aware"):
            msg = f"[InvalidTarShuffleMode] expected none or shard_aware, got={shuffle_mode!r}"
            raise ValueError(msg)
        normalized_shards = normalize_paths(shards)
        if not all(path.endswith(".tar") for path in normalized_shards):
            msg = f"[InvalidSourceType] expected .tar inputs, got={normalized_shards!r}"
            raise ValueError(msg)
        if len(normalized_shards) < runtime_context.total_slots:
            msg = (
                f"[InsufficientTarShards] got {len(normalized_shards)} tar file(s) "
                f"but {runtime_context.total_slots} slot(s) — split your tar archives "
                f"into at least {runtime_context.total_slots} shards before loading"
            )
            raise ValueError(msg)

        sidecar_specs = tuple(sidecars) if sidecars else ()

        return cls(
            context=runtime_context,
            _source=normalized_shards,
            _resample=resample,
            _source_kind="tar",
            _stages=(),
            _sidecar_specs=sidecar_specs,
            _shuffle_mode=shuffle_mode,
        )

    def _build_source_stream(self, *, context: RuntimeContext) -> Iterable[object]:
        """Build the source iterator for a runtime context."""
        return _TarSourceIterator(
            shards=self._source,
            context=context,
            resample=self._resample,
            sidecars=self._sidecar_specs,
            shuffle_mode=self._shuffle_mode,
            source_fingerprint=stable_fingerprint(self._source_fingerprint()),
        )

    def _source_fingerprint(self) -> dict[str, object]:
        """Return the source portion of the pipeline fingerprint."""
        return {
            "kind": "tar",
            "resample": self._resample,
            "shuffle_mode": self._shuffle_mode,
            "sidecars": _sidecar_fingerprint(self._sidecar_specs, self._source),
            "shards": [
                {
                    "path": shard,
                    "mtime_ns": stat.st_mtime_ns,
                    "size": stat.st_size,
                }
                for shard in self._source
                for stat in (Path(shard).stat(),)
            ],
        }
