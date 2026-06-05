"""JSONL dataset source configuration."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from mvp_dataset.core.context import RuntimeContext
from mvp_dataset.core.dataset import Dataset
from mvp_dataset.core.resume import stable_fingerprint
from mvp_dataset.core.types import ShardInput, TarUriRefFieldSpec
from mvp_dataset.utils.url import normalize_paths

from .iterator import _JsonlSourceIterator
from .sharding import split_jsonl_files
from .types import JsonlShuffleMode


@dataclass(frozen=True, slots=True)
class JsonlDataset(Dataset):
    """Dataset configuration for JSONL shards."""

    _ref_fields: tuple[TarUriRefFieldSpec, ...] = ()
    _shuffle_mode: JsonlShuffleMode = "shard_aware"

    @classmethod
    def from_source(
        cls,
        shards: ShardInput | Sequence[ShardInput],
        context: RuntimeContext | None = None,
        resample: bool = False,
        ref_fields: Sequence[TarUriRefFieldSpec] | None = None,
        shuffle_mode: JsonlShuffleMode = "shard_aware",
    ):
        """Build a dataset from local JSONL shard paths.

        Args:
            shards: Input shard path or paths.
            context: Runtime context used for sharding and deterministic randomness.
            resample: Whether to repeat the source indefinitely across rounds.
            ref_fields: JSONL reference field specifications to resolve from tar URIs.
            shuffle_mode: Source-level shuffle mode.

        Returns:
            A dataset configured for the requested source."""
        runtime_context = RuntimeContext.from_runtime() if context is None else context
        if shuffle_mode == "global":
            msg = "[UnsupportedJsonlShuffleMode] shuffle_mode='global'"
            raise ValueError(msg)
        if shuffle_mode not in ("none", "shard_aware"):
            msg = f"[InvalidJsonlShuffleMode] expected none or shard_aware, got={shuffle_mode!r}"
            raise ValueError(msg)
        normalized_shards = normalize_paths(shards)
        if not all(path.endswith(".jsonl") for path in normalized_shards):
            msg = f"[InvalidSourceType] expected .jsonl inputs, got={normalized_shards!r}"
            raise ValueError(msg)

        source_items = split_jsonl_files(normalized_shards, runtime_context.total_slots)

        ref_fields_tuple = tuple(ref_fields) if ref_fields else ()

        return cls(
            context=runtime_context,
            _source=source_items,
            _resample=resample,
            _source_kind="jsonl",
            _stages=(),
            _ref_fields=ref_fields_tuple,
            _shuffle_mode=shuffle_mode,
        )

    def _build_source_stream(self, *, context: RuntimeContext) -> Iterable[object]:
        """Build the source iterator for a runtime context."""
        return _JsonlSourceIterator(
            shards=self._source,
            context=context,
            resample=self._resample,
            shuffle_mode=self._shuffle_mode,
            ref_fields=self._ref_fields,
            source_fingerprint=stable_fingerprint(self._source_fingerprint()),
        )

    def _source_fingerprint(self) -> dict[str, object]:
        """Return the source portion of the pipeline fingerprint."""
        return {
            "kind": "jsonl",
            "resample": self._resample,
            "shuffle_mode": self._shuffle_mode,
            "ref_fields": [(field, str(base_dir)) for field, base_dir in self._ref_fields],
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
