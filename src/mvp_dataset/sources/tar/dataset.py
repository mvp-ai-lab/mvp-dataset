from collections.abc import Sequence
from dataclasses import dataclass

from mvp_dataset.core.context import RuntimeContext
from mvp_dataset.core.dataset import Dataset
from mvp_dataset.core.types import ShardInput, SidecarSpec
from mvp_dataset.utils.url import normalize_paths

from .utils import iter_tars


@dataclass(frozen=True, slots=True)
class TarDataset(Dataset):
    _sidecar_specs: tuple[SidecarSpec, ...] = ()

    @classmethod
    def from_source(
        cls,
        shards: ShardInput | Sequence[ShardInput],
        context: RuntimeContext | None = None,
        resample: bool = False,
        sidecars: Sequence[SidecarSpec] | None = None,
    ):
        """Build a dataset from local tar shard paths.

        Args:
            shards: One or more file paths, glob specs, or brace-expansion specs.
            context: Optional execution context. If omitted, inferred from runtime.
            resample: Whether to loop shards indefinitely across rounds.
            sidecars: Optional sequence of ``(name, path_resolver)`` pairs for
                shard-level sidecar merges. Each ``path_resolver`` receives a main
                tar shard path and must return the matching sidecar tar shard path.

        Returns:
            A dataset whose source is the normalized tar shard path list.

        Raises:
            ValueError: If any input path does not end with ``.tar``.
        """
        runtime_context = RuntimeContext.from_runtime() if context is None else context
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

        def _iter_source(shard_stream):
            return iter_tars(
                shard_stream,
                sidecars=sidecar_specs or None,
            )

        return cls(
            context=runtime_context,
            _source=normalized_shards,
            _resample=resample,
            _source_kind="tars",
            _stages=(),
            _iter_source_stream=_iter_source,
            _sidecar_specs=sidecar_specs,
        )
