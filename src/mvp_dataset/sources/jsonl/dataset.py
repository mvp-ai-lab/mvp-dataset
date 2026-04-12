from collections.abc import Sequence
from dataclasses import dataclass

from mvp_dataset.core.context import RuntimeContext
from mvp_dataset.core.dataset import Dataset
from mvp_dataset.core.types import RefFieldSpec, ShardInput
from mvp_dataset.utils.url import normalize_paths

from .utils import iter_jsonls, split_jsonl_files


@dataclass(frozen=True, slots=True)
class JsonlDataset(Dataset):
    _ref_fields: tuple[RefFieldSpec, ...] = ()

    @classmethod
    def from_source(
        cls,
        shards: ShardInput | Sequence[ShardInput],
        context: RuntimeContext | None = None,
        resample: bool = False,
        ref_fields: Sequence[RefFieldSpec] | None = None,
    ):
        """Build a dataset from local JSONL shard paths.

        Args:
            shards: One or more file paths, glob specs, or brace-expansion specs.
            context: Optional execution context. If omitted, inferred from runtime.
            resample: Whether to loop shards indefinitely across rounds.
            ref_fields: Optional sequence of ``(field_name, base_dir)`` pairs for
                resolving tar-referenced fields in JSONL rows.

        Returns:
            A dataset whose source is the normalized JSONL shard path list.

        Raises:
            ValueError: If any input path does not end with ``.jsonl``.
        """
        runtime_context = RuntimeContext.from_runtime() if context is None else context
        normalized_shards = normalize_paths(shards)
        if not all(path.endswith(".jsonl") for path in normalized_shards):
            msg = f"[InvalidSourceType] expected .jsonl inputs, got={normalized_shards!r}"
            raise ValueError(msg)

        source_items = split_jsonl_files(normalized_shards, runtime_context.total_slots)

        ref_fields_tuple = tuple(ref_fields) if ref_fields else ()

        def _iter_source(shard_stream):
            return iter_jsonls(
                shard_stream,
                ref_fields=ref_fields_tuple,
            )

        return cls(
            context=runtime_context,
            _source=source_items,
            _resample=resample,
            _source_kind="jsonl",
            _stages=(),
            _iter_source_stream=_iter_source,
            _ref_fields=ref_fields_tuple,
        )
