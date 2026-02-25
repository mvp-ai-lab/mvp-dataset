"""Chainable iterator dataset API for mvp-dataset."""

from __future__ import annotations

import importlib
import json
import os
import random
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Literal, cast

from ..core.types import (
    GroupedSample,
    RefFieldSpec,
    RuntimeContext,
    Sample,
    ShardInput,
    SidecarSpec,
    Stage,
)
from ..sources import iter_jsonls, iter_tars
from ..utils.sharding import iter_items
from ..utils.url import normalize_paths
from .ops import batch_samples, map_samples, shuffle_samples, unbatch_samples

SourceKind = Literal["jsonl", "tars"]
SourceStore = list[str] | list[Sample] | list[GroupedSample]
SourceShape = Literal["tar_paths", "jsonl_flat", "jsonl_grouped"]


def torch_iterabledataset_class(
    import_module: Callable[[str], ModuleType] = importlib.import_module,
) -> type:
    """Resolve ``torch.utils.data.IterableDataset`` with a no-torch fallback."""

    try:
        torch_utils_data = import_module("torch.utils.data")
    except ModuleNotFoundError:

        class _IterableDatasetFallback:
            """Fallback IterableDataset shim when torch is unavailable."""

        return _IterableDatasetFallback

    return torch_utils_data.IterableDataset


@dataclass(frozen=True, slots=True)
class Dataset(torch_iterabledataset_class()):
    """Chainable dataset built from lazy iterator stages.

    The dataset stores one of two source backends:
    - ``"tars"``: a list of tar shard paths
    - ``"jsonl"``: in-memory JSONL rows (flat or grouped)
    """

    context: RuntimeContext
    source_kind: SourceKind

    _source: SourceStore
    _source_shape: SourceShape
    _stages: tuple[Stage, ...]
    _resample: bool

    _sidecar_specs: tuple[SidecarSpec, ...]
    _ref_fields: tuple[RefFieldSpec, ...]
    _group_by: str | None

    @classmethod
    def from_source(
        cls,
        shards: ShardInput | Sequence[ShardInput],
        context: RuntimeContext | None = None,
        resample: bool = False,
    ) -> Dataset:
        """Build a dataset from local JSONL files or tar shards.

        Args:
            shards: One or more file paths, glob specs, or brace-expansion specs.
            context: Optional execution context. If omitted, inferred from runtime.
            resample: Whether to loop shards indefinitely across rounds.
        """
        runtime_context = RuntimeContext.from_runtime() if context is None else context
        normalized_shards = normalize_paths(shards)

        if all(path.endswith(".jsonl") for path in normalized_shards):
            jsonl_samples: list[Sample] = []
            for file in normalized_shards:
                with open(file, encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        try:
                            parsed = json.loads(line)
                        except json.JSONDecodeError as exc:
                            msg = f"[InvalidJsonLine] file={file!r} line={i + 1} reason={exc.msg}"
                            raise ValueError(msg) from exc
                        if not isinstance(parsed, dict):
                            msg = f"[InvalidJsonSample] file={file!r} line={i + 1} expected object row"
                            raise ValueError(msg)
                        sample: Sample = dict(parsed)
                        sample["__index_in_file__"] = i
                        sample["__file__"] = file
                        sample["__key__"] = f"{file}:{i}"
                        jsonl_samples.append(sample)
            source: SourceStore = jsonl_samples
            source_kind = "jsonl"
            source_shape: SourceShape = "jsonl_flat"
        elif all(path.endswith(".tar") for path in normalized_shards):
            source = normalized_shards
            source_kind = "tars"
            source_shape = "tar_paths"
        else:
            msg = f"[InvalidSourceType] all inputs must be .jsonl or all must be .tar, got={normalized_shards!r}"
            raise ValueError(msg)

        return cls(
            _source=source,
            _source_shape=source_shape,
            _stages=(),
            context=runtime_context,
            source_kind=source_kind,
            _resample=resample,
            _sidecar_specs=(),
            _ref_fields=(),
            _group_by=None,
        )

    def join(self, sidecars: Sequence[SidecarSpec]) -> Dataset:
        """Attach sidecar tar specs for shard-level field merges."""

        if self.source_kind != "tars":
            msg = "`join` currently supports only tar sources."
            raise ValueError(msg)
        return Dataset(
            _source=self._source,
            _source_shape=self._source_shape,
            _stages=self._stages,
            context=self.context,
            source_kind=self.source_kind,
            _resample=self._resample,
            _sidecar_specs=self._sidecar_specs + tuple(sidecars),
            _ref_fields=self._ref_fields,
            _group_by=self._group_by,
        )

    def resolve_refs(self, ref_fields: Sequence[RefFieldSpec]) -> Dataset:
        """Resolve ``tar://`` URIs in selected JSONL fields during iteration."""

        if self.source_kind != "jsonl":
            msg = "`resolve_refs` currently supports only jsonl sources."
            raise ValueError(msg)
        return Dataset(
            _source=self._source,
            _source_shape=self._source_shape,
            _stages=self._stages,
            context=self.context,
            source_kind=self.source_kind,
            _resample=self._resample,
            _sidecar_specs=self._sidecar_specs,
            _ref_fields=self._ref_fields + tuple(ref_fields),
            _group_by=self._group_by,
        )

    def group_by(self, field: str) -> Dataset:
        """Group JSONL rows by a string field (``#`` suffix ignored)."""

        if self.source_kind != "jsonl":
            msg = "`group_by` currently supports only jsonl sources."
            raise ValueError(msg)
        if self._group_by is not None:
            msg = "multiple group_by stages are not supported"
            raise ValueError(msg)
        if self._source_shape != "jsonl_flat":
            msg = "[InvalidSourceShape] group_by expects flat jsonl source"
            raise TypeError(msg)

        grouped_source: dict[str, list[Sample]] = {}
        flat_source = cast(list[Sample], self._source)
        for sample in flat_source:
            raw_value = sample.get(field)
            if not isinstance(raw_value, str):
                msg = f"[InvalidGroupKey] sample missing string key for group_by field={field!r}"
                raise ValueError(msg)
            key = raw_value.split("#", 1)[0]
            grouped_source.setdefault(key, []).append(sample)
        groups = list(grouped_source.values())

        return Dataset(
            _source=groups,
            _source_shape="jsonl_grouped",
            _stages=self._stages,
            context=self.context,
            source_kind=self.source_kind,
            _resample=self._resample,
            _sidecar_specs=self._sidecar_specs,
            _ref_fields=self._ref_fields,
            _group_by=field,
        )

    def _append_stage(self, stage: Stage) -> Dataset:
        """Return a new dataset with one extra lazy stage."""

        return Dataset(
            _source=self._source,
            _source_shape=self._source_shape,
            _stages=self._stages + (stage,),
            context=self.context,
            source_kind=self.source_kind,
            _resample=self._resample,
            _sidecar_specs=self._sidecar_specs,
            _ref_fields=self._ref_fields,
            _group_by=self._group_by,
        )

    def map(self, fn: Callable[[object], object]) -> Dataset:
        """Append a lazy map stage."""

        def stage(data: Iterable[object]) -> Iterable[object]:
            return map_samples(data, fn)

        return self._append_stage(stage)

    def shuffle(self, buffer_size: int, initial: int | None = None) -> Dataset:
        """Append a deterministic sample-level shuffle stage."""

        def stage(data: Iterable[object]) -> Iterable[object]:
            seed = self.context.sample_shuffle_seed
            rng = random.Random(seed)
            return shuffle_samples(data, buffer_size=buffer_size, initial=initial, rng=rng)

        return self._append_stage(stage)

    def batch(
        self,
        batch_size: int,
        drop_last: bool = False,
        collate_fn: Callable[[list[object]], object] | None = None,
    ) -> Dataset:
        """Append a batching stage."""

        def stage(data: Iterable[object]) -> Iterable[object]:
            return batch_samples(
                data,
                batch_size=batch_size,
                drop_last=drop_last,
                collate_fn=collate_fn,
            )

        return self._append_stage(stage)

    def unbatch(self) -> Dataset:
        """Append an unbatching stage."""

        def stage(data: Iterable[object]) -> Iterable[object]:
            return unbatch_samples(data)

        return self._append_stage(stage)

    def __iter__(self) -> Iterator[object]:
        """Materialize and run the full lazy pipeline."""

        context = self.context.resolve_current_process()
        stream: Iterable[object]

        tar_key_dot_level = int(os.environ.get("LOADER_TAR_KEY_DOT_LEVEL", 1))
        if self.source_kind == "tars":
            tar_paths = self._as_tar_source()
            shard_stream = iter_items(
                tar_paths,
                context=context,
                resample=self._resample,
            )
            stream = iter_tars(
                shard_stream,
                key_dot_level=tar_key_dot_level,
                sidecars=self._sidecar_specs,
            )
        else:
            tar_cache_size = int(os.environ.get("LOADER_JSONL_TAR_CACHE_SIZE", 8))
            if self._group_by is None:
                flat_stream = iter_items(
                    self._as_flat_jsonl_source(),
                    context=context,
                    resample=self._resample,
                    grouped=False,
                )
                jsonl_stream: Iterator[Sample | GroupedSample] = flat_stream
            else:
                grouped_stream = iter_items(
                    self._as_grouped_jsonl_source(),
                    context=context,
                    resample=self._resample,
                    grouped=True,
                )
                jsonl_stream = grouped_stream
            stream = iter_jsonls(
                jsonl_stream,
                ref_fields=self._ref_fields,
                key_dot_level=tar_key_dot_level,
                tar_cache_size=tar_cache_size,
            )

        for stage in self._stages:
            stream = stage(stream)

        yield from stream

    def _as_tar_source(self) -> list[str]:
        """Return tar shard paths with O(1) source-shape check."""

        if self._source_shape != "tar_paths":
            msg = "[InvalidSourceShape] expected tar_paths source shape"
            raise TypeError(msg)
        return cast(list[str], self._source)

    def _as_flat_jsonl_source(self) -> list[Sample]:
        """Return flat JSONL samples with O(1) source-shape check."""

        if self._source_shape != "jsonl_flat":
            msg = "[InvalidSourceShape] expected jsonl_flat source shape"
            raise TypeError(msg)
        return cast(list[Sample], self._source)

    def _as_grouped_jsonl_source(self) -> list[GroupedSample]:
        """Return grouped JSONL samples with O(1) source-shape check."""

        if self._source_shape != "jsonl_grouped":
            msg = "[InvalidSourceShape] expected jsonl_grouped source shape"
            raise TypeError(msg)
        return cast(list[GroupedSample], self._source)
