"""Chainable iterator dataset API for mm-loader."""

from __future__ import annotations

import glob
import os
import random
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..core.types import RuntimeContext
from ..distributed import split_shards
from ..distributed.sharding import split_items
from ..join.tar_join import iter_strict_tar_join
from ..sources import iter_tar_records
from .ops import batch_samples, map_samples, shuffle_samples, unbatch_samples

ShardInput = str | os.PathLike[str]
SidecarSpec = tuple[str, ShardInput | Sequence[ShardInput]]
Stage = Callable[[Iterable[object]], Iterable[object]]
SourceFactory = Callable[[], Iterator[object]]
_BRACE_RANGE = re.compile(r"^(.*)\{(\d+)\.\.(\d+)\}(.*)$")


def _expand_brace_range(spec: str) -> list[str]:
    """Expand one numeric brace range like ``a-{000..003}.tar``.

    If no numeric brace range exists, the original spec is returned.
    """

    match = _BRACE_RANGE.match(spec)
    if match is None:
        return [spec]

    prefix, start_raw, end_raw, suffix = match.groups()
    start = int(start_raw)
    end = int(end_raw)
    width = max(len(start_raw), len(end_raw))
    step = 1 if end >= start else -1

    expanded: list[str] = []
    for index in range(start, end + step, step):
        expanded.append(f"{prefix}{index:0{width}d}{suffix}")
    return expanded


def _expand_single_spec(spec: str) -> list[str]:
    expanded_specs = _expand_brace_range(spec)
    expanded_paths: list[str] = []

    for expanded in expanded_specs:
        if any(char in expanded for char in "*?["):
            matches = sorted(glob.glob(expanded))
            if matches:
                expanded_paths.extend(matches)
            else:
                expanded_paths.append(expanded)
        else:
            expanded_paths.append(expanded)
    return expanded_paths


def _normalize_shards(shards: ShardInput | Sequence[ShardInput]) -> list[str]:
    if isinstance(shards, (str, os.PathLike)):
        specs = [str(shards)]
    else:
        specs = [os.fspath(path) for path in shards]

    normalized: list[str] = []
    for spec in specs:
        for chunk in spec.split("::"):
            chunk = chunk.strip()
            if not chunk:
                continue
            normalized.extend(_expand_single_spec(chunk))

    if not normalized:
        raise ValueError("no shards were resolved from the provided shard specification")

    return [str(Path(path)) for path in normalized]


@dataclass(frozen=True, slots=True)
class Dataset:
    """Chainable dataset built from lazy iterator stages."""

    _source_factory: SourceFactory
    _stages: tuple[Stage, ...]
    context: RuntimeContext
    source_kind: str | None
    primary_shards: tuple[str, ...]

    @classmethod
    def from_tars(
        cls,
        shards: ShardInput | Sequence[ShardInput],
        *,
        context: RuntimeContext,
    ) -> Dataset:
        """Build a dataset from local tar shard paths."""

        normalized_shards = _normalize_shards(shards)

        def source() -> Iterator[object]:
            selected = split_shards(normalized_shards, context)
            for shard_path in selected:
                yield from iter_tar_records(shard_path)

        return cls(
            _source_factory=source,
            _stages=(),
            context=context,
            source_kind="tars",
            primary_shards=tuple(normalized_shards),
        )

    def _append_stage(self, stage: Stage) -> Dataset:
        return Dataset(
            _source_factory=self._source_factory,
            _stages=self._stages + (stage,),
            context=self.context,
            source_kind=self.source_kind,
            primary_shards=self.primary_shards,
        )

    def join_tars(self, sidecars: Sequence[SidecarSpec]) -> Dataset:
        """Join one or more sidecar tar groups by strict shard/key alignment.

        This method is intentionally constrained for deterministic streaming:
        - it only works for tar-based sources;
        - it must be called before any other stage;
        - sidecar shard lists must match primary shard count exactly.
        """

        if self.source_kind != "tars":
            msg = "[UnsupportedSource] join_tars currently supports only tar sources"
            raise ValueError(msg)
        if self._stages:
            msg = "[JoinOrder] join_tars must be the first stage after from_tars"
            raise ValueError(msg)
        if len(sidecars) == 0:
            msg = "[InvalidJoinConfig] sidecars cannot be empty"
            raise ValueError(msg)

        primary_shards = list(self.primary_shards)
        normalized_sidecars: list[tuple[str, tuple[str, ...]]] = []
        seen_names: set[str] = set()

        for sidecar_name, sidecar_shards in sidecars:
            if sidecar_name in seen_names:
                msg = f"[InvalidJoinConfig] duplicated sidecar_name={sidecar_name!r}"
                raise ValueError(msg)
            seen_names.add(sidecar_name)

            normalized = _normalize_shards(sidecar_shards)
            if len(normalized) != len(primary_shards):
                msg = (
                    "[ShardCountMismatch] "
                    f"sidecar_name={sidecar_name!r} "
                    f"primary_count={len(primary_shards)} "
                    f"sidecar_count={len(normalized)}"
                )
                raise ValueError(msg)

            normalized_sidecars.append((sidecar_name, tuple(normalized)))

        shard_rows: list[tuple[str, dict[str, str]]] = []
        for index, primary_shard in enumerate(primary_shards):
            sidecar_row = {
                sidecar_name: sidecar_shards[index]
                for sidecar_name, sidecar_shards in normalized_sidecars
            }
            shard_rows.append((primary_shard, sidecar_row))

        selected_shard_rows = split_items(shard_rows, self.context)

        def stage(data: Iterable[object]) -> Iterable[object]:
            return iter_strict_tar_join(data, selected_shard_pairs=selected_shard_rows)

        return self._append_stage(stage)

    def map(self, fn: Callable[[object], object]) -> Dataset:
        """Append a lazy map stage."""

        def stage(data: Iterable[object]) -> Iterable[object]:
            return map_samples(data, fn)

        return self._append_stage(stage)

    def shuffle(self, buffer_size: int, initial: int | None = None) -> Dataset:
        """Append a deterministic sample-level shuffle stage."""

        seed = self.context.sample_shuffle_seed

        def stage(data: Iterable[object]) -> Iterable[object]:
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

        stream: Iterable[object] = self._source_factory()
        for stage in self._stages:
            stream = stage(stream)
        yield from stream
