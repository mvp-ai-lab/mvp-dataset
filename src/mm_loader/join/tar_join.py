"""Strict streaming tar-to-tar join implementation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Final

from ..core.types import Sample
from ..sources import iter_tar_records

_SENTINEL: Final[object] = object()


def _require_str(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        msg = f"[InvalidSample] required string field={field!r}, got={type(value)!r}"
        raise ValueError(msg)
    return value


def _ensure_no_extra_rows(
    sidecar_iters: Mapping[str, Iterator[Sample]],
    sidecar_shards: Mapping[str, str],
    *,
    primary_shard: str,
) -> None:
    """Fail if any sidecar shard still has unread rows for the same primary shard."""

    for sidecar_name, iterator in sidecar_iters.items():
        extra = next(iterator, _SENTINEL)
        if extra is not _SENTINEL:
            sidecar_shard = sidecar_shards[sidecar_name]
            msg = (
                "[SidecarHasExtraRows] "
                f"sidecar_name={sidecar_name!r} "
                f"primary_shard={primary_shard!r} "
                f"sidecar_shard={sidecar_shard!r}"
            )
            raise ValueError(msg)


def iter_strict_tar_join(
    primary_samples: Iterable[object],
    *,
    selected_shard_pairs: Iterable[tuple[str, Mapping[str, str]]],
) -> Iterator[object]:
    """Join sidecar tar samples into primary samples with strict key/order checks.

    Args:
        primary_samples: Stream of samples from primary tar shards.
        selected_shard_pairs: Selected shard mapping in runtime order:
            ``(primary_shard, {sidecar_name: sidecar_shard, ...})``.
    """

    shard_pairs_iter = iter(selected_shard_pairs)
    current_primary_shard: str | None = None
    current_sidecar_shards: dict[str, str] = {}
    current_sidecar_iters: dict[str, Iterator[Sample]] = {}

    def open_sidecar_iters_for_primary(primary_shard: str) -> None:
        nonlocal current_primary_shard, current_sidecar_shards, current_sidecar_iters

        pair = next(shard_pairs_iter, _SENTINEL)
        if pair is _SENTINEL:
            msg = f"[ShardPairMissing] missing pair for primary_shard={primary_shard!r}"
            raise ValueError(msg)

        planned_primary_shard, sidecar_shards = pair
        if planned_primary_shard != primary_shard:
            msg = (
                "[ShardPairMissing] "
                f"expected_primary_shard={planned_primary_shard!r} "
                f"actual_primary_shard={primary_shard!r}"
            )
            raise ValueError(msg)

        current_primary_shard = primary_shard
        current_sidecar_shards = dict(sidecar_shards)
        current_sidecar_iters = {
            sidecar_name: iter(iter_tar_records(sidecar_shard))
            for sidecar_name, sidecar_shard in current_sidecar_shards.items()
        }

    for sample in primary_samples:
        if not isinstance(sample, dict):
            msg = f"[InvalidSample] expected dict sample, got={type(sample)!r}"
            raise ValueError(msg)

        primary_shard = _require_str(sample.get("__shard__"), field="__shard__")
        primary_key = _require_str(sample.get("__key__"), field="__key__")

        if primary_shard != current_primary_shard:
            if current_primary_shard is not None:
                _ensure_no_extra_rows(
                    current_sidecar_iters,
                    current_sidecar_shards,
                    primary_shard=current_primary_shard,
                )
            open_sidecar_iters_for_primary(primary_shard)

        merged = dict(sample)
        for sidecar_name, sidecar_iter in current_sidecar_iters.items():
            sidecar_sample = next(sidecar_iter, _SENTINEL)
            sidecar_shard = current_sidecar_shards[sidecar_name]

            if sidecar_sample is _SENTINEL:
                msg = (
                    "[SidecarExhaustedEarly] "
                    f"sidecar_name={sidecar_name!r} "
                    f"primary_shard={primary_shard!r} "
                    f"sidecar_shard={sidecar_shard!r} "
                    f"primary_key={primary_key!r}"
                )
                raise ValueError(msg)

            sidecar_key = _require_str(sidecar_sample.get("__key__"), field="__key__")
            if sidecar_key != primary_key:
                msg = (
                    "[KeyMismatch] "
                    f"sidecar_name={sidecar_name!r} "
                    f"primary_shard={primary_shard!r} "
                    f"sidecar_shard={sidecar_shard!r} "
                    f"primary_key={primary_key!r} "
                    f"sidecar_key={sidecar_key!r}"
                )
                raise ValueError(msg)

            for field, value in sidecar_sample.items():
                if field.startswith("__"):
                    continue
                if field in merged:
                    msg = (
                        "[FieldCollision] "
                        f"sidecar_name={sidecar_name!r} "
                        f"primary_shard={primary_shard!r} "
                        f"sidecar_shard={sidecar_shard!r} "
                        f"field={field!r} "
                        f"key={primary_key!r}"
                    )
                    raise ValueError(msg)
                merged[field] = value

        yield merged

    if current_primary_shard is not None:
        _ensure_no_extra_rows(
            current_sidecar_iters,
            current_sidecar_shards,
            primary_shard=current_primary_shard,
        )

