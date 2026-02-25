"""Tar shard source implementation for streaming sample iteration."""

from __future__ import annotations

import itertools
import tarfile
from collections.abc import Iterator, Sequence
from pathlib import PurePosixPath
from typing import Final, cast

from ..core.types import PathLikeStr, Sample, SidecarSpec


def _is_meta_member(name: str) -> bool:
    """Return ``True`` if a tar member should be treated as metadata and skipped."""

    basename = PurePosixPath(name).name
    return basename.startswith("__") and basename.endswith("__")


def _split_key_and_field(name: str, *, key_dot_level: int) -> tuple[str, str] | None:
    """Extract ``(key, field)`` from tar member names.

    ``key_dot_level`` controls how many dot-separated segments are used as key.
    For example, with ``key_dot_level=1``:
    - ``id.extra.ext`` -> key: ``id``, field: ``extra.ext``
    With ``key_dot_level=2``:
    - ``id.extra.ext`` -> key: ``id.extra``, field: ``ext``

    Returns ``None`` for files that do not follow this convention.
    """

    basename = PurePosixPath(name).name
    if "." not in basename:
        return None
    if key_dot_level <= 0:
        msg = f"[InvalidTarKeyDotLevel] key_dot_level must be > 0, got={key_dot_level}"
        raise ValueError(msg)

    parts = basename.split(".")
    if len(parts) <= key_dot_level:
        return None

    key = ".".join(parts[:key_dot_level])
    field = ".".join(parts[key_dot_level:])

    if not key or not field:
        return None
    return key, field


def iter_tar(shard_path: PathLikeStr, key_dot_level: int = 1) -> Iterator[Sample]:
    """Iterate a tar shard and yield grouped samples by key.

    Files are expected to follow the naming convention implied by ``key_dot_level``.
    Consecutive entries with the same parsed key are grouped into one sample.

    Each yielded sample contains:
    - ``__key__``: sample key
    - ``__shard__``: shard path
    - ``__index_in_shard__``: zero-based row index within the shard
    - one entry per field with raw ``bytes`` payload
    """

    shard = str(shard_path)
    current_key: str | None = None
    current_sample: Sample | None = None
    index_in_shard = 0

    with tarfile.open(shard, mode="r|*") as stream:
        for member in stream:
            if not member.isfile():
                continue
            if _is_meta_member(member.name):
                continue

            key_field = _split_key_and_field(member.name, key_dot_level=key_dot_level)
            if key_field is None:
                continue
            key, field = key_field

            extracted = stream.extractfile(member)
            if extracted is None:
                msg = f"failed to extract tar member {member.name!r} from {shard!r}"
                raise tarfile.ExtractError(msg)
            payload = extracted.read()

            if key != current_key:
                if current_sample is not None:
                    yield current_sample
                    index_in_shard += 1
                current_key = key
                current_sample = {
                    "__key__": key,
                    "__shard__": shard,
                    "__index_in_shard__": index_in_shard,
                }

            assert current_sample is not None
            if field in current_sample:
                msg = f"duplicate field {field!r} for key {key!r} in shard {shard!r}"
                raise ValueError(msg)
            current_sample[field] = payload

    if current_sample is not None:
        yield current_sample


# Sentinel used to detect length mismatches between main and sidecar iterators.
_SENTINEL: Final[object] = object()


def _require_sample_key(sample: Sample, *, shard_path: PathLikeStr, source_name: str) -> str:
    """Read and validate ``__key__`` from a decoded sample."""

    key = sample.get("__key__")
    if not isinstance(key, str):
        msg = f"[InvalidSampleKey] source={source_name!r} shard={str(shard_path)!r} sample missing string '__key__'"
        raise ValueError(msg)
    return key


def iter_tars(
    shard_paths: Iterator[PathLikeStr],
    key_dot_level: int = 1,
    sidecars: Sequence[SidecarSpec] | None = None,
) -> Iterator[Sample]:
    """Iterate multiple tar shards, optionally merging sidecar tars.

    Args:
        shard_paths: Iterator of paths to the main tar shards.
        key_dot_level: Number of dot-separated segments used as the sample key.
        sidecars: Optional list of ``(name, path_fn)`` pairs.  For each main
            shard path, ``path_fn(shard_path)`` is called to locate the
            corresponding sidecar shard.  After yielding one sample from the
            main shard and one from every sidecar, the ``__key__`` values are
            checked for equality; on mismatch a :class:`ValueError` is raised.
            Sidecar metadata fields (``__key__``, ``__shard__``,
            ``__index_in_shard__``) are dropped and the remaining fields are
            merged into the main sample dict.  Duplicate field names across
            shards raise a :class:`ValueError`.
    """
    for shard_path in shard_paths:
        if not sidecars:
            yield from iter_tar(shard_path, key_dot_level=key_dot_level)
            continue

        # Build one iterator per sidecar alongside the main iterator.
        main_iter = iter_tar(shard_path, key_dot_level=key_dot_level)
        sidecar_iters: list[tuple[str, Iterator[Sample]]] = [
            (name, iter_tar(fn(shard_path), key_dot_level=key_dot_level)) for name, fn in sidecars
        ]
        all_iters = [main_iter, *(it for _, it in sidecar_iters)]

        for group in itertools.zip_longest(*all_iters, fillvalue=_SENTINEL):
            # Detect length mismatches: if any iterator ran out early, at least
            # one slot in *group* will be the sentinel.
            if any(s is _SENTINEL for s in group):
                msg = (
                    f"[SidecarLengthMismatch] main shard and one or more sidecar "
                    f"tars have different numbers of samples in shard {shard_path!r}"
                )
                raise ValueError(msg)

            main_sample = cast(Sample, group[0])
            main_key = _require_sample_key(main_sample, shard_path=shard_path, source_name="main")
            merged: Sample = dict(main_sample)

            for i, (sidecar_name, _) in enumerate(sidecars):
                sidecar_sample = cast(Sample, group[i + 1])
                sidecar_key = _require_sample_key(
                    sidecar_sample,
                    shard_path=shard_path,
                    source_name=sidecar_name,
                )

                # Key consistency check.
                if sidecar_key != main_key:
                    msg = (
                        f"[SidecarKeyMismatch] main key {main_key!r} does not match "
                        f"sidecar {sidecar_name!r} key {sidecar_key!r} "
                        f"in shard {shard_path!r}"
                    )
                    raise ValueError(msg)

                # Merge non-metadata fields from the sidecar into the sample.
                for field, value in sidecar_sample.items():
                    if field.startswith("__") and field.endswith("__"):
                        continue  # Drop sidecar metadata fields.
                    if field in merged:
                        msg = (
                            f"[SidecarFieldConflict] field {field!r} from sidecar "
                            f"{sidecar_name!r} conflicts with an existing field "
                            f"for key {main_key!r} in shard {shard_path!r}"
                        )
                        raise ValueError(msg)
                    merged[field] = value

            yield merged
