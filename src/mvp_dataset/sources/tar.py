"""Tar shard source implementation for streaming sample iteration."""

from __future__ import annotations

import hashlib
import io
import itertools
import os
import tarfile
from functools import partial
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath
from string import hexdigits
from typing import Final, cast

from ..core.types import PathLikeStr, Sample, SidecarSpec, TarSelectValue


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


def field_matches(field: str, prefixes: Sequence[str]) -> bool:
    """Return ``True`` if *field* belongs to one of the selected field groups."""

    for prefix in prefixes:
        if field == prefix or field.startswith(f"{prefix}."):
            return True
    return False


def fingerprint_parts(*parts: str) -> str:
    """Return a short stable fingerprint for cache identity."""

    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def callable_fingerprint(fn: Callable[..., object]) -> str:
    """Return a deterministic fingerprint for one callable."""

    if isinstance(fn, partial):
        keyword_items = () if fn.keywords is None else tuple(sorted(fn.keywords.items()))
        return fingerprint_parts(
            "partial",
            callable_fingerprint(fn.func),
            repr(fn.args),
            repr(keyword_items),
        )

    code = getattr(fn, "__code__", None)
    code_fingerprint = ""
    if code is not None:
        code_fingerprint = fingerprint_parts(
            code.co_code.hex(),
            repr(code.co_consts),
            repr(code.co_names),
        )
    closure = getattr(fn, "__closure__", None)
    closure_fingerprint = ""
    if closure is not None:
        closure_fingerprint = repr(tuple(cell.cell_contents for cell in closure))
    return fingerprint_parts(
        getattr(fn, "__module__", "<unknown>"),
        getattr(fn, "__qualname__", getattr(fn, "__name__", repr(fn))),
        code_fingerprint,
        repr(getattr(fn, "__defaults__", None)),
        repr(getattr(fn, "__kwdefaults__", None)),
        closure_fingerprint,
    )


def cache_tar_dir(shard_path: PathLikeStr) -> Path:
    """Return the cache directory used for fingerprinted shard sidecars."""

    return Path(shard_path).parent / ".cache"


def cache_tar_glob(shard_path: PathLikeStr, key: str) -> str:
    """Return the glob pattern for all fingerprinted cache sidecars of one key."""

    path = Path(shard_path)
    return str(cache_tar_dir(path) / f"{path.stem}-{key}-*{path.suffix}")


def tar_source_fingerprint(shard_paths: Sequence[PathLikeStr]) -> str:
    """Return the base fingerprint used by tar map cache stages."""

    return fingerprint_parts("tar-source", *(str(path) for path in shard_paths))


def tar_map_output_fingerprint(input_fingerprint: str, fn: Callable[..., object]) -> str:
    """Return the output fingerprint for one tar map stage."""

    return fingerprint_parts("tar-map-v2", input_fingerprint, callable_fingerprint(fn))


def cache_field_name(field_name: str, *, key: str, fingerprint: str) -> str:
    """Return one field name rewritten to include the stage fingerprint."""

    if field_name == key:
        return f"{key}.{fingerprint}"
    prefix = f"{key}."
    if not field_name.startswith(prefix):
        msg = f"[InvalidCacheFieldName] field={field_name!r} does not match key prefix {key!r}"
        raise ValueError(msg)
    return f"{key}.{fingerprint}.{field_name[len(prefix):]}"


def is_cache_field_name(field_name: str, *, key: str) -> bool:
    """Return whether one field name is a fingerprinted cached output for ``key``."""

    prefix = f"{key}."
    if not field_name.startswith(prefix):
        return False
    remainder = field_name[len(prefix):]
    fingerprint = remainder.split(".", 1)[0]
    return len(fingerprint) == 16 and all(char in hexdigits for char in fingerprint)


def cache_tar_path(
    shard_path: PathLikeStr,
    key: str,
    fingerprint: str | None = None,
) -> str:
    """Return the cache tar path for one selected field group."""

    path = Path(shard_path)
    if fingerprint is not None:
        return str(cache_tar_dir(path) / f"{path.stem}-{key}-{fingerprint}{path.suffix}")
    return str(path.with_name(f"{path.stem}_{key}{path.suffix}"))


def normalize_select_output(key: str, value: TarSelectValue) -> dict[str, bytes]:
    """Normalize one select preprocessor result into tar field payloads."""

    if isinstance(value, (bytes, bytearray)):
        return {key: bytes(value)}
    if not isinstance(value, Mapping):
        msg = f"[InvalidSelectOutput] key={key!r} expected bytes or mapping, got={type(value).__name__}"
        raise TypeError(msg)

    normalized: dict[str, bytes] = {}
    for field_name, payload in value.items():
        if not isinstance(field_name, str):
            msg = f"[InvalidSelectOutput] key={key!r} field name must be str, got={type(field_name).__name__}"
            raise TypeError(msg)
        if not field_matches(field_name, (key,)):
            msg = (
                f"[InvalidSelectOutput] key={key!r} field {field_name!r} does not match "
                "the requested key prefix"
            )
            raise ValueError(msg)
        if not isinstance(payload, (bytes, bytearray)):
            msg = (
                f"[InvalidSelectOutput] key={key!r} field={field_name!r} expected bytes, "
                f"got={type(payload).__name__}"
            )
            raise TypeError(msg)
        normalized[field_name] = bytes(payload)
    if not normalized:
        msg = f"[InvalidSelectOutput] key={key!r} preprocessor returned no fields"
        raise ValueError(msg)
    return normalized


def iter_tar(
    shard_path: PathLikeStr,
    key_dot_level: int = 1,
    field_prefixes: Sequence[str] | None = None,
) -> Iterator[Sample]:
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
            if field_prefixes is not None and not field_matches(field, field_prefixes):
                continue

            extracted = stream.extractfile(member)
            if extracted is None:
                msg = f"failed to extract tar member {member.name!r} from {shard!r}"
                raise tarfile.ExtractError(msg)
            payload = extracted.read()
            if field in current_sample:
                msg = f"duplicate field {field!r} for key {key!r} in shard {shard!r}"
                raise ValueError(msg)
            current_sample[field] = payload

    if current_sample is not None:
        yield current_sample


def count_tar_samples(shard_path: PathLikeStr, *, key_dot_level: int = 1) -> int:
    """Return the number of grouped samples in one tar shard."""

    return sum(1 for _ in iter_tar(shard_path, key_dot_level=key_dot_level))


def sample_fields(sample: Sample) -> tuple[str, ...]:
    """Return non-metadata field names from one sample."""

    return tuple(field for field in sample if not (field.startswith("__") and field.endswith("__")))


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
    field_prefixes: Sequence[str] | None = None,
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
            yield from iter_tar(
                shard_path,
                key_dot_level=key_dot_level,
                field_prefixes=field_prefixes,
            )
            continue

        # Build one iterator per sidecar alongside the main iterator.
        main_iter = iter_tar(
            shard_path,
            key_dot_level=key_dot_level,
            field_prefixes=field_prefixes,
        )
        sidecar_iters: list[tuple[str, Iterator[Sample]]] = [
        ]
        for name, fn in sidecars:
            sidecar_path = Path(fn(shard_path))
            # Fingerprinted map caches are optional per shard: shards without a
            # matching key simply do not materialize that sidecar.
            if ":" in name and not sidecar_path.is_file():
                continue
            sidecar_iters.append(
                (
                    name,
                    iter_tar(
                        sidecar_path,
                        key_dot_level=key_dot_level,
                        field_prefixes=field_prefixes,
                    ),
                )
            )
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

            for i, (sidecar_name, _) in enumerate(sidecar_iters):
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


def write_select_cache(
    shard_path: PathLikeStr,
    *,
    key: str,
    samples: Iterable[Sample],
    fingerprint: str | None = None,
    expected_sample_count: int | None = None,
) -> str:
    """Write one cached select tar for the provided samples."""

    output_path = cache_tar_path(shard_path, key, fingerprint)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    temp_path = f"{output_path}.tmp"
    written_samples = 0
    with tarfile.open(temp_path, mode="w") as archive:
        for sample in samples:
            sample_key = sample.get("__key__")
            if not isinstance(sample_key, str):
                msg = f"[InvalidSampleKey] key={key!r} shard={str(shard_path)!r} sample missing string '__key__'"
                raise ValueError(msg)
            written_samples += 1
            for field_name, payload in sample.items():
                if field_name.startswith("__") and field_name.endswith("__"):
                    continue
                if not isinstance(payload, (bytes, bytearray)):
                    msg = (
                        f"[InvalidSelectOutput] key={key!r} field={field_name!r} expected bytes in cache writer, "
                        f"got={type(payload).__name__}"
                    )
                    raise TypeError(msg)
                member_name = f"{sample_key}.{field_name}"
                info = tarfile.TarInfo(name=member_name)
                payload_bytes = bytes(payload)
                info.size = len(payload_bytes)
                archive.addfile(info, io.BytesIO(payload_bytes))
    if expected_sample_count is not None and written_samples != expected_sample_count:
        Path(temp_path).unlink(missing_ok=True)
        msg = (
            f"[CacheSampleCountMismatch] key={key!r} shard={str(shard_path)!r} "
            f"expected={expected_sample_count} wrote={written_samples}"
        )
        raise ValueError(msg)
    os.replace(temp_path, output_path)
    return output_path
