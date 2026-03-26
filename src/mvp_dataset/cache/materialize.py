"""Cache materialization: source annotation, warm-up, manifest, and read path."""

from __future__ import annotations

import fcntl
import io
import itertools
import json
import os
import tarfile
import warnings
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ..core.types import RefFieldSpec, Sample, SidecarSpec
from ..log import get_logger
from ..sources.jsonl import TarManager, _parse_jsonl_line, parse_tar_uri
from ..sources.tar import iter_tar
from .codecs import decode_value, encode_value
from .fingerprint import hash_bytes

# Reserved meta member suffix inside cache tars (e.g. "0001.__cache_meta__")
_CACHE_META_SUFFIX: Final[str] = ".__cache_meta__"
# Key used inside sample dicts to carry per-sample cache metadata through pipeline
_CACHE_META_KEY: Final[str] = "__cache_meta__"


def _log_info(message: str) -> None:
    """Emit an info-level cache log via the package logger."""
    get_logger().info(message)


def _is_meta_field(name: str) -> bool:
    """Return True for dunder metadata fields (e.g. ``__key__``)."""
    return name.startswith("__") and name.endswith("__")


# ---------------------------------------------------------------------------
# Per-sample cache metadata
# ---------------------------------------------------------------------------


@dataclass
class CacheMeta:
    """Cache metadata carried inside a sample through the pre-cache stages."""

    route_shard: str
    """Source shard path that this sample is attributed to for cache routing."""

    field_sigs: dict[str, str]
    """Content-addressable signature per user field (field_name -> hex digest)."""


# ---------------------------------------------------------------------------
# Group normalization
# ---------------------------------------------------------------------------


def normalize_groups(
    groups_spec: tuple[tuple[str, ...], ...] | None,
    sample_keys: Iterable[str],
) -> list[tuple[str, ...]]:
    """Resolve a groups spec against the actual sample keys.

    Args:
        groups_spec: Explicit field grouping, or ``None`` for a single
            group containing all non-metadata keys.
        sample_keys: Field names present in the first sample.

    Returns:
        A list of field-name tuples; each tuple becomes one cache tar.
        With ``groups_spec=None`` all non-meta keys form one group.
        With an explicit spec, uncovered non-meta keys become singletons.
    """
    non_meta = [k for k in sample_keys if not _is_meta_field(k)]

    if groups_spec is None:
        return [tuple(non_meta)] if non_meta else []

    covered: set[str] = set()
    result: list[tuple[str, ...]] = []
    for group in groups_spec:
        fields = tuple(f for f in group if f in non_meta)
        if fields:
            result.append(fields)
            covered.update(fields)

    for key in non_meta:
        if key not in covered:
            result.append((key,))

    return result


def _group_label(field_names: tuple[str, ...]) -> str:
    """Build a human-readable label for a field group.

    Args:
        field_names: Sorted field names in the group.

    Returns:
        A ``+``-joined label (e.g. ``"depth+image"``), or a 16-char hash
        prefix when the label exceeds 48 characters.
    """
    label = "+".join(sorted(field_names))
    if len(label) > 48:
        return hash_bytes(label)[:16]
    return label


# ---------------------------------------------------------------------------
# Manifest and locking
# ---------------------------------------------------------------------------


def _cache_dir(shard_path: str) -> Path:
    """Return the ``.cache/`` directory co-located with *shard_path*."""
    return Path(shard_path).parent / ".cache"


def _manifest_path(shard_path: str) -> Path:
    """Return the manifest JSON path for *shard_path*."""
    return _cache_dir(shard_path) / f"{Path(shard_path).name}.manifest.json"


def _lock_path(shard_path: str) -> Path:
    """Return the lock file path for *shard_path*."""
    return _cache_dir(shard_path) / f"{Path(shard_path).name}.lock"


def read_manifest(shard_path: str, plan_fingerprint: str) -> dict[str, str] | None:
    """Read and validate a cache manifest for *shard_path*.

    Args:
        shard_path: Path to the original source shard.
        plan_fingerprint: Expected plan fingerprint to match against.

    Returns:
        A ``{group_label: tar_path}`` mapping if a valid manifest exists
        with matching fingerprint and all referenced tar files present,
        otherwise ``None``.
    """
    mp = _manifest_path(shard_path)
    if not mp.is_file():
        return None
    try:
        with mp.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("plan_fingerprint") != plan_fingerprint:
        return None
    groups: dict[str, str] = data.get("groups", {})
    if not all(Path(tp).is_file() for tp in groups.values()):
        return None
    return groups


def _write_manifest(shard_path: str, plan_fingerprint: str, groups: dict[str, str]) -> None:
    """Atomically write a cache manifest mapping group labels to tar paths.

    Args:
        shard_path: Path to the original source shard.
        plan_fingerprint: Plan fingerprint stored in the manifest.
        groups: Mapping of group label to cache tar path.
    """
    mp = _manifest_path(shard_path)
    mp.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "plan_fingerprint": plan_fingerprint,
        "source_shard": shard_path,
        "groups": groups,
    }
    tmp = mp.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    tmp.rename(mp)


class _ShardLock:
    """Context manager that acquires an exclusive flock on a per-shard lock file."""

    def __init__(self, shard_path: str) -> None:
        lp = _lock_path(shard_path)
        lp.parent.mkdir(parents=True, exist_ok=True)
        self._path = lp
        self._fh: io.IOBase | None = None

    def __enter__(self) -> _ShardLock:
        self._fh = self._path.open("w")
        fcntl.flock(self._fh, fcntl.LOCK_EX)
        return self

    def __exit__(self, *_: object) -> None:
        if self._fh is not None:
            fcntl.flock(self._fh, fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None


# ---------------------------------------------------------------------------
# Source iteration with signature annotation
# ---------------------------------------------------------------------------


def iter_tars_with_sigs(
    shard_paths: Iterator[str],
    key_dot_level: int,
    sidecars: Sequence[SidecarSpec] | None,
) -> Iterator[Sample]:
    """Iterate tar shards and annotate each sample with ``__cache_meta__``.

    Each field receives a source signature derived from the shard path,
    file stat, and tar member name.  When *sidecars* are present, sidecar
    fields get signatures from their respective sidecar tar stats.

    Args:
        shard_paths: Iterator of main tar shard paths to process.
        key_dot_level: Number of dot-separated segments used as the
            sample key inside tar member names.
        sidecars: Optional sidecar specs.  Each ``(name, path_fn)`` pair
            maps a main shard path to its corresponding sidecar tar.

    Yields:
        Sample dicts with a ``__cache_meta__`` entry containing a
        :class:`CacheMeta` instance.
    """
    _SENTINEL: Final[object] = object()

    for shard_path in shard_paths:
        shard_str = str(shard_path)
        main_stat = os.stat(shard_str)

        if not sidecars:
            for sample in iter_tar(shard_str, key_dot_level):
                sample[_CACHE_META_KEY] = _tar_sample_meta(sample, shard_str, main_stat)
                yield sample
            continue

        # Sidecar join – replicate iter_tars logic but track field origins.
        sidecar_info: list[tuple[str, str, os.stat_result]] = []
        for sc_name, sc_fn in sidecars:
            sc_path = str(sc_fn(shard_str))
            sc_stat = os.stat(sc_path)
            sidecar_info.append((sc_name, sc_path, sc_stat))

        main_iter = iter_tar(shard_str, key_dot_level)
        sidecar_iters = [iter_tar(sc_path, key_dot_level) for _, sc_path, _ in sidecar_info]
        all_iters: list[Iterator[Sample]] = [main_iter, *sidecar_iters]

        for group in itertools.zip_longest(*all_iters, fillvalue=_SENTINEL):
            if any(s is _SENTINEL for s in group):
                msg = f"[SidecarLengthMismatch] main shard and sidecars have different sample counts in {shard_str!r}"
                raise ValueError(msg)

            main_sample: Sample = group[0]  # type: ignore[assignment]
            main_key = str(main_sample.get("__key__", ""))
            merged: Sample = dict(main_sample)

            field_sigs: dict[str, str] = {}
            for field in main_sample:
                if not _is_meta_field(field):
                    member = f"{main_key}.{field}"
                    field_sigs[field] = hash_bytes(shard_str, main_stat.st_mtime_ns, main_stat.st_size, member)

            for i, (sc_name, sc_path, sc_stat) in enumerate(sidecar_info):
                sc_sample: Sample = group[i + 1]  # type: ignore[assignment]
                sc_key = str(sc_sample.get("__key__", ""))
                if sc_key != main_key:
                    msg = (
                        f"[SidecarKeyMismatch] main key {main_key!r} != "
                        f"sidecar {sc_name!r} key {sc_key!r} in {shard_str!r}"
                    )
                    raise ValueError(msg)
                for field, value in sc_sample.items():
                    if _is_meta_field(field):
                        continue
                    if field in merged:
                        msg = (
                            f"[SidecarFieldConflict] field {field!r} from sidecar "
                            f"{sc_name!r} conflicts in {shard_str!r}"
                        )
                        raise ValueError(msg)
                    merged[field] = value
                    member = f"{sc_key}.{field}"
                    field_sigs[field] = hash_bytes(sc_path, sc_stat.st_mtime_ns, sc_stat.st_size, member)

            merged[_CACHE_META_KEY] = CacheMeta(route_shard=shard_str, field_sigs=field_sigs)
            yield merged


def _tar_sample_meta(sample: Sample, shard_path: str, stat: os.stat_result) -> CacheMeta:
    """Build a :class:`CacheMeta` for a plain tar sample (no sidecars).

    Args:
        sample: Decoded tar sample dict.
        shard_path: Path to the source tar shard.
        stat: Pre-computed ``os.stat`` result for *shard_path*.

    Returns:
        A :class:`CacheMeta` with per-field source signatures.
    """
    key = str(sample.get("__key__", ""))
    field_sigs: dict[str, str] = {}
    for field in sample:
        if _is_meta_field(field):
            continue
        member = f"{key}.{field}"
        field_sigs[field] = hash_bytes(shard_path, stat.st_mtime_ns, stat.st_size, member)
    return CacheMeta(route_shard=shard_path, field_sigs=field_sigs)


def iter_jsonls_with_sigs(
    shard_paths: Iterator[str],
    ref_fields: tuple[RefFieldSpec, ...],
    key_dot_level: int,
    tar_cache_size: int,
) -> Iterator[Sample]:
    """Iterate JSONL shards, resolve ``tar://`` refs, and annotate with ``__cache_meta__``.

    Plain JSON fields receive signatures from the JSONL shard stat and
    canonical value bytes.  Tar-referenced fields receive signatures from
    the referenced tar shard stat and member name.

    Args:
        shard_paths: Iterator of JSONL shard file paths.
        ref_fields: ``(field_name, base_dir)`` pairs identifying fields
            that contain ``tar://`` URIs to resolve.
        key_dot_level: Dot-level parameter forwarded to
            :func:`~mvp_dataset.sources.jsonl.parse_tar_uri`.
        tar_cache_size: LRU cache size for opened tar files.

    Yields:
        Sample dicts with resolved tar references and a
        ``__cache_meta__`` entry.
    """
    ref_field_map = dict(ref_fields)

    with TarManager(cache_size=tar_cache_size) as manager:
        for shard_path in shard_paths:
            shard_str = str(shard_path)
            shard_stat = os.stat(shard_str)

            with open(shard_str, encoding="utf-8") as fh:
                for line_index, line in enumerate(fh):
                    sample = _parse_jsonl_line(shard_str, line_index, line, allow_preannotated=True)
                    field_sigs: dict[str, str] = {}

                    for field, value in sample.items():
                        if _is_meta_field(field):
                            continue
                        if field in ref_field_map and isinstance(value, str):
                            # Tar-referenced field: sig from referenced tar member.
                            try:
                                tar_ref = parse_tar_uri(
                                    value,
                                    base_dir=ref_field_map[field],
                                    key_dot_level=key_dot_level,
                                )
                                ref_stat = os.stat(tar_ref.shard_path)
                                member = f"{tar_ref.key}.{tar_ref.field}"
                                field_sigs[field] = hash_bytes(
                                    tar_ref.shard_path,
                                    ref_stat.st_mtime_ns,
                                    ref_stat.st_size,
                                    member,
                                )
                            except (ValueError, OSError):
                                # Fallback to JSON-field sig on parse/stat failure.
                                canonical = json.dumps(value, ensure_ascii=True, sort_keys=True)
                                field_sigs[field] = hash_bytes(
                                    shard_str,
                                    shard_stat.st_mtime_ns,
                                    shard_stat.st_size,
                                    line_index,
                                    field,
                                    canonical,
                                )
                        else:
                            canonical = json.dumps(value, ensure_ascii=True, sort_keys=True)
                            field_sigs[field] = hash_bytes(
                                shard_str,
                                shard_stat.st_mtime_ns,
                                shard_stat.st_size,
                                line_index,
                                field,
                                canonical,
                            )

                    # Resolve tar refs (matching iter_jsonls behaviour).
                    resolved = dict(sample)
                    for field, base_dir in ref_fields:
                        if field not in sample:
                            continue
                        uri = sample[field]
                        if not isinstance(uri, str):
                            continue
                        try:
                            tar_ref = parse_tar_uri(uri, base_dir=base_dir, key_dot_level=key_dot_level)
                        except ValueError as exc:
                            msg = f"[InvalidRefField] field={field!r} value={uri!r} reason={exc}"
                            raise ValueError(msg) from exc
                        resolved[field] = manager.read(tar_ref)

                    resolved[_CACHE_META_KEY] = CacheMeta(
                        route_shard=shard_str,
                        field_sigs=field_sigs,
                    )
                    yield resolved


# ---------------------------------------------------------------------------
# Cache-aware stage wrappers
# ---------------------------------------------------------------------------


def make_cache_aware_map(
    user_fn: Callable[[object], object],
    fn_fingerprint: str,
) -> Callable[[Iterable[object]], Iterator[object]]:
    """Return a stage that runs *user_fn* and propagates ``__cache_meta__``.

    Signature propagation rules:

    * **Unchanged field** (same object identity): carries forward its old sig.
    * **Modified field** (same name, different value): ``hash(old_sig, fn_fp)``.
    * **New field**: ``hash(aggregate_input_sig, fn_fp, field_name)``.

    Args:
        user_fn: The user-provided map callable.
        fn_fingerprint: Stable hash of *user_fn* used in signature
            derivation.

    Returns:
        A stage callable ``(Iterable) -> Iterator`` that applies
        *user_fn* and updates per-field signatures accordingly.
    """

    def _stage(data: Iterable[object]) -> Iterator[object]:
        for sample in data:
            if not isinstance(sample, dict) or _CACHE_META_KEY not in sample:
                # No meta (e.g. after an unsupported stage dropped it): pass through.
                yield user_fn(sample)
                continue

            meta: CacheMeta = sample[_CACHE_META_KEY]
            clean = {k: v for k, v in sample.items() if k != _CACHE_META_KEY}
            output = user_fn(clean)

            if not isinstance(output, dict):
                yield output  # boundary check happens later
                continue

            # Compute aggregate input sig for new-field signatures.
            agg_input_sig = hash_bytes(*sorted(meta.field_sigs.values()))

            new_sigs: dict[str, str] = {}
            for field, value in output.items():
                if _is_meta_field(field):
                    continue
                if field in clean and value is clean[field]:
                    # Unchanged object identity: carry forward old signature.
                    new_sigs[field] = meta.field_sigs.get(field, "")
                elif field in clean and field in meta.field_sigs:
                    # Same field name, different value.
                    new_sigs[field] = hash_bytes(meta.field_sigs[field], fn_fingerprint)
                else:
                    # New field introduced by this map fn.
                    new_sigs[field] = hash_bytes(agg_input_sig, fn_fingerprint, field)

            output[_CACHE_META_KEY] = CacheMeta(
                route_shard=meta.route_shard,
                field_sigs=new_sigs,
            )
            yield output

    return _stage


def make_cache_aware_assemble(
    assembler_factory: Callable[[], Any],
    fn_fingerprint: str,
    drop_last: bool,
) -> Callable[[Iterable[object]], Iterator[object]]:
    """Return a stage that runs the assembler and attaches ``__cache_meta__``.

    Each output sample receives per-field signatures derived from the
    aggregated input signatures, the assembler fingerprint, the output
    index, and the field name.  The output is routed to the shard of the
    last contributing input sample.

    Args:
        assembler_factory: Zero-argument callable that creates a fresh
            :class:`~mvp_dataset.core.types.Assembler` instance.
        fn_fingerprint: Stable hash of the factory callable.
        drop_last: Whether to discard unfinished tail state on stream end.

    Returns:
        A stage callable ``(Iterable) -> Iterator`` that runs the
        assembler and attaches ``__cache_meta__`` to each output.
    """

    def _stage(data: Iterable[object]) -> Iterator[object]:
        assembler = assembler_factory()
        accumulated_sigs: list[str] = []
        last_route_shard: str = ""
        output_index = 0

        def _tag_output(output: object) -> object:
            nonlocal output_index
            if not isinstance(output, dict):
                return output
            agg_sig = hash_bytes(*accumulated_sigs) if accumulated_sigs else fn_fingerprint
            field_sigs: dict[str, str] = {}
            for field in output:
                if not _is_meta_field(field):
                    field_sigs[field] = hash_bytes(agg_sig, fn_fingerprint, str(output_index), field)
            if "__key__" not in output:
                output["__key__"] = hash_bytes(*accumulated_sigs, str(output_index))[:32]
            output[_CACHE_META_KEY] = CacheMeta(
                route_shard=last_route_shard,
                field_sigs=field_sigs,
            )
            output_index += 1
            return output

        for sample in data:
            if isinstance(sample, dict) and _CACHE_META_KEY in sample:
                meta: CacheMeta = sample[_CACHE_META_KEY]
                accumulated_sigs.extend(meta.field_sigs.values())
                last_route_shard = meta.route_shard
                clean = {k: v for k, v in sample.items() if k != _CACHE_META_KEY}
                for out in assembler.push(clean):
                    yield _tag_output(out)
            else:
                for out in assembler.push(sample):
                    yield _tag_output(out)

        for out in assembler.finish(drop_last=drop_last):
            yield _tag_output(out)

    return _stage


# ---------------------------------------------------------------------------
# Cache write path
# ---------------------------------------------------------------------------


def build_shard_cache(
    shard_path: str,
    samples: list[tuple[Sample, CacheMeta]],
    groups_spec: tuple[tuple[str, ...], ...] | None,
    plan_fingerprint: str,
) -> dict[str, str]:
    """Write group tar files for *shard_path* and return ``{label: tar_path}``.

    Uses an atomic temp→rename pattern so partial writes are never seen.
    A per-shard exclusive lock prevents concurrent builds.

    Args:
        shard_path: Path to the original source shard.
        samples: Pre-cache output samples with their cache metadata.
        groups_spec: Field grouping forwarded to :func:`normalize_groups`.
        plan_fingerprint: Plan fingerprint written into the manifest.
    Returns:
        A ``{group_label: tar_path}`` mapping for the written group tars.
    """
    with _ShardLock(shard_path):
        # Re-check under lock: another process may have built it while we waited.
        existing = read_manifest(shard_path, plan_fingerprint)
        if existing is not None:
            return existing

        cache_dir = _cache_dir(shard_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        shard_stem = Path(shard_path).stem

        if not samples:
            _write_manifest(shard_path, plan_fingerprint, {})
            return {}

        first_sample = samples[0][0]
        groups = normalize_groups(groups_spec, first_sample.keys())

        group_tars: dict[str, str] = {}

        for group_fields in groups:
            label = _group_label(group_fields)

            # Compute group signature from per-field content signatures.
            sig_parts: list[str] = []
            for s_dict, s_meta in samples:
                key = str(s_dict.get("__key__", ""))
                for field in group_fields:
                    sig_parts.append(f"{key}:{field}:{s_meta.field_sigs.get(field, '')}")
            group_sig = hash_bytes(*sig_parts)[:16]

            final_name = f"{shard_stem}-{label}-{group_sig}.tar"
            final_path = cache_dir / final_name

            if not final_path.is_file():
                _write_group_tar(final_path, samples, group_fields)

            group_tars[label] = str(final_path)

        _write_manifest(shard_path, plan_fingerprint, group_tars)

        return group_tars


def _write_group_tar(
    final_path: Path,
    samples: list[tuple[Sample, CacheMeta]],
    group_fields: tuple[str, ...],
) -> None:
    """Write one group tar to a temp file, then atomically rename it.

    Args:
        final_path: Destination path for the group tar.
        samples: Sample/meta pairs to write.
        group_fields: Field names belonging to this group.
    """
    tmp_path = final_path.with_suffix(".tmp.tar")
    try:
        with tarfile.open(str(tmp_path), "w") as tf:
            for s_dict, s_meta in samples:
                key = str(s_dict.get("__key__", ""))

                # Per-field codec map stored in the sample's meta member.
                codec_map: dict[str, str] = {}

                for field in group_fields:
                    if field not in s_dict:
                        continue
                    data, codec_tag = encode_value(s_dict[field])
                    codec_map[field] = codec_tag
                    member_name = f"{key}.{field}"
                    ti = tarfile.TarInfo(name=member_name)
                    ti.size = len(data)
                    tf.addfile(ti, io.BytesIO(data))

                # Write per-sample meta member with codec + signature info.
                meta_payload: dict[str, Any] = {
                    field: {
                        "codec": codec_map[field],
                        "sig": s_meta.field_sigs.get(field, ""),
                    }
                    for field in group_fields
                    if field in codec_map
                }
                meta_bytes = json.dumps(meta_payload, separators=(",", ":")).encode("utf-8")
                meta_name = f"{key}{_CACHE_META_SUFFIX}"
                mti = tarfile.TarInfo(name=meta_name)
                mti.size = len(meta_bytes)
                tf.addfile(mti, io.BytesIO(meta_bytes))

        tmp_path.rename(final_path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Cache read path
# ---------------------------------------------------------------------------


def iter_cache_tar(tar_path: str) -> Iterator[Sample]:
    """Yield samples from a single cache group tar, restoring field values via codecs.

    Args:
        tar_path: Path to a cache group tar file.

    Yields:
        Sample dicts with ``__key__`` and decoded field values.
    """
    with tarfile.open(tar_path, "r") as tf:
        members = {m.name: m for m in tf.getmembers() if m.isfile()}

        # Collect ordered keys via meta members (preserves write order).
        ordered_keys: list[str] = []
        seen_keys: set[str] = set()
        for name in members:
            if name.endswith(_CACHE_META_SUFFIX):
                key = name[: -len(_CACHE_META_SUFFIX)]
                if key not in seen_keys:
                    ordered_keys.append(key)
                    seen_keys.add(key)

        for key in ordered_keys:
            meta_member = members.get(f"{key}{_CACHE_META_SUFFIX}")
            if meta_member is None:
                continue
            extracted_meta = tf.extractfile(meta_member)
            if extracted_meta is None:
                continue
            codec_map: dict[str, dict[str, str]] = json.loads(extracted_meta.read())

            sample: Sample = {"__key__": key}
            for field, info in codec_map.items():
                data_member = members.get(f"{key}.{field}")
                if data_member is None:
                    continue
                extracted = tf.extractfile(data_member)
                if extracted is None:
                    continue
                sample[field] = decode_value(extracted.read(), info["codec"])

            yield sample


def iter_cache_shard(shard_path: str, plan_fingerprint: str) -> Iterator[Sample]:
    """Yield merged samples from all group tars for one source shard.

    Group tars are read in parallel (zipped by position) and their fields
    are merged into a single sample dict per position.

    Args:
        shard_path: Path to the original source shard.
        plan_fingerprint: Plan fingerprint used to locate the manifest.

    Yields:
        Complete sample dicts with fields from all groups merged.

    Raises:
        RuntimeError: If no valid manifest exists for *shard_path*.
    """
    manifest = read_manifest(shard_path, plan_fingerprint)
    if manifest is None:
        msg = f"[CacheReadError] no valid manifest for shard {shard_path!r} (plan={plan_fingerprint!r})"
        raise RuntimeError(msg)

    if not manifest:
        return

    tar_paths = sorted(manifest.values())
    if len(tar_paths) == 1:
        yield from iter_cache_tar(tar_paths[0])
        return

    # Multiple groups: merge per-position (all tars share the same key ordering).
    for parts in itertools.zip_longest(*[iter_cache_tar(tp) for tp in tar_paths]):
        merged: Sample = {}
        key: str = ""
        for part in parts:
            if part is None:
                continue
            key = str(part.pop("__key__", key))
            merged.update(part)
        merged["__key__"] = key
        yield merged


# ---------------------------------------------------------------------------
# Warm-up orchestration
# ---------------------------------------------------------------------------


def warmup_cache(
    assigned_shards: list[str],
    pre_cache_stream: Iterator[object],
    groups_spec: tuple[tuple[str, ...], ...] | None,
    plan_fingerprint: str,
    show_progress: bool,
    unsupported_stage_kinds: list[str],
) -> None:
    """Run the full pre-cache pipeline and materialise cache for each shard.

    Consumes *pre_cache_stream* entirely, groups the output samples by
    their ``route_shard``, then calls :func:`build_shard_cache` for each
    assigned shard under an exclusive lock.

    Args:
        assigned_shards: Shard paths this worker is responsible for.
        pre_cache_stream: Fully composed pre-cache iterator (source +
            cache-aware stages) that yields dict samples with
            ``__cache_meta__`` attached.
        groups_spec: Field grouping forwarded to :func:`build_shard_cache`.
        plan_fingerprint: Plan fingerprint forwarded to the manifest.
        show_progress: Whether to print progress to ``stderr``.
        unsupported_stage_kinds: Stage kinds that appear before the cache
            boundary but are not tracked for invalidation.  A warning is
            emitted when non-empty.
    """
    if unsupported_stage_kinds:
        warnings.warn(
            f"[CacheInvalidationWarning] The following stage type(s) appear before "
            f".cache() but are not tracked for invalidation: "
            f"{unsupported_stage_kinds!r}. "
            f"They will affect the first cache build but changes to their logic "
            f"will NOT trigger a cache rebuild. Stale cache may be reused silently.",
            stacklevel=4,
        )

    # Collect all pre-cache outputs grouped by route_shard.
    shard_samples: dict[str, list[tuple[Sample, CacheMeta]]] = {s: [] for s in assigned_shards}

    for item in pre_cache_stream:
        if not isinstance(item, dict):
            msg = (
                f"[CacheBoundaryError] cache() boundary requires dict samples; "
                f"got {type(item).__name__!r}. "
                f"Stages like batch() must not appear before cache()."
            )
            raise TypeError(msg)
        meta: CacheMeta | None = item.pop(_CACHE_META_KEY, None)
        if meta is None:
            # Unsupported stage dropped the meta – route to first assigned shard.
            route = assigned_shards[0] if assigned_shards else ""
            meta = CacheMeta(route_shard=route, field_sigs={})
        route_shard = meta.route_shard
        if route_shard not in shard_samples:
            # Assemble may produce output routed to a shard not in assigned_shards.
            # Skip samples that belong to other slots' shards.
            continue
        shard_samples[route_shard].append((item, meta))

    # Build cache per shard (locked).
    n_shards = len(assigned_shards)
    for i, shard in enumerate(assigned_shards, 1):
        samples = shard_samples.get(shard, [])
        if show_progress:
            _log_info(f"Caching {i}/{n_shards} shards...")
        build_shard_cache(shard, samples, groups_spec, plan_fingerprint)


def wait_for_cache(
    assigned_shards: list[str],
    plan_fingerprint: str,
    *,
    show_progress: bool = False,
    poll_interval: float = 0.5,
    timeout: float = 3600.0,
) -> None:
    """Block until valid cache manifests exist for all *assigned_shards*.

    Used by non-leader model-parallel ranks (e.g. TP co-members) that skip
    the warm-up phase and wait for the leader to finish building the cache.

    Args:
        assigned_shards: Shard paths whose caches must be ready.
        plan_fingerprint: Expected plan fingerprint to validate manifests.
        show_progress: Whether to print waiting status to ``stderr``.
        poll_interval: Seconds between manifest existence checks.
        timeout: Maximum seconds to wait before raising ``TimeoutError``.

    Raises:
        TimeoutError: If the cache is not ready within *timeout* seconds.
    """
    import time

    pending = set(assigned_shards)
    if show_progress and pending:
        _log_info(f"[cache] waiting for leader to build {len(pending)} shard(s)...")

    deadline = time.monotonic() + timeout
    while pending:
        if time.monotonic() > deadline:
            msg = (
                f"[CacheWaitTimeout] timed out after {timeout}s waiting for "
                f"cache leader to build {len(pending)} shard(s): "
                f"{[Path(s).name for s in sorted(pending)]}"
            )
            raise TimeoutError(msg)
        time.sleep(poll_interval)
        still_pending = set()
        for shard in pending:
            if read_manifest(shard, plan_fingerprint) is None:
                still_pending.add(shard)
        pending = still_pending

    if show_progress:
        _log_info("[cache] leader finished, proceeding with cached data")
