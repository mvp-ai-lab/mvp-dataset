"""Cache materialization: source annotation, warm-up, manifest, and read path."""

from __future__ import annotations

import fcntl
import io
import itertools
import json
import os
import tarfile
import threading
import time
import warnings
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ..core.types import RefFieldSpec, Sample, SidecarSpec
from ..log import get_logger
from ..sources.jsonl import (
    TarManager,
    _parse_jsonl_line,
    iter_ref_field_uris,
    parse_tar_uri,
    resolve_ref_field_value,
)
from ..sources.parquet import ParquetFragment, iter_parquet
from ..sources.tar import iter_tar
from .codecs import decode_value, encode_value
from .fingerprint import _HashAccumulator, hash_bytes

# Reserved meta member suffix inside cache tars (e.g. "0001.__cache_meta__")
_CACHE_META_SUFFIX: Final[str] = ".__cache_meta__"
# Key used inside sample dicts to carry per-sample cache metadata through pipeline
_CACHE_META_KEY: Final[str] = "__cache_meta__"
_TAR_BLOCK_SIZE: Final[int] = tarfile.BLOCKSIZE
_TAR_EOF: Final[bytes] = tarfile.NUL * _TAR_BLOCK_SIZE * 2
_TAR_FORMAT: Final[int] = tarfile.GNU_FORMAT
_TAR_ERRORS: Final[str] = "surrogateescape"


def _log_info(message: str) -> None:
    get_logger().info(message)


def _format_eta(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _is_meta_field(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _stable_value_for_sig(value: Any) -> Any:
    """Convert a Python value into a JSON-serializable structure for signatures."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return {
            "__mvp_dataset_sig__": "bytes",
            "hex": bytes(value).hex(),
        }
    if isinstance(value, (list, tuple)):
        return [_stable_value_for_sig(item) for item in value]
    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value):
            return {key: _stable_value_for_sig(item) for key, item in value.items()}
        return repr(value)
    return repr(value)


def _canonical_field_value(value: Any) -> str:
    """Return a stable string representation used in source field signatures."""

    return json.dumps(
        _stable_value_for_sig(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


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
    return Path(shard_path).parent / ".cache"


def _manifest_path(shard_path: str) -> Path:
    return _cache_dir(shard_path) / f"{Path(shard_path).name}.manifest.json"


def _lock_path(shard_path: str) -> Path:
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
                        if field in ref_field_map:
                            # Tar-referenced field: sig from referenced tar member(s).
                            try:
                                uris = list(iter_ref_field_uris(value, field=field))
                                if len(uris) == 1:
                                    tar_ref = parse_tar_uri(
                                        uris[0],
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
                                else:
                                    sig_parts: list[object] = []
                                    for item_index, uri in enumerate(uris):
                                        tar_ref = parse_tar_uri(
                                            uri,
                                            base_dir=ref_field_map[field],
                                            key_dot_level=key_dot_level,
                                        )
                                        ref_stat = os.stat(tar_ref.shard_path)
                                        member = f"{tar_ref.key}.{tar_ref.field}"
                                        sig_parts.extend(
                                            [
                                                item_index,
                                                tar_ref.shard_path,
                                                ref_stat.st_mtime_ns,
                                                ref_stat.st_size,
                                                member,
                                            ]
                                        )
                                    field_sigs[field] = hash_bytes(*sig_parts)
                            except (ValueError, OSError):
                                # Fallback to JSON-field sig on parse/stat failure.
                                canonical = _canonical_field_value(value)
                                field_sigs[field] = hash_bytes(
                                    shard_str,
                                    shard_stat.st_mtime_ns,
                                    shard_stat.st_size,
                                    line_index,
                                    field,
                                    canonical,
                                )
                        else:
                            canonical = _canonical_field_value(value)
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
                        resolved[field] = resolve_ref_field_value(
                            sample[field],
                            field=field,
                            base_dir=base_dir,
                            key_dot_level=key_dot_level,
                            manager=manager,
                        )

                    resolved[_CACHE_META_KEY] = CacheMeta(
                        route_shard=shard_str,
                        field_sigs=field_sigs,
                    )
                    yield resolved


def iter_parquets_with_sigs(
    fragments: Iterator[ParquetFragment],
    *,
    columns: Sequence[str] | None,
    batch_size: int,
    use_threads: bool,
) -> Iterator[Sample]:
    """Iterate parquet shards and annotate each row sample with cache signatures."""

    for fragment in fragments:
        shard_str = fragment.path
        shard_stat = os.stat(shard_str)
        for sample in iter_parquet(
            fragment,
            columns=columns,
            batch_size=batch_size,
            use_threads=use_threads,
        ):
            index_in_file = sample.get("__index_in_file__")
            if not isinstance(index_in_file, int):
                msg = f"[InvalidParquetSample] shard={shard_str!r} sample missing int '__index_in_file__'"
                raise ValueError(msg)

            field_sigs: dict[str, str] = {}
            for field, value in sample.items():
                if _is_meta_field(field):
                    continue
                canonical = _canonical_field_value(value)
                field_sigs[field] = hash_bytes(
                    shard_str,
                    shard_stat.st_mtime_ns,
                    shard_stat.st_size,
                    index_in_file,
                    field,
                    canonical,
                )

            sample[_CACHE_META_KEY] = CacheMeta(route_shard=fragment.cache_key, field_sigs=field_sigs)
            yield sample


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


def _add_tar_member(file_obj: io.BufferedWriter, member_name: str, data: bytes) -> None:
    ti = tarfile.TarInfo(name=member_name)
    ti.size = len(data)
    file_obj.write(ti.tobuf(format=_TAR_FORMAT, encoding=tarfile.ENCODING, errors=_TAR_ERRORS))
    file_obj.write(data)
    remainder = len(data) % _TAR_BLOCK_SIZE
    if remainder:
        file_obj.write(tarfile.NUL * (_TAR_BLOCK_SIZE - remainder))


def _group_meta_payload(
    group_fields: tuple[str, ...],
    codec_map: dict[str, str],
    field_sigs: dict[str, str],
) -> bytes:
    payload: dict[str, dict[str, str]] = {}
    for field in group_fields:
        if field not in codec_map:
            continue
        payload[field] = {
            "codec": codec_map[field],
            "sig": field_sigs.get(field, ""),
        }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


@dataclass
class _GroupTarWriter:
    """Streaming writer for one cache group tar."""

    shard_stem: str
    cache_dir: Path
    label: str
    group_fields: tuple[str, ...]
    tmp_path: Path
    file_obj: io.BufferedWriter
    sig_accumulator: _HashAccumulator
    _closed: bool = False

    @classmethod
    def create(
        cls,
        *,
        shard_stem: str,
        cache_dir: Path,
        label: str,
        group_fields: tuple[str, ...],
    ) -> _GroupTarWriter:
        token = hash_bytes(shard_stem, label, os.getpid(), time.time_ns())[:16]
        tmp_path = cache_dir / f"{shard_stem}-{label}-{token}.tmp.tar"
        return cls(
            shard_stem=shard_stem,
            cache_dir=cache_dir,
            label=label,
            group_fields=group_fields,
            tmp_path=tmp_path,
            file_obj=tmp_path.open("wb"),
            sig_accumulator=_HashAccumulator(),
        )

    def append(self, sample: Sample, meta: CacheMeta) -> None:
        """Append one sample to the open temp tar and update its group signature."""

        key = str(sample.get("__key__", ""))
        codec_map: dict[str, str] = {}

        for field in self.group_fields:
            self.sig_accumulator.update(f"{key}:{field}:{meta.field_sigs.get(field, '')}")
            if field not in sample:
                continue
            data, codec_tag = encode_value(sample[field])
            codec_map[field] = codec_tag
            _add_tar_member(self.file_obj, f"{key}.{field}", data)

        meta_bytes = _group_meta_payload(self.group_fields, codec_map, meta.field_sigs)
        _add_tar_member(self.file_obj, f"{key}{_CACHE_META_SUFFIX}", meta_bytes)

    def close(self) -> None:
        """Close the underlying temp tar if it is still open."""

        if not self._closed:
            self.file_obj.write(_TAR_EOF)
            self.file_obj.close()
            self._closed = True

    def discard(self) -> None:
        """Close and remove the temp tar if it still exists."""

        self.close()
        try:
            self.tmp_path.unlink(missing_ok=True)
        except OSError:
            pass

    @property
    def final_path(self) -> Path:
        """Return the published tar path derived from the accumulated signature."""

        group_sig = self.sig_accumulator.hexdigest()[:16]
        return self.cache_dir / f"{self.shard_stem}-{self.label}-{group_sig}.tar"


@dataclass
class _ShardCacheWriter:
    """Streaming cache writer for one source shard."""

    shard_path: str
    groups_spec: tuple[tuple[str, ...], ...] | None
    plan_fingerprint: str
    _initialized: bool = False
    _group_writers: list[_GroupTarWriter] | None = None
    _existing_manifest: dict[str, str] | None = None

    def _ensure_initialized(self, sample_keys: Iterable[str]) -> None:
        if self._initialized:
            return
        self._initialized = True

        existing = read_manifest(self.shard_path, self.plan_fingerprint)
        if existing is not None:
            self._existing_manifest = existing
            self._group_writers = []
            return

        cache_dir = _cache_dir(self.shard_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        shard_stem = Path(self.shard_path).stem
        self._group_writers = []
        for group_fields in normalize_groups(self.groups_spec, sample_keys):
            self._group_writers.append(
                _GroupTarWriter.create(
                    shard_stem=shard_stem,
                    cache_dir=cache_dir,
                    label=_group_label(group_fields),
                    group_fields=group_fields,
                )
            )

    def append(self, sample: Sample, meta: CacheMeta) -> None:
        """Append one routed sample to this shard's cache temp files."""

        self._ensure_initialized(sample.keys())
        if self._existing_manifest is not None:
            return
        assert self._group_writers is not None
        for writer in self._group_writers:
            writer.append(sample, meta)

    def finalize(self) -> dict[str, str]:
        """Publish temp tars and manifest, or reuse an existing manifest."""

        if self._existing_manifest is not None:
            return self._existing_manifest

        group_writers = self._group_writers or []
        for writer in group_writers:
            writer.close()

        with _ShardLock(self.shard_path):
            existing = read_manifest(self.shard_path, self.plan_fingerprint)
            if existing is not None:
                self._existing_manifest = existing
                for writer in group_writers:
                    writer.discard()
                return existing

            if not self._initialized or not group_writers:
                _write_manifest(self.shard_path, self.plan_fingerprint, {})
                self._existing_manifest = {}
                return {}

            group_tars: dict[str, str] = {}
            for writer in group_writers:
                final_path = writer.final_path
                if not final_path.is_file():
                    writer.tmp_path.rename(final_path)
                else:
                    writer.discard()
                group_tars[writer.label] = str(final_path)

            _write_manifest(self.shard_path, self.plan_fingerprint, group_tars)
            self._existing_manifest = group_tars
            return group_tars

    def discard(self) -> None:
        """Best-effort cleanup for any un-published temp files."""

        if self._group_writers is None:
            return
        for writer in self._group_writers:
            writer.discard()


def build_shard_cache(
    shard_path: str,
    samples: list[tuple[Sample, CacheMeta]],
    groups_spec: tuple[tuple[str, ...], ...] | None,
    plan_fingerprint: str,
) -> dict[str, str]:
    """Write group tar files for *shard_path* and return ``{label: tar_path}``.

    Uses the streaming shard writer so naming, manifest publishing, and
    locking stay aligned with the warm-up path.

    Args:
        shard_path: Path to the original source shard.
        samples: Pre-cache output samples with their cache metadata.
        groups_spec: Field grouping forwarded to :func:`normalize_groups`.
        plan_fingerprint: Plan fingerprint written into the manifest.
    Returns:
        A ``{group_label: tar_path}`` mapping for the written group tars.
    """
    writer = _ShardCacheWriter(
        shard_path=shard_path,
        groups_spec=groups_spec,
        plan_fingerprint=plan_fingerprint,
    )
    try:
        for sample, meta in samples:
            writer.append(sample, meta)
        return writer.finalize()
    finally:
        writer.discard()


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


def _warmup_shards(
    worker_shards: list[str],
    stream: Iterator[object],
    groups_spec: tuple[tuple[str, ...], ...] | None,
    plan_fingerprint: str,
    *,
    show_progress: bool = False,
    progress_lock: threading.Lock | None = None,
    progress_counter: list[int] | None = None,
    total_shards: int = 0,
    started_at: float = 0.0,
) -> None:
    """Process one chunk of shards from *stream* and materialise their caches.

    This is the inner worker used by both the serial and parallel paths of
    :func:`warmup_cache`.  Each invocation owns an exclusive set of shards
    and writes independently to their cache directories.

    Args:
        worker_shards: Shards this worker is responsible for.
        stream: Pre-built sample stream for exactly *worker_shards*.
        groups_spec: Forwarded to :class:`_ShardCacheWriter`.
        plan_fingerprint: Forwarded to :class:`_ShardCacheWriter`.
        show_progress: Whether to emit per-shard log messages.
        progress_lock: Shared lock for multi-worker progress reporting.
        progress_counter: Shared ``[started_count]`` list for multi-worker ETA.
        total_shards: Total shards across all workers (for ETA denominator).
        started_at: ``time.monotonic()`` value from the warmup start.
    """
    shard_writers = {
        shard: _ShardCacheWriter(
            shard_path=shard,
            groups_spec=groups_spec,
            plan_fingerprint=plan_fingerprint,
        )
        for shard in worker_shards
    }

    local_started: set[str] = set()

    try:
        for item in stream:
            if not isinstance(item, dict):
                msg = (
                    f"[CacheBoundaryError] cache() boundary requires dict samples; "
                    f"got {type(item).__name__!r}. "
                    f"Stages like batch() must not appear before cache()."
                )
                raise TypeError(msg)
            meta: CacheMeta | None = item.pop(_CACHE_META_KEY, None)
            if meta is None:
                route = worker_shards[0] if worker_shards else ""
                meta = CacheMeta(route_shard=route, field_sigs={})
            route_shard = meta.route_shard
            if route_shard not in shard_writers:
                continue
            if show_progress and route_shard not in local_started:
                local_started.add(route_shard)
                if progress_lock is not None and progress_counter is not None:
                    with progress_lock:
                        progress_counter[0] += 1
                        started = progress_counter[0]
                    elapsed = time.monotonic() - started_at
                    avg = elapsed / started if started else 0.0
                    eta = avg * (total_shards - started)
                    _log_info(f"Caching {started}/{total_shards} shards... ETA {_format_eta(eta)}")
                else:
                    elapsed = time.monotonic() - started_at
                    avg = elapsed / len(local_started) if local_started else 0.0
                    eta = avg * (len(worker_shards) - len(local_started))
                    _log_info(f"Caching {len(local_started)}/{len(worker_shards)} shards... ETA {_format_eta(eta)}")
            shard_writers[route_shard].append(item, meta)

        for shard in worker_shards:
            shard_writers[shard].finalize()
    finally:
        for writer in shard_writers.values():
            writer.discard()


def warmup_cache(
    assigned_shards: list[str],
    pre_cache_stream: Iterator[object] | None,
    groups_spec: tuple[tuple[str, ...], ...] | None,
    plan_fingerprint: str,
    show_progress: bool,
    unsupported_stage_kinds: list[str],
    *,
    stream_factory: Callable[[list[str]], Iterator[object]] | None = None,
    num_workers: int = 1,
) -> None:
    """Run the full pre-cache pipeline and materialise cache for each shard.

    Consumes the sample stream and routes samples into per-shard temp tars.
    At the end, each shard is finalized under an exclusive lock by publishing
    the temp tars and writing the manifest.

    When *stream_factory* and *num_workers* > 1 are provided, shards are
    split across worker threads that each build their own independent stream,
    enabling parallel I/O for large datasets with many shards.

    Args:
        assigned_shards: Shard paths this worker is responsible for.
        pre_cache_stream: Fully composed pre-cache iterator (source +
            cache-aware stages).  Used when *stream_factory* is ``None``.
        groups_spec: Field grouping forwarded to :func:`build_shard_cache`.
        plan_fingerprint: Plan fingerprint forwarded to the manifest.
        show_progress: Whether to print progress to stderr.
        unsupported_stage_kinds: Stage kinds before the cache boundary that
            are not tracked for invalidation.  A warning is emitted when
            non-empty.
        stream_factory: Optional callable ``(shards) -> Iterator`` that
            builds an independent pre-cache stream for a given shard subset.
            Required for parallel warm-up (``num_workers`` > 1).
        num_workers: Number of parallel worker threads.  Only effective when
            *stream_factory* is also provided.  Defaults to 1 (serial).
    """
    if show_progress and assigned_shards:
        _log_info(f"Caching shards: {assigned_shards[0]!r} and {len(assigned_shards) - 1} more...")

    if unsupported_stage_kinds:
        warnings.warn(
            f"[CacheInvalidationWarning] The following stage type(s) appear before "
            f".cache() but are not tracked for invalidation: "
            f"{unsupported_stage_kinds!r}. "
            f"They will affect the first cache build but changes to their logic "
            f"will NOT trigger a cache rebuild. Stale cache may be reused silently.",
            stacklevel=4,
        )

    if stream_factory is not None and num_workers > 1 and len(assigned_shards) > 1:
        # Parallel path: split shards across worker threads, each with its own stream.
        # Interleaved assignment gives better load balancing than sequential chunks.
        actual_workers = min(num_workers, len(assigned_shards))
        chunks = [assigned_shards[i::actual_workers] for i in range(actual_workers)]

        started_at = time.monotonic()
        progress_lock = threading.Lock()
        progress_counter = [0]

        with ThreadPoolExecutor(max_workers=actual_workers) as pool:
            futures = [
                pool.submit(
                    _warmup_shards,
                    chunk,
                    stream_factory(chunk),
                    groups_spec,
                    plan_fingerprint,
                    show_progress=show_progress,
                    progress_lock=progress_lock,
                    progress_counter=progress_counter,
                    total_shards=len(assigned_shards),
                    started_at=started_at,
                )
                for chunk in chunks
            ]
            for fut in as_completed(futures):
                fut.result()  # re-raise any worker exception
        return

    # Serial path (stream_factory=None uses pre_cache_stream; factory+1-worker also serial).
    stream: Iterator[object]
    if stream_factory is not None:
        stream = stream_factory(assigned_shards)
    elif pre_cache_stream is not None:
        stream = pre_cache_stream
    else:
        raise ValueError("warmup_cache requires either pre_cache_stream or stream_factory")

    _warmup_shards(
        assigned_shards,
        stream,
        groups_spec,
        plan_fingerprint,
        show_progress=show_progress,
        started_at=time.monotonic(),
    )


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
