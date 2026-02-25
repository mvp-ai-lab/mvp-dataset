"""JSONL source utilities and tar reference parsing."""

from __future__ import annotations

import tarfile
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from ..core.types import GroupedSample, PathLikeStr, RefFieldSpec, Sample

_TAR_URI_PREFIX = "tar://"


@dataclass(frozen=True, slots=True)
class TarRef:
    """Parsed tar reference for one field in a JSONL sample."""

    shard_path: str
    key: str
    field: str
    raw_uri: str


def parse_tar_uri(
    uri: str,
    *,
    base_dir: PathLikeStr,
    key_dot_level: int = 1,
) -> TarRef:
    """Parse URI with grammar ``tar://<shard>#<key>.<field>``.

    ``key_dot_level`` follows the same rule as tar record parsing:
    - key: first ``key_dot_level`` dot-separated segments
    - field: remaining segments
    """

    if not uri.startswith(_TAR_URI_PREFIX):
        msg = f"[InvalidTarUri] uri={uri!r} expected_prefix={_TAR_URI_PREFIX!r}"
        raise ValueError(msg)

    body = uri[len(_TAR_URI_PREFIX) :]
    if "#" not in body:
        msg = f"[InvalidTarUri] uri={uri!r} missing '#'"
        raise ValueError(msg)

    shard_part, target = body.split("#", 1)
    if not shard_part:
        msg = f"[InvalidTarUri] uri={uri!r} empty shard path"
        raise ValueError(msg)
    if key_dot_level <= 0:
        msg = f"[InvalidTarUri] uri={uri!r} invalid key_dot_level={key_dot_level}"
        raise ValueError(msg)
    if "." not in target:
        msg = f"[InvalidTarUri] uri={uri!r} expected '<key>.<field>' target"
        raise ValueError(msg)

    target_parts = target.split(".")
    if len(target_parts) <= key_dot_level:
        msg = f"[InvalidTarUri] uri={uri!r} target does not match key_dot_level={key_dot_level}"
        raise ValueError(msg)

    key = ".".join(target_parts[:key_dot_level])
    field = ".".join(target_parts[key_dot_level:])
    if not key or not field:
        msg = f"[InvalidTarUri] uri={uri!r} invalid target key/field"
        raise ValueError(msg)

    shard_path = Path(shard_part)
    if not shard_path.is_absolute():
        shard_path = Path(base_dir) / shard_path
    return TarRef(
        shard_path=str(shard_path),
        key=key,
        field=field,
        raw_uri=uri,
    )


class TarManager:
    """Cache-aware reader for tar-referenced field payloads.

    Opens tar archives in seekable mode (``r:*``) so that individual member
    lookup is possible without reading the entire shard into memory.  Member
    metadata is indexed once per opened shard so repeated lookups do not call
    ``getmember`` repeatedly. Up to
    ``cache_size`` shard handles are kept open simultaneously; the
    least-recently-used handle is closed and evicted when the cache is full.

    Use as a context manager to ensure all handles are closed on exit::

        with TarManager(cache_size=8) as mgr:
            data = mgr.read(tar_ref)
    """

    def __init__(self, cache_size: int = 8) -> None:
        if cache_size < 1:
            msg = f"[InvalidCacheSize] cache_size must be >= 1, got={cache_size}"
            raise ValueError(msg)
        # OrderedDict used as an LRU map: most-recently-used entry is at the end.
        self._cache: OrderedDict[str, tuple[tarfile.TarFile, dict[str, tarfile.TarInfo]]] = (
            OrderedDict()
        )
        self._cache_size = cache_size

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> TarManager:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close all cached tar file handles."""
        for tf, _member_index in self._cache.values():
            try:
                tf.close()
            except Exception:  # noqa: BLE001
                pass
        self._cache.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_tar_entry(
        self,
        shard_path: str,
    ) -> tuple[tarfile.TarFile, dict[str, tarfile.TarInfo]]:
        """Return cached ``(TarFile, member_index)`` for *shard_path*.

        If the shard is already cached its entry is promoted to the
        most-recently-used position.  When the cache is full the
        least-recently-used entry is closed and removed first.
        """
        if shard_path in self._cache:
            self._cache.move_to_end(shard_path)
            return self._cache[shard_path]

        if len(self._cache) >= self._cache_size:
            _evicted_path, (evicted_tf, _evicted_member_index) = self._cache.popitem(last=False)
            try:
                evicted_tf.close()
            except Exception:  # noqa: BLE001
                pass

        # Open in seekable mode and build an O(1) member-name index once.
        tf = tarfile.open(shard_path, mode="r:*")
        members = tf.getmembers()
        # Preserve tarfile.getmember duplicate-name semantics: later entries
        # overwrite earlier ones, equivalent to searching from the end.
        member_index = {member.name: member for member in members}
        entry = (tf, member_index)
        self._cache[shard_path] = entry
        return entry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self, tar_ref: TarRef) -> bytes:
        """Read and return the raw bytes payload described by *tar_ref*.

        The tar member name is reconstructed as ``<key>.<field>``, which
        matches the naming convention produced by :func:`iter_tar`.

        Raises:
            KeyError: when the member is not found inside the shard.
            tarfile.ExtractError: when the member cannot be extracted.
        """
        member_name = f"{tar_ref.key}.{tar_ref.field}"
        tf, member_index = self._get_tar_entry(tar_ref.shard_path)
        member = member_index.get(member_name)
        if member is None:
            msg = (
                f"[TarMemberNotFound] member={member_name!r} "
                f"shard={tar_ref.shard_path!r} uri={tar_ref.raw_uri!r}"
            )
            raise KeyError(msg)
        extracted = tf.extractfile(member)
        if extracted is None:
            msg = (
                f"[TarExtractError] member={member_name!r} "
                f"shard={tar_ref.shard_path!r} uri={tar_ref.raw_uri!r}"
            )
            raise tarfile.ExtractError(msg)
        return extracted.read()


def iter_jsonls(
    stream: Iterator[Sample | GroupedSample],
    ref_fields: tuple[RefFieldSpec, ...],
    key_dot_level: int = 1,
    tar_cache_size: int = 8,
) -> Iterator[Sample]:
    """Resolve tar data references in JSONL samples on-the-fly.

    Accepts an upstream stream of either flat :class:`Sample` dicts or grouped
    ``list[Sample]`` items (produced by a ``group_by`` stage).  Grouped items
    are flattened: each sample in the group is resolved and yielded
    individually, so the output is always a flat stream of :class:`Sample`.

    For every resolved field, a ``tar://`` URI is replaced in-place with the
    raw :class:`bytes` payload read from the referenced tar member.  A shared
    :class:`TarManager` is kept alive for the lifetime of the iteration so that
    frequently accessed shards are not repeatedly opened and closed.

    Args:
        stream: Upstream iterator of flat samples or grouped sample lists.
        ref_fields: Sequence of ``(field_name, base_dir)`` pairs identifying
            fields that may contain ``tar://`` URIs.
        key_dot_level: Number of dot-separated segments that form the key
            inside a tar shard (forwarded to :func:`parse_tar_uri`).
        tar_cache_size: Maximum number of tar shard handles to keep open
            simultaneously (forwarded to :class:`TarManager`).
    """

    def _resolve_one(sample: Sample, manager: TarManager) -> Sample:
        """Resolve all ref fields in a single sample dict."""
        resolved = dict(sample)
        for field, base_dir in ref_fields:
            if field not in sample:
                continue
            value = sample[field]
            if not isinstance(value, str):
                msg = (
                    f"[InvalidRefField] expected string URI in "
                    f"field={field!r} got={type(value).__name__}"
                )
                raise ValueError(msg)
            try:
                tar_ref = parse_tar_uri(value, base_dir=base_dir, key_dot_level=key_dot_level)
            except ValueError as exc:
                msg = (
                    f"[InvalidRefField] failed to parse tar URI in "
                    f"field={field!r} value={value!r} reason={exc}"
                )
                raise ValueError(msg) from exc
            resolved[field] = manager.read(tar_ref)
        return resolved

    with TarManager(cache_size=tar_cache_size) as manager:
        for item in stream:
            if isinstance(item, list):
                # Grouped path: flatten the group and yield each resolved
                # sample individually.
                for sample in item:
                    if not isinstance(sample, dict):
                        msg = "[InvalidGroupedSample] expected sample dict in grouped JSONL stream"
                        raise ValueError(msg)
                    yield _resolve_one(sample, manager)
            else:
                # Flat path: resolve and yield a single sample.
                yield _resolve_one(item, manager)
