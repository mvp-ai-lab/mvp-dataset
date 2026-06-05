"""Tar reference handling for JSONL sources."""

from __future__ import annotations

import tarfile
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from ...core.types import PathLikeStr

_TAR_URI_PREFIX = "tar://"


@dataclass(frozen=True, slots=True)
class TarUriRef:
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
) -> TarUriRef:
    """Parse URI with grammar ``tar://<shard>#<key>.<field>`` or ``<shard>#<key>.<field>``.

    Args:
        uri: The uri argument.
        base_dir: Base directory used to resolve relative tar paths.
        key_dot_level: Number of dot-separated suffix components removed from tar member keys.

    Returns:
        A parsed tar URI reference."""

    body = uri[len(_TAR_URI_PREFIX) :] if uri.startswith(_TAR_URI_PREFIX) else uri
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
    return TarUriRef(
        shard_path=str(shard_path),
        key=key,
        field=field,
        raw_uri=uri,
    )


def iter_ref_field_uris(value: object, *, field: str) -> Iterator[str]:
    """Yield one-or-many tar reference URIs from a JSONL field value.

    Args:
        value: Reference value to inspect or resolve.
        field: Reference field name.

    Returns:
        An iterator over tar URI strings found in the value."""

    if isinstance(value, str):
        yield value
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            if not isinstance(item, str):
                msg = (
                    f"[InvalidRefField] expected string URI in field={field!r} list item "
                    f"{index}, got={type(item).__name__}"
                )
                raise ValueError(msg)
            yield item
        return
    msg = f"[InvalidRefField] expected string URI or list of string URIs in field={field!r} got={type(value).__name__}"
    raise ValueError(msg)


def resolve_ref_field_value(
    value: object,
    *,
    field: str,
    base_dir: PathLikeStr,
    key_dot_level: int,
    manager: TarManager,
) -> object:
    """Resolve one JSONL reference field to bytes or a list of bytes.

    Args:
        value: Reference value to inspect or resolve.
        field: Reference field name.
        base_dir: Base directory used to resolve relative tar paths.
        key_dot_level: Number of dot-separated suffix components removed from tar member keys.
        manager: Tar manager used to read referenced entries.

    Returns:
        The value with tar URI references replaced by bytes."""

    if isinstance(value, str):
        try:
            tar_ref = parse_tar_uri(value, base_dir=base_dir, key_dot_level=key_dot_level)
        except ValueError as exc:
            msg = f"[InvalidRefField] failed to parse tar URI in field={field!r} value={value!r} reason={exc}"
            raise ValueError(msg) from exc
        return manager.read(tar_ref)

    uris = list(iter_ref_field_uris(value, field=field))
    resolved: list[bytes] = []
    for uri in uris:
        try:
            tar_ref = parse_tar_uri(uri, base_dir=base_dir, key_dot_level=key_dot_level)
        except ValueError as exc:
            msg = f"[InvalidRefField] failed to parse tar URI in field={field!r} value={uri!r} reason={exc}"
            raise ValueError(msg) from exc
        resolved.append(manager.read(tar_ref))
    return resolved


class TarManager:
    """LRU reader for tar-referenced field payloads."""

    def __init__(self, max_open_files: int = 8) -> None:
        """Initialize the object."""
        if max_open_files < 1:
            msg = f"[InvalidTarManagerSize] max_open_files must be >= 1, got={max_open_files}"
            raise ValueError(msg)
        self._entries: OrderedDict[str, tuple[tarfile.TarFile, dict[str, tarfile.TarInfo]]] = OrderedDict()
        self._max_open_files = max_open_files

    def __enter__(self) -> TarManager:
        """Enter the context manager and return this object."""
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        """Close resources when leaving the context manager."""
        self.close()

    def close(self) -> None:
        """Close owned resources.

        Returns:
            None."""
        for tf, _member_index in self._entries.values():
            try:
                tf.close()
            except Exception:  # noqa: BLE001
                pass
        self._entries.clear()

    def _get_tar_entry(
        self,
        shard_path: str,
    ) -> tuple[tarfile.TarFile, dict[str, tarfile.TarInfo]]:
        """Return one tar member payload by name."""
        if shard_path in self._entries:
            self._entries.move_to_end(shard_path)
            return self._entries[shard_path]

        if len(self._entries) >= self._max_open_files:
            _evicted_path, (evicted_tf, _evicted_member_index) = self._entries.popitem(last=False)
            try:
                evicted_tf.close()
            except Exception:  # noqa: BLE001
                pass

        tf = tarfile.open(shard_path, mode="r:*")
        members = tf.getmembers()
        member_index = {member.name: member for member in members}
        entry = (tf, member_index)
        self._entries[shard_path] = entry
        return entry

    def read(self, tar_ref: TarUriRef) -> bytes:
        """Read and return the raw bytes payload described by *tar_ref*.

        Args:
            tar_ref: Parsed tar URI reference to read.

        Returns:
            Raw bytes for the referenced tar entry."""

        member_name = f"{tar_ref.key}.{tar_ref.field}"
        tf, member_index = self._get_tar_entry(tar_ref.shard_path)
        member = member_index.get(member_name)
        if member is None:
            msg = f"[TarMemberNotFound] member={member_name!r} shard={tar_ref.shard_path!r} uri={tar_ref.raw_uri!r}"
            raise KeyError(msg)
        extracted = tf.extractfile(member)
        if extracted is None:
            msg = f"[TarExtractError] member={member_name!r} shard={tar_ref.shard_path!r} uri={tar_ref.raw_uri!r}"
            raise tarfile.ExtractError(msg)
        return extracted.read()
