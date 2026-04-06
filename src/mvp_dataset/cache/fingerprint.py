"""Stable fingerprinting utilities for callables and composite values."""

from __future__ import annotations

import functools
import hashlib
import struct
import types
from collections.abc import Callable
from typing import Any


def _update_hash(h: hashlib._Hash, *parts: Any) -> None:
    """Update *h* using the same typed framing scheme as :func:`hash_bytes`."""

    for part in parts:
        if isinstance(part, (bytes, bytearray)):
            data = bytes(part)
            h.update(b"B")
            h.update(struct.pack(">Q", len(data)))
            h.update(data)
        elif isinstance(part, bool):
            h.update(b"b")
            h.update(b"\x01" if part else b"\x00")
        elif isinstance(part, int):
            h.update(b"I")
            h.update(struct.pack(">q", part))
        elif isinstance(part, float):
            h.update(b"F")
            h.update(struct.pack(">d", part))
        elif isinstance(part, str):
            encoded = part.encode("utf-8")
            h.update(b"S")
            h.update(struct.pack(">Q", len(encoded)))
            h.update(encoded)
        else:
            raise TypeError(f"unsupported hash part type {type(part)!r}")


class _HashAccumulator:
    """Incremental SHA-256 accumulator compatible with :func:`hash_bytes`."""

    def __init__(self) -> None:
        self._hash = hashlib.sha256()

    def update(self, *parts: Any) -> None:
        _update_hash(self._hash, *parts)

    def hexdigest(self) -> str:
        return self._hash.hexdigest()


def hash_bytes(*parts: Any) -> str:
    """Return a SHA-256 hex digest over a sequence of typed parts.

    Each part is type-tagged and length-prefixed so that adjacent parts
    cannot collide.

    Args:
        *parts: Values to hash.  Supported types: ``bytes``, ``str``,
            ``int``, ``float``, ``bool``.

    Returns:
        A 64-character lowercase hex digest string.

    Raises:
        TypeError: If any part has an unsupported type.
    """
    h = hashlib.sha256()
    _update_hash(h, *parts)
    return h.hexdigest()


def callable_fingerprint(fn: Callable[..., Any]) -> str:
    """Return a stable fingerprint for a Python callable.

    Covers: bytecode, filename, name, defaults, kwdefaults, and closure cell
    contents.

    Args:
        fn: The callable to fingerprint.  May be a function, lambda,
            :class:`functools.partial`, or any object with a ``__code__``
            attribute.

    Returns:
        A 64-character hex digest uniquely identifying the callable's
        logic and captured state.

    Raises:
        ValueError: If a closure cell contains a value that cannot be
            stably encoded (e.g. an arbitrary object instance).
    """
    parts: list[Any] = []
    _collect_callable(fn, parts, seen=set())
    return hash_bytes(*parts)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_callable(fn: Any, parts: list[Any], seen: set[int]) -> None:
    fn_id = id(fn)
    if fn_id in seen:
        parts.append(b"<recursive>")
        return
    seen.add(fn_id)

    if isinstance(fn, functools.partial):
        parts.append(b"<partial>")
        for arg in fn.args:
            _collect_value(arg, parts)
        for k in sorted(fn.keywords):
            parts.append(k)
            _collect_value(fn.keywords[k], parts)
        _collect_callable(fn.func, parts, seen)
        return

    code = getattr(fn, "__code__", None)
    if code is None:
        # Built-in or extension callable: use qualified name as proxy.
        parts.append(f"{type(fn).__module__}.{type(fn).__qualname__}")
        return

    raw_bytecode = code.co_code if isinstance(code.co_code, bytes) else bytes(code.co_code)
    parts.append(raw_bytecode)
    parts.append(code.co_filename)
    parts.append(code.co_name)

    for d in getattr(fn, "__defaults__", None) or ():
        _collect_value(d, parts)
    for k in sorted(getattr(fn, "__kwdefaults__", None) or {}):
        parts.append(k)
        _collect_value((fn.__kwdefaults__ or {})[k], parts)

    for cell in getattr(fn, "__closure__", None) or ():
        try:
            val = cell.cell_contents
        except ValueError:
            continue  # empty cell
        _collect_value(val, parts)


def _collect_value(val: Any, parts: list[Any]) -> None:
    """Encode a scalar-ish value into *parts* for hashing.

    Raises :class:`ValueError` for values that cannot be stably encoded.
    """
    if val is None:
        parts.append(b"\x00None")
    elif isinstance(val, bool):
        parts.append(b"\x01True" if val else b"\x01False")
    elif isinstance(val, int):
        parts.append(val)
    elif isinstance(val, float):
        parts.append(val)
    elif isinstance(val, str):
        parts.append(val)
    elif isinstance(val, (bytes, bytearray)):
        parts.append(bytes(val))
    elif isinstance(val, (list, tuple)):
        parts.append(b"<seq>")
        for item in val:
            _collect_value(item, parts)
        parts.append(b"</seq>")
    elif isinstance(val, dict):
        parts.append(b"<dict>")
        for k in sorted(str(k) for k in val):
            parts.append(k)
            _collect_value(val[k], parts)  # type: ignore[index]
        parts.append(b"</dict>")
    elif isinstance(val, types.ModuleType):
        # Represent a module by its fully-qualified name.
        parts.append(f"<module:{val.__name__}>")
    elif callable(val):
        _collect_callable(val, parts, seen=set())
    elif hasattr(val, "__fingerprint__"):
        fp = val.__fingerprint__()
        if not isinstance(fp, str):
            raise TypeError(f"__fingerprint__ must return str, got {type(fp).__name__!r}")
        parts.append(f"<fingerprint:{fp}>")
    else:
        msg = (
            f"[CacheFingerprintError] cannot stably encode closure value of type "
            f"{type(val).__name__!r}; supported: None, bool, int, float, str, bytes, "
            f"list, tuple, dict, module, callable, or objects with __fingerprint__()"
        )
        raise ValueError(msg)
