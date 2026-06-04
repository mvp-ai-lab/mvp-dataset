"""Resume protocol primitives for dataset pipelines."""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import textwrap
from collections.abc import Callable
from typing import Protocol, runtime_checkable

RESUME_STATE_VERSION = 1


class UnsupportedResume(RuntimeError):
    """Raised when a source or stage cannot describe resumable state."""


class ResumeStateError(ValueError):
    """Raised when a resume state is malformed or incompatible."""


@runtime_checkable
class StatefulIterator(Protocol):
    """Iterator that can persist and restore its future output state."""

    def __iter__(self) -> StatefulIterator: ...

    def __next__(self) -> object: ...

    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state: dict[str, object]) -> None: ...

    def fingerprint(self) -> str: ...


@runtime_checkable
class StatefulStage(Protocol):
    """Pipeline stage that can persist and restore its internal state."""

    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state: dict[str, object]) -> None: ...

    def fingerprint(self) -> str: ...


def callable_fingerprint(fn: Callable[..., object] | None) -> dict[str, object] | None:
    """Return a stable payload describing callable identity, config, and code."""

    if fn is None:
        return None
    if isinstance(fn, functools.partial):
        return {
            "kind": "partial",
            "func": callable_fingerprint(fn.func),
            "args": repr(fn.args),
            "keywords": repr(sorted((fn.keywords or {}).items())),
        }

    custom = getattr(fn, "fingerprint", None)
    if callable(custom):
        try:
            value = custom()
        except TypeError:
            value = None
        if value is not None:
            return {
                "kind": "custom",
                "callable": _callable_name(fn),
                "type": _type_name(fn),
                "fingerprint": str(value),
            }

    target = _source_target(fn)
    return {
        "kind": "callable",
        "callable": _callable_name(fn),
        "type": _type_name(fn),
        "source_hash": _source_hash(target),
        "code_hash": _code_hash(target),
        "defaults": _defaults(fn),
        "closure": _closure(fn),
        "config": None if inspect.isfunction(fn) or inspect.ismethod(fn) or inspect.isclass(fn) else repr(fn),
    }


def _callable_name(fn: Callable[..., object]) -> str:
    module = getattr(fn, "__module__", None) or type(fn).__module__
    qualname = getattr(fn, "__qualname__", None)
    if qualname is None:
        qualname = type(fn).__qualname__
    return f"{module}.{qualname}"


def _type_name(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _source_target(fn: Callable[..., object]) -> object:
    if isinstance(fn, functools.partial):
        return fn.func
    if inspect.isclass(fn) or inspect.isfunction(fn) or inspect.ismethod(fn):
        return fn
    return type(fn).__call__


def _source_hash(target: object) -> str | None:
    try:
        source = inspect.getsource(target)
    except (OSError, TypeError):
        return None
    source = textwrap.dedent(source).strip()
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _code_hash(target: object) -> str | None:
    code = getattr(target, "__code__", None)
    if code is None and inspect.ismethod(target):
        code = getattr(target.__func__, "__code__", None)
    if code is None:
        return None
    payload = {
        "argcount": code.co_argcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "posonlyargcount": code.co_posonlyargcount,
        "code": code.co_code.hex(),
        "consts": repr(code.co_consts),
        "names": code.co_names,
        "varnames": code.co_varnames,
    }
    return stable_fingerprint(payload)


def _defaults(fn: Callable[..., object]) -> dict[str, object] | None:
    defaults = getattr(fn, "__defaults__", None)
    kwdefaults = getattr(fn, "__kwdefaults__", None)
    if defaults is None and kwdefaults is None:
        return None
    return {
        "defaults": repr(defaults),
        "kwdefaults": repr(sorted((kwdefaults or {}).items())),
    }


def _closure(fn: Callable[..., object]) -> list[str] | None:
    closure = getattr(fn, "__closure__", None)
    if not closure:
        return None
    values: list[str] = []
    for cell in closure:
        try:
            values.append(repr(cell.cell_contents))
        except ValueError:
            values.append("<empty>")
    return values


def stable_fingerprint(payload: object) -> str:
    """Return a stable SHA256 fingerprint for a JSON-serializable payload."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
