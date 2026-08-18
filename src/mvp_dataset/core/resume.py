"""Resume protocols: identity for configuration, state for live iterators."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import inspect
import json
import textwrap
from collections.abc import Mapping, Sequence, Set
from typing import Protocol, runtime_checkable

from mvp_dataset.log import get_logger

RESUME_STATE_VERSION = 3
_MAX_IDENTITY_DEPTH = 16


class UnsupportedResume(RuntimeError):
    """Raised when a source or stage cannot describe resumable state."""


class ResumeStateError(ValueError):
    """Raised when a resume state is malformed or incompatible."""


def warn_if_iterator_replaced(previous: object | None) -> None:
    """Warn when iter() replaces an iterator that has not reached StopIteration."""
    if previous is None or getattr(previous, "_exhausted", True):
        return
    get_logger().warning(
        "iter() called while a previous iterator is still active; "
        "state_dict() on this handle will follow the new iterator."
    )


@runtime_checkable
class Stateful(Protocol):
    """Live object that can persist and restore its in-flight state."""

    def state_dict(self) -> dict[str, object]:
        """Return the live iterator/source state."""
        ...

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore live state in place."""
        ...


def digest(payload: object) -> str:
    """Return a SHA256 digest of a JSON-serializable payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def checkpoint(identity_payload: object, state: object | None) -> dict[str, object]:
    """Build a resume envelope."""
    return {"version": RESUME_STATE_VERSION, "identity": identity_payload, "state": state}


def checkpoint_from_active_iter(identity_payload: object, active_iter: object | None) -> dict[str, object]:
    """Checkpoint the handle's current iterator, or a fresh start if none is active."""
    if active_iter is not None and not getattr(active_iter, "_exhausted", True):
        live_state = getattr(active_iter, "live_state", None)
        if callable(live_state):
            return checkpoint(identity_payload, live_state())
    return checkpoint(identity_payload, None)


def parse_checkpoint(blob: object) -> tuple[object, object | None]:
    """Validate a resume envelope and return ``(identity, state)``."""
    if not isinstance(blob, dict):
        raise ResumeStateError("[InvalidResumeState] checkpoint must be a dict")
    version = blob.get("version")
    if version != RESUME_STATE_VERSION:
        raise ResumeStateError(f"[InvalidResumeStateVersion] expected={RESUME_STATE_VERSION} got={version!r}")
    if "identity" not in blob:
        raise ResumeStateError("[InvalidResumeState] checkpoint is missing identity")
    if "state" not in blob:
        raise ResumeStateError("[InvalidResumeState] checkpoint is missing state")
    return blob["identity"], blob["state"]


def check_identity(expected: object, actual: object) -> None:
    """Raise if two identity trees differ, reporting path and both values."""
    diff = _identity_diff(expected, actual, "identity")
    if diff is not None:
        path, left, right = diff
        raise ResumeStateError(f"[ResumeIdentityMismatch] path={path} expected={left!r} actual={right!r}")


def identity(value: object) -> object:
    """Return a JSON-safe, process-stable identity for resume checks.

    Raises:
        ResumeStateError: If a stable identity cannot be produced.
    """
    return _identity(value, depth=0, seen=frozenset())


def _identity(value: object, *, depth: int, seen: frozenset[int]) -> object:
    if depth > _MAX_IDENTITY_DEPTH:
        raise ResumeStateError(f"[UnstableResumeIdentity] identity graph exceeded max depth for {_type_name(value)}")
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            return {"kind": "float", "value": repr(value)}
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"kind": "bytes", "sha256": hashlib.sha256(bytes(value)).hexdigest()}
    if isinstance(value, range):
        return {"kind": "range", "start": value.start, "stop": value.stop, "step": value.step}

    value_id = id(value)
    if value_id in seen:
        raise ResumeStateError(f"[UnstableResumeIdentity] cyclic identity for {_type_name(value)}")
    next_seen = seen | {value_id}

    custom = _object_identity(value)
    if custom is not None:
        return {"type": _type_name(value), "id": custom}
    if isinstance(value, functools.partial):
        keywords = value.keywords or {}
        return {
            "kind": "partial",
            "func": _identity(value.func, depth=depth + 1, seen=next_seen),
            "args": [_identity(arg, depth=depth + 1, seen=next_seen) for arg in value.args],
            "keywords": [
                {
                    "key": str(key),
                    "value": _identity(item, depth=depth + 1, seen=next_seen),
                }
                for key, item in sorted(keywords.items(), key=lambda item: str(item[0]))
            ],
        }
    if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isclass(value):
        return _callable_identity(value, depth=depth, seen=next_seen)
    if isinstance(value, Mapping):
        return {
            "kind": "mapping",
            "type": _type_name(value),
            "items": _mapping_items(value, depth=depth, seen=next_seen),
        }
    if isinstance(value, Set):
        items = [_identity(item, depth=depth + 1, seen=next_seen) for item in value]
        return {
            "kind": "set",
            "type": _type_name(value),
            "items": sorted(items, key=_sort_key),
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "type": _type_name(value),
            "items": [_identity(item, depth=depth + 1, seen=next_seen) for item in value],
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return {
            "kind": "sequence",
            "type": _type_name(value),
            "items": [_identity(item, depth=depth + 1, seen=next_seen) for item in value],
        }
    if not isinstance(value, type) and dataclasses.is_dataclass(value):
        return {
            "kind": "dataclass",
            "type": _type_name(value),
            "fields": {
                field.name: _identity(getattr(value, field.name), depth=depth + 1, seen=next_seen)
                for field in dataclasses.fields(value)
            },
        }
    config = _scalar_instance_config(value)
    if callable(value) and not inspect.isclass(value):
        payload = _callable_identity(value, depth=depth, seen=next_seen)
        if config:
            payload["config"] = config
        return payload
    if config:
        return {"kind": "object", "type": _type_name(value), "config": config}
    raise ResumeStateError(f"[UnstableResumeIdentity] {_type_name(value)} has no identity(); implement identity().")


def _object_identity(value: object) -> object | None:
    method = getattr(value, "identity", None)
    if not callable(method):
        return None
    if inspect.isfunction(method) and _required_positional_count(method) > 0:
        return None
    return method()


def _callable_identity(fn: object, *, depth: int, seen: frozenset[int]) -> dict[str, object]:
    if inspect.ismethod(fn):
        return {
            "kind": "method",
            "func": _callable_identity(fn.__func__, depth=depth, seen=seen),
            "self": _identity(fn.__self__, depth=depth + 1, seen=seen),
        }
    target = fn if inspect.isfunction(fn) or inspect.isclass(fn) else type(fn).__call__
    payload: dict[str, object] = {
        "kind": "callable",
        "callable": _callable_name(fn),
        "type": _type_name(fn),
        "source_hash": _source_hash(target),
    }
    if inspect.isfunction(fn):
        defaults = getattr(fn, "__defaults__", None)
        if defaults:
            payload["defaults"] = [_identity(item, depth=depth + 1, seen=seen) for item in defaults]
        kwdefaults = getattr(fn, "__kwdefaults__", None)
        if kwdefaults:
            payload["kwdefaults"] = [
                {"key": str(key), "value": _identity(item, depth=depth + 1, seen=seen)}
                for key, item in sorted(kwdefaults.items(), key=lambda item: str(item[0]))
            ]
        closure = getattr(fn, "__closure__", None)
        if closure:
            cells: list[object] = []
            for cell in closure:
                try:
                    contents = cell.cell_contents
                except ValueError:
                    cells.append({"kind": "empty"})
                else:
                    cells.append(_identity(contents, depth=depth + 1, seen=seen))
            payload["closure"] = cells
    return payload


def _mapping_items(value: Mapping[object, object], *, depth: int, seen: frozenset[int]) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for key, item in value.items():
        if key is not None and not isinstance(key, (bool, int, float, str)):
            raise ResumeStateError(
                f"[UnstableResumeIdentity] mapping key {_type_name(key)} is not a JSON scalar; "
                "use a scalar key or implement identity()."
            )
        items.append(
            {
                "key": key,
                "key_type": _type_name(key) if key is not None else "none",
                "value": _identity(item, depth=depth + 1, seen=seen),
            }
        )
    items.sort(key=lambda item: (str(item["key_type"]), repr(item["key"])))
    return items


def _scalar_instance_config(value: object) -> dict[str, object]:
    attributes = getattr(value, "__dict__", None)
    if not isinstance(attributes, dict):
        slots = getattr(type(value), "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        attributes = {
            name: getattr(value, name)
            for name in slots
            if name not in {"__dict__", "__weakref__"} and hasattr(value, name)
        }
    config: dict[str, object] = {}
    for key, item in sorted(attributes.items(), key=lambda pair: str(pair[0])):
        if item is None or isinstance(item, (bool, int, float, str)):
            config[str(key)] = item
            continue
        raise ResumeStateError(
            f"[UnstableResumeIdentity] {_type_name(value)}.{key} is not a scalar; implement identity()."
        )
    return config


def _required_positional_count(fn: object) -> int:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return 0
    count = 0
    for parameter in signature.parameters.values():
        if parameter.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}:
            if parameter.default is inspect.Parameter.empty:
                count += 1
    return count


def _callable_name(fn: object) -> str:
    module = getattr(fn, "__module__", None) or type(fn).__module__
    qualname = getattr(fn, "__qualname__", None)
    if qualname is None:
        qualname = type(fn).__qualname__
    return f"{module}.{qualname}"


def _type_name(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _source_hash(target: object) -> str | None:
    try:
        source = inspect.getsource(target)
    except (OSError, TypeError):
        return None
    return hashlib.sha256(textwrap.dedent(source).strip().encode("utf-8")).hexdigest()


def _sort_key(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _identity_diff(left: object, right: object, path: str) -> tuple[str, object, object] | None:
    if isinstance(left, dict) and isinstance(right, dict):
        keys = set(left) | set(right)
        for key in sorted(keys, key=str):
            child = f"{path}.{key}"
            if key not in left:
                return child, None, right[key]
            if key not in right:
                return child, left[key], None
            found = _identity_diff(left[key], right[key], child)
            if found is not None:
                return found
        return None
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return path, left, right
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            found = _identity_diff(left_item, right_item, f"{path}[{index}]")
            if found is not None:
                return found
        return None
    return None if left == right else (path, left, right)
