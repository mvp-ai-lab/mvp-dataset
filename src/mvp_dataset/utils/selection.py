"""Helpers for validating selected tar field-group keys."""

from collections.abc import Sequence


def normalize_selected_keys(keys: Sequence[str]) -> tuple[str, ...]:
    """Validate and deduplicate selected tar field-group keys."""

    normalized: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if not isinstance(key, str):
            msg = f"[InvalidSelectKey] expected str key, got={type(key).__name__}"
            raise TypeError(msg)
        candidate = key.strip()
        if not candidate:
            msg = "[InvalidSelectKey] select keys must be non-empty strings"
            raise ValueError(msg)
        if "/" in candidate or "\\" in candidate:
            msg = f"[InvalidSelectKey] key={candidate!r} must not contain path separators"
            raise ValueError(msg)
        if candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)
    if not normalized:
        msg = "[InvalidSelectKey] select requires at least one key"
        raise ValueError(msg)
    return tuple(normalized)
