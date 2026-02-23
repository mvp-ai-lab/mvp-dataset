"""Join extension points for combining fields from secondary sources."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Protocol

from ..core.types import Sample


class JoinProvider(Protocol):
    """Protocol for key-based field providers.

    Implementations can load additional fields for an existing sample key from
    sidecar storage (for example, separate depth tar shards).
    """

    def get_fields(self, key: str) -> Mapping[str, object] | None:
        """Return extra fields for ``key`` or ``None`` if key is unavailable."""


def apply_join(
    sample: Sample,
    provider: JoinProvider,
    *,
    on_missing: Literal["error", "ignore"] = "error",
) -> Sample:
    """Return a new sample merged with provider fields.

    This function is intentionally minimal in step 1. It defines stable behavior
    for future join integrations without wiring join into the main dataset pipeline yet.

    Missing-key semantics:
    - ``on_missing=\"error\"`` raises ``KeyError`` when provider has no data.
    - ``on_missing=\"ignore\"`` returns a shallow copy of the original sample.

    Field collisions are rejected to avoid silent overwrites.
    """

    key = sample.get("__key__")
    if not isinstance(key, str):
        raise KeyError("sample must include a string '__key__' field")

    extra_fields = provider.get_fields(key)
    if extra_fields is None:
        if on_missing == "ignore":
            return dict(sample)
        raise KeyError(f"no join fields found for key {key!r}")

    merged = dict(sample)
    for field, value in extra_fields.items():
        if field in merged:
            raise ValueError(f"join field collision on {field!r} for key {key!r}")
        merged[field] = value
    return merged
