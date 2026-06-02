"""Resume protocol primitives for dataset pipelines."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
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

    def __call__(self, upstream: Iterable[object]) -> Iterable[object]: ...

    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state: dict[str, object]) -> None: ...

    def fingerprint(self) -> str: ...


def stable_fingerprint(payload: object) -> str:
    """Return a stable SHA256 fingerprint for a JSON-serializable payload."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
