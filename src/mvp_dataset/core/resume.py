"""Small helpers for resumable dataset iteration."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from .context import RuntimeContext
from .types import SourceKind, StageSpec

RESUME_STATE_VERSION = 1
RESUME_TOKEN_KEY = "__resume_token__"
SUPPORTED_RESUME_STAGES = {"map", "select", "batch"}


def runtime_fingerprint(context: RuntimeContext) -> dict[str, object]:
    """Return the runtime values that must match for safe resume."""

    mesh = context.mesh
    mesh_payload = None if mesh is None else {"dp_rank": mesh.dp_rank, "dp_size": mesh.dp_size}
    return {
        "rank": context.rank,
        "world_size": context.world_size,
        "local_rank": context.local_rank,
        "local_world_size": context.local_world_size,
        "node_rank": context.node_rank,
        "num_nodes": context.num_nodes,
        "worker_id": context.worker_id,
        "num_workers": context.num_workers,
        "epoch": context.epoch,
        "seed": context.seed,
        "mesh": mesh_payload,
    }


def _path_payload(path: str) -> dict[str, object] | None:
    try:
        stat = Path(path).stat()
    except OSError:
        return None
    return {
        "path": path,
        "is_dir": Path(path).is_dir(),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
    }


def _normalize(value: object) -> object:
    if dataclasses.is_dataclass(value):
        return _normalize(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _normalize(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
            if key not in {"handle", "index_handle"}
        }
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        payload = _path_payload(value) if isinstance(value, str) else None
        return {"path_stat": payload} if payload is not None else value
    return repr(value)


def source_fingerprint(source: object) -> str:
    payload = _normalize(source)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stage_payload(spec: StageSpec) -> dict[str, object]:
    if spec.kind not in SUPPORTED_RESUME_STAGES:
        msg = (
            f"[UnsupportedResumeStage] stage={spec.kind!r} is not resumable in this implementation; "
            "supported stages are map, select, and batch"
        )
        raise ValueError(msg)
    return {"kind": spec.kind, "apply": _normalize(spec.apply)}


def pipeline_fingerprint(stages: Iterable[StageSpec]) -> str:
    payload = [_stage_payload(spec) for spec in stages]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_resume_state(
    *,
    source_kind: SourceKind,
    source: object,
    stages: Iterable[StageSpec],
    context: RuntimeContext,
    source_cursor: object,
    step: int,
) -> dict[str, object]:
    return {
        "version": RESUME_STATE_VERSION,
        "step": step,
        "source_kind": source_kind,
        "runtime_fingerprint": runtime_fingerprint(context),
        "source_fingerprint": source_fingerprint(source),
        "pipeline_fingerprint": pipeline_fingerprint(stages),
        "source_cursor": source_cursor,
    }


def validate_resume_state(
    state: Mapping[str, object],
    *,
    source_kind: SourceKind,
    source: object,
    stages: Iterable[StageSpec],
    context: RuntimeContext,
) -> None:
    if state.get("version") != RESUME_STATE_VERSION:
        msg = f"[InvalidResumeState] unsupported version={state.get('version')!r}"
        raise ValueError(msg)
    expected = make_resume_state(
        source_kind=source_kind,
        source=source,
        stages=stages,
        context=context,
        source_cursor=state.get("source_cursor"),
        step=int(state.get("step", 0)),
    )
    for key in ("source_kind", "runtime_fingerprint", "source_fingerprint", "pipeline_fingerprint"):
        if state.get(key) != expected[key]:
            msg = f"[InvalidResumeState] {key} does not match current dataset"
            raise ValueError(msg)


def token_matches(token: object, cursor: object) -> bool:
    return isinstance(token, Mapping) and isinstance(cursor, Mapping) and dict(token) == dict(cursor)


def token_kind(cursor: object) -> str | None:
    if isinstance(cursor, Mapping):
        kind = cursor.get("kind")
        return kind if isinstance(kind, str) else None
    return None
