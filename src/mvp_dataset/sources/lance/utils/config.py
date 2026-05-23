"""Lance source config-file parsing."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from mvp_dataset.core.types import ShardInput
from mvp_dataset.utils.url import normalize_paths


def _resolve_config_uri(uri: object, *, base_dir: Path) -> str:
    if not isinstance(uri, str) or not uri:
        msg = f"[InvalidLanceConfig] expected a non-empty URI string, got {uri!r}"
        raise ValueError(msg)
    if "://" in uri or Path(uri).is_absolute():
        return uri
    return str(base_dir / uri)


def _resolve_config_uri_or_list(uri: object, *, base_dir: Path) -> str | list[str]:
    if isinstance(uri, list):
        if not uri:
            msg = "[InvalidLanceConfig] expected a non-empty URI list"
            raise ValueError(msg)
        return [_resolve_config_uri(item, base_dir=base_dir) for item in uri]
    return _resolve_config_uri(uri, base_dir=base_dir)


def _load_lance_source_config(config_path: Path) -> tuple[list[str], dict[str, dict[str, object]] | None]:
    if not config_path.exists():
        msg = f"[MissingLanceConfig] Lance source config was not found: {config_path}"
        raise FileNotFoundError(msg)
    if not config_path.is_file():
        msg = f"[InvalidLanceConfig] Lance source config must be a JSON file: {config_path}"
        raise ValueError(msg)

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"[InvalidLanceConfig] failed to parse JSON config {config_path}: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(config, dict):
        msg = f"[InvalidLanceConfig] expected a JSON object in {config_path}"
        raise ValueError(msg)

    base_dir = config_path.resolve().parent
    raw_shards = config.get("main_uri", config.get("shards", config.get("uri")))
    if raw_shards is None:
        msg = f"[InvalidLanceConfig] {config_path} must contain 'main_uri', 'shards', or 'uri'"
        raise ValueError(msg)
    if isinstance(raw_shards, str):
        resolved_shards = [_resolve_config_uri(raw_shards, base_dir=base_dir)]
    elif isinstance(raw_shards, list):
        resolved_shards = [_resolve_config_uri(uri, base_dir=base_dir) for uri in raw_shards]
    else:
        msg = f"[InvalidLanceConfig] main_uri/shards/uri must be a string or list of strings in {config_path}"
        raise ValueError(msg)

    raw_ref_columns = config.get("ref_columns")
    if raw_ref_columns is None:
        return resolved_shards, None
    if not isinstance(raw_ref_columns, dict):
        msg = f"[InvalidLanceConfig] ref_columns must be a mapping in {config_path}"
        raise ValueError(msg)

    resolved_ref_columns: dict[str, dict[str, object]] = {}
    for column, ref_config in raw_ref_columns.items():
        if not isinstance(column, str) or not column:
            msg = f"[InvalidLanceConfig] ref column names must be non-empty strings in {config_path}"
            raise ValueError(msg)
        if not isinstance(ref_config, dict):
            msg = f"[InvalidLanceConfig] ref config for {column!r} must be a mapping in {config_path}"
            raise ValueError(msg)
        resolved_ref_config: dict[str, object] = {}
        for key, value in ref_config.items():
            if key == "uri":
                resolved_ref_config[key] = _resolve_config_uri_or_list(value, base_dir=base_dir)
            elif isinstance(value, str):
                resolved_ref_config[key] = value
            else:
                msg = f"[InvalidLanceConfig] ref config value {column!r}.{key} must be a string in {config_path}"
                raise ValueError(msg)
        resolved_ref_columns[column] = resolved_ref_config

    return resolved_shards, resolved_ref_columns


def resolve_lance_source_config(
    shards: ShardInput | Sequence[ShardInput],
    ref_columns: dict[str, dict[str, object]] | None,
) -> tuple[list[str], dict[str, dict[str, object]] | None]:
    normalized_shards = normalize_paths(shards)
    json_config_paths = [Path(path) for path in normalized_shards if Path(path).suffix.lower() == ".json"]
    if not json_config_paths:
        return normalized_shards, ref_columns
    if len(normalized_shards) != 1:
        msg = "[InvalidLanceConfig] a Lance JSON source config must be provided as the only shard"
        raise ValueError(msg)

    resolved_shards, config_ref_columns = _load_lance_source_config(json_config_paths[0])
    if ref_columns is not None:
        return resolved_shards, ref_columns
    return resolved_shards, config_ref_columns
