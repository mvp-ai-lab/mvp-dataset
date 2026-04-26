import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from mvp_dataset.core.context import RuntimeContext
from mvp_dataset.core.dataset import Dataset
from mvp_dataset.core.types import ShardInput
from mvp_dataset.utils.url import normalize_paths

from .utils import (
    LanceSourceSpec,
    assign_items,
    attach_lance_ref_columns,
    iter_lance,
    list_lance_sources,
)


def _resolve_config_uri(uri: object, *, base_dir: Path) -> str:
    if not isinstance(uri, str) or not uri:
        msg = f"[InvalidLanceConfig] expected a non-empty URI string, got {uri!r}"
        raise ValueError(msg)
    if "://" in uri or Path(uri).is_absolute():
        return uri
    return str(base_dir / uri)


def _load_lance_source_config(config_path: Path) -> tuple[list[str], dict[str, dict[str, str]] | None]:
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

    resolved_ref_columns: dict[str, dict[str, str]] = {}
    for column, ref_config in raw_ref_columns.items():
        if not isinstance(column, str) or not column:
            msg = f"[InvalidLanceConfig] ref column names must be non-empty strings in {config_path}"
            raise ValueError(msg)
        if not isinstance(ref_config, dict):
            msg = f"[InvalidLanceConfig] ref config for {column!r} must be a mapping in {config_path}"
            raise ValueError(msg)
        resolved_ref_config: dict[str, str] = {}
        for key, value in ref_config.items():
            if key == "uri":
                resolved_ref_config[key] = _resolve_config_uri(value, base_dir=base_dir)
            elif isinstance(value, str):
                resolved_ref_config[key] = value
            else:
                msg = f"[InvalidLanceConfig] ref config value {column!r}.{key} must be a string in {config_path}"
                raise ValueError(msg)
        resolved_ref_columns[column] = resolved_ref_config

    return resolved_shards, resolved_ref_columns


def _resolve_lance_source_config(
    shards: ShardInput | Sequence[ShardInput],
    ref_columns: dict[str, dict[str, str]] | None,
) -> tuple[list[str], dict[str, dict[str, str]] | None]:
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


@dataclass(frozen=True, slots=True)
class _LanceSourceIter:
    source: LanceSourceSpec
    columns: Sequence[str] | None = None
    batch_size: int = 65536
    load_in_memory: bool = False

    handles_sharding = True

    def __call__(self, source_stream):
        return iter_lance(
            self.source,
            source_stream,
            columns=self.columns,
            batch_size=self.batch_size,
            load_in_memory=self.load_in_memory,
        )


@dataclass(frozen=True, slots=True)
class LanceDataset(Dataset):
    _global_shuffle: bool = False
    _load_in_memory: bool = False

    @classmethod
    def from_source(
        cls,
        shards: ShardInput | Sequence[ShardInput],
        context: RuntimeContext | None = None,
        resample: bool = False,
        columns: Sequence[str] | None = None,
        batch_size: int = 1024,
        global_shuffle: bool = False,
        load_in_memory: bool = False,
        ref_columns: dict[str, dict[str, str]] | None = None,
    ):
        """Build a dataset from local Lance dataset paths.

        Args:
            shards: One or more Lance dataset URIs or directory paths. A single
                    JSON file may also be provided; it must contain ``main_uri``
                    (or ``shards``/``uri``) and optional ``ref_columns``.
            context: Optional execution context. If omitted, inferred from runtime.
            resample: Whether to loop shards indefinitely across rounds.
            columns: Optional list of column names to read.
            batch_size: Number of rows per Arrow batch during iteration.
            global_shuffle: Whether to shuffle rows globally across all datasets.
            load_in_memory: Whether to load entire datasets into memory (recommended
                            if you provide a metadata lance dataset
                            and link other data via reference columns).
            ref_columns: Optional mapping of source column names to explicit Lance
                         reference configs containing uri, key_column, and value_column.
        Returns:
            A lance dataset.
        """
        runtime_context = RuntimeContext.from_runtime() if context is None else context
        normalized_shards, resolved_ref_columns = _resolve_lance_source_config(shards, ref_columns)
        sources = list_lance_sources(
            normalized_shards,
        )
        source = attach_lance_ref_columns(sources[0], resolved_ref_columns)
        sources = [source]

        return cls(
            context=runtime_context,
            _source=sources,
            _resample=resample,
            _source_kind="lance",
            _stages=(),
            _iter_source_stream=_LanceSourceIter(
                source=sources[0],
                columns=columns,
                batch_size=batch_size,
                load_in_memory=load_in_memory,
            ),
            _global_shuffle=global_shuffle,
            _load_in_memory=load_in_memory,
        )

    def _build_source_stream(self, *, context: RuntimeContext) -> Iterable[object]:
        assert len(self._source) == 1, "Multiple Lance sources are not supported in this implementation"
        source_shard_stream = assign_items(
            self._source,
            context=context,
            resample=self._resample,
            shuffle=self._global_shuffle,
        )
        return self._iter_source_stream(source_shard_stream)

    def shuffle(self, *args, **kwargs) -> Dataset:
        raise NotImplementedError("LanceDataset.shuffle() is not supported.")
