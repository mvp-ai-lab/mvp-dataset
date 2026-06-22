"""Lance ref-column config parsing."""

from __future__ import annotations

from ..types import (
    LanceRefIndexConfig,
    LanceRefIndexConfigInput,
    LanceRefSpec,
    LanceSource,
)


def parse_lance_ref_columns(ref_columns: object) -> tuple[LanceRefSpec, ...]:
    """Parse the public ref_columns mapping into immutable ref specs.

    Args:
        ref_columns: Mapping from output column name to a config dict with ``uri``, ``key_column``, and
            ``value_column``. ``uri`` may be one reference Lance URI or a non-empty list of URIs.

    Returns:
        Normalized Lance reference specifications."""

    if ref_columns is None:
        return ()
    if not isinstance(ref_columns, dict):
        msg = "[InvalidLanceRefColumns] ref_columns must be a mapping of column name to reference config"
        raise TypeError(msg)

    specs: list[LanceRefSpec] = []
    for column, config in ref_columns.items():
        if not isinstance(column, str) or not column:
            msg = "[InvalidLanceRefColumn] ref column names must be non-empty strings"
            raise ValueError(msg)
        if not isinstance(config, dict):
            msg = f"[InvalidLanceRefColumn] config for {column!r} must be a mapping"
            raise TypeError(msg)
        missing = [key for key in ("uri", "key_column", "value_column") if key not in config]
        if missing:
            msg = f"[InvalidLanceRefColumn] config for {column!r} is missing: {', '.join(missing)}"
            raise ValueError(msg)
        raw_uri = config["uri"]
        if isinstance(raw_uri, (list, tuple)):
            if not raw_uri:
                msg = f"[InvalidLanceRefColumn] config for {column!r} has an empty uri list"
                raise ValueError(msg)
            if not all(isinstance(uri, str) and uri for uri in raw_uri):
                msg = f"[InvalidLanceRefColumn] config for {column!r}.uri must contain non-empty strings"
                raise ValueError(msg)
            uri: str | tuple[str, ...] = tuple(raw_uri)
        elif isinstance(raw_uri, str) and raw_uri:
            uri = raw_uri
        else:
            msg = f"[InvalidLanceRefColumn] config for {column!r}.uri must be a string or list of strings"
            raise ValueError(msg)

        specs.append(
            LanceRefSpec(
                column=column,
                uri=uri,
                key_column=str(config["key_column"]),
                value_column=str(config["value_column"]),
            )
        )
    return tuple(specs)


def attach_lance_ref_columns(source: LanceSource, ref_columns: object) -> LanceSource:
    """Attach Lance reference resolution stages to a dataset.

    Args:
        source: Lance source specification.
        ref_columns: Mapping from output column name to a config dict with ``uri``, ``key_column``, and
            ``value_column``.

    Returns:
        A Lance source specification with reference metadata attached."""
    return LanceSource(datasets=source.datasets, ref_columns=parse_lance_ref_columns(ref_columns))


def resolve_ref_index_config(index: LanceRefIndexConfigInput) -> LanceRefIndexConfig:
    """Return validated reference index configuration.

    ``index`` may contain ``scope``, ``build_strategy``, and ``bucket_count``.
    See ``LanceDataset.resolve_ref`` for accepted values and defaults.
    """
    if index is None:
        return LanceRefIndexConfig()
    if not isinstance(index, dict):
        msg = "[InvalidLanceRefIndexConfig] index must be a mapping"
        raise TypeError(msg)
    allowed_keys = {"scope", "build_strategy", "bucket_count"}
    unknown_keys = sorted(set(index) - allowed_keys)
    if unknown_keys:
        msg = f"[InvalidLanceRefIndexConfig] unknown config key(s): {', '.join(unknown_keys)}"
        raise ValueError(msg)

    scope = index.get("scope")
    if scope is not None and scope not in ("shared", "node_local", "process"):
        msg = f"[InvalidLanceRefIndexScope] expected shared, node_local, or process, got {scope!r}"
        raise ValueError(msg)
    build_strategy = index.get("build_strategy")
    if build_strategy is not None and build_strategy not in ("auto", "in_memory", "bucketed"):
        msg = f"[InvalidLanceRefIndexBuildStrategy] expected auto, in_memory, or bucketed, got {build_strategy!r}"
        raise ValueError(msg)
    bucket_count = index.get("bucket_count")
    if bucket_count is not None:
        bucket_count = int(bucket_count)
        if bucket_count <= 0:
            msg = f"[InvalidLanceRefIndexBucketCount] bucket_count must be > 0, got {bucket_count}"
            raise ValueError(msg)

    return LanceRefIndexConfig(
        scope=scope,
        build_strategy=build_strategy,
        bucket_count=bucket_count,
    )
