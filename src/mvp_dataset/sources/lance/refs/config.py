"""Lance ref-column config parsing."""

from __future__ import annotations

from ..types import LanceRefSpec, LanceSourceSpec


def parse_lance_ref_columns(ref_columns: object) -> tuple[LanceRefSpec, ...]:
    """Parse the public ref_columns mapping into immutable ref specs.

    Args:
        ref_columns: Lance reference column configuration.

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


def attach_lance_ref_columns(source: LanceSourceSpec, ref_columns: object) -> LanceSourceSpec:
    """Attach Lance reference resolution stages to a dataset.

    Args:
        source: Lance source specification.
        ref_columns: Lance reference column configuration.

    Returns:
        A Lance source specification with reference metadata attached."""
    return LanceSourceSpec(datasets=source.datasets, ref_columns=parse_lance_ref_columns(ref_columns))
