"""Path normalization helpers for shard specifications."""

import glob
import os
import re
from collections.abc import Sequence
from pathlib import Path

from ..core.types import ShardInput

_BRACE_RANGE = re.compile(r"^(.*)\{(\d+)\.\.(\d+)\}(.*)$")


def _expand_brace_range(spec: str) -> list[str]:
    """Expand one numeric brace range like ``a-{000..003}.tar``.

    If no numeric brace range exists, the original spec is returned.
    """

    match = _BRACE_RANGE.match(spec)
    if match is None:
        return [spec]

    prefix, start_raw, end_raw, suffix = match.groups()
    start = int(start_raw)
    end = int(end_raw)
    width = max(len(start_raw), len(end_raw))
    step = 1 if end >= start else -1

    expanded: list[str] = []
    for index in range(start, end + step, step):
        expanded.append(f"{prefix}{index:0{width}d}{suffix}")
    return expanded


def _expand_single_spec(spec: str) -> list[str]:
    """Expand one shard spec with optional brace range and glob tokens."""

    expanded_specs = _expand_brace_range(spec)
    expanded_paths: list[str] = []

    for expanded in expanded_specs:
        if any(char in expanded for char in "*?["):
            matches = sorted(glob.glob(expanded))
            if matches:
                expanded_paths.extend(matches)
            else:
                expanded_paths.append(expanded)
        else:
            expanded_paths.append(expanded)
    return expanded_paths


def normalize_paths(inputs: ShardInput | Sequence[ShardInput]) -> list[str]:
    """Normalize shard input specs into a flat list of file-system paths.

    Supported syntax:
    - ``a::b``: concatenate multiple specs
    - ``*.tar``: glob expansion
    - ``shard_{000..127}.tar``: numeric brace range expansion
    """

    if isinstance(inputs, (str, os.PathLike)):
        specs = [str(inputs)]
    else:
        specs = [os.fspath(path) for path in inputs]

    normalized: list[str] = []
    for spec in specs:
        for chunk in spec.split("::"):
            chunk = chunk.strip()
            if not chunk:
                continue
            normalized.extend(_expand_single_spec(chunk))

    if not normalized:
        raise ValueError("no shards were resolved from the provided shard specification")

    return [str(Path(path)) for path in normalized]
