#!/usr/bin/env python3
"""Write demo Lance datasets with media stored behind reference columns."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa

TAR_URI_PREFIX = "tar://"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT = REPO_ROOT / "examples" / "demo_data" / "samples.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "examples" / "demo_data" / "samples.lance"
DEFAULT_MEDIA_OUTPUT = REPO_ROOT / "examples" / "demo_data" / "media.lance"


class TarResolver:
    def __init__(self) -> None:
        self._handles: dict[Path, tarfile.TarFile] = {}
        self._members: dict[Path, dict[str, tarfile.TarInfo]] = {}

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        self._members.clear()

    def read(self, uri: str, *, base_dir: Path) -> bytes:
        if not uri.startswith(TAR_URI_PREFIX):
            raise ValueError(f"expected tar URI, got {uri!r}")

        body = uri[len(TAR_URI_PREFIX) :]
        if "#" not in body:
            raise ValueError(f"invalid tar URI without '#': {uri!r}")

        shard_part, member_name = body.split("#", 1)
        shard_path = Path(shard_part)
        if not shard_path.is_absolute():
            shard_path = base_dir / shard_path
        shard_path = shard_path.resolve()

        if shard_path not in self._handles:
            handle = tarfile.open(shard_path, mode="r:*")
            self._handles[shard_path] = handle
            self._members[shard_path] = {member.name: member for member in handle.getmembers()}

        member = self._members[shard_path].get(member_name)
        if member is None:
            raise KeyError(f"member {member_name!r} not found in {shard_path}")

        extracted = self._handles[shard_path].extractfile(member)
        if extracted is None:
            raise ValueError(f"failed to extract {member_name!r} from {shard_path}")
        return extracted.read()


def collect_media_ref(
    value: Any,
    *,
    base_dir: Path,
    resolver: TarResolver,
    media_by_id: dict[str, dict[str, Any]],
) -> Any:
    if isinstance(value, str) and value.startswith(TAR_URI_PREFIX):
        member_name = value.rsplit("#", 1)[-1]
        media_id = Path(member_name).stem
        if media_id not in media_by_id:
            media_by_id[media_id] = {"media_id": media_id, "payload": resolver.read(value, base_dir=base_dir)}
        return media_id
    if isinstance(value, list):
        return [
            collect_media_ref(item, base_dir=base_dir, resolver=resolver, media_by_id=media_by_id) for item in value
        ]
    return value


def load_rows(jsonl_path: Path, *, base_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sample_rows: list[dict[str, Any]] = []
    media_by_id: dict[str, dict[str, Any]] = {}
    resolver = TarResolver()
    try:
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"line {line_number} is not a JSON object")
                sample_rows.append(
                    {
                        key: collect_media_ref(
                            value,
                            base_dir=base_dir,
                            resolver=resolver,
                            media_by_id=media_by_id,
                        )
                        for key, value in row.items()
                    }
                )
    finally:
        resolver.close()
    return sample_rows, list(media_by_id.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write demo Lance sample/media reference datasets.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input JSONL file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output sample Lance dataset path.")
    parser.add_argument(
        "--media-output",
        type=Path,
        default=DEFAULT_MEDIA_OUTPUT,
        help="Output media Lance dataset path.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=REPO_ROOT,
        help="Base directory used to resolve relative tar:// shard paths.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    media_output_path = args.media_output.resolve()
    base_dir = args.base_dir.resolve()

    sample_rows, media_rows = load_rows(input_path, base_dir=base_dir)
    if not sample_rows:
        raise ValueError(f"no rows found in {input_path}")
    if not media_rows:
        raise ValueError(f"no media references found in {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    media_output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_table = pa.Table.from_pylist(sample_rows)
    media_table = pa.Table.from_pylist(media_rows)
    lance.write_dataset(sample_table, str(output_path), mode="overwrite")
    lance.write_dataset(media_table, str(media_output_path), mode="overwrite")
    print(f"wrote {len(sample_rows)} sample rows to {output_path}")
    print(f"wrote {len(media_rows)} media rows to {media_output_path}")


if __name__ == "__main__":
    main()
