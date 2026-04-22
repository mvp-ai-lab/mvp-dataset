from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def build_records(count: int = 6) -> list[dict[str, object]]:
    return [
        {
            "id": f"sample-{index}",
            "text": f"text-{index}",
            "value": index,
        }
        for index in range(count)
    ]


def write_tar_shards(root: Path, records: list[dict[str, object]], *, num_shards: int) -> list[str]:
    shard_paths: list[str] = []
    for shard_index in range(num_shards):
        shard_path = root / f"shard_{shard_index:06d}.tar"
        shard_paths.append(str(shard_path))
        shard_records = records[shard_index::num_shards]
        with tarfile.open(shard_path, mode="w") as archive:
            for record in shard_records:
                record_id = str(record["id"])
                for field in ("id", "text", "value"):
                    payload = str(record[field]).encode("utf-8")
                    info = tarfile.TarInfo(name=f"{record_id}.{field}")
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
    return shard_paths


def write_jsonl_file(root: Path, records: list[dict[str, object]]) -> str:
    path = root / "samples.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return str(path)


def write_parquet_file(root: Path, records: list[dict[str, object]], *, row_group_size: int | None = None) -> str:
    path = root / "samples.parquet"
    table = pa.table(
        {
            "id": [str(record["id"]) for record in records],
            "text": [str(record["text"]) for record in records],
            "value": [int(record["value"]) for record in records],
        }
    )
    pq.write_table(table, path, row_group_size=row_group_size)
    return str(path)


def write_nested_parquet_file(root: Path, records: list[dict[str, object]], *, row_group_size: int = 2) -> str:
    path = root / "nested_samples.parquet"
    table = pa.table(
        {
            "id": [str(record["id"]) for record in records],
            "meta": [dict(record["meta"]) for record in records],
            "tags": [list(record["tags"]) for record in records],
        }
    )
    pq.write_table(table, path, row_group_size=row_group_size)
    return str(path)


def write_lance_dataset(root: Path, records: list[dict[str, object]]) -> str:
    import lance

    path = root / "samples.lance"
    table = pa.table(
        {
            "id": [str(record["id"]) for record in records],
            "text": [str(record["text"]) for record in records],
            "value": [int(record["value"]) for record in records],
        }
    )
    lance.write_dataset(table, str(path))
    return str(path)


def normalize_sample(sample: dict[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for field in ("id", "text", "value"):
        value = sample[field]
        if isinstance(value, (bytes, bytearray)):
            value = value.decode("utf-8")
        if field == "value":
            value = int(value)
        normalized[field] = value
    return normalized
