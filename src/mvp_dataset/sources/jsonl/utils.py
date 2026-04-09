"""JSONL source utilities, spill sharding, and tar reference parsing."""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from ...core.types import PathLikeStr, RefFieldSpec, Sample

_TAR_URI_PREFIX = "tar://"


@dataclass(frozen=True, slots=True)
class TarRef:
    """Parsed tar reference for one field in a JSONL sample."""

    shard_path: str
    key: str
    field: str
    raw_uri: str


def parse_tar_uri(
    uri: str,
    *,
    base_dir: PathLikeStr,
    key_dot_level: int = 1,
) -> TarRef:
    """Parse URI with grammar ``tar://<shard>#<key>.<field>`` or ``<shard>#<key>.<field>``."""

    body = uri[len(_TAR_URI_PREFIX) :] if uri.startswith(_TAR_URI_PREFIX) else uri
    if "#" not in body:
        msg = f"[InvalidTarUri] uri={uri!r} missing '#'"
        raise ValueError(msg)

    shard_part, target = body.split("#", 1)
    if not shard_part:
        msg = f"[InvalidTarUri] uri={uri!r} empty shard path"
        raise ValueError(msg)
    if key_dot_level <= 0:
        msg = f"[InvalidTarUri] uri={uri!r} invalid key_dot_level={key_dot_level}"
        raise ValueError(msg)
    if "." not in target:
        msg = f"[InvalidTarUri] uri={uri!r} expected '<key>.<field>' target"
        raise ValueError(msg)

    target_parts = target.split(".")
    if len(target_parts) <= key_dot_level:
        msg = f"[InvalidTarUri] uri={uri!r} target does not match key_dot_level={key_dot_level}"
        raise ValueError(msg)

    key = ".".join(target_parts[:key_dot_level])
    field = ".".join(target_parts[key_dot_level:])
    if not key or not field:
        msg = f"[InvalidTarUri] uri={uri!r} invalid target key/field"
        raise ValueError(msg)

    shard_path = Path(shard_part)
    if not shard_path.is_absolute():
        shard_path = Path(base_dir) / shard_path
    return TarRef(
        shard_path=str(shard_path),
        key=key,
        field=field,
        raw_uri=uri,
    )


def iter_ref_field_uris(value: object, *, field: str) -> Iterator[str]:
    """Yield one-or-many tar reference URIs from a JSONL field value."""

    if isinstance(value, str):
        yield value
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            if not isinstance(item, str):
                msg = (
                    f"[InvalidRefField] expected string URI in field={field!r} list item "
                    f"{index}, got={type(item).__name__}"
                )
                raise ValueError(msg)
            yield item
        return
    msg = f"[InvalidRefField] expected string URI or list of string URIs in field={field!r} got={type(value).__name__}"
    raise ValueError(msg)


def resolve_ref_field_value(
    value: object,
    *,
    field: str,
    base_dir: PathLikeStr,
    key_dot_level: int,
    manager: TarManager,
) -> object:
    """Resolve one JSONL reference field to bytes or a list of bytes."""

    if isinstance(value, str):
        try:
            tar_ref = parse_tar_uri(value, base_dir=base_dir, key_dot_level=key_dot_level)
        except ValueError as exc:
            msg = f"[InvalidRefField] failed to parse tar URI in field={field!r} value={value!r} reason={exc}"
            raise ValueError(msg) from exc
        return manager.read(tar_ref)

    uris = list(iter_ref_field_uris(value, field=field))
    resolved: list[bytes] = []
    for uri in uris:
        try:
            tar_ref = parse_tar_uri(uri, base_dir=base_dir, key_dot_level=key_dot_level)
        except ValueError as exc:
            msg = f"[InvalidRefField] failed to parse tar URI in field={field!r} value={uri!r} reason={exc}"
            raise ValueError(msg) from exc
        resolved.append(manager.read(tar_ref))
    return resolved


class TarManager:
    """Cache-aware reader for tar-referenced field payloads."""

    def __init__(self, cache_size: int = 8) -> None:
        if cache_size < 1:
            msg = f"[InvalidCacheSize] cache_size must be >= 1, got={cache_size}"
            raise ValueError(msg)
        self._cache: OrderedDict[str, tuple[tarfile.TarFile, dict[str, tarfile.TarInfo]]] = OrderedDict()
        self._cache_size = cache_size

    def __enter__(self) -> TarManager:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        for tf, _member_index in self._cache.values():
            try:
                tf.close()
            except Exception:  # noqa: BLE001
                pass
        self._cache.clear()

    def _get_tar_entry(
        self,
        shard_path: str,
    ) -> tuple[tarfile.TarFile, dict[str, tarfile.TarInfo]]:
        if shard_path in self._cache:
            self._cache.move_to_end(shard_path)
            return self._cache[shard_path]

        if len(self._cache) >= self._cache_size:
            _evicted_path, (evicted_tf, _evicted_member_index) = self._cache.popitem(last=False)
            try:
                evicted_tf.close()
            except Exception:  # noqa: BLE001
                pass

        tf = tarfile.open(shard_path, mode="r:*")
        members = tf.getmembers()
        member_index = {member.name: member for member in members}
        entry = (tf, member_index)
        self._cache[shard_path] = entry
        return entry

    def read(self, tar_ref: TarRef) -> bytes:
        """Read and return the raw bytes payload described by *tar_ref*."""

        member_name = f"{tar_ref.key}.{tar_ref.field}"
        tf, member_index = self._get_tar_entry(tar_ref.shard_path)
        member = member_index.get(member_name)
        if member is None:
            msg = f"[TarMemberNotFound] member={member_name!r} shard={tar_ref.shard_path!r} uri={tar_ref.raw_uri!r}"
            raise KeyError(msg)
        extracted = tf.extractfile(member)
        if extracted is None:
            msg = f"[TarExtractError] member={member_name!r} shard={tar_ref.shard_path!r} uri={tar_ref.raw_uri!r}"
            raise tarfile.ExtractError(msg)
        return extracted.read()


def materialize_jsonl_shards(
    files: Sequence[str],
    *,
    group_key: str | None,
    num_shards: int | None,
    target_samples_per_shard: int | None,
    spill_buckets: int,
    output_dir: PathLikeStr | None,
) -> list[str]:
    """Spill raw JSONL rows into balanced local shard files."""

    if spill_buckets <= 0:
        msg = f"[InvalidSpillBucketCount] spill_buckets must be > 0, got={spill_buckets}"
        raise ValueError(msg)
    if num_shards is not None and num_shards <= 0:
        msg = f"[InvalidShardCount] num_shards must be > 0, got={num_shards}"
        raise ValueError(msg)
    if target_samples_per_shard is not None and target_samples_per_shard <= 0:
        msg = f"[InvalidTargetSamplesPerShard] target_samples_per_shard must be > 0, got={target_samples_per_shard}"
        raise ValueError(msg)

    fingerprint = _jsonl_shard_plan_fingerprint(
        files=files,
        group_key=group_key,
        num_shards=num_shards,
        target_samples_per_shard=target_samples_per_shard,
        spill_buckets=spill_buckets,
    )
    root = Path(output_dir) if output_dir is not None else Path(".mvp_dataset_jsonl_shards")
    dataset_dir = root / fingerprint
    manifest_path = dataset_dir / "manifest.json"
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        shard_paths = payload.get("shards")
        if isinstance(shard_paths, list) and all(isinstance(path, str) for path in shard_paths):
            return [str(dataset_dir / path) for path in shard_paths]

    dataset_dir.mkdir(parents=True, exist_ok=True)
    bucket_dir = dataset_dir / "buckets"
    bucket_dir.mkdir(parents=True, exist_ok=True)

    bucket_handles: dict[int, object] = {}
    bucket_counts = [0] * spill_buckets
    total_rows = 0
    try:
        for file in files:
            with open(file, encoding="utf-8") as handle:
                for i, line in enumerate(handle):
                    sample = _parse_jsonl_line(file, i, line)
                    bucket_id = _bucket_id_for_sample(sample, group_key=group_key, spill_buckets=spill_buckets)
                    bucket_counts[bucket_id] += 1
                    total_rows += 1
                    bucket_handle = bucket_handles.get(bucket_id)
                    if bucket_handle is None:
                        bucket_path = bucket_dir / f"bucket_{bucket_id:05d}.jsonl"
                        bucket_handle = bucket_path.open("w", encoding="utf-8")
                        bucket_handles[bucket_id] = bucket_handle
                    bucket_handle.write(json.dumps(sample, ensure_ascii=True) + "\n")
    finally:
        for handle in bucket_handles.values():
            handle.close()

    if total_rows == 0:
        msg = "[EmptyJsonlSource] no rows found in input files"
        raise ValueError(msg)

    final_shard_count = _resolve_final_shard_count(
        total_rows=total_rows,
        num_shards=num_shards,
        target_samples_per_shard=target_samples_per_shard,
    )
    shard_targets = _balanced_shard_targets(total_rows=total_rows, shard_count=final_shard_count)

    shard_paths = [dataset_dir / f"shard_{index:05d}.jsonl" for index in range(final_shard_count)]
    shard_handles = [path.open("w", encoding="utf-8") for path in shard_paths]
    try:
        current_shard = 0
        rows_in_current_shard = 0
        for bucket_id in sorted(i for i, c in enumerate(bucket_counts) if c > 0):
            bucket_path = bucket_dir / f"bucket_{bucket_id:05d}.jsonl"
            with bucket_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if current_shard < final_shard_count - 1 and rows_in_current_shard >= shard_targets[current_shard]:
                        current_shard += 1
                        rows_in_current_shard = 0
                    shard_handles[current_shard].write(line)
                    rows_in_current_shard += 1
        manifest = {
            "files": list(files),
            "group_key": group_key,
            "num_shards": final_shard_count,
            "target_samples_per_shard": target_samples_per_shard,
            "spill_buckets": spill_buckets,
            "shards": [path.name for path in shard_paths],
        }
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=True, indent=2, sort_keys=True)
    finally:
        for handle in shard_handles:
            handle.close()

    return [str(path) for path in shard_paths]


def iter_jsonls(
    shard_paths: Iterator[PathLikeStr],
    ref_fields: tuple[RefFieldSpec, ...],
) -> Iterator[Sample]:
    """Resolve tar data references while streaming JSONL shard files."""
    key_dot_level = int(os.environ.get("MVP_DATASET_TAR_KEY_DOT_LEVEL", "1"))
    tar_cache_size = int(os.environ.get("MVP_DATASET_TAR_CACHE_SIZE", "8"))

    def _resolve_one(sample: Sample, manager: TarManager) -> Sample:
        resolved = dict(sample)
        for field, base_dir in ref_fields:
            if field not in sample:
                continue
            resolved[field] = resolve_ref_field_value(
                sample[field],
                field=field,
                base_dir=base_dir,
                key_dot_level=key_dot_level,
                manager=manager,
            )
        return resolved

    with TarManager(cache_size=tar_cache_size) as manager:
        for shard_path in shard_paths:
            with open(shard_path, encoding="utf-8") as handle:
                for line_index, line in enumerate(handle):
                    sample = _parse_jsonl_line(str(shard_path), line_index, line, allow_preannotated=True)
                    yield _resolve_one(sample, manager)


def _bucket_id_for_sample(sample: Sample, *, group_key: str | None, spill_buckets: int) -> int:
    if group_key is None:
        key = str(sample["__key__"])
    else:
        value = sample.get(group_key)
        if isinstance(value, str):
            key = value.split("#", 1)[0]
        elif isinstance(value, list):
            if not value:
                key = str(sample["__key__"])
            elif all(isinstance(item, str) for item in value):
                key = value[0].split("#", 1)[0]
            else:
                msg = f"[InvalidGroupKey] sample missing string refs for group_key={group_key!r}"
                raise ValueError(msg)
        else:
            msg = f"[InvalidGroupKey] sample missing string key for group_key={group_key!r}"
            raise ValueError(msg)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % spill_buckets


def _jsonl_shard_plan_fingerprint(
    *,
    files: Sequence[str],
    group_key: str | None,
    num_shards: int | None,
    target_samples_per_shard: int | None,
    spill_buckets: int,
) -> str:
    payload = {
        "files": [(file, s.st_mtime_ns, s.st_size) for file in files for s in (Path(file).stat(),)],
        "group_key": group_key,
        "num_shards": num_shards,
        "target_samples_per_shard": target_samples_per_shard,
        "spill_buckets": spill_buckets,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_jsonl_line(
    file: str,
    index_in_file: int,
    line: str,
    *,
    allow_preannotated: bool = False,
) -> Sample:
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError as exc:
        msg = f"[InvalidJsonLine] file={file!r} line={index_in_file + 1} reason={exc.msg}"
        raise ValueError(msg) from exc
    if not isinstance(parsed, dict):
        msg = f"[InvalidJsonSample] file={file!r} line={index_in_file + 1} expected object row"
        raise ValueError(msg)

    sample: Sample = dict(parsed)
    if allow_preannotated and _has_jsonl_metadata(sample):
        return sample

    sample["__index_in_file__"] = index_in_file
    sample["__file__"] = file
    sample["__key__"] = f"{file}:{index_in_file}"
    return sample


def _has_jsonl_metadata(sample: Sample) -> bool:
    return (
        isinstance(sample.get("__index_in_file__"), int)
        and isinstance(sample.get("__file__"), str)
        and isinstance(sample.get("__key__"), str)
    )


def _resolve_final_shard_count(
    *,
    total_rows: int,
    num_shards: int | None,
    target_samples_per_shard: int | None,
) -> int:
    if num_shards is not None:
        return num_shards
    if target_samples_per_shard is None:
        return 1
    return max(1, (total_rows + target_samples_per_shard - 1) // target_samples_per_shard)


def _balanced_shard_targets(*, total_rows: int, shard_count: int) -> list[int]:
    """Return per-shard row targets whose totals differ by at most one."""

    base = total_rows // shard_count
    remainder = total_rows % shard_count
    return [base + (1 if index < remainder else 0) for index in range(shard_count)]
