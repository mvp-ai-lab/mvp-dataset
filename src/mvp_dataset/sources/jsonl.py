"""JSONL source utilities, spill sharding, and tar reference parsing."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tarfile
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from ..core.types import PathLikeStr, RefFieldSpec, Sample, SidecarSpec
from .tar import fingerprint_parts

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
    """Parse URI with grammar ``tar://<shard>#<key>.<field>``."""

    if not uri.startswith(_TAR_URI_PREFIX):
        msg = f"[InvalidTarUri] uri={uri!r} expected_prefix={_TAR_URI_PREFIX!r}"
        raise ValueError(msg)

    body = uri[len(_TAR_URI_PREFIX) :]
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
        for bucket_id in sorted(_non_empty_bucket_ids(bucket_counts)):
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


def jsonl_cache_dir(shard_path: PathLikeStr) -> Path:
    """Return the cache directory used for fingerprinted JSONL sidecars."""

    return Path(shard_path).parent / ".cache"


def jsonl_cache_glob(shard_path: PathLikeStr, key: str) -> str:
    """Return the glob pattern for all fingerprinted JSONL cache sidecars of one key."""

    path = Path(shard_path)
    return str(jsonl_cache_dir(path) / f"{path.stem}-{key}-*.jsonl")


def jsonl_cache_path(
    shard_path: PathLikeStr,
    key: str,
    fingerprint: str,
) -> str:
    """Return the JSONL cache path for one keyed map stage."""

    path = Path(shard_path)
    return str(jsonl_cache_dir(path) / f"{path.stem}-{key}-{fingerprint}.jsonl")


def jsonl_source_fingerprint(
    shard_paths: Sequence[PathLikeStr],
    *,
    ref_fields: Sequence[RefFieldSpec] = (),
) -> str:
    """Return the base fingerprint used by JSONL map cache stages."""

    return fingerprint_parts(
        "jsonl-source",
        *(str(path) for path in shard_paths),
        *(f"{field}:{base_dir}" for field, base_dir in ref_fields),
    )


def count_jsonl_samples(path: PathLikeStr) -> int:
    """Return the number of rows in one JSONL shard or cache file."""

    with open(path, encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def iter_jsonls(
    shard_paths: Iterator[PathLikeStr],
    ref_fields: tuple[RefFieldSpec, ...],
    key_dot_level: int = 1,
    tar_cache_size: int = 8,
    cache_specs: tuple[SidecarSpec, ...] = (),
) -> Iterator[Sample]:
    """Resolve tar data references while streaming JSONL shard files."""

    def _resolve_one(sample: Sample, manager: TarManager) -> Sample:
        resolved = dict(sample)
        for field, base_dir in ref_fields:
            if field not in sample:
                continue
            value = sample[field]
            if not isinstance(value, str):
                msg = f"[InvalidRefField] expected string URI in field={field!r} got={type(value).__name__}"
                raise ValueError(msg)
            try:
                tar_ref = parse_tar_uri(value, base_dir=base_dir, key_dot_level=key_dot_level)
            except ValueError as exc:
                msg = f"[InvalidRefField] failed to parse tar URI in field={field!r} value={value!r} reason={exc}"
                raise ValueError(msg) from exc
            resolved[field] = manager.read(tar_ref)
        return resolved

    with TarManager(cache_size=tar_cache_size) as manager:
        for shard_path in shard_paths:
            cache_handles = _open_jsonl_cache_handles(str(shard_path), cache_specs)
            try:
                with open(shard_path, encoding="utf-8") as handle:
                    for line_index, line in enumerate(handle):
                        sample = _parse_jsonl_line(str(shard_path), line_index, line, allow_preannotated=True)
                        resolved = _resolve_one(sample, manager)
                        for cache_handle in cache_handles:
                            cache_line = cache_handle.readline()
                            if cache_line == "":
                                msg = (
                                    f"[JsonlCacheCountMismatch] cache={cache_handle.name!r} "
                                    f"ended before shard={str(shard_path)!r}"
                                )
                                raise ValueError(msg)
                            resolved.update(
                                _parse_jsonl_cache_line(
                                    cache_handle.name,
                                    line_index,
                                    cache_line,
                                    expected_key=str(resolved["__key__"]),
                                )
                            )
                        yield resolved
                for cache_handle in cache_handles:
                    if cache_handle.readline() != "":
                        msg = (
                            f"[JsonlCacheCountMismatch] cache={cache_handle.name!r} "
                            f"contains more rows than shard={str(shard_path)!r}"
                        )
                        raise ValueError(msg)
            finally:
                for cache_handle in cache_handles:
                    cache_handle.close()


def _bucket_id_for_sample(sample: Sample, *, group_key: str | None, spill_buckets: int) -> int:
    if group_key is None:
        key = str(sample["__key__"])
    else:
        value = sample.get(group_key)
        if not isinstance(value, str):
            msg = f"[InvalidGroupKey] sample missing string key for group_key={group_key!r}"
            raise ValueError(msg)
        key = value.split("#", 1)[0]
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
        "files": [(file, Path(file).stat().st_mtime_ns, Path(file).stat().st_size) for file in files],
        "group_key": group_key,
        "num_shards": num_shards,
        "target_samples_per_shard": target_samples_per_shard,
        "spill_buckets": spill_buckets,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _non_empty_bucket_ids(bucket_counts: Sequence[int]) -> Iterator[int]:
    for bucket_id, count in enumerate(bucket_counts):
        if count > 0:
            yield bucket_id


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


def write_jsonl_cache(
    shard_path: PathLikeStr,
    *,
    key: str,
    samples: Iterable[Sample],
    fingerprint: str,
    expected_sample_count: int,
) -> None:
    """Write one cached JSONL sidecar for the provided samples."""

    output_path = Path(jsonl_cache_path(shard_path, key, fingerprint))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    written_samples = 0
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            for sample in samples:
                cache_fields = sample.get("__cache_fields__")
                if not isinstance(cache_fields, dict):
                    msg = f"[InvalidJsonlCacheSample] key={key!r} missing __cache_fields__ mapping"
                    raise ValueError(msg)
                line = {
                    "__key__": sample["__key__"],
                    "__index_in_file__": sample["__index_in_file__"],
                    "__file__": sample["__file__"],
                    "__cache_fields__": {
                        field_name: base64.b64encode(payload).decode("ascii")
                        for field_name, payload in cache_fields.items()
                    },
                }
                handle.write(json.dumps(line, sort_keys=True, ensure_ascii=True) + "\n")
                written_samples += 1
        if written_samples != expected_sample_count:
            msg = (
                f"[JsonlCacheSampleCountMismatch] key={key!r} shard={str(shard_path)!r} "
                f"expected={expected_sample_count} wrote={written_samples}"
            )
            raise ValueError(msg)
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _open_jsonl_cache_handles(
    shard_path: str,
    cache_specs: tuple[SidecarSpec, ...],
) -> list[object]:
    handles: list[object] = []
    for _name, resolver in cache_specs:
        cache_path = resolver(shard_path)
        handles.append(open(cache_path, encoding="utf-8"))
    return handles


def _parse_jsonl_cache_line(
    file: str,
    index_in_file: int,
    line: str,
    *,
    expected_key: str,
) -> dict[str, bytes]:
    sample = _parse_jsonl_line(file, index_in_file, line, allow_preannotated=True)
    if sample["__key__"] != expected_key:
        msg = (
            f"[JsonlCacheKeyMismatch] cache={file!r} line={index_in_file + 1} "
            f"expected_key={expected_key!r} got={sample['__key__']!r}"
        )
        raise ValueError(msg)
    cache_fields = sample.get("__cache_fields__")
    if not isinstance(cache_fields, dict):
        msg = f"[InvalidJsonlCacheLine] file={file!r} line={index_in_file + 1} missing __cache_fields__"
        raise ValueError(msg)
    decoded: dict[str, bytes] = {}
    for field_name, payload in cache_fields.items():
        if not isinstance(field_name, str) or not isinstance(payload, str):
            msg = f"[InvalidJsonlCacheLine] file={file!r} line={index_in_file + 1} invalid cache field payload"
            raise ValueError(msg)
        try:
            decoded[field_name] = base64.b64decode(payload.encode("ascii"))
        except Exception as exc:  # noqa: BLE001
            msg = (
                f"[InvalidJsonlCacheLine] file={file!r} line={index_in_file + 1} "
                f"field={field_name!r} invalid base64 payload"
            )
            raise ValueError(msg) from exc
    return decoded
