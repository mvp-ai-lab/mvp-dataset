# Persistent Cache

`mvp-dataset` stores all derived data under one cache root. Set it independently on each node:

```bash
export MVP_DATASET_CACHE_DIR=/mnt/shared-cache/mvp-dataset
```

Different local mount paths may point to the same shared storage. Absolute source and cache paths are never included in
cache fingerprints or persisted manifests.

This cache has no legacy-path discovery or migration. Existing source-adjacent cache directories are ignored.

## Layout

```text
<cache-root>/
  <source-fingerprint>/
    source.json
    jsonl-split-v1/
      <artifact-fingerprint>/
        manifest.json
        complete
        ...
    lance-ref-index-v1/
      <artifact-fingerprint>/
        manifest.json
        complete
        ...
    lance-filter-index-v1/
      <artifact-fingerprint>/
        manifest.json
        complete
        ...
    dataset-snapshot-v1/
      <artifact-fingerprint>/
        manifest.json
        complete
        slot-00000000/data.lance/
        slot-00000001/data.lance/
        ...
  _locks/
  _tmp/
```

The first fingerprint identifies the source. The cache-kind directory identifies the artifact format. The final
fingerprint covers cache parameters and dependent sources.

Local file fingerprints use source-relative paths, sizes, and nanosecond modification times by default. To hash full
file contents instead:

```bash
export MVP_DATASET_CACHE_FINGERPRINT_MODE=content
```

Supported values are `metadata` and `content`.

Cache builders and waiters time out after 30 minutes by default. Override the timeout in seconds when large artifacts
need more time to build:

```bash
export MVP_DATASET_CACHE_WAIT_TIMEOUT_SECONDS=10800
```

## Builder API

`CacheManager.ensure()` supports both complete-artifact and distributed part builders:

```python
entry = manager.ensure(..., build=build_artifact)

entry = manager.ensure(
    ...,
    parts=plan_parts,
    assigned_parts=(current_part,),
    build=build_one_part,
)
```

The presence of `parts` selects the callback signature: whole-artifact builds receive a temporary directory, while
partitioned builds receive a part name and temporary directory. Both modes produce the same `CacheEntry` contract.
Part builders automatically claim available work through per-part leases and continue with unfinished parts after
completing their current part. `assigned_parts` optionally restricts a caller to specific parts; this is used when each
runtime slot must build only its own upstream stream.

## Crash and Concurrency Safety

Builders never write into a final artifact directory. Each build uses a unique directory under `_tmp`, writes and
validates all declared files, then writes `manifest.json` and `complete`. The completed directory is published with a
same-filesystem atomic rename.

An interrupted build therefore leaves only an ignored temporary directory. Build ownership uses advisory `flock`
leases on stable files under `_locks`; lock files are never deleted or replaced. The kernel releases a lease when its
process exits, including `SIGKILL`, and the next owner removes abandoned temporary directories before rebuilding.
Shared cache filesystems must provide cross-host POSIX advisory lock semantics.

Concurrent builders for the same key share one completed result. A cache entry is usable only when its fingerprints,
manifest, completion marker, declared files, and file sizes all validate.

Artifacts such as Lance filter indexes use partitioned builds. Concurrent ranks and workers claim independent parts,
write them through part-specific temporary directories, and publish each part under its own advisory file lease. Parts
completed before a process failure remain reusable, while abandoned part temporary directories are replaced by the
next owner. A generation ID fences off workers from abandoned attempts, and the finalizer locks and validates every
part before atomically publishing the complete artifact.

Dataset snapshots use a partitioned build with one Lance part per slot in the topology that first creates the cache.
Each participating slot consumes its existing upstream stream and writes only its named part. Published parts form one
stable slot-major row space. Later readers may use a different runtime topology; a valid cache hit never plans new
parts or constructs the upstream iterator.

## Management API

```python
from mvp_dataset import clear_cache, list_cache_entries

entries = list_cache_entries()
removed = clear_cache(kind="jsonl-split")
removed = clear_cache(source_fingerprint="...")
```

Both functions accept an optional cache root as their first argument. `clear_cache()` only removes completed artifacts;
it does not modify source data.
