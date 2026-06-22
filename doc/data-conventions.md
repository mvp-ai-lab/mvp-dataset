# Data Conventions

This page documents source-level sample conventions used by `mvp-dataset`.

## Sample Shape

Samples are Python dictionaries. Source readers may attach internal metadata keys using the `__name__` convention. Selection stages preserve metadata keys when projecting fields.

Common metadata keys include:

- `__key__`: stable sample key when available.
- `__file__`: source file path when available.
- `__index_in_file__`: row or item position within a source file when available.

## Tar Member Naming

Tar samples are grouped by member name. A member name is interpreted as:

```text
<key>.<field>
```

Examples:

- `abc.jpg` -> key `abc`, field `jpg`
- `abc.txt` -> key `abc`, field `txt`

The key split can be adjusted with the `LOADER_TAR_KEY_DOT_LEVEL` environment variable. Its default value is `1`.

## Tar Sidecars

Tar sidecars join related tar shards to the main shard stream:

```python
dataset = Dataset.from_source(
    "tar",
    "/data/images/shard_{000000..000127}.tar",
    sidecars=[
        ("depth", lambda shard: shard.replace("/images/", "/depth/")),
    ],
)
```

Each sidecar resolver receives the main shard path and returns the related sidecar shard path.

## JSONL Tar References

JSONL reference fields can resolve values stored inside tar shards. Supported reference values are strings or lists of strings.

Reference URI forms:

```text
tar://<shard_path>#<key>.<field>
<shard_path>#<key>.<field>
```

Example:

```text
tar:///data/images/shard_000000.tar#sample_001.jpg
```

Configure reference fields with `(field_name, base_dir)` entries:

```python
dataset = Dataset.from_source(
    "jsonl",
    "/data/metadata/train.jsonl",
    ref_fields=[("image", "/data")],
)
```

## Parquet Rows

Parquet rows are yielded as dictionaries. `columns=` controls projection at read time.

Parquet scheduling works on chunks built from row groups, which makes resume and distributed assignment cheaper than row-by-row skipping.

## Lance Rows

Lance rows are yielded as dictionaries. `columns=` controls projection at read time.

Lance supports these source shuffle modes:

- `none`: deterministic slot-stride order.
- `global`: deterministic global permutation without materializing a full index list.
- `chunk`: deterministic chunk-window shuffle with bounded row-permutation state.
- `chunk` can be tuned with `chunk_shuffle={"chunk_size": ..., "k": ..., "row_order": ...}`.

## Environment Variables

Runtime context may read common distributed environment variables when explicit values are not provided:

- `RANK`
- `WORLD_SIZE`
- `WORKER`
- `NUM_WORKERS`

Source behavior variables:

- `LOADER_TAR_KEY_DOT_LEVEL`: tar key parsing level, default `1`.
- `MVP_DATASET_TAR_MAX_OPEN_FILES`: maximum cached tar file handles used while resolving JSONL tar references.
