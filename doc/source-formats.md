# Source Formats

All sources are constructed with `Dataset.from_source(source_kind, ...)`.

## Tar

```python
Dataset.from_source(
    "tar",
    shards,
    context=None,
    resample=False,
    sidecars=None,
    shuffle_mode="shard_aware",
)
```

Arguments:

- `shards`: tar file path or sequence of tar paths. Brace expansion is supported by path normalization.
- `sidecars`: optional sequence of `(name, path_resolver)` pairs. Each resolver maps a main shard path to a sidecar shard path.
- `shuffle_mode`: `"none"` or `"shard_aware"`. `"global"` is intentionally unsupported for tar.

Tar source notes:

- Inputs must end with `.tar`.
- The number of tar files must be at least `context.total_slots`.
- Sidecars are joined by sample key.

## JSONL

```python
Dataset.from_source(
    "jsonl",
    shards,
    context=None,
    resample=False,
    ref_fields=None,
    shuffle_mode="shard_aware",
)
```

Arguments:

- `shards`: JSONL file path or sequence of paths.
- `ref_fields`: optional sequence of `(field_name, base_dir)` entries for resolving tar URI references.
- `shuffle_mode`: `"none"` or `"shard_aware"`. `"global"` is intentionally unsupported for JSONL.

JSONL source notes:

- Inputs must end with `.jsonl`.
- Files are split into slot-aligned logical shards.
- Samples are JSON objects represented as Python dictionaries.

## Parquet

```python
Dataset.from_source(
    "parquet",
    shards,
    context=None,
    resample=False,
    min_row_groups_per_chunk=1,
    columns=None,
    use_threads=True,
    shuffle_mode="chunk_aware",
)
```

Arguments:

- `shards`: Parquet file path or sequence of paths.
- `min_row_groups_per_chunk`: minimum row groups grouped into one scheduling chunk.
- `columns`: optional projection.
- `use_threads`: forwarded to the Parquet reader.
- `shuffle_mode`: `"none"` or `"chunk_aware"`. `"global"` is intentionally unsupported for Parquet.

Parquet source notes:

- Inputs must end with `.parquet`.
- Work is scheduled by chunk and row group.
- Each row is yielded as a sample dictionary.

## Lance

```python
Dataset.from_source(
    "lance",
    shards,
    context=None,
    resample=False,
    columns=None,
    read_batch_size=1024,
    shuffle_mode="none",
    chunk_shuffle=None,
    ref_columns=None,
    ref_index_scope=None,
)
```

Arguments:

- `shards`: Lance dataset URI or sequence of URIs.
- `columns`: optional projection.
- `read_batch_size`: number of row indexes aggregated into one Lance read call.
- `shuffle_mode`: `"none"`, `"global"`, or `"chunk"`.
- `chunk_shuffle`: optional chunk shuffle configuration when `shuffle_mode="chunk"`.
- `ref_columns`: optional Lance reference column configuration.
- `ref_index_scope`: optional reference-index storage scope.

Lance source notes:

- Lance supports exact deterministic global shuffle without materializing a full epoch index list.
- `chunk` shuffle randomizes chunk order and rows within a bounded chunk window.
- Lance reference columns can be resolved with `resolve_ref(...)`.
- Use `chunk_shuffle={"chunk_size": 65536, "k": 4, "row_order": "sequential"}` to tune chunk shuffle.
- `row_order` can be `"permuted"` or `"sequential"`. The default is `"permuted"`.

### Lance Reference Resolution

```python
dataset = Dataset.from_source(
    "lance",
    "/data/train.lance",
    ref_columns={
        "image": {
            "uri": "/data/images.lance",
            "key_column": "id",
            "value_column": "bytes",
        }
    },
)

dataset = dataset.resolve_ref(
    ["image"],
    ref_index_build_strategy="auto",
    ref_index_bucket_count=4096,
)
```

`resolve_ref(...)` appends a stateful assembly stage that resolves configured reference values lazily.
Missing reference indexes are built with `ref_index_build_strategy="auto"` by default. Small sources use the in-memory
builder, while larger sources use a bucketed on-disk join to avoid keeping all reference keys in memory.
