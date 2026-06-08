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
    min_row_groups_per_fragment=1,
    columns=None,
    use_threads=True,
    shuffle_mode="fragment_aware",
)
```

Arguments:

- `shards`: Parquet file path or sequence of paths.
- `min_row_groups_per_fragment`: minimum row groups grouped into one scheduling fragment.
- `columns`: optional projection.
- `use_threads`: forwarded to the Parquet reader.
- `shuffle_mode`: `"none"` or `"fragment_aware"`. `"global"` is intentionally unsupported for Parquet.

Parquet source notes:

- Inputs must end with `.parquet`.
- Work is scheduled by fragment and row group.
- Each row is yielded as a sample dictionary.

## Lance

```python
Dataset.from_source(
    "lance",
    shards,
    context=None,
    resample=False,
    columns=None,
    batch_size=1024,
    shuffle_mode="none",
    chunk_aware_shuffle_chunk_size=250_000,
    chunk_aware_shuffle_k=8,
    ref_columns=None,
    ref_index_scope=None,
)
```

Arguments:

- `shards`: Lance dataset URI or sequence of URIs.
- `columns`: optional projection.
- `batch_size`: internal Lance read batch size.
- `shuffle_mode`: `"none"`, `"global"`, `"chunk_aware"`, or `"fragment_aware"`.
- `chunk_aware_shuffle_chunk_size`: rows per chunk for `shuffle_mode="chunk_aware"`.
- `chunk_aware_shuffle_k`: number of shuffled chunks in each chunk-aware window.
- `ref_columns`: optional Lance reference column configuration.
- `ref_index_scope`: optional reference-index storage scope.

Lance source notes:

- Lance supports exact deterministic global shuffle without materializing a full epoch index list.
- `chunk_aware` shuffle randomizes chunk order and rows within a bounded chunk window.
- `fragment_aware` shuffle preserves deterministic order while staying friendlier to fragment locality.
- Lance reference columns can be resolved with `resolve_ref(...)`.

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

dataset = dataset.resolve_ref(["image"])
```

`resolve_ref(...)` appends a stateful assembly stage that resolves configured reference values lazily.
