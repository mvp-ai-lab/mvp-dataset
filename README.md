# MVP Dataset

`mvp-dataset` is a efficient data loading library for multimodal training.

It is built around a few ideas:

- one immutable `Dataset` pipeline
- deterministic sharding across distributed ranks and worker processes
- simple on-the-fly transforms
- optional `TorchLoader` integration for PyTorch `DataLoader`

## Installation

```bash
uv pip install "git+https://github.com/mvp-ai-lab/mvp-dataset.git"

# Development install
uv pip install -e .
```

If you use `TorchLoader`, install PyTorch separately.

## Public API

Top-level exports:

- `Dataset`
- `RuntimeContext`
- `DataLoadMesh`
- `TorchLoader`
- `set_logger(...)` / `get_logger(...)`
- `set_log_level(...)` / `get_log_level(...)`

Dataset construction uses a single entrypoint:

```python
Dataset.from_source(source_kind, ...)
```

Supported `source_kind` values:

- `"tars"`
- `"jsonl"`
- `"parquet"`
- `"lance"`

Chainable dataset stages:

- `.map(fn)`
- `.select(fields)`
- `.shuffle(buffer_size, initial=None)`
- `.assemble(factory, drop_last=False)`
- `.batch(batch_size, drop_last=False, collate_fn=None)`
- `.unbatch()`

`TorchLoader` supports post-merge stages:

- `.unbatch()`
- `.shuffle(buffer_size, initial=None, seed=None)`
- `.assemble(factory, drop_last=False)`
- `.batch(batch_size, drop_last=False, collate_fn=None)`

## Core Concepts

### Dataset

`Dataset` is an immutable iterable pipeline. Every stage returns a new dataset.

```python
from mvp_dataset import Dataset

dataset = (
    Dataset.from_source("tars", "/data/shards/shard_{000000..000127}.tar")
    .shuffle(buffer_size=1024)
    .map(preprocess)
)
```

### RuntimeContext

`RuntimeContext` carries rank, worker, seed, and optional mesh information.

Most users do not need to construct it manually. If `context=` is omitted, runtime values are inferred from:

- `torch.distributed`
- `torch.utils.data.get_worker_info()`
- environment variables such as `RANK` and `WORLD_SIZE`

### DataLoadMesh

`DataLoadMesh` is used when the training mesh contains model-parallel dimensions such as tensor parallelism.

Its purpose is to separate:

- data-parallel dimensions: different data
- model-parallel dimensions: same data

This lets TP co-members receive identical dataset shards while DP ranks remain distinct.

## Quick Start

### TAR

```python
from mvp_dataset import Dataset, TorchLoader

dataset = (
    Dataset.from_source("tars", "/data/shards/shard_{000000..000006}.tar", resample=True)
    .shuffle(buffer_size=1024)
    .map(decode_sample)
)

loader = (
    TorchLoader(dataset, num_workers=8, batch_size=32, collate_fn=lambda x: x)
    .unbatch()
    .shuffle(buffer_size=4096)
    .assemble(pack_samples)
    .batch(batch_size=64)
)

for batch in loader:
    train_step(batch)
```

### TAR With Sidecars

Use `sidecars=` when related modalities live in separate tar shards with matching shard layouts.

```python
from mvp_dataset import Dataset

dataset = Dataset.from_source(
    "tars",
    "/data/images/shard_{000000..000006}.tar",
    sidecars=[
        ("depth", lambda shard: shard.replace("/images/", "/depth/")),
    ],
)
```

### JSONL With TAR References

```python
from mvp_dataset import Dataset

dataset = Dataset.from_source(
    "jsonl",
    "/data/meta/train.jsonl",
    ref_fields=[("image", "/data")],
)
```

### Parquet

```python
from mvp_dataset import Dataset

def add_length(sample: dict[str, object]) -> dict[str, object]:
    return {**sample, "length": len(str(sample["text"]))}

dataset = (
    Dataset.from_source(
        "parquet",
        "/data/meta/train_{000000..000003}.parquet",
        columns=["text", "label"],
        batch_size=8192,
    )
    .map(add_length)
)
```

### Lance

```python
from mvp_dataset import Dataset

dataset = Dataset.from_source(
    "lance",
    "/data/lance/train.lance",
    columns=["tokens", "label"],
)
```

## Source Types

### TAR

```python
Dataset.from_source(
    "tars",
    shards,
    context=None,
    resample=False,
    sidecars=None,
)
```

Notes:

- inputs must resolve to `.tar` files
- tar shard count must be at least `context.total_slots`
- samples are parsed from tar members as keyed field dictionaries

### JSONL

```python
Dataset.from_source(
    "jsonl",
    shards,
    context=None,
    resample=False,
    ref_fields=None,
)
```

Notes:

- inputs must resolve to `.jsonl` files
- JSONL inputs are internally split into slot-aligned logical shards
- `ref_fields=[(field_name, base_dir), ...]` resolves `tar://...`-style references

### Parquet

```python
Dataset.from_source(
    "parquet",
    shards,
    context=None,
    resample=False,
    columns=None,
    batch_size=65536,
    use_threads=True,
)
```

Notes:

- inputs must resolve to `.parquet` files
- work is scheduled by parquet fragment / row group
- each row becomes one sample dict

### Lance

```python
Dataset.from_source(
    "lance",
    shards,
    context=None,
    resample=False,
    columns=None,
    batch_size=65536,
    shuffle_mode="none",
)
```

Notes:

- inputs are local Lance dataset paths or URIs
- `shuffle_mode="none"` preserves ordered reads
- `shuffle_mode="global"` performs an exact global permutation via `take(...)`
- `shuffle_mode="fragment_aware"` shuffles assigned fragments/chunks and row blocks while keeping each slot close to fewer fragments

## Pipeline Semantics

### `map`

Applies a pure per-sample transform.

### `select`

Projects user fields while preserving internal metadata such as `__key__`.

### `shuffle`

Applies bounded-memory sample shuffle with deterministic seeding derived from runtime context.

### `assemble`

Builds a stateful stream assembler from `factory(context)` and allows many-input to many-output transformations. It is very useful for data packing in LLM/VLM training.

The `factory` receives the runtime-resolved `RuntimeContext`, so it can safely depend on:

- `worker_id`
- `num_workers`
- distributed rank information
- mesh-aware DP semantics

### `batch` / `unbatch`

Operate directly on the dataset stream.

## Distributed and Tensor-Parallel Behavior

Without a mesh, sharding is based on global `rank` and `world_size`.

With a `DataLoadMesh`, sharding is based on data-parallel dimensions:

- `RuntimeContext.slot` uses `dp_rank`
- `RuntimeContext.total_slots` uses `dp_size`
- TP co-members receive identical data
- DP ranks receive distinct data

Example:

```python
from mvp_dataset import Dataset, RuntimeContext

ctx = RuntimeContext.from_runtime(
    device_mesh=mesh,
    dp_dims=("replicate", "shard"),
)

dataset = Dataset.from_source("tars", shards, context=ctx)
```

## TorchLoader

`TorchLoader` wraps PyTorch `DataLoader` and applies additional post-merge stages in the main process.

Recommended high-throughput pattern:

1. use worker micro-batches with `batch_size` and `collate_fn=lambda x: x`
2. call `.unbatch()`
3. call loader-side `.shuffle(...)`
4. optionally call loader-side `.assemble(...)`
5. call loader-side `.batch(...)`

Example:

```python
from mvp_dataset import Dataset, TorchLoader

dataset = Dataset.from_source("tars", shards).map(preprocess)

loader = (
    TorchLoader(dataset, num_workers=8, batch_size=32, collate_fn=lambda x: x)
    .unbatch()
    .shuffle(buffer_size=4096)
    .assemble(pack_samples)
    .batch(batch_size=128)
)
```

Loader-side shuffle uses the upstream dataset context when available, so TP co-members stay aligned by default.
Loader-side assemble also resolves runtime context from the upstream dataset when available.

## Data Conventions

### TAR Member Naming

Tar members are parsed as `<key>.<field>`.

Examples:

- `abc.jpg` -> key=`abc`, field=`jpg`
- `abc.extra.jpg` with `key_dot_level=2` -> key=`abc.extra`, field=`jpg`

Related environment variable:

- `LOADER_TAR_KEY_DOT_LEVEL` default `1`

### JSONL TAR References

Supported URI grammar:

```text
tar://<shard_path>#<key>.<field>
<shard_path>#<key>.<field>
```

Example:

```text
tar://examples/data/tars/shard_000000.tar#image_001.jpg
```

Reference fields may be either:

- one URI string
- a list of URI strings

### Parquet Samples

Each parquet row is exposed as a sample dict containing:

- `__file__`
- `__index_in_file__`
- `__key__`

Parquet scheduling uses row groups as units of work.

## Runtime and Environment Variables

Distributed / runtime context:

- `RANK`
- `WORLD_SIZE`
- `WORKER`
- `NUM_WORKERS`

Source behavior:

- `LOADER_TAR_KEY_DOT_LEVEL`
- `MVP_DATASET_TAR_MAX_OPEN_FILES`

## Demo Commands

```bash
python3 examples/tar.py examples/demo_data/image/shard_{000000..000003}.tar --max-batches 2
python3 examples/jsonl.py examples/demo_data/samples.jsonl --max-batches 2
python3 examples/parquet.py "examples/demo_data/*.parquet" --max-batches 2
```
