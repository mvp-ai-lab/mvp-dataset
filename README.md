# MVP Dataset

A minimal, high-performance data loading library for multimodal training pipelines.

`mvp-dataset` focuses on a small API surface, deterministic behavior, and practical throughput for local shard-based datasets.

## Why this project

- Minimal: only a few core abstractions (`Dataset`, `TorchLoader`, `RuntimeContext`)
- Fast: lazy iterator stages, bounded-memory shuffle, tar sidecar join, and loader-side pipeline composition
- Deterministic: seed-aware sharding and shuffle behavior across distributed + worker processes
- Extensible: source-level operations for TAR, JSONL-with-TAR-reference, and parquet workflows

## Features

- `Dataset.from_source("tars", ...)` for local `.tar` shard inputs
- `Dataset.from_source("jsonl", ...)` for local `.jsonl` inputs
- `Dataset.from_source("parquet", ...)` for local `.parquet` inputs
- `Dataset.from_source("lance", ...)` for local Lance dataset inputs
- Chainable pipeline ops:
  - `.map(...)`
  - `.select([...])`
  - `.shuffle(buffer_size, initial=...)`
  - `.assemble(factory, drop_last=...)`
  - `.batch(batch_size, drop_last=..., collate_fn=...)`
  - `.unbatch()`
  - `.cache(cache_dir=..., cache_num_workers=...)`
- Tar workflows:
  - sample parsing from shard members
  - optional sidecar merge via `from_source("tars", sidecars=[...])`
- JSONL workflows:
  - optional linked-tar reference resolution via `from_source("jsonl", ref_fields=[...])`
  - JSONL files are split into slot-aligned logical shards internally
- Parquet workflows:
  - row-group-parallel sample iteration from local `.parquet` shards
  - optional column projection via `from_source("parquet", columns=[...])`
  - each row is exposed as one `dict[str, object]` sample with `__file__`, `__index_in_file__`, and `__key__`
- Lance workflows:
  - fragment-parallel iteration from local Lance datasets
  - optional column projection via `from_source("lance", columns=[...])`
- `TorchLoader` for PyTorch `DataLoader` integration with post-merge stages

## Installation

Python requirement: `>=3.12,<3.13`

```bash
uv pip install "git+https://github.com/mvp-ai-lab/mvp-dataset.git"

# For development:
uv pip install -e .
```

If you use `TorchLoader`, install PyTorch separately.

## Quick start

### TAR pipeline

```python
from mvp_dataset import Dataset, TorchLoader

dataset = (
    Dataset.from_source("tars", "/data/shards/shard_{000000..000006}.tar", resample=True)
    .shuffle(buffer_size=1024)
    .map(lambda s: s)
)

loader = (
    TorchLoader(dataset, num_workers=8, batch_size=32, collate_fn=lambda x: x)
    .unbatch()
    .shuffle(buffer_size=4096)
    .batch(batch_size=64)
)

for batch in loader:
    train_step(batch)
```

### Joint TAR pipeline

```python
from mvp_dataset import Dataset, TorchLoader

dataset = (
    Dataset.from_source(
        "tars",
        "/data/images/shard_{000000..000006}.tar",
        resample=True,
        sidecars=[
            ("depth", lambda shard: shard.replace("/images/", "/depth/")),
        ],
    )
    .shuffle(buffer_size=1024)
    .map(lambda s: s)
)

loader = (
    TorchLoader(dataset, num_workers=8, batch_size=32, collate_fn=lambda x: x)
    .unbatch()
    .shuffle(buffer_size=4096)
    .batch(batch_size=64)
)

for batch in loader:
    train_step(batch)
```

### JSONL + TAR reference pipeline

```python
from mvp_dataset import Dataset

dataset = (
    Dataset.from_source(
        "jsonl",
        "/data/meta/train.jsonl",
        resample=True,
        ref_fields=[("image", "/data")],
    )
    .map(lambda s: s)
)

for sample in dataset:
    consume(sample)
```

### Parquet pipeline

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
    .cache()
)

for sample in dataset:
    consume(sample)
```

### Demo

Generate demo shards and metadata:

Read image/depth tar shards:

```bash
python3 examples/tar.py examples/demo_data/image/shard_{000000..000003}.tar --max-batches 2
```

Read JSONL with `tar://` references:

```bash
python3 examples/jsonl.py examples/demo_data/samples.jsonl --max-batches 2
```

Read parquet shards:

```bash
python3 examples/parquet.py "examples/demo_data/*.parquet" --max-batches 2
```

## Caching

`.cache()` materializes all upstream outputs into a local Lance dataset on disk so that expensive preprocessing (decoding, tokenization, augmentation, etc.) runs once and is reused on every subsequent iteration.

### Basic usage

```python
ds = (
    Dataset.from_source("tars", "shards/shard_{000000..000099}.tar")
    .map(expensive_preprocess)   # runs only on first iteration
    .cache()                     # <-- cache boundary
    .shuffle(buffer_size=4096)   # runs every iteration
)

for sample in ds:  # 1st iter: builds cache
    train_step(sample)

for sample in ds:  # 2nd iter: reads from cache, skips expensive_preprocess
    train_step(sample)
```

Cached Lance datasets are written under `cache_dir` (default: `.cache` in the current working directory).

### How invalidation works

The cache key is a SHA-256 plan fingerprint over:

- the source listing / fragment identity
- every pre-cache stage marked as traceable

Today:

- `.map()` participates in cache invalidation
- `.select()` participates in cache invalidation
- `.assemble()` participates in cache invalidation, including `drop_last`
- `.shuffle()` is intentionally allowed before `.cache()`, but it does not participate in invalidation; a warning is logged when this happens

Any lambda or local closure used before `.cache()` is rejected because its identity is not stable across process restarts.

### Supported pre-cache stages

- Supported: `map`, `select`, `shuffle`, `assemble`
- Rejected: `batch`, `unbatch`

Cache warm-up preserves full-stream semantics:

- only the leading contiguous `map(...)` prefix is parallelized with `cache_num_workers`
- once a non-`map` stage is reached, all remaining pre-cache stages run on the merged stream in declaration order

This keeps `shuffle` / `assemble` semantics identical between cached and uncached pipelines.

### Post-cache stages

Stages appended **after** `.cache()` run on every iteration, reading from the cached Lance dataset:

```python
ds = (
    Dataset.from_source("tars", shards)
    .map(tokenize)       # cached
    .cache()
    .map(random_augment) # NOT cached — runs every time
    .shuffle(buffer_size=2048)
)
```

### Distributed / tensor-parallel awareness

When using a `DataLoadMesh` with model-parallel dimensions (for example tensor parallelism), multiple ranks may intentionally receive identical input shards. `RuntimeContext.slot` and `total_slots` are derived from the mesh's data-parallel dimensions, so TP co-members share data while DP ranks remain distinct.

The cache layer is mesh-aware as well:

- per-slot cache shards are keyed by `dp_rank`, not global `rank`
- only one DP leader writes each per-slot cache shard
- the final merge consumes one cache shard per DP slot, so TP co-members do not duplicate cached samples

```python
from mvp_dataset import Dataset, RuntimeContext

ctx = RuntimeContext.from_runtime(
    device_mesh=mesh,
    dp_dims=("replicate", "shard"),  # TP dim excluded → co-members share shards
)

ds = (
    Dataset.from_source("tars", shards, context=ctx)
    .map(expensive_fn)
    .cache()
)
```

## Data conventions

### TAR member naming

Tar members are parsed as `<key>.<field>` (or multi-segment keys controlled by `key_dot_level`).

Examples:
- `abc.jpg` -> key=`abc`, field=`jpg`
- `abc.extra.jpg` with `key_dot_level=2` -> key=`abc.extra`, field=`jpg`

### JSONL TAR reference URI

Supported URI grammar:

```text
tar://<shard_path>#<key>.<field>
<shard_path>#<key>.<field>
```

Example:

```text
tar://examples/data/tars/shard_000000.tar#image_001.jpg
```

Reference fields can be either a single string URI or a list of string URIs. For example:

```json
{"images": ["images/train-00000.tar#image_00.jpg", "images/train-00000.tar#image_01.jpg"]}
```

### Parquet row samples

Parquet rows are yielded as ordinary sample dicts. Each sample includes:

- `__file__`: parquet file path
- `__index_in_file__`: zero-based row index
- `__key__`: synthesized as `<file>:<row_index>`

Source-specific options remain source-local:

- `sidecars=...` is tar-only
- `ref_fields=...` is jsonl-only

Parquet scheduling uses row groups as the unit of work, so different slots can
read different row groups from the same parquet file.

## Runtime and environment variables

- Distributed/runtime context:
  - `RANK`
  - `WORLD_SIZE`
  - `WORKER`
  - `NUM_WORKERS`
- Source behavior:
  - `LOADER_TAR_KEY_DOT_LEVEL` (default: `1`)
  - `LOADER_JSONL_TAR_CACHE_SIZE` (default: `8`)
- JSONL preprocessing:
  - JSONL logical splitting is internal to `from_source("jsonl", ...)`

## Performance notes

- Iterator pipeline is lazy; data transforms are applied on demand.
- Shuffle uses bounded buffer memory (`buffer_size`, optional `initial`).
- JSONL reference resolution uses LRU tar-handle caching.
- `from_source("jsonl", ...)` internally splits JSONL inputs into slot-aligned logical shards.
- parquet row groups are scheduled as shards and streamed row-by-row via `pyarrow`.
- Recommended high-throughput pattern with PyTorch:
  1. worker micro-batch (`batch_size` on `TorchLoader` + identity collate)
  2. `.unbatch()`
  3. loader-side `.shuffle(...)`
  4. loader-side `.batch(...)`
