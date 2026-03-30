# MVP Dataset

A minimal, high-performance data loading library for multimodal training pipelines.

`mvp-dataset` focuses on a small API surface, deterministic behavior, and practical throughput for local shard-based datasets.

## Why this project

- Minimal: only a few core abstractions (`Dataset`, `TorchLoader`, `RuntimeContext`)
- Fast: lazy iterator stages, bounded-memory shuffle, tar sidecar join, and loader-side pipeline composition
- Deterministic: seed-aware sharding and shuffle behavior across distributed + worker processes
- Extensible: source-level operations for TAR and JSONL-with-TAR-reference workflows

## Features

- `Dataset.from_tars(...)` for local `.tar` shard inputs
- `Dataset.from_jsonl(...)` for local `.jsonl` inputs
- `Dataset.from_source(...)` as a compatibility dispatcher for mixed call sites
- Chainable pipeline ops:
  - `.map(...)`
  - `.shuffle(buffer_size, initial=...)`
  - `.batch(batch_size, drop_last=..., collate_fn=...)`
  - `.unbatch()`
  - `.cache(groups=...)` — materialize expensive preprocessing to disk with automatic invalidation
- Tar workflows:
  - sample parsing from shard members
  - optional sidecar merge via `.join([...])`. In this way, you can store different modalities in separate tars and join them on the fly via member naming conventions.
- JSONL workflows:
  - optional linked-tar reference resolution via `.resolve_refs([...])`. In this way, you can use JSONL to store data like conversations and reference external image data in tar shards.
  - optional spill sharding via `from_jsonl(..., group_key=..., num_shards=...)` for bounded-memory preprocessing and better tar locality.
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
    Dataset.from_tars("/data/shards/shard_{000000..000006}.tar", resample=True)
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
    Dataset.from_tars("/data/images/shard_{000000..000006}.tar", resample=True)
    .join([
      ("depth", lambda s: s.replace("image", "depth))
    ])
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
    Dataset.from_jsonl(
        "/data/meta/train.jsonl",
        resample=True,
        group_key="image",
        num_shards=64,
    )
    .resolve_refs([("image", "/data")])  # tar://... URI base dir
    .map(lambda s: s)
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

## Caching

`.cache()` materializes all upstream stages into tar files on disk so that expensive preprocessing (decoding, tokenization, augmentation, etc.) runs once and is reused on every subsequent iteration.

### Basic usage

```python
ds = (
    Dataset.from_tars("shards/shard_{000000..000099}.tar")
    .map(expensive_preprocess)   # runs only on first iteration
    .cache()                     # <-- cache boundary
    .shuffle(buffer_size=4096)   # runs every iteration
)

for sample in ds:  # 1st iter: builds cache
    train_step(sample)

for sample in ds:  # 2nd iter: reads from cache, skips expensive_preprocess
    train_step(sample)
```

Cache tars are written to a `.cache/` directory next to the source shards.

### How invalidation works

Each `.map()` / `.assemble()` stage is fingerprinted from its bytecode, defaults, and closure state. A **plan fingerprint** (SHA-256 of all pre-cache stage fingerprints + the groups spec) is stored in the manifest. When any pre-cache stage changes, the fingerprint changes and the cache is rebuilt automatically.

### Field grouping

By default all fields go into one tar per shard. Pass `groups` to split fields into separate tars — useful when only a subset of fields change between experiments:

```python
ds = (
    Dataset.from_tars(shards)
    .map(preprocess)
    .cache(groups=[["image"], ["label"]])
    # "image", "label" each get their own tar; any remaining fields
    # (e.g. "tag") become singleton groups automatically.
)
```

Only the group tars whose content signatures changed are rewritten.

### Post-cache stages

Stages appended **after** `.cache()` run on every iteration, reading from the cached tars:

```python
ds = (
    Dataset.from_tars(shards)
    .map(tokenize)       # cached
    .cache()
    .map(random_augment) # NOT cached — runs every time
    .shuffle(buffer_size=2048)
)
```

### Distributed / tensor-parallel awareness

When using a `DataLoadMesh` with model-parallel dimensions (e.g. tensor parallelism), multiple ranks receive the same shards. The cache layer automatically elects one **cache leader** per model-parallel group (the rank whose non-DP local ranks are all 0) to build the cache. All other co-members wait for the leader to finish and then read directly from the cached tars, avoiding redundant computation.

```python
from mvp_dataset import Dataset
from mvp_dataset.core.types import RuntimeContext

ctx = RuntimeContext.from_runtime(
    device_mesh=mesh,
    dp_dims=("replicate", "shard"),  # TP dim excluded → co-members share shards
)

ds = (
    Dataset.from_tars(shards, context=ctx)
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
  - `from_jsonl(..., group_key=..., num_shards=..., spill_buckets=..., output_dir=...)`

## Performance notes

- Iterator pipeline is lazy; data transforms are applied on demand.
- Shuffle uses bounded buffer memory (`buffer_size`, optional `initial`).
- JSONL reference resolution uses LRU tar-handle caching.
- `from_jsonl(...)` can pre-materialize balanced local JSONL shards so workers shard by file instead of holding the full JSONL in memory.
- Recommended high-throughput pattern with PyTorch:
  1. worker micro-batch (`batch_size` on `TorchLoader` + identity collate)
  2. `.unbatch()`
  3. loader-side `.shuffle(...)`
  4. loader-side `.batch(...)`
