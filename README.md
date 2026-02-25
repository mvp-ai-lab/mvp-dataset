# MVP Dataset

A minimal, high-performance data loading library for multimodal training pipelines.

`mvp-dataset` focuses on a small API surface, deterministic behavior, and practical throughput for local shard-based datasets.

## Why this project

- Minimal: only a few core abstractions (`Dataset`, `TorchLoader`, `RuntimeContext`)
- Fast: lazy iterator stages, bounded-memory shuffle, tar sidecar join, and loader-side pipeline composition
- Deterministic: seed-aware sharding and shuffle behavior across distributed + worker processes
- Extensible: source-level operations for TAR and JSONL-with-TAR-reference workflows

## Features

- `Dataset.from_source(...)` for local `.tar` or `.jsonl` shard inputs
- Chainable pipeline ops:
  - `.map(...)`
  - `.shuffle(buffer_size, initial=...)`
  - `.batch(batch_size, drop_last=..., collate_fn=...)`
  - `.unbatch()`
- Tar workflows:
  - sample parsing from shard members
  - optional sidecar merge via `.join([...])`. In this way, you can store different modalities in separate tars and join them on the fly via member naming conventions.
- JSONL workflows:
  - optional `tar://` reference resolution via `.resolve_refs([...])`. In this way, you can use JSONL to store data like conversations and reference external image data in tar shards.
  - optional grouping via `.group_by(field)` for faster reference resolution and TAR joining.
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
    Dataset.from_source("/data/shards/shard_{000000..000006}.tar", resample=True)
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
    Dataset.from_source("/data/images/shard_{000000..000006}.tar", resample=True)
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
    Dataset.from_source("/data/meta/train.jsonl", resample=True)
    .group_by("image")
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
```

Example:

```text
tar://examples/data/tars/shard_000000.tar#image_001.jpg
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

## Performance notes

- Iterator pipeline is lazy; data transforms are applied on demand.
- Shuffle uses bounded buffer memory (`buffer_size`, optional `initial`).
- JSONL reference resolution uses LRU tar-handle caching.
- Recommended high-throughput pattern with PyTorch:
  1. worker micro-batch (`batch_size` on `TorchLoader` + identity collate)
  2. `.unbatch()`
  3. loader-side `.shuffle(...)`
  4. loader-side `.batch(...)`
