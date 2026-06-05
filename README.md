# MVP Dataset

`mvp-dataset` is a small, resume-aware data loading library for multimodal training. It provides a chainable `Dataset` abstraction, deterministic distributed sharding, resumable iterators, and optional PyTorch `DataLoader` integration through `TorchLoader`.

The project is designed for training code that needs predictable data order, checkpointable input pipelines, and local source formats commonly used in large-scale multimodal datasets.

## Highlights

- Immutable, chainable `Dataset` pipelines.
- Source support for tar, JSONL, Parquet, and Lance.
- Deterministic sharding across ranks, data-parallel meshes, and PyTorch worker processes.
- Resumable source and stage iterators through `iterator.state_dict()` / `load_state_dict()`.
- Source-level shuffle modes where the source can support them efficiently.
- Post-worker `TorchLoader` stages for global shuffle, assembly, batching, and resume.
- No hard PyTorch dependency for core dataset usage; install PyTorch separately when using `TorchLoader`.

## Installation

```bash
uv pip install "git+https://github.com/mvp-ai-lab/mvp-dataset.git"

# Development install
uv pip install -e .
```

Python `>=3.12,<3.13` is required.

## Quick Start

```python
from mvp_dataset import Dataset, TorchLoader


def preprocess(sample: dict[str, object]) -> dict[str, object]:
    return sample


dataset = (
    Dataset.from_source("tar", "/data/train/shard_{000000..000127}.tar", resample=True)
    .shuffle(buffer_size=4096)
    .map(preprocess)
)

loader = TorchLoader(
    dataset,
    num_workers=8,
    batch_size=32,
    collate_fn=lambda batch: batch,
    pin_memory=True,
)

for batch in loader:
    train_step(batch)
```

## Public API

Top-level exports:

- `Dataset`
- `TorchLoader`
- `RuntimeContext`
- `DataLoadMesh`
- `UnsupportedResume`
- `ResumeStateError`
- logging helpers: `set_logger`, `get_logger`, `reset_logger`, `set_log_level`, `get_log_level`, `reset_log_level`

Construct datasets through one entrypoint:

```python
Dataset.from_source(source_kind, ...)
```

Supported `source_kind` values:

- `"tar"`
- `"jsonl"`
- `"parquet"`
- `"lance"`

Dataset stages:

- `.map(fn)`
- `.select(fields)`
- `.shuffle(buffer_size, initial=None)`
- `.assemble(factory, drop_last=False)`
- `.batch(batch_size, drop_last=False, collate_fn=None)`
- `.unbatch()`

TorchLoader stages:

- `.unbatch()`
- `.shuffle(buffer_size, initial=None, seed=None)`
- `.assemble(factory, drop_last=False)`
- `.batch(batch_size, drop_last=False, collate_fn=None)`

## Resume

Checkpoint the active iterator, not the dataset object itself:

```python
dataset = Dataset.from_source("lance", "/data/train.lance")
it = iter(dataset)

sample = next(it)
state = it.state_dict()

resumed_dataset = Dataset.from_source("lance", "/data/train.lance").load_state_dict(state)
resumed_it = iter(resumed_dataset)
```

`Dataset.state_dict()` and `TorchLoader.state_dict()` exist only as initial-state convenience APIs. For in-progress training, call `state_dict()` on the iterator returned by `iter(dataset)` or `iter(loader)`.

## Documentation

Detailed docs live in [`doc/`](doc/):

- [Documentation Index](doc/index.md)
- [Overview](doc/overview.md)
- [Dataset API](doc/dataset-api.md)
- [Source Formats](doc/source-formats.md)
- [TorchLoader](doc/torch-loader.md)
- [Resume and Checkpointing](doc/resume.md)
- [Distributed Loading](doc/distributed.md)
- [Data Conventions](doc/data-conventions.md)

## Development

```bash
uv sync
uv run ruff check src tests
uv run pytest -q
```

The package uses Python 3.12, Ruff, and pytest.
