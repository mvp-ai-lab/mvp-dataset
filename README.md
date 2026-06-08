<p align="center">
  <img src="doc/assets/logo.svg" alt="MVP Dataset logo" width="112">
</p>

<h1 align="center">MVP Dataset</h1>

<p align="center">
  <a href="https://github.com/mvp-ai-lab/mvp-dataset/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/mvp-ai-lab/mvp-dataset?style=flat-square"></a>
  <img alt="Version" src="https://img.shields.io/badge/version-0.1.0-111827?style=flat-square">
  <img alt="Python" src="https://img.shields.io/badge/python-3.12-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="Sources" src="https://img.shields.io/badge/sources-tar%20%7C%20jsonl%20%7C%20parquet%20%7C%20lance-F5C86B?style=flat-square">
</p>

<p align="center">
  <b>Unified data loading for multimodal training.</b><br>
  Chain immutable datasets, shard deterministically, checkpoint iterator state, and scale through PyTorch workers when needed.
</p>

`mvp-dataset` provides a chainable `Dataset` abstraction, deterministic distributed sharding, resumable iterators, and optional PyTorch `DataLoader` integration through `TorchLoader`.

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

Python `>=3.12` is required.

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
