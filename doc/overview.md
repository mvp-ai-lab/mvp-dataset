# Overview

`mvp-dataset` provides a deterministic iterable data pipeline for multimodal training workloads.

The core abstraction is `Dataset`: an immutable configuration object that becomes a concrete iterator only when `iter(dataset)` is called. Runtime cursor state belongs to that iterator, which makes checkpointing explicit and avoids hidden active-iterator state on the dataset object.

## Design Goals

- Keep dataset construction declarative and immutable.
- Make distributed sharding deterministic and inspectable.
- Keep source readers local and format-specific.
- Make resume compatibility explicit through runtime and pipeline fingerprints.
- Preserve PyTorch interoperability without requiring PyTorch for core dataset usage.

## Main Concepts

### Dataset

A `Dataset` is a lazy pipeline made from one source and zero or more stages. Stages return new dataset objects:

```python
dataset = (
    Dataset.from_source("parquet", "/data/train.parquet", columns=["text", "label"])
    .select(["text", "label"])
    .map(tokenize)
    .batch(32)
)
```

### DatasetIterator

`iter(dataset)` creates a materialized iterator. It owns source cursor state and stage state. Use this iterator for in-progress checkpoints:

```python
it = iter(dataset)
next(it)
state = it.state_dict()
```

### RuntimeContext

`RuntimeContext` carries rank, world size, worker id, worker count, epoch, seed, and optional data-load mesh information. It is used to assign source items to runtime slots and derive deterministic shuffle seeds.

### TorchLoader

`TorchLoader` wraps PyTorch `DataLoader` and adds resume-aware post-worker stages. It is useful when training code wants PyTorch worker processes plus checkpointable loader-side transformations.
