# Distributed Loading

`mvp-dataset` uses `RuntimeContext` to map data to runtime slots.

A slot is determined from:

- `rank`
- `world_size`
- `worker_id`
- `num_workers`
- optional data-parallel mesh information

## RuntimeContext

Most users can omit `context`; it is inferred from PyTorch distributed state, PyTorch DataLoader worker info, and environment variables.

```python
dataset = Dataset.from_source("tar", shards)
```

For explicit control:

```python
from mvp_dataset import RuntimeContext

context = RuntimeContext(rank=0, world_size=8, worker_id=0, num_workers=4, seed=123)
dataset = Dataset.from_source("parquet", shards, context=context)
```

## DataLoadMesh

`DataLoadMesh` separates data-parallel dimensions from model-parallel dimensions. This lets model-parallel co-members see the same data while data-parallel ranks receive different shards.

```python
from mvp_dataset import DataLoadMesh, RuntimeContext

mesh = DataLoadMesh(dp_rank=0, dp_size=8)
context = RuntimeContext(mesh=mesh, seed=123)
```

## Sharding Behavior

- Tar and JSONL schedule shard-level work across runtime slots.
- Parquet schedules chunk / row-group work across runtime slots.
- Lance maps logical positions to physical rows deterministically for each slot.

For finite datasets, each slot receives its assigned portion. With `resample=True`, the source repeats across deterministic rounds.

## Seeds

`RuntimeContext.seed` feeds deterministic source and stage shuffle behavior. Changing the seed changes fingerprints and shuffle order, and incompatible resume states are rejected.
