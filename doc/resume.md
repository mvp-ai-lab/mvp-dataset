# Resume and Checkpointing

Resume is based on two facts: the pipeline **identity** and the live iterator **state**.

## Dataset Resume

```python
dataset = Dataset.from_source("parquet", "/data/train.parquet")
it = iter(dataset)

for _ in range(100):
    next(it)

blob = it.state_dict()

resumed = Dataset.from_source("parquet", "/data/train.parquet")
resumed.load_state_dict(blob)
resumed_it = iter(resumed)
```

`load_state_dict` mutates the handle in place. The restored state is consumed by the next `iter(...)` only. A later `iter(...)` starts a new epoch.

## TorchLoader Resume

```python
loader = TorchLoader(dataset, num_workers=2, batch_size=8)
it = iter(loader)

batch = next(it)
blob = it.state_dict()

resumed = TorchLoader(dataset, num_workers=2, batch_size=8)
resumed.load_state_dict(blob)
resumed_it = iter(resumed)
```

## Checkpoint shape

```python
{
  "version": 3,
  "identity": { "runtime", "source", "stages", "loader" },
  "state": { "num_yielded", "source", "stages", "loader" } | None,
}
```

`Dataset.state_dict()` / `TorchLoader.state_dict()` save identity with `state=None` (no live iterator). Use the live iterator's `state_dict()` to checkpoint mid-epoch.

## Compatibility Checks

`load_state_dict(...)` compares identity once. A mismatch raises `ResumeStateError` with `[ResumeIdentityMismatch] path=...`.

`identity()` walks configuration: primitives, dataclasses, mappings, sequences, `partial`, and objects that implement `identity()`. A default object `repr` that embeds a memory address is rejected with `[UnstableResumeIdentity]`.

`UnsupportedResume` is raised when a source, stage, or loader configuration cannot provide resumable state.

## Training Loop Sketch

```python
loader = TorchLoader(dataset, num_workers=8, batch_size=32)
loader_iter = iter(loader)

while training:
    batch = next(loader_iter)
    train_step(batch)

    if should_checkpoint:
        save_checkpoint({"step": step, "data": loader_iter.state_dict()})
```

On restore:

```python
loader.load_state_dict(checkpoint["data"])
loader_iter = iter(loader)
```
