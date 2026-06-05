# Resume and Checkpointing

Resume is based on explicit iterator state.

## Dataset Resume

```python
dataset = Dataset.from_source("parquet", "/data/train.parquet")
it = iter(dataset)

for _ in range(100):
    next(it)

state = it.state_dict()

resumed_dataset = Dataset.from_source("parquet", "/data/train.parquet").load_state_dict(state)
resumed_it = iter(resumed_dataset)
```

The resumed iterator continues from the next item that the original iterator would have produced.

## TorchLoader Resume

```python
loader = TorchLoader(dataset, num_workers=2, batch_size=8)
it = iter(loader)

batch = next(it)
state = it.state_dict()

resumed_loader = TorchLoader(dataset, num_workers=2, batch_size=8).load_state_dict(state)
resumed_it = iter(resumed_loader)
```

`TorchLoader` stores pending prefetched outputs in the checkpoint state so resumed output matches the original continued stream.

## Initial-State Convenience APIs

These APIs exist, but are not intended for active checkpointing:

```python
initial_dataset_state = dataset.state_dict()
initial_loader_state = loader.state_dict()
```

They create fresh initial iterator states and emit warnings. In a training loop, store the iterator and call `iterator.state_dict()`.

## Compatibility Checks

Resume state stores fingerprints for:

- resume schema version
- runtime context
- source configuration and source metadata
- dataset stages
- TorchLoader configuration and loader-side stages

`load_state_dict(...)` raises `ResumeStateError` if the saved state does not match the new pipeline.

`UnsupportedResume` is raised when a source, stage, or loader configuration cannot provide resumable state.

## Training Loop Sketch

```python
loader = TorchLoader(dataset, num_workers=8, batch_size=32)
loader_iter = iter(loader)

while training:
    batch = next(loader_iter)
    train_step(batch)

    if should_checkpoint:
        checkpoint = {
            "step": step,
            "loader": loader_iter.state_dict(),
        }
        save_checkpoint(checkpoint)
```

On restore:

```python
loader = TorchLoader(dataset, num_workers=8, batch_size=32).load_state_dict(checkpoint["loader"])
loader_iter = iter(loader)
```
