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

`Dataset.state_dict()` / `TorchLoader.state_dict()` save the active iterator created by the last `iter()`. If there is no active iterator, or it is exhausted, `state` is `None`.

## Snapshot ownership

`TorchLoader` keeps the native version-3 checkpoint shape and worker messages.
Each save owns its live worker snapshots through the returned payload; the
running merge iterator does not cache these graphs between saves. A caller or
asynchronous serializer must keep that payload alive until serialization ends.
Pending outputs are copied into checkpoint lists without advancing their order.
Finished workers are different: the iterator retains their final states because
they cannot answer later snapshot requests. Consecutive saves use distinct
request generations, and a failed capture releases the request in `finally`.

On resume, worker bootstrap mappings are temporary. After process creation the
parent releases its copy; each worker releases its mapping and local reference
only after `load_live_state` and `iter(dataset)` succeed. With zero workers this
handoff happens lazily on first read (or snapshot), not at loader construction.
The loader's pending restore is still consumed once, and caller-owned checkpoint
mappings are never cleared. Source/stage objects continue owning the state they
actually need for execution. These lifetime changes do not encode states as
opaque bytes, change worker IPC, or discard terminal cursors.

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
