# TorchLoader

`TorchLoader` wraps PyTorch `DataLoader` and adds resume-aware iterator state.

```python
from mvp_dataset import Dataset, TorchLoader

dataset = Dataset.from_source("tar", "/data/train/shard_{000000..000127}.tar")
loader = TorchLoader(dataset, num_workers=8, batch_size=32, collate_fn=lambda batch: batch)
```

PyTorch must be installed separately.

## Constructor

```python
TorchLoader(
    dataset,
    *,
    num_workers=0,
    batch_size=None,
    pin_memory=False,
    persistent_workers=False,
    prefetch_factor=2,
    multiprocessing_context=None,
    collate_fn=None,
    drop_last=False,
    **loader_kwargs,
)
```

Arguments:

- `dataset`: upstream iterable dataset.
- `num_workers`: PyTorch worker process count.
- `batch_size`: worker-side batch size. `None` means sample mode.
- `pin_memory`: forwarded to PyTorch DataLoader.
- `persistent_workers`: forwarded when workers are used.
- `prefetch_factor`: forwarded when workers are used.
- `multiprocessing_context`: optional context such as `"spawn"` or `"fork"`.
- `collate_fn`: optional worker-side collate function.
- `drop_last`: drops incomplete worker batches when `batch_size` is set.
- `loader_kwargs`: additional PyTorch DataLoader keyword arguments.

## Loader-Side Stages

Loader-side stages run after worker outputs are merged in the main process.

```python
loader = (
    TorchLoader(dataset, num_workers=8, batch_size=32, collate_fn=lambda x: x)
    .unbatch()
    .shuffle(buffer_size=8192)
    .batch(64)
)
```

Available stages:

- `.unbatch()`
- `.shuffle(buffer_size, initial=None, seed=None)`
- `.assemble(factory, drop_last=False)`
- `.batch(batch_size, drop_last=False, collate_fn=None)`
- `.balance(drop_last=True, dummy_factory=None, buffer_size=32, chunk_size=16, ...)`

### Distributed Balance

`balance()` is a loader-side distributed stage for already materialized packs or
batches. It runs in the main process after worker outputs are merged, performs a
small all-rank control sync once per chunk, and uses sparse point-to-point
transfers only when a rank needs items from another rank.

```python
loader = (
    TorchLoader(dataset, num_workers=8)
    .assemble(make_pack)
    .balance(chunk_size=16, drop_last=True)
)
```

- Use it as a runtime fallback when an offline packing plan is unavailable.
- Prefer placing it after pack or batch construction, not at sample granularity.
- `drop_last=True` synchronously drops a final incomplete distributed step.
- `drop_last=False` requires `dummy_factory`, which must return a batch-compatible
  dummy item that the training step can mask out.
- `process_group` is used for control sync, Python object payloads, and CPU tensor
  payloads.
- `device_process_group` is used for non-CPU tensor payloads, such as CUDA/NPU
  tensors. Tensors are stripped out of the Python batch structure, coalesced by
  device and dtype, transferred as flat tensors, then restored on the receiving
  rank.
- Resume is not supported for this stage.

## Resume

Use the active loader iterator for checkpoints:

```python
loader = TorchLoader(dataset, num_workers=2, batch_size=4)
it = iter(loader)

batch = next(it)
state = it.state_dict()

resumed_loader = TorchLoader(dataset, num_workers=2, batch_size=4).load_state_dict(state)
resumed_it = iter(resumed_loader)
```

`TorchLoader.state_dict()` returns only an initial-state snapshot and emits a warning. It does not inspect an already-running iterator.

## Practical Pattern

For global sample shuffle with workers:

1. Use worker-side `batch_size` with `collate_fn=lambda x: x` for throughput.
2. Call `.unbatch()` after worker merge.
3. Call `.shuffle(...)` in the main process.
4. Assemble or batch final training inputs with loader-side stages.
