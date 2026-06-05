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
