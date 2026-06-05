# Dataset API

Import the public API from the package root:

```python
from mvp_dataset import Dataset, RuntimeContext, ResumeStateError, UnsupportedResume
```

## `Dataset.from_source(...)`

```python
Dataset.from_source(source_kind, *args, **kwargs) -> Dataset
```

Creates a dataset from a supported source. `source_kind` must be one of:

- `"tar"`
- `"jsonl"`
- `"parquet"`
- `"lance"`

All source constructors accept these common arguments:

- `context`: optional `RuntimeContext`; inferred from runtime when omitted.
- `resample`: repeat the source indefinitely across rounds when `True`.

Source-specific arguments are documented in [Source Formats](source-formats.md).

## Stages

All dataset stages are lazy and immutable. Calling a stage returns a new dataset.

### `map(fn)`

```python
dataset = dataset.map(fn)
```

Applies `fn(sample)` to each upstream sample.

### `select(fields)`

```python
dataset = dataset.select(["id", "text"])
```

Keeps selected keys from dictionary samples. Internal metadata keys such as `__key__` are preserved.

### `shuffle(buffer_size, initial=None)`

```python
dataset = dataset.shuffle(buffer_size=4096, initial=None)
```

Applies deterministic bounded-buffer sample shuffle after the source and previous stages.

- `buffer_size` must be positive.
- `initial` defaults to `buffer_size`.
- The stage stores its buffer and RNG state for resume.

### `assemble(factory, drop_last=False)`

```python
dataset = dataset.assemble(make_assembler, drop_last=False)
```

Builds a stateful assembler from `factory(RuntimeContext)` and runs it over the stream. The assembler must implement `StatefulAssembler` for resume support.

### `batch(batch_size, drop_last=False, collate_fn=None)`

```python
dataset = dataset.batch(32, collate_fn=lambda samples: samples)
```

Groups samples into batches. If `collate_fn` is provided, it receives the list of samples and returns the yielded batch object.

### `unbatch()`

```python
dataset = dataset.unbatch()
```

Expands list, tuple, or dictionary-style batches back into samples.

## Iteration

```python
for sample in dataset:
    consume(sample)
```

Each `iter(dataset)` call creates a fresh `DatasetIterator`. A second iterator starts from the beginning unless the dataset was created with `load_state_dict(...)`.

## State APIs

```python
it = iter(dataset)
state = it.state_dict()
resumed = Dataset.from_source("tar", shards).load_state_dict(state)
```

- Use `iterator.state_dict()` for active checkpoints.
- `Dataset.state_dict()` creates a fresh initial iterator state and emits a warning.
- `Dataset.load_state_dict(state)` validates runtime and pipeline fingerprints and returns a new dataset with pending resume state.
