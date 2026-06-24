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

## Subset Operations

`split` and `sample` derive new datasets that cover or sample the source. They
are supported on all sources except `mixed`. Each derived dataset reads only its
own data.

### `split(fractions)`

```python
train, val = dataset.split([0.8, 0.2])          # [8, 2] is equivalent
train, val, test = dataset.split([0.8, 0.1, 0.1])
```

Partitions the dataset into disjoint subsets that together cover all data.

- `fractions` are normalized internally, so `[8, 2]` and `[0.8, 0.2]` are the same.
- Returns a tuple of datasets, one per fraction, in input order.
- The partition follows source order. Source-level shuffle can be used before
  splitting when random membership is needed.

### `sample(fraction, *, seed=0)`

```python
subset = dataset.sample(0.1, seed=0)
```

Returns a dataset over a seeded random subset.

- `fraction` must be in `(0, 1]` — sampling is without replacement and cannot oversample.
- Reproducible for a given `seed`.

### Granularity

Selection granularity depends on what the source can read efficiently:

- **lance** — row-exact. `split([0.8, 0.2])` yields exactly 80% / 20% of rows,
  and `sample(f)` keeps exactly `round(f * num_rows)` rows, reading only those rows.
- **parquet** — chunk-level, weighted by row count.
- **tar** / **jsonl** — whole-shard level. Fractions are approximate and bounded
  by the shard/file count; a single tar or `.jsonl` file is one shard and cannot
  be split internally.

A subset must keep at least `context.total_slots` units (shards/chunks) for
unit-based sources, otherwise a `ValueError` is raised. `split`/`sample` cannot
be stacked on an already-subset `lance` dataset; apply them on the base dataset first.

## Terminal Operations

### `consume(factory)`

```python
result = dataset.consume(lambda context: consumer)
```

Eagerly consumes the pipeline and returns `consumer.finish()`. The consumer is
created with the same resolved `RuntimeContext` used by the dataset iterator.

- `consumer.push(item)` receives each pipeline output.
- Returning `False` from `push()` stops consumption early.
- Returning `True` or `None` continues consumption.
- Exceptions from the upstream pipeline or consumer propagate, and `finish()` is
  not called after a failed `push()`.

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
