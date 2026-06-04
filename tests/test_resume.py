from __future__ import annotations

import importlib.util

import pytest

from mvp_dataset import (
    Dataset,
    ResumeStateError,
    RuntimeContext,
    TorchLoader,
    UnsupportedResume,
)
from mvp_dataset.core.resume import RESUME_STATE_VERSION

from .helpers import (
    build_records,
    normalize_sample,
    write_jsonl_file,
    write_lance_dataset,
    write_lance_table,
    write_parquet_file,
    write_tar_shards,
)


def _jsonl_dataset(tmp_path, *, seed: int = 0) -> Dataset:
    path = write_jsonl_file(tmp_path, build_records())
    return Dataset.from_source("jsonl", shards=path, context=RuntimeContext(seed=seed))


def _consume(stream, count: int) -> list[dict[str, object]]:
    return [normalize_sample(next(stream)) for _ in range(count)]


def _add_marker(sample: dict[str, object]) -> dict[str, object]:
    return {**sample, "marker": f"marked-{sample['id']}"}


def _normalize_stage_sample(sample: dict[str, object]) -> dict[str, object]:
    marker = sample["marker"]
    if isinstance(marker, (bytes, bytearray)):
        marker = marker.decode("utf-8")
    sample_id = sample["id"]
    if isinstance(sample_id, (bytes, bytearray)):
        sample_id = sample_id.decode("utf-8")
    return {"id": sample_id, "marker": marker}


def _normalize_batch(batch: list[dict[str, object]]) -> list[dict[str, object]]:
    return [normalize_sample(sample) for sample in batch]


def _decode_value(value: object) -> object:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return value


def _as_list(value: object) -> list[object]:
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _normalize_loader_output(output: object) -> object:
    if isinstance(output, dict) and isinstance(output.get("id"), (list, tuple)):
        ids = _as_list(output["id"])
        texts = _as_list(output["text"])
        values = _as_list(output["value"])
        return [
            {
                "id": _decode_value(sample_id),
                "text": _decode_value(text),
                "value": int(_decode_value(value)),
            }
            for sample_id, text, value in zip(ids, texts, values, strict=True)
        ]
    if isinstance(output, list):
        if output and isinstance(output[0], dict):
            return [normalize_sample(sample) for sample in output]
        return [_decode_value(item) for item in output]
    if isinstance(output, dict):
        return normalize_sample(output)
    return output


def _consume_loader_outputs(stream, count: int) -> list[object]:
    return [_normalize_loader_output(next(stream)) for _ in range(count)]


def _resume_torch_loader(dataset: Dataset, **kwargs) -> TorchLoader:
    if kwargs.get("num_workers", 0) > 0:
        kwargs.setdefault("multiprocessing_context", "forkserver")
    return TorchLoader(dataset, **kwargs)


def _collate_ids(batch: list[dict[str, object]]) -> list[object]:
    return [sample["id"] for sample in batch]


def _collate_columns(batch: list[dict[str, object]]) -> dict[str, list[object]]:
    return {
        "id": [sample["id"] for sample in batch],
        "text": [sample["text"] for sample in batch],
        "value": [sample["value"] for sample in batch],
    }


_ASSEMBLER_FINGERPRINT_VERSION = "v1"


class _PairOutputAssembler:
    def __init__(self) -> None:
        self.pending: list[str] = []

    def push(self, sample: dict[str, object]) -> list[dict[str, object]]:
        sample_id = sample["id"]
        if isinstance(sample_id, (bytes, bytearray)):
            sample_id = sample_id.decode("utf-8")
        self.pending.append(str(sample_id))
        if len(self.pending) < 2:
            return []
        left, right = self.pending
        self.pending = []
        pair = f"{left}+{right}"
        return [{"pair": pair, "slot": 0}, {"pair": pair, "slot": 1}]

    def finish(self, *, drop_last: bool = False) -> list[dict[str, object]]:
        if drop_last or not self.pending:
            return []
        tail = self.pending.pop()
        return [{"pair": tail, "slot": 0}]

    def state_dict(self) -> dict[str, object]:
        return {"pending": list(self.pending)}

    def load_state_dict(self, state: dict[str, object]) -> None:
        pending = state.get("pending")
        if not isinstance(pending, list):
            raise ResumeStateError("[InvalidResumeState] assembler pending must be a list")
        self.pending = [str(item) for item in pending]

    def fingerprint(self) -> str:
        return f"pair-output-assembler:{_ASSEMBLER_FINGERPRINT_VERSION}"


class _NonStatefulAssembler:
    def push(self, sample: dict[str, object]) -> list[dict[str, object]]:
        return [sample]

    def finish(self, *, drop_last: bool = False) -> list[dict[str, object]]:
        return []


def _build_pair_output_assembler(_context: RuntimeContext) -> _PairOutputAssembler:
    return _PairOutputAssembler()


def _build_non_stateful_assembler(_context: RuntimeContext) -> _NonStatefulAssembler:
    return _NonStatefulAssembler()


def _compatible_state(dataset: Dataset) -> dict[str, object]:
    return {
        "version": RESUME_STATE_VERSION,
        "runtime_fingerprint": dataset.context.fingerprint(),
        "pipeline_fingerprint": dataset._pipeline_fingerprint(),
        "num_yielded": 0,
        "source": {
            "kind": dataset._source_kind,
            "fingerprint": "source-fingerprint",
            "state": {},
        },
        "stages": [],
    }


def test_iter_rejects_unsupported_assemble_stage_with_clear_error(tmp_path) -> None:
    pytest.importorskip("lance")

    path = write_lance_dataset(tmp_path, build_records())
    dataset = Dataset.from_source("lance", shards=path).assemble(_build_non_stateful_assembler)

    with pytest.raises(UnsupportedResume, match=r"\[UnsupportedResume\] stage kind='assemble'"):
        iter(dataset)


def test_load_state_dict_rejects_unknown_schema_version(tmp_path) -> None:
    dataset = _jsonl_dataset(tmp_path)

    with pytest.raises(ResumeStateError, match=r"\[InvalidResumeStateVersion\]"):
        dataset.load_state_dict({"version": RESUME_STATE_VERSION + 1})


def test_load_state_dict_rejects_runtime_fingerprint_mismatch(tmp_path) -> None:
    dataset = _jsonl_dataset(tmp_path, seed=1)
    state = _compatible_state(dataset)
    changed_runtime = Dataset.from_source(
        "jsonl",
        shards=dataset._source,
        context=RuntimeContext(seed=2),
    )

    with pytest.raises(ResumeStateError, match=r"\[ResumeRuntimeMismatch\]"):
        changed_runtime.load_state_dict(state)


def test_load_state_dict_rejects_pipeline_fingerprint_mismatch(tmp_path) -> None:
    dataset = _jsonl_dataset(tmp_path)
    state = _compatible_state(dataset)
    changed_pipeline = dataset.select(["id"])

    with pytest.raises(ResumeStateError, match=r"\[ResumePipelineMismatch\]"):
        changed_pipeline.load_state_dict(state)


def test_load_state_dict_attaches_validated_state_to_new_dataset(tmp_path) -> None:
    dataset = _jsonl_dataset(tmp_path)
    state = _compatible_state(dataset)

    with pytest.warns(UserWarning, match="Dataset.load_state_dict"):
        resumed = dataset.load_state_dict(state)

    assert resumed is not dataset
    assert resumed._resume_state == state
    assert dataset._resume_state is None


def test_tar_source_is_stateful(tmp_path) -> None:
    dataset = Dataset.from_source("tars", shards=write_tar_shards(tmp_path, build_records(), num_shards=1))

    assert [normalize_sample(sample) for sample in dataset] == build_records()


def test_runtime_context_fingerprint_is_stable_and_seed_sensitive() -> None:
    first = RuntimeContext(seed=1).fingerprint()
    second = RuntimeContext(seed=1).fingerprint()
    different_seed = RuntimeContext(seed=2).fingerprint()

    assert first == second
    assert first != different_seed


@pytest.mark.parametrize("total_rows", [1, 2, 3, 7, 8, 9, 17, 31, 33, 100])
def test_lance_global_shuffle_permute_index_is_bijective(total_rows: int) -> None:
    pytest.importorskip("lance")

    from mvp_dataset.sources.lance.utils.shuffle import permute_index

    observed = [permute_index(position, total_rows=total_rows, seed=41) for position in range(total_rows)]

    assert sorted(observed) == list(range(total_rows))


def test_lance_pipeline_fingerprint_is_stable_for_same_source(tmp_path) -> None:
    pytest.importorskip("lance")

    path = write_lance_dataset(tmp_path, build_records())

    first = Dataset.from_source("lance", shards=path)._pipeline_fingerprint()
    second = Dataset.from_source("lance", shards=path)._pipeline_fingerprint()

    assert first == second


def test_lance_pipeline_fingerprint_changes_with_columns(tmp_path) -> None:
    pytest.importorskip("lance")

    path = write_lance_dataset(tmp_path, build_records())

    all_columns = Dataset.from_source("lance", shards=path)
    selected_columns = Dataset.from_source("lance", shards=path, columns=["id"])

    assert all_columns._pipeline_fingerprint() != selected_columns._pipeline_fingerprint()


def test_lance_pipeline_fingerprint_changes_with_batch_size(tmp_path) -> None:
    pytest.importorskip("lance")

    path = write_lance_dataset(tmp_path, build_records())

    small_batch = Dataset.from_source("lance", shards=path, batch_size=2)
    large_batch = Dataset.from_source("lance", shards=path, batch_size=4)

    assert small_batch._pipeline_fingerprint() != large_batch._pipeline_fingerprint()


def test_lance_pipeline_fingerprint_changes_with_shuffle_mode(tmp_path) -> None:
    pytest.importorskip("lance")

    path = write_lance_dataset(tmp_path, build_records())
    no_shuffle = Dataset.from_source("lance", shards=path, shuffle_mode="none")
    global_shuffle = Dataset.from_source("lance", shards=path, shuffle_mode="global")

    assert no_shuffle._pipeline_fingerprint() != global_shuffle._pipeline_fingerprint()


def test_lance_pipeline_fingerprint_changes_with_ref_column_config(tmp_path) -> None:
    pytest.importorskip("lance")

    meta_path = write_lance_table(tmp_path, "meta.lance", [{"id": "sample-0", "image": "image-0"}])
    image_path = write_lance_table(tmp_path, "image.lance", [{"image_id": "image-0", "image": b"zero"}])
    label_path = write_lance_table(tmp_path, "label.lance", [{"label_id": "image-0", "label": "cat"}])

    image_ref = Dataset.from_source(
        "lance",
        shards=meta_path,
        ref_columns={"image": {"uri": image_path, "key_column": "image_id", "value_column": "image"}},
    )
    label_ref = Dataset.from_source(
        "lance",
        shards=meta_path,
        ref_columns={"image": {"uri": label_path, "key_column": "label_id", "value_column": "label"}},
    )

    assert image_ref._pipeline_fingerprint() != label_ref._pipeline_fingerprint()


@pytest.mark.parametrize("shuffle_mode", ["none", "global", "fragment_aware"])
@pytest.mark.parametrize("checkpoint_after", [0, 1, 3, 7])
def test_lance_source_resume_matches_continued_iterator(
    tmp_path,
    shuffle_mode: str,
    checkpoint_after: int,
) -> None:
    pytest.importorskip("lance")

    records = build_records(count=7)
    path = write_lance_dataset(tmp_path, records, max_rows_per_file=2)
    dataset = Dataset.from_source(
        "lance",
        shards=path,
        context=RuntimeContext(seed=13),
        batch_size=2,
        shuffle_mode=shuffle_mode,
    )
    iterator = iter(dataset)

    consumed = _consume(iterator, checkpoint_after)
    state = iterator.state_dict()
    continued = [normalize_sample(sample) for sample in iterator]
    expected = [normalize_sample(sample) for sample in dataset]

    with pytest.warns(UserWarning, match="Dataset.load_state_dict"):
        resumed_dataset = Dataset.from_source(
            "lance",
            shards=path,
            context=RuntimeContext(seed=13),
            batch_size=2,
            shuffle_mode=shuffle_mode,
        ).load_state_dict(state)
    resumed = [normalize_sample(sample) for sample in resumed_dataset]

    assert consumed + continued == expected
    assert resumed == continued


@pytest.mark.parametrize("checkpoint_after", [0, 1, 4, 9])
def test_tar_source_resume_matches_continued_iterator(tmp_path, checkpoint_after: int) -> None:
    records = build_records(count=9)
    shards = write_tar_shards(tmp_path, records, num_shards=3)
    dataset = Dataset.from_source("tars", shards=shards, context=RuntimeContext(seed=23))
    iterator = iter(dataset)

    consumed = _consume(iterator, checkpoint_after)
    state = iterator.state_dict()
    continued = [normalize_sample(sample) for sample in iterator]
    expected = [normalize_sample(sample) for sample in dataset]

    with pytest.warns(UserWarning, match="Dataset.load_state_dict"):
        resumed_dataset = Dataset.from_source(
            "tars",
            shards=shards,
            context=RuntimeContext(seed=23),
        ).load_state_dict(state)
    resumed = [normalize_sample(sample) for sample in resumed_dataset]

    assert state["source"]["kind"] == "tars"
    assert "sample_index" in state["source"]["state"]
    assert consumed + continued == expected
    assert resumed == continued


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_torch_loader_plain_iterable_iteration_does_not_require_resume_state() -> None:
    assert list(TorchLoader([1, 2, 3], num_workers=0)) == [1, 2, 3]

    iterator = iter(TorchLoader([1, 2, 3], num_workers=0))
    assert next(iterator) == 1
    with pytest.raises(UnsupportedResume, match=r"\[UnsupportedResume\] TorchLoader dataset iterator"):
        iterator.state_dict()


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_torch_loader_multi_worker_resume_matches_continued_iterator(tmp_path) -> None:
    num_workers = 2
    prefetch_factor = 2
    records = build_records(count=8)
    shards = write_tar_shards(tmp_path, records, num_shards=4)
    context = RuntimeContext(seed=43)
    dataset = Dataset.from_source("tars", shards=shards, context=context)
    loader = _resume_torch_loader(dataset, num_workers=num_workers, batch_size=None, prefetch_factor=prefetch_factor)
    iterator = iter(loader)

    consumed = _consume_loader_outputs(iterator, 1)
    state = iterator.state_dict()
    continued = [_normalize_loader_output(sample) for sample in iterator]

    resumed_dataset = Dataset.from_source("tars", shards=shards, context=context)
    with pytest.warns(UserWarning, match="TorchLoader.load_state_dict"):
        resumed_loader = _resume_torch_loader(
            resumed_dataset,
            num_workers=num_workers,
            batch_size=None,
            prefetch_factor=prefetch_factor,
        ).load_state_dict(state)
    resumed = [_normalize_loader_output(sample) for sample in resumed_loader]

    assert set(state["workers"]).issubset({str(worker_id) for worker_id in range(max(1, num_workers))})
    assert "pending_outputs" in state
    assert resumed == continued
    assert len(consumed) == 1


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_torch_loader_single_process_resume_matches_continued_iterator(tmp_path) -> None:
    records = build_records(count=8)
    shards = write_tar_shards(tmp_path, records, num_shards=4)
    context = RuntimeContext(seed=45)
    dataset = Dataset.from_source("tars", shards=shards, context=context)
    loader = _resume_torch_loader(dataset, num_workers=0, batch_size=None)
    iterator = iter(loader)

    consumed = _consume_loader_outputs(iterator, 1)
    state = iterator.state_dict()
    continued = [_normalize_loader_output(sample) for sample in iterator]

    resumed_dataset = Dataset.from_source("tars", shards=shards, context=context)
    with pytest.warns(UserWarning, match="TorchLoader.load_state_dict"):
        resumed_loader = _resume_torch_loader(resumed_dataset, num_workers=0).load_state_dict(state)
    resumed = [_normalize_loader_output(sample) for sample in resumed_loader]

    assert resumed == continued
    assert len(consumed) == 1


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_torch_loader_prefetch_factor_one_resume_matches_continued_iterator(tmp_path) -> None:
    records = build_records(count=8)
    shards = write_tar_shards(tmp_path, records, num_shards=4)
    context = RuntimeContext(seed=46)
    dataset = Dataset.from_source("tars", shards=shards, context=context)
    loader = _resume_torch_loader(dataset, num_workers=2, batch_size=None, prefetch_factor=1)
    iterator = iter(loader)

    consumed = _consume_loader_outputs(iterator, 1)
    state = iterator.state_dict()
    continued = [_normalize_loader_output(sample) for sample in iterator]

    resumed_dataset = Dataset.from_source("tars", shards=shards, context=context)
    with pytest.warns(UserWarning, match="TorchLoader.load_state_dict"):
        resumed_loader = _resume_torch_loader(
            resumed_dataset,
            num_workers=2,
            batch_size=None,
            prefetch_factor=1,
        ).load_state_dict(state)
    resumed = [_normalize_loader_output(sample) for sample in resumed_loader]

    assert resumed == continued
    assert len(consumed) == 1


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
@pytest.mark.parametrize("collate_fn", [None, _normalize_batch])
def test_torch_loader_batch_resume_matches_continued_iterator(tmp_path, collate_fn) -> None:
    records = build_records(count=8)
    shards = write_tar_shards(tmp_path, records, num_shards=4)
    context = RuntimeContext(seed=47)
    dataset = Dataset.from_source("tars", shards=shards, context=context)
    loader = _resume_torch_loader(
        dataset,
        num_workers=2,
        batch_size=2,
        collate_fn=collate_fn,
        prefetch_factor=2,
    )
    iterator = iter(loader)

    consumed = _consume_loader_outputs(iterator, 1)
    state = iterator.state_dict()
    continued = [_normalize_loader_output(batch) for batch in iterator]

    resumed_dataset = Dataset.from_source("tars", shards=shards, context=context)
    with pytest.warns(UserWarning, match="TorchLoader.load_state_dict"):
        resumed_loader = _resume_torch_loader(
            resumed_dataset,
            num_workers=2,
            batch_size=2,
            collate_fn=collate_fn,
            prefetch_factor=2,
        ).load_state_dict(state)
    resumed = [_normalize_loader_output(batch) for batch in resumed_loader]

    assert resumed == continued
    assert len(consumed) == 1


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_torch_loader_custom_collate_resume_matches_continued_iterator(tmp_path) -> None:
    shards = write_tar_shards(tmp_path, build_records(count=8), num_shards=4)
    context = RuntimeContext(seed=49)
    dataset = Dataset.from_source("tars", shards=shards, context=context)
    loader = _resume_torch_loader(dataset, num_workers=2, batch_size=2, collate_fn=_collate_ids, prefetch_factor=2)
    iterator = iter(loader)

    consumed = _consume_loader_outputs(iterator, 1)
    state = iterator.state_dict()
    continued = [_normalize_loader_output(batch) for batch in iterator]

    resumed_dataset = Dataset.from_source("tars", shards=shards, context=context)
    with pytest.warns(UserWarning, match="TorchLoader.load_state_dict"):
        resumed_loader = _resume_torch_loader(
            resumed_dataset,
            num_workers=2,
            batch_size=2,
            collate_fn=_collate_ids,
            prefetch_factor=2,
        ).load_state_dict(state)
    resumed = [_normalize_loader_output(batch) for batch in resumed_loader]

    assert resumed == continued
    assert len(consumed) == 1


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_torch_loader_batch_resume_respects_drop_last(tmp_path) -> None:
    records = build_records(count=10)
    shards = write_tar_shards(tmp_path, records, num_shards=4)
    context = RuntimeContext(seed=51)
    dataset = Dataset.from_source("tars", shards=shards, context=context)
    loader = _resume_torch_loader(dataset, num_workers=2, batch_size=3, drop_last=True, collate_fn=_normalize_batch)
    iterator = iter(loader)

    consumed = _consume_loader_outputs(iterator, 1)
    state = iterator.state_dict()
    continued = [_normalize_loader_output(batch) for batch in iterator]

    resumed_dataset = Dataset.from_source("tars", shards=shards, context=context)
    with pytest.warns(UserWarning, match="TorchLoader.load_state_dict"):
        resumed_loader = _resume_torch_loader(
            resumed_dataset,
            num_workers=2,
            batch_size=3,
            drop_last=True,
            collate_fn=_normalize_batch,
        ).load_state_dict(state)
    resumed = [_normalize_loader_output(batch) for batch in resumed_loader]

    assert resumed == continued
    assert len(consumed) == 1


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_torch_loader_resume_supports_persistent_workers_and_pin_memory(tmp_path) -> None:
    records = build_records(count=8)
    shards = write_tar_shards(tmp_path, records, num_shards=4)
    context = RuntimeContext(seed=52)
    dataset = Dataset.from_source("tars", shards=shards, context=context)
    loader = _resume_torch_loader(
        dataset,
        num_workers=2,
        batch_size=2,
        collate_fn=_normalize_batch,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
    )
    iterator = iter(loader)

    consumed = _consume_loader_outputs(iterator, 1)
    state = iterator.state_dict()
    continued = [_normalize_loader_output(batch) for batch in iterator]

    resumed_dataset = Dataset.from_source("tars", shards=shards, context=context)
    with pytest.warns(UserWarning, match="TorchLoader.load_state_dict"):
        resumed_loader = _resume_torch_loader(
            resumed_dataset,
            num_workers=2,
            batch_size=2,
            collate_fn=_normalize_batch,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True,
        ).load_state_dict(state)
    resumed = [_normalize_loader_output(batch) for batch in resumed_loader]

    assert resumed == continued
    assert len(consumed) == 1


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_torch_loader_state_dict_returns_initial_state(tmp_path) -> None:
    shards = write_tar_shards(tmp_path, build_records(count=4), num_shards=2)
    dataset = Dataset.from_source("tars", shards=shards, context=RuntimeContext(seed=53))
    loader = _resume_torch_loader(dataset, num_workers=2, batch_size=2, collate_fn=_normalize_batch)

    with pytest.warns(UserWarning, match="TorchLoader.state_dict"):
        state = loader.state_dict()
    with pytest.warns(UserWarning, match="TorchLoader.load_state_dict"):
        resumed_loader = _resume_torch_loader(
            dataset,
            num_workers=2,
            batch_size=2,
            collate_fn=_normalize_batch,
        ).load_state_dict(state)

    assert state["workers"] == {}
    assert state["pending_outputs"] == {}
    assert [_normalize_loader_output(batch) for batch in resumed_loader] == [
        _normalize_loader_output(batch) for batch in loader
    ]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_torch_loader_state_dict_does_not_track_active_iterator(tmp_path) -> None:
    shards = write_tar_shards(tmp_path, build_records(count=4), num_shards=2)
    dataset = Dataset.from_source("tars", shards=shards, context=RuntimeContext(seed=54))
    loader = _resume_torch_loader(dataset, num_workers=0, batch_size=None)
    iterator = iter(loader)

    next(iterator)
    iterator_state = iterator.state_dict()
    with pytest.warns(UserWarning, match="TorchLoader.state_dict"):
        loader_state = loader.state_dict()

    assert iterator_state["num_yielded"] == 1
    assert loader_state["num_yielded"] == 0
    assert loader_state["workers"] == {}


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
@pytest.mark.parametrize(
    "changed_kwargs",
    [
        {"num_workers": 1},
        {"batch_size": 3},
        {"prefetch_factor": 2},
        {"collate_fn": _collate_ids},
    ],
)
def test_torch_loader_resume_rejects_loader_fingerprint_mismatch(tmp_path, changed_kwargs: dict[str, object]) -> None:
    shards = write_tar_shards(tmp_path, build_records(count=4), num_shards=2)
    dataset = Dataset.from_source("tars", shards=shards, context=RuntimeContext(seed=55))
    kwargs = {"num_workers": 2, "batch_size": 2, "collate_fn": _normalize_batch, "prefetch_factor": 1}
    with pytest.warns(UserWarning, match="TorchLoader.state_dict"):
        state = _resume_torch_loader(dataset, **kwargs).state_dict()
    kwargs.update(changed_kwargs)
    changed_loader = _resume_torch_loader(dataset, **kwargs)

    with pytest.raises(ResumeStateError, match=r"\[ResumeLoaderMismatch\]"):
        changed_loader.load_state_dict(state)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_torch_loader_side_batch_resume_matches_continued_iterator(tmp_path) -> None:
    shards = write_tar_shards(tmp_path, build_records(count=8), num_shards=4)
    context = RuntimeContext(seed=57)
    dataset = Dataset.from_source("tars", shards=shards, context=context)
    loader = _resume_torch_loader(dataset, num_workers=0, batch_size=None).batch(3, collate_fn=_normalize_batch)
    iterator = iter(loader)

    consumed = [_normalize_loader_output(next(iterator))]
    state = iterator.state_dict()
    continued = [_normalize_loader_output(batch) for batch in iterator]

    resumed_dataset = Dataset.from_source("tars", shards=shards, context=context)
    with pytest.warns(UserWarning, match="TorchLoader.load_state_dict"):
        resumed_loader = (
            _resume_torch_loader(resumed_dataset, num_workers=0, batch_size=None)
            .batch(3, collate_fn=_normalize_batch)
            .load_state_dict(state)
        )
    resumed = [_normalize_loader_output(batch) for batch in resumed_loader]

    assert state["stages"][0]["kind"] == "batch"
    assert resumed == continued
    assert len(consumed) == 1


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_torch_loader_side_unbatch_resume_matches_continued_iterator(tmp_path) -> None:
    shards = write_tar_shards(tmp_path, build_records(count=8), num_shards=4)
    context = RuntimeContext(seed=58)
    dataset = Dataset.from_source("tars", shards=shards, context=context)
    loader = _resume_torch_loader(dataset, num_workers=0, batch_size=2, collate_fn=_collate_columns).unbatch()
    iterator = iter(loader)

    consumed = _consume_loader_outputs(iterator, 3)
    state = iterator.state_dict()
    continued = [_normalize_loader_output(sample) for sample in iterator]

    resumed_dataset = Dataset.from_source("tars", shards=shards, context=context)
    with pytest.warns(UserWarning, match="TorchLoader.load_state_dict"):
        resumed_loader = (
            _resume_torch_loader(resumed_dataset, num_workers=0, batch_size=2, collate_fn=_collate_columns)
            .unbatch()
            .load_state_dict(state)
        )
    resumed = [_normalize_loader_output(sample) for sample in resumed_loader]

    assert state["stages"][0]["kind"] == "unbatch"
    assert resumed == continued
    assert len(consumed) == 3


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_torch_loader_side_shuffle_resume_matches_continued_iterator(tmp_path) -> None:
    shards = write_tar_shards(tmp_path, build_records(count=8), num_shards=4)
    context = RuntimeContext(seed=59)
    dataset = Dataset.from_source("tars", shards=shards, context=context)
    loader = _resume_torch_loader(dataset, num_workers=0, batch_size=None).shuffle(buffer_size=3, initial=2)
    iterator = iter(loader)

    consumed = _consume_loader_outputs(iterator, 2)
    state = iterator.state_dict()
    continued = [_normalize_loader_output(sample) for sample in iterator]

    resumed_dataset = Dataset.from_source("tars", shards=shards, context=context)
    with pytest.warns(UserWarning, match="TorchLoader.load_state_dict"):
        resumed_loader = (
            _resume_torch_loader(resumed_dataset, num_workers=0, batch_size=None)
            .shuffle(buffer_size=3, initial=2)
            .load_state_dict(state)
        )
    resumed = [_normalize_loader_output(sample) for sample in resumed_loader]

    assert state["stages"][0]["kind"] == "shuffle"
    assert resumed == continued
    assert len(consumed) == 2


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_torch_loader_side_assemble_resume_matches_continued_iterator(tmp_path) -> None:
    shards = write_tar_shards(tmp_path, build_records(count=8), num_shards=4)
    context = RuntimeContext(seed=60)
    dataset = Dataset.from_source("tars", shards=shards, context=context)
    loader = _resume_torch_loader(dataset, num_workers=0, batch_size=None).assemble(_build_pair_output_assembler)
    iterator = iter(loader)

    consumed = [next(iterator)]
    state = iterator.state_dict()
    continued = list(iterator)

    resumed_dataset = Dataset.from_source("tars", shards=shards, context=context)
    with pytest.warns(UserWarning, match="TorchLoader.load_state_dict"):
        resumed_loader = (
            _resume_torch_loader(resumed_dataset, num_workers=0, batch_size=None)
            .assemble(_build_pair_output_assembler)
            .load_state_dict(state)
        )
    resumed = list(resumed_loader)

    assert state["stages"][0]["kind"] == "assemble"
    assert resumed == continued
    assert len(consumed) == 1


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_torch_loader_resume_rejects_loader_side_stage_config_mismatch(tmp_path) -> None:
    shards = write_tar_shards(tmp_path, build_records(count=8), num_shards=4)
    dataset = Dataset.from_source("tars", shards=shards, context=RuntimeContext(seed=61))
    iterator = iter(_resume_torch_loader(dataset, num_workers=0, batch_size=None).shuffle(buffer_size=3))

    next(iterator)
    state = iterator.state_dict()
    changed_loader = _resume_torch_loader(dataset, num_workers=0, batch_size=None).shuffle(buffer_size=4)

    with pytest.raises(ResumeStateError, match=r"\[ResumeLoaderMismatch\]"):
        changed_loader.load_state_dict(state)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_torch_loader_side_assemble_rejects_non_stateful_assembler(tmp_path) -> None:
    shards = write_tar_shards(tmp_path, build_records(count=4), num_shards=2)
    dataset = Dataset.from_source("tars", shards=shards, context=RuntimeContext(seed=62))
    loader = _resume_torch_loader(dataset, num_workers=0, batch_size=None).assemble(_build_non_stateful_assembler)

    with pytest.raises(UnsupportedResume, match=r"\[UnsupportedResume\] loader stage kind='assemble'"):
        iter(loader)


def test_tar_source_resume_supports_resample_across_rounds(tmp_path) -> None:
    records = build_records(count=4)
    shards = write_tar_shards(tmp_path, records, num_shards=2)
    dataset = Dataset.from_source("tars", shards=shards, context=RuntimeContext(seed=29), resample=True)
    iterator = iter(dataset)

    _consume(iterator, 6)
    state = iterator.state_dict()
    continued = _consume(iterator, 5)

    with pytest.warns(UserWarning, match="Dataset.load_state_dict"):
        resumed_dataset = Dataset.from_source(
            "tars",
            shards=shards,
            context=RuntimeContext(seed=29),
            resample=True,
        ).load_state_dict(state)
    resumed = _consume(iter(resumed_dataset), 5)

    assert state["source"]["state"]["round_index"] >= 0
    assert resumed == continued


def test_load_state_dict_rejects_tar_shuffle_mode_change(tmp_path) -> None:
    shards = write_tar_shards(tmp_path, build_records(count=4), num_shards=2)
    dataset = Dataset.from_source("tars", shards=shards, context=RuntimeContext(seed=31))
    state = iter(dataset).state_dict()
    changed_pipeline = Dataset.from_source(
        "tars",
        shards=shards,
        context=RuntimeContext(seed=31),
        shuffle_mode="none",
    )

    with pytest.raises(ResumeStateError, match=r"\[ResumePipelineMismatch\]"):
        changed_pipeline.load_state_dict(state)


@pytest.mark.parametrize("checkpoint_after", [0, 1, 3, 6])
def test_jsonl_source_resume_matches_continued_iterator(tmp_path, checkpoint_after: int) -> None:
    records = build_records(count=6)
    path = write_jsonl_file(tmp_path, records)
    dataset = Dataset.from_source("jsonl", shards=path, context=RuntimeContext(seed=23))
    iterator = iter(dataset)

    consumed = _consume(iterator, checkpoint_after)
    state = iterator.state_dict()
    continued = [normalize_sample(sample) for sample in iterator]
    expected = [normalize_sample(sample) for sample in dataset]

    with pytest.warns(UserWarning, match="Dataset.load_state_dict"):
        resumed_dataset = Dataset.from_source(
            "jsonl",
            shards=path,
            context=RuntimeContext(seed=23),
        ).load_state_dict(state)
    resumed = [normalize_sample(sample) for sample in resumed_dataset]

    assert state["source"]["kind"] == "jsonl"
    assert "byte_offset" in state["source"]["state"]
    assert consumed + continued == expected
    assert resumed == continued


def test_jsonl_source_resume_supports_resample_across_rounds(tmp_path) -> None:
    records = build_records(count=4)
    path = write_jsonl_file(tmp_path, records)
    dataset = Dataset.from_source("jsonl", shards=path, context=RuntimeContext(seed=29), resample=True)
    iterator = iter(dataset)

    _consume(iterator, 6)
    state = iterator.state_dict()
    continued = _consume(iterator, 5)

    with pytest.warns(UserWarning, match="Dataset.load_state_dict"):
        resumed_dataset = Dataset.from_source(
            "jsonl",
            shards=path,
            context=RuntimeContext(seed=29),
            resample=True,
        ).load_state_dict(state)
    resumed = _consume(iter(resumed_dataset), 5)

    assert state["source"]["state"]["round_index"] >= 0
    assert resumed == continued


def test_load_state_dict_rejects_jsonl_source_file_change(tmp_path) -> None:
    path = write_jsonl_file(tmp_path, build_records(count=3))
    dataset = Dataset.from_source("jsonl", shards=path, context=RuntimeContext(seed=31))
    state = iter(dataset).state_dict()

    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"id":"sample-new","text":"text-new","value":999}\n')

    with pytest.raises(ResumeStateError, match=r"\[ResumePipelineMismatch\]"):
        dataset.load_state_dict(state)


def test_load_state_dict_rejects_jsonl_shuffle_mode_change(tmp_path) -> None:
    path = write_jsonl_file(tmp_path, build_records(count=3))
    dataset = Dataset.from_source("jsonl", shards=path, context=RuntimeContext(seed=31))
    state = iter(dataset).state_dict()
    changed_pipeline = Dataset.from_source(
        "jsonl",
        shards=path,
        context=RuntimeContext(seed=31),
        shuffle_mode="none",
    )

    with pytest.raises(ResumeStateError, match=r"\[ResumePipelineMismatch\]"):
        changed_pipeline.load_state_dict(state)


@pytest.mark.parametrize("checkpoint_after", [0, 1, 4, 7])
def test_parquet_source_resume_matches_continued_iterator(tmp_path, checkpoint_after: int) -> None:
    records = build_records(count=7)
    path = write_parquet_file(tmp_path, records, row_group_size=2)
    dataset = Dataset.from_source(
        "parquet",
        shards=path,
        context=RuntimeContext(seed=31),
        min_row_groups_per_fragment=2,
    )
    iterator = iter(dataset)

    consumed = _consume(iterator, checkpoint_after)
    state = iterator.state_dict()
    continued = [normalize_sample(sample) for sample in iterator]
    expected = [normalize_sample(sample) for sample in dataset]

    with pytest.warns(UserWarning, match="Dataset.load_state_dict"):
        resumed_dataset = Dataset.from_source(
            "parquet",
            shards=path,
            context=RuntimeContext(seed=31),
            min_row_groups_per_fragment=2,
        ).load_state_dict(state)
    resumed = [normalize_sample(sample) for sample in resumed_dataset]

    assert state["source"]["kind"] == "parquet"
    assert "row_index" not in state["source"]["state"]
    assert "row_group_index" in state["source"]["state"]
    assert "row_in_row_group" in state["source"]["state"]
    assert consumed + continued == expected
    assert resumed == continued


def test_parquet_source_resume_supports_resample_across_rounds(tmp_path) -> None:
    records = build_records(count=4)
    path = write_parquet_file(tmp_path, records, row_group_size=2)
    dataset = Dataset.from_source(
        "parquet",
        shards=path,
        context=RuntimeContext(seed=37),
        min_row_groups_per_fragment=1,
        resample=True,
    )
    iterator = iter(dataset)

    _consume(iterator, 6)
    state = iterator.state_dict()
    continued = _consume(iterator, 5)

    with pytest.warns(UserWarning, match="Dataset.load_state_dict"):
        resumed_dataset = Dataset.from_source(
            "parquet",
            shards=path,
            context=RuntimeContext(seed=37),
            min_row_groups_per_fragment=1,
            resample=True,
        ).load_state_dict(state)
    resumed = _consume(iter(resumed_dataset), 5)

    assert state["source"]["state"]["round_index"] >= 0
    assert resumed == continued


def test_load_state_dict_rejects_parquet_shuffle_mode_change(tmp_path) -> None:
    path = write_parquet_file(tmp_path, build_records(), row_group_size=2)
    dataset = Dataset.from_source("parquet", shards=path, context=RuntimeContext(seed=41))
    state = iter(dataset).state_dict()
    changed_pipeline = Dataset.from_source(
        "parquet",
        shards=path,
        context=RuntimeContext(seed=41),
        shuffle_mode="none",
    )

    with pytest.raises(ResumeStateError, match=r"\[ResumePipelineMismatch\]"):
        changed_pipeline.load_state_dict(state)


@pytest.mark.parametrize("shuffle_mode", ["none", "global", "fragment_aware"])
@pytest.mark.parametrize("checkpoint_after", [0, 1, 3, 7])
def test_lance_map_select_resume_matches_continued_iterator(
    tmp_path,
    shuffle_mode: str,
    checkpoint_after: int,
) -> None:
    pytest.importorskip("lance")

    records = build_records(count=7)
    path = write_lance_dataset(tmp_path, records, max_rows_per_file=2)
    dataset = (
        Dataset.from_source(
            "lance",
            shards=path,
            context=RuntimeContext(seed=17),
            batch_size=2,
            shuffle_mode=shuffle_mode,
        )
        .map(_add_marker)
        .select(["id", "marker"])
    )
    iterator = iter(dataset)

    consumed = [_normalize_stage_sample(next(iterator)) for _ in range(checkpoint_after)]
    state = iterator.state_dict()
    continued = [_normalize_stage_sample(sample) for sample in iterator]
    expected = [_normalize_stage_sample(sample) for sample in dataset]

    with pytest.warns(UserWarning, match="Dataset.load_state_dict"):
        resumed_dataset = (
            Dataset.from_source(
                "lance",
                shards=path,
                context=RuntimeContext(seed=17),
                batch_size=2,
                shuffle_mode=shuffle_mode,
            )
            .map(_add_marker)
            .select(["id", "marker"])
            .load_state_dict(state)
        )
    resumed = [_normalize_stage_sample(sample) for sample in resumed_dataset]

    assert [stage["kind"] for stage in state["stages"]] == ["map", "select"]
    assert consumed + continued == expected
    assert resumed == continued


def test_lance_stage_resume_rejects_stage_fingerprint_mismatch(tmp_path) -> None:
    pytest.importorskip("lance")

    path = write_lance_dataset(tmp_path, build_records())
    dataset = Dataset.from_source("lance", shards=path).select(["id"])
    iterator = iter(dataset)
    next(iterator)
    state = iterator.state_dict()
    state["stages"][0]["fingerprint"] = "changed"

    with pytest.warns(UserWarning, match="Dataset.load_state_dict"):
        resumed_dataset = Dataset.from_source("lance", shards=path).select(["id"]).load_state_dict(state)

    with pytest.raises(ResumeStateError, match=r"\[ResumeStageMismatch\]"):
        list(resumed_dataset)


def test_load_state_dict_rejects_stateless_stage_pipeline_change(tmp_path) -> None:
    pytest.importorskip("lance")

    path = write_lance_dataset(tmp_path, build_records())
    dataset = Dataset.from_source("lance", shards=path).map(_add_marker).select(["id", "marker"])
    state = iter(dataset).state_dict()
    changed_pipeline = Dataset.from_source("lance", shards=path).select(["id", "marker"]).map(_add_marker)

    with pytest.raises(ResumeStateError, match=r"\[ResumePipelineMismatch\]"):
        changed_pipeline.load_state_dict(state)


@pytest.mark.parametrize("initial", [None, 2])
@pytest.mark.parametrize("checkpoint_after", [0, 1, 4, 9])
def test_lance_dataset_shuffle_resume_matches_continued_iterator(
    tmp_path,
    initial: int | None,
    checkpoint_after: int,
) -> None:
    pytest.importorskip("lance")

    records = build_records(count=9)
    path = write_lance_dataset(tmp_path, records, max_rows_per_file=2)
    dataset = Dataset.from_source(
        "lance",
        shards=path,
        context=RuntimeContext(seed=37),
        batch_size=2,
        shuffle_mode="none",
    ).shuffle(buffer_size=3, initial=initial)
    iterator = iter(dataset)

    consumed = _consume(iterator, checkpoint_after)
    state = iterator.state_dict()
    continued = [normalize_sample(sample) for sample in iterator]
    expected = [normalize_sample(sample) for sample in dataset]

    with pytest.warns(UserWarning, match="Dataset.load_state_dict"):
        resumed_dataset = (
            Dataset.from_source(
                "lance",
                shards=path,
                context=RuntimeContext(seed=37),
                batch_size=2,
                shuffle_mode="none",
            )
            .shuffle(buffer_size=3, initial=initial)
            .load_state_dict(state)
        )
    resumed = [normalize_sample(sample) for sample in resumed_dataset]

    assert state["stages"][0]["kind"] == "shuffle"
    assert set(state["stages"][0]["state"]) == {"buffer", "rng_state", "upstream_exhausted"}
    assert consumed + continued == expected
    assert resumed == continued


@pytest.mark.parametrize(
    "changed_dataset",
    [
        lambda path: Dataset.from_source("lance", shards=path).shuffle(buffer_size=4),
        lambda path: Dataset.from_source("lance", shards=path).shuffle(buffer_size=3, initial=2),
    ],
)
def test_load_state_dict_rejects_dataset_shuffle_config_change(tmp_path, changed_dataset) -> None:
    pytest.importorskip("lance")

    path = write_lance_dataset(tmp_path, build_records())
    dataset = Dataset.from_source("lance", shards=path).shuffle(buffer_size=3)
    state = iter(dataset).state_dict()

    with pytest.raises(ResumeStateError, match=r"\[ResumePipelineMismatch\]"):
        changed_dataset(path).load_state_dict(state)


@pytest.mark.parametrize("shuffle_mode", ["none", "global", "fragment_aware"])
@pytest.mark.parametrize("checkpoint_after", [0, 1, 2, 3])
def test_lance_batch_resume_matches_continued_iterator(
    tmp_path,
    shuffle_mode: str,
    checkpoint_after: int,
) -> None:
    pytest.importorskip("lance")

    records = build_records(count=7)
    path = write_lance_dataset(tmp_path, records, max_rows_per_file=2)
    dataset = Dataset.from_source(
        "lance",
        shards=path,
        context=RuntimeContext(seed=43),
        batch_size=2,
        shuffle_mode=shuffle_mode,
    ).batch(3)
    iterator = iter(dataset)

    consumed = [_normalize_batch(next(iterator)) for _ in range(checkpoint_after)]
    state = iterator.state_dict()
    continued = [_normalize_batch(batch) for batch in iterator]
    expected = [_normalize_batch(batch) for batch in dataset]

    with pytest.warns(UserWarning, match="Dataset.load_state_dict"):
        resumed_dataset = (
            Dataset.from_source(
                "lance",
                shards=path,
                context=RuntimeContext(seed=43),
                batch_size=2,
                shuffle_mode=shuffle_mode,
            )
            .batch(3)
            .load_state_dict(state)
        )
    resumed = [_normalize_batch(batch) for batch in resumed_dataset]

    assert state["stages"][0]["kind"] == "batch"
    assert state["stages"][0]["state"]["emitted"] == checkpoint_after
    assert consumed + continued == expected
    assert resumed == continued


def test_lance_batch_resume_uses_pending_samples(tmp_path) -> None:
    pytest.importorskip("lance")

    records = build_records(count=5)
    path = write_lance_dataset(tmp_path, records, max_rows_per_file=2)
    source_dataset = Dataset.from_source("lance", shards=path, batch_size=2, shuffle_mode="none")
    source_iterator = iter(source_dataset)
    pending = [next(source_iterator), next(source_iterator)]
    state = source_iterator.state_dict()

    batch_dataset = Dataset.from_source("lance", shards=path, batch_size=2, shuffle_mode="none").batch(3)
    batch_fingerprint = iter(batch_dataset).stages[0].fingerprint()
    state["pipeline_fingerprint"] = batch_dataset._pipeline_fingerprint()
    state["num_yielded"] = 0
    state["stages"] = [
        {
            "kind": "batch",
            "fingerprint": batch_fingerprint,
            "state": {"pending": pending, "emitted": 0},
        }
    ]

    expected_first = _normalize_batch([*pending, next(source_iterator)])
    expected_tail = [_normalize_batch(list(source_iterator))]

    with pytest.warns(UserWarning, match="Dataset.load_state_dict"):
        resumed_dataset = batch_dataset.load_state_dict(state)
    resumed = [_normalize_batch(batch) for batch in resumed_dataset]

    assert resumed == [expected_first, *expected_tail]


@pytest.mark.parametrize(
    "changed_dataset",
    [
        lambda path: Dataset.from_source("lance", shards=path).batch(4),
        lambda path: Dataset.from_source("lance", shards=path).batch(3, drop_last=True),
        lambda path: Dataset.from_source("lance", shards=path).batch(3, collate_fn=_collate_ids),
    ],
)
def test_load_state_dict_rejects_batch_stage_config_change(tmp_path, changed_dataset) -> None:
    pytest.importorskip("lance")

    path = write_lance_dataset(tmp_path, build_records())
    dataset = Dataset.from_source("lance", shards=path).batch(3)
    state = iter(dataset).state_dict()

    with pytest.raises(ResumeStateError, match=r"\[ResumePipelineMismatch\]"):
        changed_dataset(path).load_state_dict(state)


@pytest.mark.parametrize("checkpoint_after", [0, 1, 4, 7])
def test_lance_unbatch_resume_matches_continued_iterator(tmp_path, checkpoint_after: int) -> None:
    pytest.importorskip("lance")

    records = build_records(count=7)
    path = write_lance_dataset(tmp_path, records, max_rows_per_file=2)
    dataset = Dataset.from_source("lance", shards=path, batch_size=2).batch(3).unbatch()
    iterator = iter(dataset)

    consumed = _consume(iterator, checkpoint_after)
    state = iterator.state_dict()
    continued = [normalize_sample(sample) for sample in iterator]
    expected = [normalize_sample(sample) for sample in dataset]

    with pytest.warns(UserWarning, match="Dataset.load_state_dict"):
        resumed_dataset = (
            Dataset.from_source("lance", shards=path, batch_size=2).batch(3).unbatch().load_state_dict(state)
        )
    resumed = [normalize_sample(sample) for sample in resumed_dataset]

    assert [stage["kind"] for stage in state["stages"]] == ["batch", "unbatch"]
    if checkpoint_after == 1:
        assert len(state["stages"][1]["state"]["pending"]) == 2
    assert consumed + continued == expected
    assert resumed == continued


def test_lance_unbatch_resume_expands_dict_batch(tmp_path) -> None:
    pytest.importorskip("lance")

    records = build_records(count=5)
    path = write_lance_dataset(tmp_path, records, max_rows_per_file=2)
    dataset = Dataset.from_source("lance", shards=path, batch_size=2).batch(3, collate_fn=_collate_columns).unbatch()
    iterator = iter(dataset)

    consumed = _consume(iterator, 1)
    state = iterator.state_dict()
    continued = [normalize_sample(sample) for sample in iterator]
    expected = [normalize_sample(sample) for sample in dataset]

    with pytest.warns(UserWarning, match="Dataset.load_state_dict"):
        resumed_dataset = (
            Dataset.from_source("lance", shards=path, batch_size=2)
            .batch(3, collate_fn=_collate_columns)
            .unbatch()
            .load_state_dict(state)
        )
    resumed = [normalize_sample(sample) for sample in resumed_dataset]

    assert len(state["stages"][1]["state"]["pending"]) == 2
    assert consumed + continued == expected
    assert resumed == continued


@pytest.mark.parametrize("checkpoint_after", [0, 1, 2, 4, 5])
def test_lance_assemble_resume_matches_continued_iterator(tmp_path, checkpoint_after: int) -> None:
    pytest.importorskip("lance")

    records = build_records(count=5)
    path = write_lance_dataset(tmp_path, records, max_rows_per_file=2)
    dataset = Dataset.from_source("lance", shards=path, batch_size=2).assemble(_build_pair_output_assembler)
    iterator = iter(dataset)

    consumed = [next(iterator) for _ in range(checkpoint_after)]
    state = iterator.state_dict()
    continued = list(iterator)
    expected = list(dataset)

    with pytest.warns(UserWarning, match="Dataset.load_state_dict"):
        resumed_dataset = (
            Dataset.from_source("lance", shards=path, batch_size=2)
            .assemble(_build_pair_output_assembler)
            .load_state_dict(state)
        )
    resumed = list(resumed_dataset)

    assert state["stages"][0]["kind"] == "assemble"
    if checkpoint_after == 1:
        assert state["stages"][0]["state"]["pending_outputs"] == [{"pair": "sample-0+sample-1", "slot": 1}]
    assert consumed + continued == expected
    assert resumed == continued


def test_lance_assemble_resume_rejects_non_stateful_assembler(tmp_path) -> None:
    pytest.importorskip("lance")

    path = write_lance_dataset(tmp_path, build_records())
    dataset = Dataset.from_source("lance", shards=path).assemble(_build_non_stateful_assembler)

    with pytest.raises(UnsupportedResume, match=r"\[UnsupportedResume\] stage kind='assemble'"):
        iter(dataset)


def test_lance_assemble_resume_rejects_assembler_fingerprint_change(tmp_path) -> None:
    pytest.importorskip("lance")

    global _ASSEMBLER_FINGERPRINT_VERSION

    path = write_lance_dataset(tmp_path, build_records())
    dataset = Dataset.from_source("lance", shards=path).assemble(_build_pair_output_assembler)
    iterator = iter(dataset)
    next(iterator)
    state = iterator.state_dict()

    _ASSEMBLER_FINGERPRINT_VERSION = "v2"
    try:
        with pytest.warns(UserWarning, match="Dataset.load_state_dict"):
            resumed_dataset = (
                Dataset.from_source("lance", shards=path).assemble(_build_pair_output_assembler).load_state_dict(state)
            )
        with pytest.raises(ResumeStateError, match=r"\[ResumeStageMismatch\]"):
            list(resumed_dataset)
    finally:
        _ASSEMBLER_FINGERPRINT_VERSION = "v1"


@pytest.mark.parametrize("shuffle_mode", ["none", "global", "fragment_aware"])
def test_lance_source_resume_supports_multiple_lance_datasets(tmp_path, shuffle_mode: str) -> None:
    pytest.importorskip("lance")

    records_a = build_records(count=3)
    records_b = [{**record, "id": f"b-{record['id']}"} for record in build_records(count=4)]
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    path_a = write_lance_dataset(root_a, records_a, max_rows_per_file=2)
    path_b = write_lance_dataset(root_b, records_b, max_rows_per_file=2)

    dataset = Dataset.from_source(
        "lance",
        shards=[path_a, path_b],
        context=RuntimeContext(seed=19),
        batch_size=3,
        shuffle_mode=shuffle_mode,
    )
    iterator = iter(dataset)

    consumed = _consume(iterator, 4)
    state = iterator.state_dict()
    continued = [normalize_sample(sample) for sample in iterator]
    expected = [normalize_sample(sample) for sample in dataset]

    resumed_dataset = Dataset.from_source(
        "lance",
        shards=[path_a, path_b],
        context=RuntimeContext(seed=19),
        batch_size=3,
        shuffle_mode=shuffle_mode,
    )
    with pytest.warns(UserWarning, match="Dataset.load_state_dict"):
        resumed_dataset = resumed_dataset.load_state_dict(state)
    resumed = [normalize_sample(sample) for sample in resumed_dataset]

    assert consumed + continued == expected
    assert resumed == continued


@pytest.mark.parametrize("shuffle_mode", ["none", "global", "fragment_aware"])
def test_lance_source_resume_supports_resample_across_rounds(tmp_path, shuffle_mode: str) -> None:
    pytest.importorskip("lance")

    records = build_records(count=5)
    path = write_lance_dataset(tmp_path, records, max_rows_per_file=2)
    dataset = Dataset.from_source(
        "lance",
        shards=path,
        context=RuntimeContext(seed=23),
        batch_size=2,
        shuffle_mode=shuffle_mode,
        resample=True,
    )
    iterator = iter(dataset)

    _consume(iterator, 7)
    state = iterator.state_dict()
    continued = _consume(iterator, 6)

    with pytest.warns(UserWarning, match="Dataset.load_state_dict"):
        resumed_dataset = Dataset.from_source(
            "lance",
            shards=path,
            context=RuntimeContext(seed=23),
            batch_size=2,
            shuffle_mode=shuffle_mode,
            resample=True,
        ).load_state_dict(state)
    resumed = _consume(iter(resumed_dataset), 6)

    assert state["source"]["state"]["round_index"] >= 0
    assert resumed == continued


@pytest.mark.parametrize("shuffle_mode", ["none", "global"])
def test_lance_none_and_global_shuffle_do_not_materialize_round_order(tmp_path, shuffle_mode: str) -> None:
    pytest.importorskip("lance")

    path = write_lance_dataset(tmp_path, build_records(count=16))
    dataset = Dataset.from_source(
        "lance",
        shards=path,
        context=RuntimeContext(seed=31),
        batch_size=4,
        shuffle_mode=shuffle_mode,
    )
    iterator = iter(dataset)

    _consume(iterator, 5)

    assert iterator.source._index_order == []
    assert iterator.source._index_order_round is None


def test_lance_source_resume_rejects_source_fingerprint_mismatch(tmp_path) -> None:
    pytest.importorskip("lance")

    path = write_lance_dataset(tmp_path, build_records())
    dataset = Dataset.from_source("lance", shards=path, batch_size=2, shuffle_mode="none")
    with pytest.warns(UserWarning, match="Dataset.state_dict"):
        state = dataset.state_dict()
    state["source"]["fingerprint"] = "changed"
    with pytest.warns(UserWarning, match="Dataset.load_state_dict"):
        resumed_dataset = Dataset.from_source(
            "lance",
            shards=path,
            batch_size=2,
            shuffle_mode="none",
        ).load_state_dict(state)

    with pytest.raises(ResumeStateError, match=r"\[ResumeSourceMismatch\]"):
        list(resumed_dataset)


def test_lance_dataset_state_dict_returns_initial_iterator_state(tmp_path) -> None:
    pytest.importorskip("lance")

    records = build_records(count=4)
    path = write_lance_dataset(tmp_path, records)
    dataset = Dataset.from_source("lance", shards=path, batch_size=2, shuffle_mode="none")

    with pytest.warns(UserWarning, match="Dataset.state_dict"):
        state = dataset.state_dict()
    with pytest.warns(UserWarning, match="Dataset.load_state_dict"):
        resumed_dataset = Dataset.from_source("lance", shards=path, batch_size=2).load_state_dict(state)
    resumed = [normalize_sample(sample) for sample in resumed_dataset]

    assert state["num_yielded"] == 0
    assert state["source"]["state"]["position_in_round"] == 0
    assert resumed == records


@pytest.mark.parametrize("shuffle_mode", ["none", "global", "fragment_aware"])
def test_lance_iterators_can_checkpoint_independently(tmp_path, shuffle_mode: str) -> None:
    pytest.importorskip("lance")

    records = build_records(count=6)
    path = write_lance_dataset(tmp_path, records, max_rows_per_file=2)
    dataset = Dataset.from_source(
        "lance",
        shards=path,
        context=RuntimeContext(seed=29),
        batch_size=2,
        shuffle_mode=shuffle_mode,
    )
    first_iterator = iter(dataset)
    second_iterator = iter(dataset)

    first_consumed = _consume(first_iterator, 1)
    second_consumed = _consume(second_iterator, 3)
    first_state = first_iterator.state_dict()
    second_state = second_iterator.state_dict()
    first_continued = [normalize_sample(sample) for sample in first_iterator]
    second_continued = [normalize_sample(sample) for sample in second_iterator]
    expected = [normalize_sample(sample) for sample in dataset]

    with pytest.warns(UserWarning, match="Dataset.load_state_dict"):
        first_resumed_dataset = Dataset.from_source(
            "lance",
            shards=path,
            context=RuntimeContext(seed=29),
            batch_size=2,
            shuffle_mode=shuffle_mode,
        ).load_state_dict(first_state)
    with pytest.warns(UserWarning, match="Dataset.load_state_dict"):
        second_resumed_dataset = Dataset.from_source(
            "lance",
            shards=path,
            context=RuntimeContext(seed=29),
            batch_size=2,
            shuffle_mode=shuffle_mode,
        ).load_state_dict(second_state)
    first_resumed = [normalize_sample(sample) for sample in first_resumed_dataset]
    second_resumed = [normalize_sample(sample) for sample in second_resumed_dataset]

    assert first_consumed + first_continued == expected
    assert second_consumed + second_continued == expected
    assert first_resumed == first_continued
    assert second_resumed == second_continued
