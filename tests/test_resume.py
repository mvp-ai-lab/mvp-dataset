from __future__ import annotations

import pytest

from mvp_dataset import Dataset, ResumeStateError, RuntimeContext, UnsupportedResume
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


def _source_path(tmp_path, source_kind: str) -> str | list[str]:
    records = build_records()
    if source_kind == "jsonl":
        return write_jsonl_file(tmp_path, records)
    if source_kind == "parquet":
        return write_parquet_file(tmp_path, records)
    if source_kind == "tars":
        return write_tar_shards(tmp_path, records, num_shards=1)
    raise AssertionError(f"unexpected source_kind={source_kind!r}")


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


@pytest.mark.parametrize("source_kind", ["jsonl", "parquet", "tars"])
def test_state_dict_rejects_unsupported_sources(tmp_path, source_kind: str) -> None:
    dataset = Dataset.from_source(source_kind, shards=_source_path(tmp_path, source_kind))

    with pytest.warns(UserWarning, match="Dataset.state_dict"):
        with pytest.raises(UnsupportedResume, match=rf"\[UnsupportedResume\] source kind='{source_kind}'"):
            dataset.state_dict()


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


def test_iterating_non_stateful_source_is_rejected(tmp_path) -> None:
    dataset = _jsonl_dataset(tmp_path)

    with pytest.raises(UnsupportedResume, match=r"\[UnsupportedResume\] source kind='jsonl'"):
        list(dataset)


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
