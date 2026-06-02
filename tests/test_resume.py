from __future__ import annotations

import pytest

from mvp_dataset import Dataset, ResumeStateError, RuntimeContext, UnsupportedResume
from mvp_dataset.core.resume import RESUME_STATE_VERSION

from .helpers import (
    build_records,
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


def _compatible_state(dataset: Dataset) -> dict[str, object]:
    return {
        "version": RESUME_STATE_VERSION,
        "runtime_fingerprint": dataset.context.fingerprint(),
        "pipeline_fingerprint": dataset._pipeline_fingerprint(),
        "output_step": 0,
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

    with pytest.raises(UnsupportedResume, match=rf"\[UnsupportedResume\] source kind='{source_kind}'"):
        dataset.state_dict()


def test_state_dict_rejects_unsupported_stage_with_clear_error(tmp_path) -> None:
    dataset = _jsonl_dataset(tmp_path).map(lambda sample: sample)

    with pytest.raises(UnsupportedResume, match=r"\[UnsupportedResume\] stage kind='map' index=0"):
        dataset.state_dict()


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

    resumed = dataset.load_state_dict(state)

    assert resumed is not dataset
    assert resumed._resume_state == state
    assert dataset._resume_state is None


def test_runtime_context_fingerprint_is_stable_and_seed_sensitive() -> None:
    first = RuntimeContext(seed=1).fingerprint()
    second = RuntimeContext(seed=1).fingerprint()
    different_seed = RuntimeContext(seed=2).fingerprint()

    assert first == second
    assert first != different_seed


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


def test_lance_pipeline_fingerprint_changes_with_load_in_memory(tmp_path) -> None:
    pytest.importorskip("lance")

    path = write_lance_dataset(tmp_path, build_records())
    disk_backed = Dataset.from_source("lance", shards=path, load_in_memory=False)
    memory_backed = Dataset.from_source("lance", shards=path, load_in_memory=True)

    assert disk_backed._pipeline_fingerprint() != memory_backed._pipeline_fingerprint()


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
