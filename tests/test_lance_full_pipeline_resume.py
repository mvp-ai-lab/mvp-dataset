from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from mvp_dataset import Dataset, ResumeStateError, RuntimeContext

from .helpers import build_records, write_lance_dataset
from .test_resume import (
    _add_marker,
    _add_marker_v2,
    _build_pair_output_assembler,
    _consume,
    _remaining,
)


def _build_lance_source(
    tmp_path: Path,
    *,
    seed: int,
    shuffle_mode: str,
    count: int = 9,
) -> Callable[[], Dataset]:
    path = write_lance_dataset(tmp_path, build_records(count=count), max_rows_per_file=2)
    context = RuntimeContext(seed=seed)
    chunk_shuffle = {"chunk_size": 3, "k": 2} if shuffle_mode == "chunk" else None

    def build() -> Dataset:
        return Dataset.from_source(
            "lance",
            shards=path,
            context=context,
            read_batch_size=2,
            shuffle_mode=shuffle_mode,
            chunk_shuffle=chunk_shuffle,
        )

    return build


def _full_pipeline(
    dataset: Dataset, *, map_fn: Callable[[dict[str, object]], dict[str, object]] = _add_marker
) -> Dataset:
    return (
        dataset.map(map_fn)
        .select(["id", "marker"])
        .shuffle(buffer_size=3, initial=2)
        .batch(3)
        .unbatch()
        .assemble(_build_pair_output_assembler)
    )


def _build_full_pipeline(tmp_path: Path, *, seed: int, shuffle_mode: str) -> Callable[[], Dataset]:
    build_source = _build_lance_source(tmp_path, seed=seed, shuffle_mode=shuffle_mode)
    return lambda: _full_pipeline(build_source())


@pytest.mark.parametrize("shuffle_mode", ["global", "chunk"])
@pytest.mark.parametrize("checkpoint_after", [1, 5])
def test_lance_full_pipeline_resume_matches_continued(tmp_path, shuffle_mode: str, checkpoint_after: int) -> None:
    pytest.importorskip("lance")

    build_dataset = _build_full_pipeline(tmp_path, seed=13, shuffle_mode=shuffle_mode)
    iterator = iter(build_dataset())

    consumed = _consume(iterator, checkpoint_after)
    state = iterator.state_dict()
    continued = _remaining(iterator)
    expected = _remaining(iter(build_dataset()))

    resumed_dataset = build_dataset()
    resumed_dataset.load_state_dict(state)
    resumed = _remaining(iter(resumed_dataset))

    assert consumed + continued == expected
    assert resumed == continued
    assert [stage["kind"] for stage in state["identity"]["stages"]] == [
        "map",
        "select",
        "shuffle",
        "batch",
        "unbatch",
        "assemble",
    ]


@pytest.mark.parametrize("shuffle_mode", ["global", "chunk"])
def test_lance_full_pipeline_resume_pending_state_consumed_once(tmp_path, shuffle_mode: str) -> None:
    pytest.importorskip("lance")

    build_dataset = _build_full_pipeline(tmp_path, seed=17, shuffle_mode=shuffle_mode)
    iterator = iter(build_dataset())
    _consume(iterator, 4)
    state = iterator.state_dict()
    continued = _remaining(iterator)

    resumed = build_dataset()
    resumed.load_state_dict(state)
    first = _remaining(iter(resumed))
    second = _remaining(iter(resumed))

    assert first == continued
    assert second == _remaining(iter(build_dataset()))


@pytest.mark.parametrize("shuffle_mode", ["global", "chunk"])
def test_lance_full_pipeline_resume_rejects_map_identity_mismatch(tmp_path, shuffle_mode: str) -> None:
    pytest.importorskip("lance")

    build_source = _build_lance_source(tmp_path, seed=19, shuffle_mode=shuffle_mode)
    iterator = iter(_full_pipeline(build_source()))
    _consume(iterator, 3)
    state = iterator.state_dict()

    changed = _full_pipeline(build_source(), map_fn=_add_marker_v2)
    with pytest.raises(ResumeStateError, match=r"\[ResumeIdentityMismatch\]"):
        changed.load_state_dict(state)
