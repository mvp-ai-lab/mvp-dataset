from __future__ import annotations

import importlib.util
from collections.abc import Callable

import pytest

from mvp_dataset import (
    Dataset,
    ResumeStateError,
    RuntimeContext,
    TorchLoader,
    UnsupportedResume,
)
from mvp_dataset.core.resume import RESUME_STATE_VERSION, callable_fingerprint

from .helpers import (
    build_records,
    write_jsonl_file,
    write_lance_dataset,
    write_lance_table,
    write_parquet_file,
    write_tar_shards,
)


def _normalize(value: object) -> object:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    if hasattr(value, "tolist"):
        return _normalize(value.tolist())
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def _consume(stream, count: int) -> list[object]:
    return [_normalize(next(stream)) for _ in range(count)]


def _remaining(stream) -> list[object]:
    return [_normalize(item) for item in stream]


def _add_marker(sample: dict[str, object]) -> dict[str, object]:
    return {**sample, "marker": f"marked-{_normalize(sample['id'])}"}


def _add_marker_v2(sample: dict[str, object]) -> dict[str, object]:
    return {**sample, "marker": f"changed-{_normalize(sample['id'])}"}


class _CallablePlusOne:
    def __call__(self, value: int) -> int:
        return value + 1


class _CallablePlusTwo:
    def __call__(self, value: int) -> int:
        return value + 2


def _normalize_batch(batch: list[dict[str, object]]) -> list[dict[str, object]]:
    return [_normalize(sample) for sample in batch]


def _collate_ids(batch: list[dict[str, object]]) -> list[object]:
    return [sample["id"] for sample in batch]


def _collate_columns(batch: list[dict[str, object]]) -> dict[str, list[object]]:
    return {key: [sample[key] for sample in batch] for key in batch[0]}


_ASSEMBLER_FINGERPRINT_VERSION = "v1"


class _PairOutputAssembler:
    def __init__(self) -> None:
        self.pending: list[str] = []

    def push(self, sample: dict[str, object]) -> list[dict[str, object]]:
        self.pending.append(str(_normalize(sample["id"])))
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


def _source_factory(
    tmp_path,
    source: str,
    *,
    seed: int = 0,
    rank: int = 0,
    world_size: int = 1,
    resample: bool = False,
    lance_shuffle_mode: str = "none",
) -> Callable[[], Dataset]:
    records = build_records(count=9)
    context = RuntimeContext(rank=rank, world_size=world_size, seed=seed)

    if source == "jsonl":
        path = write_jsonl_file(tmp_path, records)

        def build() -> Dataset:
            return Dataset.from_source("jsonl", shards=path, context=context, resample=resample)

        return build

    if source == "tar":
        shards = write_tar_shards(tmp_path, records, num_shards=3)

        def build() -> Dataset:
            return Dataset.from_source("tar", shards=shards, context=context, resample=resample)

        return build

    if source == "parquet":
        path = write_parquet_file(tmp_path, records, row_group_size=2)

        def build() -> Dataset:
            return Dataset.from_source(
                "parquet",
                shards=path,
                context=context,
                min_row_groups_per_chunk=1,
                resample=resample,
            )

        return build

    if source == "lance":
        pytest.importorskip("lance")
        path = write_lance_dataset(tmp_path, records, max_rows_per_file=2)
        chunk_shuffle = {"chunk_size": 3, "k": 2} if lance_shuffle_mode == "chunk" else None

        def build() -> Dataset:
            return Dataset.from_source(
                "lance",
                shards=path,
                context=context,
                read_batch_size=2,
                shuffle_mode=lance_shuffle_mode,
                chunk_shuffle=chunk_shuffle,
                resample=resample,
            )

        return build

    raise AssertionError(f"unknown source {source!r}")


def _full_dataset_pipeline(dataset: Dataset) -> Dataset:
    return (
        dataset.map(_add_marker)
        .select(["id", "marker"])
        .shuffle(buffer_size=3, initial=2)
        .batch(3)
        .unbatch()
        .assemble(_build_pair_output_assembler)
    )


def _assert_dataset_resume_matches_continued(
    build_dataset: Callable[[], Dataset], checkpoint_after: int
) -> dict[str, object]:
    dataset = build_dataset()
    iterator = iter(dataset)

    consumed = _consume(iterator, checkpoint_after)
    state = iterator.state_dict()
    continued = _remaining(iterator)
    expected = _remaining(iter(build_dataset()))

    with pytest.warns(UserWarning, match="Dataset.load_state_dict"):
        resumed_dataset = build_dataset().load_state_dict(state)
    resumed = _remaining(iter(resumed_dataset))

    assert consumed + continued == expected
    assert resumed == continued
    return state


def _resume_torch_loader(dataset: Dataset, **kwargs) -> TorchLoader:
    if kwargs.get("num_workers", 0) > 0:
        kwargs.setdefault("multiprocessing_context", "forkserver")
    return TorchLoader(dataset, **kwargs)


def _full_loader_pipeline(loader: TorchLoader) -> TorchLoader:
    return (
        loader.unbatch()
        .shuffle(buffer_size=3, initial=2)
        .batch(3, collate_fn=_normalize_batch)
        .unbatch()
        .assemble(_build_pair_output_assembler)
    )


def _assert_loader_resume_matches_continued(
    build_loader: Callable[[], TorchLoader], checkpoint_after: int
) -> dict[str, object]:
    loader = build_loader()
    iterator = iter(loader)

    consumed = _consume(iterator, checkpoint_after)
    state = iterator.state_dict()
    continued = _remaining(iterator)
    expected = _remaining(iter(build_loader()))

    with pytest.warns(UserWarning, match="TorchLoader.load_state_dict"):
        resumed_loader = build_loader().load_state_dict(state)
    resumed = _remaining(iter(resumed_loader))

    assert consumed + continued == expected
    assert resumed == continued
    return state


def test_callable_fingerprint_includes_function_and_callable_class_code() -> None:
    first_fn = callable_fingerprint(_add_marker)
    second_fn = callable_fingerprint(_add_marker_v2)
    first_callable = callable_fingerprint(_CallablePlusOne())
    second_callable = callable_fingerprint(_CallablePlusTwo())

    assert first_fn != second_fn
    assert first_callable != second_callable
    assert first_fn["source_hash"] is not None
    assert first_callable["source_hash"] is not None


def test_load_state_dict_rejects_unknown_schema_version(tmp_path) -> None:
    dataset = _source_factory(tmp_path, "jsonl")()

    with pytest.raises(ResumeStateError, match=r"\[InvalidResumeStateVersion\]"):
        dataset.load_state_dict({"version": RESUME_STATE_VERSION + 1})


def test_load_state_dict_rejects_runtime_fingerprint_mismatch(tmp_path) -> None:
    build_dataset = _source_factory(tmp_path, "jsonl", seed=1)
    state = iter(build_dataset()).state_dict()
    changed_runtime = _source_factory(tmp_path, "jsonl", seed=2)()

    with pytest.raises(ResumeStateError, match=r"\[ResumeRuntimeMismatch\]"):
        changed_runtime.load_state_dict(state)


def test_load_state_dict_rejects_pipeline_fingerprint_mismatch(tmp_path) -> None:
    build_dataset = _source_factory(tmp_path, "jsonl")
    state = iter(_full_dataset_pipeline(build_dataset())).state_dict()
    changed_pipeline = build_dataset().map(_add_marker).select(["id", "marker"]).shuffle(buffer_size=4)

    with pytest.raises(ResumeStateError, match=r"\[ResumePipelineMismatch\]"):
        changed_pipeline.load_state_dict(state)


def test_load_state_dict_attaches_validated_state_to_new_dataset(tmp_path) -> None:
    dataset = _source_factory(tmp_path, "jsonl")()
    state = iter(dataset).state_dict()

    with pytest.warns(UserWarning, match="Dataset.load_state_dict"):
        resumed = dataset.load_state_dict(state)

    assert resumed is not dataset
    assert resumed._resume_state == state
    assert dataset._resume_state is None


def test_runtime_context_fingerprint_is_stable_and_seed_sensitive() -> None:
    assert RuntimeContext(seed=1).fingerprint() == RuntimeContext(seed=1).fingerprint()
    assert RuntimeContext(seed=1).fingerprint() != RuntimeContext(seed=2).fingerprint()


@pytest.mark.parametrize("total_rows", [1, 2, 7, 8, 100])
def test_lance_global_shuffle_permute_index_is_bijective(total_rows: int) -> None:
    pytest.importorskip("lance")

    from mvp_dataset.sources.lance.order import permute_index

    observed = [permute_index(position, total_rows=total_rows, seed=41) for position in range(total_rows)]

    assert sorted(observed) == list(range(total_rows))


@pytest.mark.parametrize(
    ("source", "lance_shuffle_mode"),
    [
        ("jsonl", "none"),
        ("tar", "none"),
        ("parquet", "none"),
        ("lance", "none"),
        ("lance", "global"),
        ("lance", "chunk"),
    ],
)
def test_dataset_resume_full_pipeline_matches_continued_stream(tmp_path, source: str, lance_shuffle_mode: str) -> None:
    build_source = _source_factory(tmp_path, source, seed=13, lance_shuffle_mode=lance_shuffle_mode)

    state = _assert_dataset_resume_matches_continued(lambda: _full_dataset_pipeline(build_source()), checkpoint_after=1)

    assert [stage["kind"] for stage in state["stages"]] == ["map", "select", "shuffle", "batch", "unbatch", "assemble"]


@pytest.mark.parametrize("checkpoint_after", [0, 9])
def test_dataset_resume_full_pipeline_covers_initial_and_end_checkpoints(tmp_path, checkpoint_after: int) -> None:
    build_source = _source_factory(tmp_path, "jsonl", seed=17)

    _assert_dataset_resume_matches_continued(lambda: _full_dataset_pipeline(build_source()), checkpoint_after)


@pytest.mark.parametrize("rank", [0, 1])
def test_dataset_resume_full_pipeline_with_distributed_context(tmp_path, rank: int) -> None:
    build_source = _source_factory(tmp_path, "jsonl", seed=19, rank=rank, world_size=2)

    state = _assert_dataset_resume_matches_continued(lambda: _full_dataset_pipeline(build_source()), checkpoint_after=1)

    assert state["runtime_fingerprint"] == _full_dataset_pipeline(build_source()).context.fingerprint()


@pytest.mark.parametrize(
    ("source", "lance_shuffle_mode"),
    [
        ("jsonl", "none"),
        ("tar", "none"),
        ("parquet", "none"),
        ("lance", "global"),
    ],
)
def test_source_resume_supports_resample_across_rounds(tmp_path, source: str, lance_shuffle_mode: str) -> None:
    build_source = _source_factory(
        tmp_path,
        source,
        seed=23,
        resample=True,
        lance_shuffle_mode=lance_shuffle_mode,
    )
    dataset = build_source()
    iterator = iter(dataset)

    _consume(iterator, 11)
    state = iterator.state_dict()
    continued = _consume(iterator, 6)

    with pytest.warns(UserWarning, match="Dataset.load_state_dict"):
        resumed_dataset = build_source().load_state_dict(state)
    resumed = _consume(iter(resumed_dataset), 6)

    assert state["source"]["state"]["round_index"] >= 1
    assert resumed == continued


def test_lance_resume_supports_multiple_datasets(tmp_path) -> None:
    pytest.importorskip("lance")

    records_a = build_records(count=4)
    records_b = [{**record, "id": f"b-{record['id']}"} for record in build_records(count=5)]
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    path_a = write_lance_dataset(root_a, records_a, max_rows_per_file=2)
    path_b = write_lance_dataset(root_b, records_b, max_rows_per_file=2)

    def build_dataset() -> Dataset:
        return Dataset.from_source(
            "lance",
            shards=[path_a, path_b],
            context=RuntimeContext(seed=29),
            read_batch_size=3,
            shuffle_mode="global",
        )

    _assert_dataset_resume_matches_continued(build_dataset, checkpoint_after=4)


@pytest.mark.parametrize(
    ("shuffle_mode", "resolve_ref"),
    [
        ("global", False),
        ("global", True),
        ("chunk", False),
        ("chunk", True),
    ],
)
def test_lance_shuffle_resume_with_and_without_resolve_ref(
    tmp_path,
    monkeypatch,
    shuffle_mode: str,
    resolve_ref: bool,
) -> None:
    pytest.importorskip("lance")

    main_records = [
        {"id": f"sample-{index}", "text": f"text-{index}", "value": index, "image_ref": f"img-{index % 5}"}
        for index in range(17)
    ]
    ref_records = [{"image_id": f"img-{index}", "image_value": f"resolved-{index}"} for index in range(5)]
    main_path = write_lance_table(tmp_path, "main.lance", main_records)
    ref_path = write_lance_table(tmp_path, "refs.lance", ref_records)

    def build_dataset() -> Dataset:
        dataset = Dataset.from_source(
            "lance",
            shards=main_path,
            context=RuntimeContext(seed=31),
            read_batch_size=4,
            shuffle_mode=shuffle_mode,
            chunk_shuffle={"chunk_size": 4, "k": 3} if shuffle_mode == "chunk" else None,
            ref_columns={
                "image_ref": {
                    "uri": ref_path,
                    "key_column": "image_id",
                    "value_column": "image_value",
                }
            },
        )
        if not resolve_ref:
            return dataset
        return dataset.resolve_ref(
            ["image_ref"],
            resolve_batch_size=3,
            index={
                "scope": "process",
                "build_strategy": "bucketed",
                "bucket_count": 3,
            },
        )

    state = _assert_dataset_resume_matches_continued(build_dataset, checkpoint_after=5)
    expected_image_refs = (
        {f"resolved-{index}" for index in range(5)} if resolve_ref else {f"img-{index}" for index in range(5)}
    )
    observed_image_refs = {sample["image_ref"] for sample in _remaining(iter(build_dataset()))}

    assert observed_image_refs == expected_image_refs
    assert state["source"]["state"]["shuffle_mode"] == shuffle_mode
    assert [stage["kind"] for stage in state["stages"]] == (["assemble"] if resolve_ref else [])


@pytest.mark.parametrize("shuffle_mode", ["none", "global", "chunk"])
def test_lance_non_global_shuffle_do_not_materialize_full_round_order(tmp_path, shuffle_mode: str) -> None:
    pytest.importorskip("lance")

    path = write_lance_dataset(tmp_path, build_records(count=16))
    iterator = iter(
        Dataset.from_source(
            "lance",
            shards=path,
            context=RuntimeContext(seed=31),
            read_batch_size=4,
            shuffle_mode=shuffle_mode,
            chunk_shuffle={"chunk_size": 4, "k": 3} if shuffle_mode == "chunk" else None,
        )
    )

    _consume(iterator, 5)

    assert not hasattr(iterator.source, "_index_order")
    assert not hasattr(iterator.source, "_index_order_round")


def test_resume_rejects_source_fingerprint_mismatch(tmp_path) -> None:
    build_dataset = _source_factory(tmp_path, "jsonl")
    dataset = build_dataset()
    with pytest.warns(UserWarning, match="Dataset.state_dict"):
        state = dataset.state_dict()
    state["source"]["fingerprint"] = "changed"

    with pytest.warns(UserWarning, match="Dataset.load_state_dict"):
        resumed_dataset = build_dataset().load_state_dict(state)

    with pytest.raises(ResumeStateError, match=r"\[ResumeSourceMismatch\]"):
        list(resumed_dataset)


def test_dataset_state_dict_returns_initial_iterator_state(tmp_path) -> None:
    build_dataset = _source_factory(tmp_path, "jsonl")

    with pytest.warns(UserWarning, match="Dataset.state_dict"):
        state = build_dataset().state_dict()
    with pytest.warns(UserWarning, match="Dataset.load_state_dict"):
        resumed_dataset = build_dataset().load_state_dict(state)

    assert state["num_yielded"] == 0
    assert state["source"]["state"]
    assert _remaining(iter(resumed_dataset)) == _remaining(iter(build_dataset()))


def test_iterators_can_checkpoint_independently(tmp_path) -> None:
    build_dataset = _source_factory(tmp_path, "jsonl", seed=37)
    dataset = build_dataset()
    first_iterator = iter(dataset)
    second_iterator = iter(dataset)

    first_consumed = _consume(first_iterator, 1)
    second_consumed = _consume(second_iterator, 3)
    first_state = first_iterator.state_dict()
    second_state = second_iterator.state_dict()
    first_continued = _remaining(first_iterator)
    second_continued = _remaining(second_iterator)
    expected = _remaining(iter(build_dataset()))

    with pytest.warns(UserWarning, match="Dataset.load_state_dict"):
        first_resumed = _remaining(iter(build_dataset().load_state_dict(first_state)))
    with pytest.warns(UserWarning, match="Dataset.load_state_dict"):
        second_resumed = _remaining(iter(build_dataset().load_state_dict(second_state)))

    assert first_consumed + first_continued == expected
    assert second_consumed + second_continued == expected
    assert first_resumed == first_continued
    assert second_resumed == second_continued


def test_assemble_rejects_non_stateful_assembler(tmp_path) -> None:
    build_dataset = _source_factory(tmp_path, "jsonl")
    dataset = build_dataset().assemble(_build_non_stateful_assembler)

    with pytest.raises(UnsupportedResume, match=r"\[UnsupportedResume\] stage kind=.*assemble"):
        iter(dataset)


def test_assemble_rejects_assembler_fingerprint_change(tmp_path) -> None:
    global _ASSEMBLER_FINGERPRINT_VERSION

    build_source = _source_factory(tmp_path, "jsonl")
    iterator = iter(build_source().assemble(_build_pair_output_assembler))
    next(iterator)
    state = iterator.state_dict()

    _ASSEMBLER_FINGERPRINT_VERSION = "v2"
    try:
        with pytest.raises(ResumeStateError, match=r"\[ResumePipelineMismatch\]"):
            build_source().assemble(_build_pair_output_assembler).load_state_dict(state)
    finally:
        _ASSEMBLER_FINGERPRINT_VERSION = "v1"


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
@pytest.mark.parametrize("num_workers", [0, 2])
def test_torch_loader_resume_full_pipeline_matches_continued_stream(tmp_path, num_workers: int) -> None:
    build_dataset = _source_factory(tmp_path, "tar", seed=41)

    def build_loader() -> TorchLoader:
        loader = _resume_torch_loader(
            build_dataset(),
            num_workers=num_workers,
            batch_size=2,
            collate_fn=_collate_columns,
            prefetch_factor=2,
            persistent_workers=num_workers > 0,
            pin_memory=num_workers > 0,
        )
        return _full_loader_pipeline(loader)

    state = _assert_loader_resume_matches_continued(build_loader, checkpoint_after=1)

    assert [stage["kind"] for stage in state["stages"]] == ["unbatch", "shuffle", "batch", "unbatch", "assemble"]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_torch_loader_state_dict_is_initial_not_active_iterator_state(tmp_path) -> None:
    build_dataset = _source_factory(tmp_path, "jsonl", seed=43)
    loader = _resume_torch_loader(build_dataset(), num_workers=0, batch_size=None)
    iterator = iter(loader)

    next(iterator)
    iterator_state = iterator.state_dict()
    with pytest.warns(UserWarning, match="TorchLoader.state_dict"):
        loader_state = loader.state_dict()

    assert iterator_state["num_yielded"] == 1
    assert loader_state["num_yielded"] == 0
    assert loader_state["workers"] == {}


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_torch_loader_resume_rejects_loader_config_change(tmp_path) -> None:
    build_dataset = _source_factory(tmp_path, "jsonl", seed=47)
    with pytest.warns(UserWarning, match="TorchLoader.state_dict"):
        state = _resume_torch_loader(build_dataset(), num_workers=0, batch_size=2).state_dict()
    changed_loader = _resume_torch_loader(build_dataset(), num_workers=0, batch_size=3)

    with pytest.raises(ResumeStateError, match=r"\[ResumeLoaderMismatch\]"):
        changed_loader.load_state_dict(state)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_torch_loader_plain_iterable_iteration_does_not_support_resume_state() -> None:
    assert list(TorchLoader([1, 2, 3], num_workers=0)) == [1, 2, 3]

    iterator = iter(TorchLoader([1, 2, 3], num_workers=0))
    assert next(iterator) == 1
    with pytest.raises(UnsupportedResume, match=r"\[UnsupportedResume\] TorchLoader dataset iterator"):
        iterator.state_dict()


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_torch_loader_assemble_rejects_non_stateful_assembler(tmp_path) -> None:
    build_dataset = _source_factory(tmp_path, "jsonl", seed=53)
    loader = _resume_torch_loader(build_dataset(), num_workers=0, batch_size=None).assemble(
        _build_non_stateful_assembler
    )

    with pytest.raises(UnsupportedResume, match=r"\[UnsupportedResume\] loader stage kind=.*assemble"):
        iter(loader)
