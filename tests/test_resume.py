from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pytest

from mvp_dataset import (
    Dataset,
    ResumeStateError,
    RuntimeContext,
    TorchLoader,
    UnsupportedResume,
    reset_logger,
    set_logger,
)
from mvp_dataset.core.resume import RESUME_STATE_VERSION, check_identity, identity

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

    def identity(self) -> str:
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

    resumed_dataset = build_dataset()
    resumed_dataset.load_state_dict(state)
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

    resumed_loader = build_loader()
    resumed_loader.load_state_dict(state)
    resumed = _remaining(iter(resumed_loader))

    assert consumed + continued == expected
    assert resumed == continued
    return state


def test_identity_includes_function_and_callable_class_code() -> None:
    first_fn = identity(_add_marker)
    second_fn = identity(_add_marker_v2)
    first_callable = identity(_CallablePlusOne())
    second_callable = identity(_CallablePlusTwo())

    assert first_fn != second_fn
    assert first_callable != second_callable
    assert first_fn["source_hash"] is not None
    assert first_callable["source_hash"] is not None


class _AddressedHandler:
    def __init__(self, pad: int) -> None:
        self.pad = pad
        self.live = object()


class _StableHandler:
    def __init__(self, pad: int) -> None:
        self.pad = pad

    def identity(self) -> str:
        return f"handler:{self.pad}"


@dataclass
class _SampleSpec:
    schema_handler: object


def _from_row(row: object, *, sample_spec: _SampleSpec) -> object:
    del sample_spec
    return row


class _ScalarCollator:
    def __init__(self, pad_token_id: int, ignore_index: int = -100) -> None:
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, batch: list[object]) -> list[object]:
        return batch


def test_identity_methodcaller_includes_name_and_kwargs() -> None:
    from operator import methodcaller

    first = identity(methodcaller("to_model_inputs"))
    second = identity(methodcaller("to_model_inputs"))
    renamed = identity(methodcaller("to_tokens"))
    flagged = identity(methodcaller("to_model_inputs", load_media=True))

    assert first == second
    assert first != renamed
    assert first != flagged
    assert first["kind"] == "methodcaller"
    assert first["name"] == "to_model_inputs"
    assert flagged["keywords"] == [{"key": "load_media", "value": True}]


def test_identity_uses_nested_identity_not_repr() -> None:
    first = identity(partial(_from_row, sample_spec=_SampleSpec(_StableHandler(1))))
    second = identity(partial(_from_row, sample_spec=_SampleSpec(_StableHandler(1))))
    changed = identity(partial(_from_row, sample_spec=_SampleSpec(_StableHandler(2))))

    encoded = json.dumps(first)
    assert first == second
    assert first != changed
    assert " at 0x" not in encoded
    keywords = {item["key"]: item["value"] for item in first["keywords"]}
    assert keywords["sample_spec"]["fields"]["schema_handler"]["id"] == "handler:1"


def test_identity_rejects_address_bearing_nested_object() -> None:
    fn = partial(_from_row, sample_spec=_SampleSpec(_AddressedHandler(1)))
    with pytest.raises(ResumeStateError, match=r"\[UnstableResumeIdentity\]"):
        identity(fn)


def test_identity_typeerror_from_method_propagates() -> None:
    class _Broken:
        def identity(self) -> str:
            raise TypeError("broken identity")

    with pytest.raises(TypeError, match="broken identity"):
        identity(_Broken())


def test_check_identity_includes_both_values() -> None:
    with pytest.raises(ResumeStateError, match=r"path=identity.seed expected=1 actual=2"):
        check_identity({"seed": 1}, {"seed": 2})


def test_identity_distinguishes_bound_methods() -> None:
    class _Owner:
        def __init__(self, value: int) -> None:
            self.value = value

        def method(self, sample: object) -> object:
            return sample

    assert identity(_Owner(1).method) != identity(_Owner(2).method)


def test_identity_distinguishes_closures() -> None:
    def make_fn(marker: int):
        def fn(sample: object) -> object:
            return {**sample, "marker": marker} if isinstance(sample, dict) else sample

        return fn

    assert identity(make_fn(1)) != identity(make_fn(2))


def test_identity_rejects_non_scalar_mapping_keys() -> None:
    with pytest.raises(ResumeStateError, match=r"\[UnstableResumeIdentity\]"):
        identity({object(): 1})


def test_identity_distinguishes_int_and_str_mapping_keys() -> None:
    assert identity({1: "a"}) != identity({"1": "a"})


def test_identity_includes_callable_instance_state() -> None:
    assert identity(_ScalarCollator(0)) == identity(_ScalarCollator(0))
    assert identity(_ScalarCollator(0)) != identity(_ScalarCollator(1))


def test_identity_walks_dataclass_and_collections() -> None:
    payload = identity({"flags": {"b", "a"}, "spec": _SampleSpec(_StableHandler(3))})
    items = {item["key"]: item["value"] for item in payload["items"]}
    assert items["flags"]["items"] == ["a", "b"]
    assert items["spec"]["fields"]["schema_handler"]["id"] == "handler:3"


def test_identity_partial_is_stable_across_hash_seeds() -> None:
    script = textwrap.dedent(
        """
        from dataclasses import dataclass
        from functools import partial
        from mvp_dataset.core.resume import digest, identity

        class Handler:
            def __init__(self, pad: int) -> None:
                self.pad = pad
            def identity(self) -> str:
                return f"handler:{self.pad}"

        @dataclass
        class Spec:
            schema_handler: object

        def from_row(row, *, sample_spec):
            return row

        print(digest(identity(partial(from_row, sample_spec=Spec(Handler(1))))))
        """
    )
    env_base = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    outputs = []
    for seed in ("1", "2"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env={**env_base, "PYTHONHASHSEED": seed},
        )
        outputs.append(result.stdout.strip())
    assert outputs[0]
    assert outputs[0] == outputs[1]


def test_load_state_dict_rejects_unknown_schema_version(tmp_path) -> None:
    dataset = _source_factory(tmp_path, "jsonl")()

    with pytest.raises(ResumeStateError, match=r"\[InvalidResumeStateVersion\]"):
        dataset.load_state_dict({"version": RESUME_STATE_VERSION + 1})


def test_load_state_dict_rejects_runtime_identity_mismatch(tmp_path) -> None:
    build_dataset = _source_factory(tmp_path, "jsonl", seed=1)
    state = iter(build_dataset()).state_dict()
    changed_runtime = _source_factory(tmp_path, "jsonl", seed=2)()

    with pytest.raises(ResumeStateError, match=r"\[ResumeIdentityMismatch\]"):
        changed_runtime.load_state_dict(state)


def test_load_state_dict_rejects_pipeline_identity_mismatch(tmp_path) -> None:
    build_dataset = _source_factory(tmp_path, "jsonl")
    state = iter(_full_dataset_pipeline(build_dataset())).state_dict()
    changed_pipeline = build_dataset().map(_add_marker).select(["id", "marker"]).shuffle(buffer_size=4)

    with pytest.raises(ResumeStateError, match=r"\[ResumeIdentityMismatch\]"):
        changed_pipeline.load_state_dict(state)


def test_load_state_dict_stages_pending_on_same_dataset(tmp_path) -> None:
    dataset = _source_factory(tmp_path, "jsonl")()
    state = iter(dataset).state_dict()
    dataset.load_state_dict(state)
    assert dataset._pending_state == state["state"]


def test_pending_state_is_consumed_once(tmp_path) -> None:
    build_dataset = _source_factory(tmp_path, "jsonl")
    iterator = iter(build_dataset())
    _consume(iterator, 3)
    state = iterator.state_dict()
    continued = _remaining(iterator)

    resumed = build_dataset()
    resumed.load_state_dict(state)
    first = _remaining(iter(resumed))
    second = _remaining(iter(resumed))

    assert first == continued
    assert second == _remaining(iter(build_dataset()))


def test_dataset_iterator_load_accepts_own_envelope(tmp_path) -> None:
    build_dataset = _source_factory(tmp_path, "jsonl")
    iterator = iter(build_dataset())
    _consume(iterator, 2)
    blob = iterator.state_dict()
    continued = _remaining(iterator)

    resumed_iter = iter(build_dataset())
    resumed_iter.load_state_dict(blob)
    assert _remaining(resumed_iter) == continued


def test_live_state_does_not_embed_identity(tmp_path) -> None:
    iterator = iter(_source_factory(tmp_path, "jsonl")())
    live = iterator.live_state()
    assert "identity" not in live
    assert "version" not in live


def test_exhausted_iterator_checkpoint_has_null_state(tmp_path) -> None:
    build_dataset = _source_factory(tmp_path, "jsonl")
    iterator = iter(build_dataset())
    _remaining(iterator)
    blob = iterator.state_dict()
    assert blob["state"] is None
    resumed = build_dataset()
    resumed.load_state_dict(blob)
    assert _remaining(iter(resumed)) == _remaining(iter(build_dataset()))


def test_runtime_context_identity_is_stable_and_seed_sensitive() -> None:
    assert RuntimeContext(seed=1).identity() == RuntimeContext(seed=1).identity()
    assert RuntimeContext(seed=1).identity() != RuntimeContext(seed=2).identity()


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

    assert [stage["kind"] for stage in state["identity"]["stages"]] == [
        "map",
        "select",
        "shuffle",
        "batch",
        "unbatch",
        "assemble",
    ]


@pytest.mark.parametrize("checkpoint_after", [0, 9])
def test_dataset_resume_full_pipeline_covers_initial_and_end_checkpoints(tmp_path, checkpoint_after: int) -> None:
    build_source = _source_factory(tmp_path, "jsonl", seed=17)

    _assert_dataset_resume_matches_continued(lambda: _full_dataset_pipeline(build_source()), checkpoint_after)


@pytest.mark.parametrize("rank", [0, 1])
def test_dataset_resume_full_pipeline_with_distributed_context(tmp_path, rank: int) -> None:
    build_source = _source_factory(tmp_path, "jsonl", seed=19, rank=rank, world_size=2)

    state = _assert_dataset_resume_matches_continued(lambda: _full_dataset_pipeline(build_source()), checkpoint_after=1)

    assert state["identity"]["runtime"] == _full_dataset_pipeline(build_source()).context.identity()


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

    resumed_dataset = build_source()
    resumed_dataset.load_state_dict(state)
    resumed = _consume(iter(resumed_dataset), 6)

    assert state["state"]["source"]["round_index"] >= 1
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
    assert state["identity"]["source"]["shuffle_mode"] == shuffle_mode
    assert [stage["kind"] for stage in state["identity"]["stages"]] == (["assemble"] if resolve_ref else [])


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


def test_resume_rejects_source_identity_mismatch(tmp_path) -> None:
    build_dataset = _source_factory(tmp_path, "jsonl")
    state = build_dataset().state_dict()
    state["identity"]["source"]["source_fingerprint"] = "changed"
    resumed_dataset = build_dataset()
    with pytest.raises(ResumeStateError, match=r"\[ResumeIdentityMismatch\]"):
        resumed_dataset.load_state_dict(state)


def test_dataset_state_dict_without_live_iterator_replays_from_start(tmp_path) -> None:
    build_dataset = _source_factory(tmp_path, "jsonl")
    state = build_dataset().state_dict()
    resumed_dataset = build_dataset()
    resumed_dataset.load_state_dict(state)

    assert state["state"] is None
    assert _remaining(iter(resumed_dataset)) == _remaining(iter(build_dataset()))


def test_handle_state_dict_saves_active_iterator(tmp_path) -> None:
    build_dataset = _source_factory(tmp_path, "jsonl")
    dataset = build_dataset()
    iterator = iter(dataset)
    consumed = _consume(iterator, 3)
    state = dataset.state_dict()
    continued = _remaining(iterator)

    assert state["state"] is not None
    resumed = build_dataset()
    resumed.load_state_dict(state)
    assert _remaining(iter(resumed)) == continued
    assert consumed + continued == _remaining(iter(build_dataset()))


def test_handle_state_dict_is_fresh_after_iterator_exhausts(tmp_path) -> None:
    dataset = _source_factory(tmp_path, "jsonl")()
    _remaining(iter(dataset))
    assert dataset.state_dict()["state"] is None


class _ListLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def debug(self, msg: object, *args: object, **kwargs: object) -> None:
        return None

    def info(self, msg: object, *args: object, **kwargs: object) -> None:
        return None

    def warning(self, msg: object, *args: object, **kwargs: object) -> None:
        self.messages.append(str(msg))

    def error(self, msg: object, *args: object, **kwargs: object) -> None:
        return None


def test_iter_warns_only_while_previous_iterator_is_active(tmp_path) -> None:
    logger = _ListLogger()
    set_logger(logger)
    try:
        dataset = _source_factory(tmp_path, "jsonl")()
        first = iter(dataset)
        second = iter(dataset)
        assert any("previous iterator is still active" in message for message in logger.messages)
        _remaining(first)
        _remaining(second)
        logger.messages.clear()
        iter(dataset)
        assert not any("previous iterator is still active" in message for message in logger.messages)
    finally:
        reset_logger()


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

    first_handle = build_dataset()
    first_handle.load_state_dict(first_state)
    first_resumed = _remaining(iter(first_handle))
    second_handle = build_dataset()
    second_handle.load_state_dict(second_state)
    second_resumed = _remaining(iter(second_handle))

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
        changed = build_source().assemble(_build_pair_output_assembler)
        with pytest.raises(ResumeStateError, match=r"\[ResumeIdentityMismatch\]"):
            changed.load_state_dict(state)
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

    assert [stage["kind"] for stage in state["identity"]["loader"]["stages"]] == [
        "unbatch",
        "shuffle",
        "batch",
        "unbatch",
        "assemble",
    ]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_torch_loader_state_dict_is_initial_not_active_iterator_state(tmp_path) -> None:
    build_dataset = _source_factory(tmp_path, "jsonl", seed=43)
    loader = _resume_torch_loader(build_dataset(), num_workers=0, batch_size=None)
    iterator = iter(loader)

    next(iterator)
    iterator_state = iterator.state_dict()
    loader_state = loader.state_dict()

    assert iterator_state["state"]["num_yielded"] == 1
    assert loader_state["state"] is None


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_torch_loader_resume_rejects_loader_config_change(tmp_path) -> None:
    build_dataset = _source_factory(tmp_path, "jsonl", seed=47)
    state = _resume_torch_loader(build_dataset(), num_workers=0, batch_size=2).state_dict()
    changed_loader = _resume_torch_loader(build_dataset(), num_workers=0, batch_size=3)

    with pytest.raises(ResumeStateError, match=r"\[ResumeIdentityMismatch\]"):
        changed_loader.load_state_dict(state)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_torch_loader_plain_iterable_iteration_does_not_support_resume_state() -> None:
    assert list(TorchLoader([1, 2, 3], num_workers=0)) == [1, 2, 3]

    iterator = iter(TorchLoader([1, 2, 3], num_workers=0))
    assert next(iterator) == 1
    with pytest.raises(
        UnsupportedResume, match=r"\[UnsupportedResume\] TorchLoader dataset does not implement identity"
    ):
        iterator.state_dict()


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_torch_loader_assemble_rejects_non_stateful_assembler(tmp_path) -> None:
    build_dataset = _source_factory(tmp_path, "jsonl", seed=53)
    loader = _resume_torch_loader(build_dataset(), num_workers=0, batch_size=None).assemble(
        _build_non_stateful_assembler
    )

    with pytest.raises(UnsupportedResume, match=r"\[UnsupportedResume\] loader stage kind=.*assemble"):
        iter(loader)
