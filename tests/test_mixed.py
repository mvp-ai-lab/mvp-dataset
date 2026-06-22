from __future__ import annotations

import pytest

from mvp_dataset import Dataset, RuntimeContext

from .helpers import write_jsonl_file, write_parquet_file


def _records(prefix: str, count: int) -> list[dict[str, object]]:
    return [{"id": f"{prefix}-{index}", "text": f"text-{index}", "value": index} for index in range(count)]


def _ids(stream) -> list[str]:
    return [str(sample["id"]) for sample in stream]


def _build_mixed_dataset(
    tmp_path,
    *,
    context: RuntimeContext | None = None,
    strategy: str = "weighted_round_robin",
) -> Dataset:
    runtime_context = RuntimeContext(seed=11) if context is None else context
    jsonl_dir = tmp_path / "jsonl"
    parquet_dir = tmp_path / "parquet"
    jsonl_dir.mkdir(exist_ok=True)
    parquet_dir.mkdir(exist_ok=True)

    jsonl = Dataset.from_source(
        "jsonl",
        shards=write_jsonl_file(jsonl_dir, _records("jsonl", 4)),
        context=runtime_context,
        shuffle_mode="none",
    )
    parquet = Dataset.from_source(
        "parquet",
        shards=write_parquet_file(parquet_dir, _records("parquet", 2)),
        context=runtime_context,
        shuffle_mode="none",
    )
    return Dataset.from_source(
        "mixed",
        sources={"jsonl": jsonl, "parquet": parquet},
        strategy=strategy,
        weights={"jsonl": 2, "parquet": 1},
    )


@pytest.mark.parametrize(
    ("strategy", "expected_ids", "expected_sources"),
    [
        (
            "concat",
            ["jsonl-0", "jsonl-1", "jsonl-2", "jsonl-3", "parquet-0", "parquet-1"],
            ["jsonl", "jsonl", "jsonl", "jsonl", "parquet", "parquet"],
        ),
        (
            "round_robin",
            ["jsonl-0", "parquet-0", "jsonl-1", "parquet-1", "jsonl-2", "jsonl-3"],
            ["jsonl", "parquet", "jsonl", "parquet", "jsonl", "jsonl"],
        ),
        (
            "weighted_round_robin",
            ["jsonl-0", "parquet-0", "jsonl-1", "jsonl-2", "parquet-1", "jsonl-3"],
            ["jsonl", "parquet", "jsonl", "jsonl", "parquet", "jsonl"],
        ),
    ],
)
def test_mixed_source_uses_ordered_strategy(tmp_path, strategy, expected_ids, expected_sources) -> None:
    dataset = _build_mixed_dataset(tmp_path, strategy=strategy)

    observed = list(dataset)

    assert _ids(observed) == expected_ids
    assert [sample["__source__"] for sample in observed] == expected_sources


@pytest.mark.parametrize("strategy", ["random", "weighted_random"])
def test_mixed_source_random_strategy_is_deterministic(tmp_path, strategy) -> None:
    dataset = _build_mixed_dataset(tmp_path, strategy=strategy)

    first = list(dataset)
    second = list(dataset)

    assert _ids(first) == _ids(second)
    assert sorted(_ids(first)) == ["jsonl-0", "jsonl-1", "jsonl-2", "jsonl-3", "parquet-0", "parquet-1"]


def test_mixed_source_resume_matches_continued_iteration(tmp_path) -> None:
    dataset = _build_mixed_dataset(tmp_path)
    iterator = iter(dataset)

    consumed = [next(iterator) for _ in range(3)]
    state = iterator.state_dict()
    continued = list(iterator)

    with pytest.warns(UserWarning, match="Dataset.load_state_dict"):
        resumed = dataset.load_state_dict(state)

    assert _ids(consumed + continued) == _ids(dataset)
    assert _ids(resumed) == _ids(continued)


def test_mixed_source_rejects_context_mismatch(tmp_path) -> None:
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left_dir.mkdir()
    right_dir.mkdir()
    left = Dataset.from_source(
        "jsonl",
        shards=write_jsonl_file(left_dir, _records("left", 1)),
        context=RuntimeContext(seed=1),
    )
    right = Dataset.from_source(
        "jsonl",
        shards=write_jsonl_file(right_dir, _records("right", 1)),
        context=RuntimeContext(seed=2),
    )

    with pytest.raises(ValueError, match="MixedSourceContextMismatch"):
        Dataset.from_source("mixed", sources={"left": left, "right": right})
