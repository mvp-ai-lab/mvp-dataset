#!/usr/bin/env python3
"""Benchmark read throughput for user-provided mvp-dataset sources.

Examples:
    python examples/benchmark.py \
      --source parquet=parquet:/data/openbee/**/*.parquet \
      --source lance=lance:/data/openbee-lance/ref_columns.json \
      --max-samples 10000 --repeats 3

    python examples/benchmark.py \
      --source parquet=parquet:/data/openbee/Caption \
      --source lance=lance:/data/openbee-lance/stage3-dev/ref_columns.json \
      --columns images,conversations \
      --use-loader --num-workers 4 --worker-batch-size 8 \
      --prefetch-factor 1 --multiprocessing-context spawn
"""

from __future__ import annotations

import argparse
import glob
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from mvp_dataset import Dataset, TorchLoader
from mvp_dataset.core.context import RuntimeContext

SOURCE_KINDS = {"jsonl", "tars", "parquet", "lance"}


@dataclass(frozen=True, slots=True)
class SourceSpec:
    name: str
    kind: str
    path: str


@dataclass(frozen=True, slots=True)
class TimingResult:
    samples: int
    seconds: float
    payload_bytes: int | None = None

    @property
    def samples_per_second(self) -> float:
        return self.samples / self.seconds if self.seconds > 0 else float("inf")

    @property
    def mb_per_second(self) -> float | None:
        if self.payload_bytes is None:
            return None
        return (self.payload_bytes / 1024 / 1024) / self.seconds if self.seconds > 0 else float("inf")


def identity_collate(batch: list[object]) -> list[object]:
    return batch


def _parse_source_spec(raw_spec: str) -> SourceSpec:
    """Parse NAME=KIND:PATH or KIND:PATH into a SourceSpec."""

    if "=" in raw_spec:
        name, raw_kind_path = raw_spec.split("=", 1)
        name = name.strip()
        if not name:
            msg = f"[InvalidBenchmarkSource] source name is empty in {raw_spec!r}"
            raise ValueError(msg)
    else:
        name = ""
        raw_kind_path = raw_spec

    if ":" not in raw_kind_path:
        msg = f"[InvalidBenchmarkSource] expected --source NAME=KIND:PATH or KIND:PATH, got {raw_spec!r}"
        raise ValueError(msg)

    kind, path = raw_kind_path.split(":", 1)
    kind = kind.strip()
    path = path.strip()
    if kind not in SOURCE_KINDS:
        msg = f"[InvalidBenchmarkSource] kind must be one of {sorted(SOURCE_KINDS)}, got {kind!r}"
        raise ValueError(msg)
    if not path:
        msg = f"[InvalidBenchmarkSource] path is empty in {raw_spec!r}"
        raise ValueError(msg)

    if not name:
        path_name = Path(path).name or Path(path).parent.name or kind
        name = f"{kind}:{path_name}"

    return SourceSpec(name=name, kind=kind, path=path)


def _parse_columns(raw_columns: str | None) -> list[str] | None:
    if raw_columns is None:
        return None
    columns = [column.strip() for column in raw_columns.split(",") if column.strip()]
    return columns or None


def _expand_directory_source(spec: SourceSpec, *, recursive: bool) -> str | list[str]:
    """Expand glob and directory inputs where sensible."""

    expanded_path = str(Path(spec.path).expanduser())
    if any(char in expanded_path for char in "*?["):
        matches = sorted(glob.glob(expanded_path, recursive=True))
        if matches:
            return matches
        msg = f"[EmptyBenchmarkSource] glob did not match any paths: {expanded_path}"
        raise ValueError(msg)

    path = Path(expanded_path)
    if not path.is_dir() or spec.kind == "lance":
        return expanded_path

    suffix_by_kind = {
        "jsonl": ".jsonl",
        "parquet": ".parquet",
        "tars": ".tar",
    }
    suffix = suffix_by_kind.get(spec.kind)
    if suffix is None:
        return spec.path

    iterator = path.rglob(f"*{suffix}") if recursive else path.glob(f"*{suffix}")
    files = sorted(str(file_path) for file_path in iterator if file_path.is_file())
    if not files:
        msg = f"[EmptyBenchmarkSource] no {suffix} files found under {path}"
        raise ValueError(msg)
    return files


def _build_dataset(spec: SourceSpec, args: argparse.Namespace) -> Dataset:
    context = RuntimeContext(seed=args.seed)
    source_path = _expand_directory_source(spec, recursive=args.recursive)
    columns = _parse_columns(args.columns)

    if spec.kind == "parquet":
        return Dataset.from_source(
            "parquet",
            source_path,
            context=context,
            resample=args.resample,
            columns=columns,
            min_row_groups_per_fragment=args.parquet_min_row_groups_per_fragment,
            use_threads=args.parquet_use_threads,
        )

    if spec.kind == "lance":
        return Dataset.from_source(
            "lance",
            source_path,
            context=context,
            resample=args.resample,
            columns=columns,
            batch_size=args.lance_batch_size,
            global_shuffle=args.global_shuffle,
            load_in_memory=args.lance_load_in_memory,
        )

    if columns is not None:
        msg = f"[UnsupportedBenchmarkColumns] --columns is only supported for parquet/lance, got {spec.kind!r}"
        raise ValueError(msg)

    return Dataset.from_source(spec.kind, source_path, context=context, resample=args.resample)


def _iter_dataset(ds: Dataset, args: argparse.Namespace) -> Iterable[object]:
    if not args.use_loader:
        return ds

    return TorchLoader(
        ds,
        num_workers=args.num_workers,
        batch_size=args.worker_batch_size,
        collate_fn=identity_collate,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        multiprocessing_context=args.multiprocessing_context,
    ).unbatch()


def _payload_size(value: object) -> int:
    if isinstance(value, bytes | bytearray | memoryview):
        return len(value)
    if isinstance(value, dict):
        return sum(_payload_size(item) for item in value.values())
    if isinstance(value, list | tuple):
        return sum(_payload_size(item) for item in value)
    return 0


def _drain(ds: Dataset, args: argparse.Namespace, *, limit: int) -> TimingResult:
    count = 0
    payload_bytes = 0 if args.measure_bytes else None
    started_at = time.perf_counter()

    for sample in _iter_dataset(ds, args):
        count += 1
        if payload_bytes is not None:
            payload_bytes += _payload_size(sample)
        if limit > 0 and count >= limit:
            break

    return TimingResult(
        samples=count,
        seconds=time.perf_counter() - started_at,
        payload_bytes=payload_bytes,
    )


def _time_source(spec: SourceSpec, args: argparse.Namespace) -> list[TimingResult]:
    results: list[TimingResult] = []

    if args.warmup_samples > 0:
        print(f"  warmup {spec.name}: {args.warmup_samples} samples", flush=True)
        ds = _build_dataset(spec, args)
        _drain(ds, args, limit=args.warmup_samples)

    for repeat_i in range(args.repeats):
        print(f"  run {spec.name}: repeat {repeat_i + 1}/{args.repeats}", flush=True)
        ds = _build_dataset(spec, args)
        results.append(_drain(ds, args, limit=args.max_samples))

    return results


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _format_result(name: str, kind: str, results: list[TimingResult]) -> str:
    samples = [result.samples for result in results]
    seconds = [result.seconds for result in results]
    rates = [result.samples_per_second for result in results]
    best_rate = max(rates)
    avg_rate = _mean(rates)
    avg_seconds = _mean(seconds)
    sample_label = str(min(samples)) if min(samples) == max(samples) else f"{min(samples)}-{max(samples)}"

    parts = [
        f"{name:<24}",
        f"{kind:<8}",
        f"samples {sample_label:>9}",
        f"avg {avg_seconds:8.3f}s",
        f"avg {avg_rate:10.1f} samples/s",
        f"best {best_rate:10.1f} samples/s",
    ]

    mbps_values = [result.mb_per_second for result in results if result.mb_per_second is not None]
    if mbps_values:
        parts.append(f"avg {_mean(mbps_values):9.1f} MB/s")

    return "  ".join(parts)


def _print_report(all_results: dict[SourceSpec, list[TimingResult]]) -> None:
    print()
    print("=" * 118)
    print("mvp-dataset read benchmark")
    print("=" * 118)
    for spec, results in all_results.items():
        print(_format_result(spec.name, spec.kind, results))
    print("=" * 118)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark read speed for user-provided mvp-dataset sources")
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help=("Source spec. Repeatable. Format: NAME=KIND:PATH or KIND:PATH. Kinds: jsonl, tars, parquet, lance."),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10_000,
        help="Samples to read per repeat. Use 0 to drain finite sources fully.",
    )
    parser.add_argument("--warmup-samples", type=int, default=0, help="Optional warmup samples before timing")
    parser.add_argument("--repeats", type=int, default=3, help="Timed repeats per source")
    parser.add_argument("--seed", type=int, default=42, help="Runtime context seed")
    parser.add_argument("--resample", action="store_true", help="Loop sources indefinitely")
    parser.add_argument("--recursive", action="store_true", help="When a source path is a directory, scan recursively")
    parser.add_argument(
        "--columns",
        default=None,
        help="Comma-separated projection for parquet/lance sources, for example images,conversations",
    )
    parser.add_argument(
        "--measure-bytes",
        action="store_true",
        help="Recursively count bytes payloads and report approximate MB/s. Adds CPU overhead.",
    )

    parser.add_argument("--parquet-min-row-groups-per-fragment", type=int, default=1)
    parser.add_argument(
        "--parquet-use-threads",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Arrow threaded Parquet reads",
    )
    parser.add_argument("--lance-batch-size", type=int, default=1024)
    parser.add_argument("--global-shuffle", action="store_true", help="Enable Lance global row shuffle")
    parser.add_argument(
        "--lance-load-in-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load Lance tables in memory before reading. Not recommended for large media columns.",
    )

    parser.add_argument("--use-loader", action="store_true", help="Wrap each Dataset with TorchLoader")
    parser.add_argument("--num-workers", type=int, default=0, help="TorchLoader worker count")
    parser.add_argument("--worker-batch-size", type=int, default=16, help="TorchLoader worker micro-batch size")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--multiprocessing-context", choices=["fork", "spawn", "forkserver"], default=None)
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("--repeats must be > 0")
    if args.max_samples < 0:
        raise ValueError("--max-samples must be >= 0")
    if args.warmup_samples < 0:
        raise ValueError("--warmup-samples must be >= 0")
    if args.use_loader and args.worker_batch_size <= 0:
        raise ValueError("--worker-batch-size must be > 0 when --use-loader is set")
    if args.resample and args.max_samples <= 0:
        raise ValueError("--max-samples must be > 0 when --resample is set")

    source_specs = [_parse_source_spec(raw_spec) for raw_spec in args.source]
    duplicate_names = {spec.name for spec in source_specs if [item.name for item in source_specs].count(spec.name) > 1}
    if duplicate_names:
        raise ValueError(f"duplicate source names: {sorted(duplicate_names)}")

    all_results: dict[SourceSpec, list[TimingResult]] = {}
    for spec in source_specs:
        print(f"benchmarking {spec.name} ({spec.kind}) from {spec.path}", flush=True)
        all_results[spec] = _time_source(spec, args)

    _print_report(all_results)


if __name__ == "__main__":
    main()
