"""CLI helpers for converting Parquet trees into Lance reference datasets."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import queue
import random
import re
import threading
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq

_END = object()
_REF_ID_COLUMN = "ref_id"
_SOURCE_COLUMN = "__source__"
_DEFAULT_BATCH_SIZE = 1024
_DEFAULT_QUEUE_SIZE = 1
_DEFAULT_MAX_ROWS_PER_FILE = 8192
_DEFAULT_MAX_ROWS_PER_GROUP = 512
_DEFAULT_MAX_BYTES_PER_FILE = 512 * 1024 * 1024
_DEFAULT_SCANNER_BATCH_READAHEAD = 1
_DEFAULT_SCANNER_FRAGMENT_READAHEAD = 1
_SAFE_MAX_QUEUE_SIZE = 16
_CONVERSION_MANIFEST = "ref_columns.json"
_SAFE_DATASET_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True, slots=True)
class ConvertPlan:
    input_path: Path
    output_path: Path
    parquet_files: tuple[Path, ...]
    ref_columns: tuple[str, ...]
    ref_dataset_names: dict[str, str]
    parquet_schema: pa.Schema
    main_schema: pa.Schema
    ref_schemas: dict[str, pa.Schema]
    total_rows: int
    batch_size: int
    workers: int
    queue_size: int
    max_rows_per_file: int
    max_rows_per_group: int
    max_bytes_per_file: int
    cleanup_old_versions: bool
    show_progress: bool
    parquet_reader: str
    parquet_use_threads: bool
    scanner_batch_readahead: int
    scanner_fragment_readahead: int

    @property
    def main_uri(self) -> Path:
        return self.output_path / "samples.lance"

    def ref_uri(self, column: str) -> Path:
        return self.output_path / self.ref_dataset_names[column]


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    plan = build_convert_plan(
        input_path=args.input,
        output_path=args.output,
        ref_columns=_parse_ref_columns(args.ref_columns),
        batch_size=args.batch_size,
        workers=args.workers,
        queue_size=args.queue_size,
        max_rows_per_file=args.max_rows_per_file,
        max_rows_per_group=args.max_rows_per_group,
        max_bytes_per_file=args.max_bytes_per_file,
        cleanup_old_versions=args.cleanup_old_versions,
        show_progress=not args.no_progress,
        parquet_reader=args.parquet_reader,
        parquet_use_threads=args.parquet_use_threads,
        scanner_batch_readahead=args.scanner_batch_readahead,
        scanner_fragment_readahead=args.scanner_fragment_readahead,
        allow_large_queue=args.allow_large_queue,
    )
    result = convert_parquets_to_lance(plan)
    print(f"converted {result['row_count']} rows from {result['parquet_count']} parquet file(s) to {plan.main_uri}")
    if plan.ref_columns:
        print(f"wrote {len(plan.ref_columns)} ref dataset(s) under {plan.output_path}")
        print(f"wrote ref column manifest to {plan.output_path / _CONVERSION_MANIFEST}")


def build_convert_plan(
    *,
    input_path: Path,
    output_path: Path,
    ref_columns: Sequence[str],
    batch_size: int = _DEFAULT_BATCH_SIZE,
    workers: int | None = None,
    queue_size: int | None = None,
    max_rows_per_file: int = _DEFAULT_MAX_ROWS_PER_FILE,
    max_rows_per_group: int = _DEFAULT_MAX_ROWS_PER_GROUP,
    max_bytes_per_file: int = _DEFAULT_MAX_BYTES_PER_FILE,
    cleanup_old_versions: bool = False,
    show_progress: bool = True,
    parquet_reader: str = "scanner",
    parquet_use_threads: bool = False,
    scanner_batch_readahead: int = _DEFAULT_SCANNER_BATCH_READAHEAD,
    scanner_fragment_readahead: int = _DEFAULT_SCANNER_FRAGMENT_READAHEAD,
    allow_large_queue: bool = False,
) -> ConvertPlan:
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    parquet_files = tuple(_discover_parquet_files(input_path))
    if not parquet_files:
        msg = f"[NoParquetFiles] no parquet files found under {input_path}"
        raise ValueError(msg)

    workers = _resolve_workers(workers, len(parquet_files))
    queue_size = queue_size if queue_size is not None else _DEFAULT_QUEUE_SIZE
    if batch_size <= 0:
        msg = f"[InvalidBatchSize] --batch-size must be > 0, got {batch_size}"
        raise ValueError(msg)
    if queue_size <= 0:
        msg = f"[InvalidQueueSize] --queue-size must be > 0, got {queue_size}"
        raise ValueError(msg)
    if queue_size > _SAFE_MAX_QUEUE_SIZE and not allow_large_queue:
        msg = (
            f"[UnsafeQueueSize] --queue-size={queue_size} is likely to OOM for large binary ref columns. "
            f"The value is per Lance writer, not global; use --queue-size 1 or pass --allow-large-queue "
            f"if you really want to allow more than {_SAFE_MAX_QUEUE_SIZE} queued batches per writer."
        )
        raise ValueError(msg)
    if max_rows_per_file <= 0:
        msg = f"[InvalidLanceRowsPerFile] --max-rows-per-file must be > 0, got {max_rows_per_file}"
        raise ValueError(msg)
    if max_rows_per_group <= 0:
        msg = f"[InvalidLanceRowsPerGroup] --max-rows-per-group must be > 0, got {max_rows_per_group}"
        raise ValueError(msg)
    if max_bytes_per_file <= 0:
        msg = f"[InvalidLanceBytesPerFile] --max-bytes-per-file must be > 0, got {max_bytes_per_file}"
        raise ValueError(msg)
    if parquet_reader not in {"scanner", "file"}:
        msg = f"[InvalidParquetReader] --parquet-reader must be 'scanner' or 'file', got {parquet_reader!r}"
        raise ValueError(msg)
    if scanner_batch_readahead <= 0:
        msg = f"[InvalidScannerBatchReadahead] --scanner-batch-readahead must be > 0, got {scanner_batch_readahead}"
        raise ValueError(msg)
    if scanner_fragment_readahead <= 0:
        msg = (
            f"[InvalidScannerFragmentReadahead] --scanner-fragment-readahead must be > 0, "
            f"got {scanner_fragment_readahead}"
        )
        raise ValueError(msg)

    parquet_schema, total_rows = _validate_parquet_schemas(parquet_files, workers=workers)
    normalized_ref_columns = tuple(dict.fromkeys(ref_columns))
    ref_dataset_names = _build_ref_dataset_names(normalized_ref_columns)
    main_schema, ref_schemas = _build_output_schemas(parquet_schema, normalized_ref_columns)

    return ConvertPlan(
        input_path=input_path,
        output_path=output_path,
        parquet_files=parquet_files,
        ref_columns=normalized_ref_columns,
        ref_dataset_names=ref_dataset_names,
        parquet_schema=parquet_schema,
        main_schema=main_schema,
        ref_schemas=ref_schemas,
        total_rows=total_rows,
        batch_size=batch_size,
        workers=workers,
        queue_size=queue_size,
        max_rows_per_file=max_rows_per_file,
        max_rows_per_group=max_rows_per_group,
        max_bytes_per_file=max_bytes_per_file,
        cleanup_old_versions=cleanup_old_versions,
        show_progress=show_progress,
        parquet_reader=parquet_reader,
        parquet_use_threads=parquet_use_threads,
        scanner_batch_readahead=scanner_batch_readahead,
        scanner_fragment_readahead=scanner_fragment_readahead,
    )


def convert_parquets_to_lance(plan: ConvertPlan) -> dict[str, Any]:
    producer_futures = []
    shuffle_files = list(plan.parquet_files)
    random.shuffle(shuffle_files)

    plan.output_path.mkdir(parents=True, exist_ok=True)

    output_queues: dict[str, queue.Queue[object]] = {
        "samples": queue.Queue(maxsize=plan.queue_size),
        **{column: queue.Queue(maxsize=plan.queue_size) for column in plan.ref_columns},
    }
    stop_event = threading.Event()
    row_counts: dict[Path, int] = {}
    progress_bar = _build_progress_bar(plan)

    with (
        progress_bar as progress,
        ThreadPoolExecutor(
            max_workers=len(output_queues),
            thread_name_prefix="lance-writer",
        ) as writer_pool,
    ):
        writer_futures = [
            writer_pool.submit(
                _write_lance_from_queue,
                output_queues["samples"],
                plan.main_uri,
                plan.main_schema,
                plan,
                stop_event,
            )
        ]
        for column in plan.ref_columns:
            writer_futures.append(
                writer_pool.submit(
                    _write_lance_from_queue,
                    output_queues[column],
                    plan.ref_uri(column),
                    plan.ref_schemas[column],
                    plan,
                    stop_event,
                )
            )

        try:
            with ThreadPoolExecutor(max_workers=plan.workers, thread_name_prefix="parquet-converter") as pool:
                producer_futures = [
                    pool.submit(_convert_parquet_file, parquet_path, plan, output_queues, stop_event, progress)
                    for parquet_path in shuffle_files
                ]
                for future in as_completed(producer_futures):
                    parquet_path, row_count = future.result()
                    row_counts[parquet_path] = row_count
                    if progress is not None:
                        progress.set_postfix_str(f"files={len(row_counts)}/{len(plan.parquet_files)}")
                    _raise_finished_writer_errors(writer_futures)
        except Exception as exc:
            stop_event.set()
            for future in producer_futures:
                future.cancel()
            _signal_outputs(output_queues.values(), exc, discard_when_full=True)
            raise
        else:
            _signal_outputs(output_queues.values(), _END)

        for future in writer_futures:
            future.result()

    if plan.cleanup_old_versions:
        _cleanup_lance_versions(plan)

    manifest = _write_conversion_manifest(plan, row_count=sum(row_counts.values()))
    return manifest


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert all parquet files under a path into Lance datasets.")
    parser.add_argument("--input", type=Path, required=True, help="Input parquet file or directory.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for generated Lance datasets.")
    parser.add_argument(
        "--ref-columns",
        type=str,
        default="",
        help="Comma-separated columns to extract into one separate .lance dataset per column.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_DEFAULT_BATCH_SIZE,
        help="Parquet record batch size used while converting. Lower this for large binary columns.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parquet conversion workers. Defaults to min(file_count, CPU count).",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=None,
        help="Maximum queued Arrow batches per output writer. Defaults to 1 to keep binary payload memory bounded.",
    )
    parser.add_argument(
        "--allow-large-queue",
        action="store_true",
        help="Allow --queue-size values above the built-in OOM safety threshold.",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=_DEFAULT_MAX_ROWS_PER_FILE,
        help="Lance writer max_rows_per_file.",
    )
    parser.add_argument(
        "--max-rows-per-group",
        type=int,
        default=_DEFAULT_MAX_ROWS_PER_GROUP,
        help="Lance writer max_rows_per_group.",
    )
    parser.add_argument(
        "--max-bytes-per-file",
        type=int,
        default=_DEFAULT_MAX_BYTES_PER_FILE,
        help="Lance writer max_bytes_per_file.",
    )
    parser.add_argument(
        "--cleanup-old-versions",
        action="store_true",
        help="After successful overwrite, keep only the latest Lance version for every generated dataset.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar.",
    )
    parser.add_argument(
        "--parquet-reader",
        choices=["scanner", "file"],
        default="scanner",
        help="Parquet reader backend. 'scanner' supports more nested schemas; 'file' is faster when it works.",
    )
    parser.add_argument(
        "--parquet-use-threads",
        action="store_true",
        help="Enable PyArrow's internal threaded Parquet decoding inside each converter worker.",
    )
    parser.add_argument(
        "--scanner-batch-readahead",
        type=int,
        default=_DEFAULT_SCANNER_BATCH_READAHEAD,
        help="Scanner batch readahead. Higher values can speed reads but use more memory.",
    )
    parser.add_argument(
        "--scanner-fragment-readahead",
        type=int,
        default=_DEFAULT_SCANNER_FRAGMENT_READAHEAD,
        help="Scanner fragment readahead. Higher values can speed reads but use more memory.",
    )
    return parser.parse_args(argv)


def _parse_ref_columns(raw_value: str) -> tuple[str, ...]:
    if not raw_value:
        return ()
    columns = tuple(column.strip() for column in raw_value.split(",") if column.strip())
    if not columns:
        return ()
    return columns


def _discover_parquet_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in {".parquet", ".parq"}:
            msg = f"[InvalidInputFile] expected a parquet file, got {input_path}"
            raise ValueError(msg)
        return [input_path]
    if not input_path.exists():
        msg = f"[InputNotFound] input path does not exist: {input_path}"
        raise FileNotFoundError(msg)
    if not input_path.is_dir():
        msg = f"[InvalidInputPath] input path must be a parquet file or directory: {input_path}"
        raise ValueError(msg)
    return sorted(
        path for path in input_path.rglob("*") if path.is_file() and path.suffix.lower() in {".parquet", ".parq"}
    )


def _resolve_workers(workers: int | None, parquet_count: int) -> int:
    if workers is None:
        workers = min(parquet_count, os.cpu_count() or 1)
    if workers <= 0:
        msg = f"[InvalidWorkers] --workers must be > 0, got {workers}"
        raise ValueError(msg)
    return min(workers, parquet_count)


def _validate_parquet_schemas(parquet_files: Sequence[Path], *, workers: int) -> tuple[pa.Schema, int]:
    first_file = pq.ParquetFile(parquet_files[0])
    first_schema = first_file.schema_arrow.remove_metadata()
    total_rows = first_file.metadata.num_rows
    if _SOURCE_COLUMN in first_schema.names:
        msg = f"[ReservedColumn] input parquet already contains reserved column {_SOURCE_COLUMN!r}"
        raise ValueError(msg)
    if not parquet_files[1:]:
        return first_schema, total_rows

    def read_schema(path: Path) -> tuple[Path, pa.Schema, int]:
        parquet_file = pq.ParquetFile(path)
        return path, parquet_file.schema_arrow.remove_metadata(), parquet_file.metadata.num_rows

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="parquet-schema") as pool:
        futures = [pool.submit(read_schema, path) for path in parquet_files[1:]]
        for future in as_completed(futures):
            path, schema, num_rows = future.result()
            if schema != first_schema:
                msg = (
                    f"[SchemaMismatch] parquet schema for {path} does not match {parquet_files[0]}\n"
                    f"expected: {first_schema}\n"
                    f"actual: {schema}"
                )
                raise ValueError(msg)
            total_rows += num_rows
    return first_schema, total_rows


def _build_progress_bar(plan: ConvertPlan):
    if not plan.show_progress:
        return contextlib.nullcontext(None)
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError as exc:
        msg = "[TqdmUnavailable] mvp_dataset.convert progress requires tqdm. Use --no-progress to disable it."
        raise RuntimeError(msg) from exc
    return tqdm(
        total=plan.total_rows,
        desc="convert",
        unit="row",
        unit_scale=True,
        dynamic_ncols=True,
        smoothing=0.1,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} rows [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )


def _build_ref_dataset_names(ref_columns: Sequence[str]) -> dict[str, str]:
    names: dict[str, str] = {}
    used_names = {"samples.lance"}
    for column in ref_columns:
        safe_name = _SAFE_DATASET_NAME.sub("_", column).strip("._")
        if not safe_name:
            msg = f"[InvalidRefColumnName] cannot build a dataset name for ref column {column!r}"
            raise ValueError(msg)
        dataset_name = f"{safe_name}.lance"
        if dataset_name in used_names:
            msg = f"[DuplicateRefDatasetName] ref column {column!r} maps to duplicate dataset {dataset_name!r}"
            raise ValueError(msg)
        used_names.add(dataset_name)
        names[column] = dataset_name
    return names


def _build_output_schemas(
    parquet_schema: pa.Schema,
    ref_columns: Sequence[str],
) -> tuple[pa.Schema, dict[str, pa.Schema]]:
    schema_by_name = {field.name: field for field in parquet_schema}
    missing = [column for column in ref_columns if column not in schema_by_name]
    if missing:
        msg = f"[MissingRefColumns] ref columns not found in parquet schema: {', '.join(missing)}"
        raise ValueError(msg)
    if _REF_ID_COLUMN in ref_columns:
        msg = f"[ReservedRefColumn] {_REF_ID_COLUMN!r} cannot be used as a ref column"
        raise ValueError(msg)

    main_fields: list[pa.Field] = []
    ref_schemas: dict[str, pa.Schema] = {}
    ref_column_set = set(ref_columns)
    for field in parquet_schema:
        if field.name in ref_column_set:
            main_fields.append(pa.field(field.name, _ref_key_type(field.type), nullable=field.nullable))
            ref_schemas[field.name] = pa.schema(
                [
                    pa.field(_REF_ID_COLUMN, pa.string(), nullable=False),
                    pa.field(field.name, _ref_value_type(field.type), nullable=True),
                    pa.field(_SOURCE_COLUMN, pa.string(), nullable=False),
                ]
            )
        else:
            main_fields.append(field)
    main_fields.append(pa.field(_SOURCE_COLUMN, pa.string(), nullable=False))
    return pa.schema(main_fields), ref_schemas


def _ref_key_type(value_type: pa.DataType) -> pa.DataType:
    if pa.types.is_list(value_type):
        return pa.list_(pa.string())
    if pa.types.is_large_list(value_type):
        return pa.large_list(pa.string())
    return pa.string()


def _ref_value_type(value_type: pa.DataType) -> pa.DataType:
    if pa.types.is_list(value_type) or pa.types.is_large_list(value_type):
        return value_type.value_type
    return value_type


def _convert_parquet_file(
    parquet_path: Path,
    plan: ConvertPlan,
    output_queues: dict[str, queue.Queue[object]],
    stop_event: threading.Event,
    progress: Any,
) -> tuple[Path, int]:
    try:
        source = parquet_path.parent.name
        relative_path = _relative_parquet_path(plan.input_path, parquet_path)
        total_rows = 0

        for batch in _iter_parquet_batches(parquet_path, plan):
            if stop_event.is_set():
                msg = "[ConversionStopped] conversion stopped because another worker failed"
                raise RuntimeError(msg)

            main_batch, ref_batches = _convert_record_batch(
                batch,
                plan=plan,
                source=source,
                relative_path=relative_path,
                row_offset=total_rows,
            )
            _put_output(output_queues["samples"], main_batch, stop_event)
            for column, ref_batch in ref_batches.items():
                if ref_batch.num_rows:
                    _put_output(output_queues[column], ref_batch, stop_event)
            total_rows += batch.num_rows
            if progress is not None:
                progress.update(batch.num_rows)
            del batch, main_batch, ref_batches
    finally:
        _release_arrow_unused_memory()

    return parquet_path, total_rows


def _iter_parquet_batches(parquet_path: Path, plan: ConvertPlan) -> Iterable[pa.RecordBatch]:
    if plan.parquet_reader == "file":
        parquet_file = pq.ParquetFile(parquet_path)
        yield from parquet_file.iter_batches(
            batch_size=plan.batch_size,
            columns=plan.parquet_schema.names,
            use_threads=plan.parquet_use_threads,
        )
        return

    dataset = pads.dataset(str(parquet_path), format="parquet")
    scanner = pads.Scanner.from_dataset(
        dataset,
        columns=plan.parquet_schema.names,
        batch_size=plan.batch_size,
        batch_readahead=plan.scanner_batch_readahead,
        fragment_readahead=plan.scanner_fragment_readahead,
        use_threads=plan.parquet_use_threads,
    )
    yield from scanner.to_batches()


def _convert_record_batch(
    batch: pa.RecordBatch,
    *,
    plan: ConvertPlan,
    source: str,
    relative_path: str,
    row_offset: int,
) -> tuple[pa.RecordBatch, dict[str, pa.RecordBatch]]:
    source_array = pa.array([source] * batch.num_rows, type=pa.string())
    ref_column_set = set(plan.ref_columns)
    ref_batches: dict[str, pa.RecordBatch] = {}
    main_arrays: list[pa.Array] = []

    for column_index, field in enumerate(plan.parquet_schema):
        column_name = field.name
        values = batch.column(column_index)
        if column_name not in ref_column_set:
            main_arrays.append(values)
            continue

        ids = _build_ref_ids(
            source=source,
            relative_path=relative_path,
            column=column_name,
            row_offset=row_offset,
            num_rows=batch.num_rows,
        )
        if pa.types.is_list(values.type) or pa.types.is_large_list(values.type):
            main_ref_ids, ref_ids, ref_values, ref_sources = _convert_list_ref_column(values, ids, source)
        else:
            main_ref_ids, ref_ids, ref_values, ref_sources = _convert_scalar_ref_column(values, ids, source_array)

        main_arrays.append(main_ref_ids)
        ref_batches[column_name] = pa.RecordBatch.from_arrays(
            [ref_ids, ref_values, ref_sources],
            schema=plan.ref_schemas[column_name],
        )

    main_arrays.append(source_array)
    main_batch = pa.RecordBatch.from_arrays(main_arrays, schema=plan.main_schema)
    return main_batch, ref_batches


def _build_ref_ids(
    *,
    source: str,
    relative_path: str,
    column: str,
    row_offset: int,
    num_rows: int,
) -> pa.Array:
    return pa.array(
        [f"{source}:{relative_path}:{row_offset + row_index}:{column}" for row_index in range(num_rows)],
        type=pa.string(),
    )


def _convert_scalar_ref_column(
    values: pa.Array,
    ids: pa.Array,
    source_array: pa.Array,
) -> tuple[pa.Array, pa.Array, pa.Array, pa.Array]:
    if values.null_count:
        valid_mask = pc.is_valid(values)
        main_ids = pc.if_else(valid_mask, ids, pa.nulls(len(values), type=pa.string()))
        ref_ids = pc.filter(ids, valid_mask)
        ref_values = pc.filter(values, valid_mask)
        ref_sources = pc.filter(source_array, valid_mask)
        return main_ids, ref_ids, ref_values, ref_sources
    return ids, ids, values, source_array


def _convert_list_ref_column(
    values: pa.Array,
    ids: pa.Array,
    source: str,
) -> tuple[pa.Array, pa.Array, pa.Array, pa.Array]:
    flattened_values = pc.list_flatten(values)
    flattened_valid = pc.is_valid(flattened_values).to_pylist()
    parent_valid = pc.is_valid(values).to_pylist()
    offsets = values.offsets.to_pylist()

    main_ids: list[list[str] | None] = []
    ref_ids: list[str] = []
    flattened_position = 0

    for row_id, row_is_valid, start, end in zip(
        ids.to_pylist(),
        parent_valid,
        offsets[:-1],
        offsets[1:],
        strict=True,
    ):
        if not row_is_valid:
            main_ids.append(None)
            continue
        row_ref_ids: list[str] = []
        for value_position in range(end - start):
            if flattened_valid[flattened_position]:
                ref_id = f"{row_id}:{value_position}"
                row_ref_ids.append(ref_id)
                ref_ids.append(ref_id)
            flattened_position += 1
        main_ids.append(row_ref_ids)

    key_type = _ref_key_type(values.type)
    ref_values = pc.filter(flattened_values, pc.is_valid(flattened_values))
    return (
        pa.array(main_ids, type=key_type),
        pa.array(ref_ids, type=pa.string()),
        ref_values,
        pa.array([source] * len(ref_ids), type=pa.string()),
    )


def _relative_parquet_path(input_path: Path, parquet_path: Path) -> str:
    if input_path.is_file():
        return parquet_path.name
    try:
        return parquet_path.relative_to(input_path).as_posix()
    except ValueError:
        return parquet_path.name


def _put_output(
    output_queue: queue.Queue[object],
    item: object,
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        try:
            output_queue.put(item, timeout=0.1)
            return
        except queue.Full:
            continue
    msg = "[ConversionStopped] conversion stopped before an output batch could be queued"
    raise RuntimeError(msg)


def _signal_outputs(
    output_queues: Iterable[queue.Queue[object]],
    item: object,
    *,
    discard_when_full: bool = False,
) -> None:
    for output_queue in output_queues:
        while True:
            try:
                output_queue.put(item, timeout=0.1)
                break
            except queue.Full:
                if discard_when_full:
                    try:
                        output_queue.get_nowait()
                    except queue.Empty:
                        pass
                continue


def _write_lance_from_queue(
    output_queue: queue.Queue[object],
    uri: Path,
    schema: pa.Schema,
    plan: ConvertPlan,
    stop_event: threading.Event,
) -> None:
    try:
        lance.write_dataset(
            _iter_queue_batches(output_queue),
            str(uri),
            schema=schema,
            mode="overwrite",
            max_rows_per_file=plan.max_rows_per_file,
            max_rows_per_group=plan.max_rows_per_group,
            max_bytes_per_file=plan.max_bytes_per_file,
        )
    except Exception:
        stop_event.set()
        raise


def _iter_queue_batches(output_queue: queue.Queue[object]):
    while True:
        item = output_queue.get()
        if item is _END:
            return
        if isinstance(item, Exception):
            raise item
        yield item


def _raise_finished_writer_errors(writer_futures) -> None:
    for future in writer_futures:
        if future.done():
            future.result()


def _release_arrow_unused_memory() -> None:
    release_unused = getattr(pa.default_memory_pool(), "release_unused", None)
    if callable(release_unused):
        release_unused()


def _cleanup_lance_versions(plan: ConvertPlan) -> None:
    uris = [plan.main_uri, *(plan.ref_uri(column) for column in plan.ref_columns)]
    for uri in uris:
        lance.dataset(str(uri)).cleanup_old_versions(retain_versions=1, delete_unverified=True)


def _write_conversion_manifest(plan: ConvertPlan, *, row_count: int) -> dict[str, Any]:
    ref_columns = {
        column: {
            "uri": str(plan.ref_uri(column)),
            "key_column": _REF_ID_COLUMN,
            "value_column": column,
        }
        for column in plan.ref_columns
    }
    manifest: dict[str, Any] = {
        "version": 1,
        "input": str(plan.input_path),
        "output": str(plan.output_path),
        "main_uri": str(plan.main_uri),
        "parquet_count": len(plan.parquet_files),
        "row_count": row_count,
        "source_column": _SOURCE_COLUMN,
        "ref_columns": ref_columns,
    }
    manifest_path = plan.output_path / _CONVERSION_MANIFEST
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    main()
