import argparse
import os
import time
from pathlib import Path

from mvp_dataset import Dataset, TorchLoader, set_log_level


def map_func(sample: dict) -> dict:
    sample["mapped"] = True
    return sample


def identity_collate(batch: list[object]) -> list[object]:
    return batch


def _maybe_init_torch_distributed(backend: str | None) -> tuple[bool, object | None]:
    """Initialize torch.distributed when launched under torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, None

    try:
        import torch
        import torch.distributed as dist
    except ModuleNotFoundError as exc:
        msg = "[TorchUnavailable] torch distributed launch was requested via WORLD_SIZE>1, but torch is not installed"
        raise RuntimeError(msg) from exc

    if not dist.is_available():
        msg = "[TorchDistributedUnavailable] torch.distributed is not available"
        raise RuntimeError(msg)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        resolved_backend = backend
        if resolved_backend is None:
            resolved_backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=resolved_backend, init_method="env://")

    return True, dist


def _build_progress_bar(*, rank: int, total: int | None):
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError as exc:
        msg = "[TqdmUnavailable] examples/load.py requires tqdm. Install it with `uv add tqdm` or `pip install tqdm`."
        raise RuntimeError(msg) from exc

    return tqdm(
        total=total,
        desc=f"rank {rank}",
        unit="sample",
        dynamic_ncols=True,
        mininterval=0.25,
        disable=rank != 0,
    )


def main(
    source: str,
    source_kind: str,
    *,
    dist_backend: str | None,
    max_samples: int | None,
    log_level: str,
    cache: bool,
    shuffle: bool,
    resample: bool,
    num_workers: int,
    worker_batch_size: int,
    prefetch_factor: int,
    pin_memory: bool,
    persistent_workers: bool,
    multiprocessing_context: str | None,
    no_ref: bool,
    ref_column: str,
    ref_uri: str | None,
    ref_key_column: str,
    ref_value_column: str | None,
):
    set_log_level(log_level)
    if worker_batch_size <= 0:
        msg = f"--worker-batch-size must be > 0, got {worker_batch_size}"
        raise ValueError(msg)

    dist_enabled, dist = _maybe_init_torch_distributed(dist_backend)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    print(
        f"<MVP Dataset - rank {rank}/{world_size} local_rank={local_rank}> "
        f"Loading {source_kind} dataset from {source}...",
        flush=True,
    )

    if source_kind == "lance":
        resolved_ref_columns = None
        source_is_config = Path(source).suffix.lower() == ".json"
        uses_default_ref_args = (
            ref_column == "images" and ref_uri is None and ref_key_column == "ref_id" and ref_value_column is None
        )
        if no_ref:
            resolved_ref_columns = {}
        elif not source_is_config or not uses_default_ref_args:
            resolved_ref_uri = ref_uri or str(Path(source).expanduser().parent / f"{ref_column}.lance")
            resolved_ref_columns = {
                ref_column: {
                    "uri": resolved_ref_uri,
                    "key_column": ref_key_column,
                    "value_column": ref_value_column or ref_column,
                },
            }
        ds = Dataset.from_source(
            source_kind,
            shards=source,
            resample=resample,
            global_shuffle=shuffle,
            ref_columns=resolved_ref_columns,
        ).map(map_func)
    else:
        ds = Dataset.from_source(source_kind, shards=source, resample=resample).map(map_func)
        ds = ds.shuffle(1000)
    if cache:
        ds = ds.cache(cache_num_workers=8)

    loader = TorchLoader(
        ds,
        num_workers=num_workers,
        batch_size=worker_batch_size,
        collate_fn=identity_collate,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        multiprocessing_context=multiprocessing_context,
    )

    count = 0
    progress = _build_progress_bar(rank=rank, total=max_samples)
    try:
        for i, _sample in enumerate(loader):
            if i == 0:
                t0 = time.monotonic()
            count += worker_batch_size
            progress.update(worker_batch_size)
            if max_samples is not None and i * worker_batch_size + 1 >= max_samples:
                break
    finally:
        progress.close()
    elapsed = time.monotonic() - t0
    rate = count / elapsed if elapsed > 0 else float("inf")
    print(
        f"<MVP Dataset - rank {rank}> read {count} samples in {elapsed:.3f}s ({rate:.1f} samples/s)",
        flush=True,
    )

    if dist_enabled and dist is not None:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a dataset from a source.")
    parser.add_argument("--source", type=str, help="The source to load the dataset from.")
    parser.add_argument(
        "--source_kind",
        type=str,
        choices=["jsonl", "tars", "parquet", "lance"],
        help="The kind of the source.",
    )
    parser.add_argument(
        "--dist-backend",
        type=str,
        choices=["gloo", "nccl"],
        default=None,
        help=(
            "Optional torch.distributed backend. When omitted, the example auto-detects "
            "'nccl' for CUDA launches and 'gloo' otherwise."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on the number of samples printed per rank.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Package log level, for example DEBUG, INFO, WARNING, or ERROR.",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Whether to cache the dataset after loading. Caching is recommended for better "
        "performance when the dataset is large and/or expensive to load.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Whether to shuffle the dataset after loading.",
    )
    parser.add_argument(
        "--resample",
        action="store_true",
        help="Whether to resample the dataset after loading.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="PyTorch DataLoader worker count.",
    )
    parser.add_argument(
        "--worker-batch-size",
        type=int,
        default=16,
        help="Worker-side micro-batch size before unbatching in the main process.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="PyTorch DataLoader prefetch factor when num_workers > 0.",
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable DataLoader pin_memory.",
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep DataLoader workers alive when num_workers > 0.",
    )
    parser.add_argument(
        "--multiprocessing-context",
        choices=["fork", "spawn", "forkserver"],
        default=None,
        help="Optional PyTorch DataLoader multiprocessing context, e.g. spawn for Lance.",
    )
    parser.add_argument(
        "--no-ref",
        action="store_true",
        help="Disable Lance reference-column resolution.",
    )
    parser.add_argument(
        "--ref-column",
        default="images",
        help="Lance source column to resolve through a reference dataset.",
    )
    parser.add_argument(
        "--ref-uri",
        default=None,
        help="Reference Lance dataset URI. Defaults to <source parent>/<ref-column>.lance.",
    )
    parser.add_argument(
        "--ref-key-column",
        default="ref_id",
        help="Reference dataset key column.",
    )
    parser.add_argument(
        "--ref-value-column",
        default=None,
        help="Reference dataset value column. Defaults to --ref-column.",
    )

    args = parser.parse_args()

    main(
        args.source,
        args.source_kind,
        dist_backend=args.dist_backend,
        max_samples=args.max_samples,
        log_level=args.log_level,
        cache=args.cache,
        shuffle=args.shuffle,
        resample=args.resample,
        num_workers=args.num_workers,
        worker_batch_size=args.worker_batch_size,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        multiprocessing_context=args.multiprocessing_context,
        no_ref=args.no_ref,
        ref_column=args.ref_column,
        ref_uri=args.ref_uri,
        ref_key_column=args.ref_key_column,
        ref_value_column=args.ref_value_column,
    )
