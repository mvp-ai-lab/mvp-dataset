import argparse
import os
import time

from mvp_dataset import Dataset, set_log_level


def map_func(sample: dict) -> dict:
    time.sleep(0.1)  # Simulate some processing time
    sample["mapped"] = True
    return sample


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


def main(
    source: str,
    source_kind: str,
    *,
    dist_backend: str | None,
    max_samples: int | None,
    log_level: str,
    cache: bool,
):
    set_log_level(log_level)
    dist_enabled, dist = _maybe_init_torch_distributed(dist_backend)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    print(
        f"<MVP Dataset - rank {rank}/{world_size} local_rank={local_rank}> "
        f"Loading {source_kind} dataset from {source}...",
        flush=True,
    )

    ds = Dataset.from_source(source_kind, shards=source).map(map_func).shuffle(2)
    if cache:
        ds = ds.cache()

    t0 = time.monotonic()
    count = 0
    for i, _sample in enumerate(ds):
        count += 1
        if max_samples is not None and i + 1 >= max_samples:
            break
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
    args = parser.parse_args()

    main(
        args.source,
        args.source_kind,
        dist_backend=args.dist_backend,
        max_samples=args.max_samples,
        log_level=args.log_level,
        cache=args.cache,
    )
