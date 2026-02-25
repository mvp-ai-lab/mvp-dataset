"""Example: load JSONL samples and resolve tar references."""

from __future__ import annotations

import argparse
import io
import time
from collections.abc import Sequence

from mvp_dataset import Dataset, TorchLoader


def summarize_sample(sample: object) -> object:
    """Attach lightweight summaries to one decoded JSONL sample."""

    if not isinstance(sample, dict):
        return sample
    summarized = dict(sample)
    for key, value in sample.items():
        if isinstance(value, (bytes, bytearray)):
            summarized[f"{key}_num_bytes"] = len(value)

    from PIL import Image

    image_payload = summarized.get("image")
    if isinstance(image_payload, bytes):
        image = Image.open(io.BytesIO(image_payload))
        image.load()
        summarized["image_pil"] = image
    return summarized


def identity_collate(batch: list[object]) -> list[object]:
    """Return worker micro-batches as plain lists."""

    return batch


def build_loader(dataset: Dataset, args: argparse.Namespace) -> TorchLoader:
    """Build a loader with either direct-sample or worker-microbatch pipeline."""

    if args.loader_mode == "simple":
        return (
            TorchLoader(
                dataset,
                num_workers=args.num_workers,
                batch_size=None,
            )
            .shuffle(args.shuffle_buffer)
            .batch(batch_size=args.batch_size)
        )

    if args.worker_batch_size <= 0:
        msg = "--worker-batch-size must be > 0 when --loader-mode=microbatch"
        raise ValueError(msg)

    return (
        TorchLoader(
            dataset,
            num_workers=args.num_workers,
            batch_size=args.worker_batch_size,
            collate_fn=identity_collate,
            persistent_workers=args.persistent_workers and args.num_workers > 0,
            prefetch_factor=args.prefetch_factor,
            pin_memory=args.pin_memory,
        )
        .unbatch()
        .shuffle(args.shuffle_buffer)
        .batch(batch_size=args.batch_size)
    )


def main() -> None:
    """Run the JSONL reference-resolution example from CLI arguments."""

    parser = argparse.ArgumentParser(description="Resolve tar references from JSONL rows")
    parser.add_argument("files", nargs="+", help="JSONL file paths or patterns")
    parser.add_argument("--seed", type=int, default=0, help="Global seed")
    parser.add_argument("--epoch", type=int, default=0, help="Epoch index")
    parser.add_argument("--tar-cache-size", type=int, default=8, help="LRU tar cache size")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="PyTorch DataLoader worker count",
    )
    parser.add_argument(
        "--loader-mode",
        choices=("simple", "microbatch"),
        default="microbatch",
        help="simple: direct sample stream; microbatch: worker batch -> unbatch -> shuffle",
    )
    parser.add_argument(
        "--worker-batch-size",
        type=int,
        default=32,
        help="Worker micro-batch size (used in microbatch mode)",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="PyTorch prefetch factor when num_workers > 0",
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable pin_memory in TorchLoader (recommended for GPU training)",
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep worker processes alive across epochs when num_workers > 0",
    )
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=1024,
        help="Global shuffle buffer size (loader side)",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=100,
        help="Stop after N printed batches",
    )
    args = parser.parse_args()

    ref_fields: Sequence[tuple[str, str]] = [("image", ".")]
    dataset = (
        Dataset.from_source(args.files, resample=True).group_by("image").resolve_refs(ref_fields).map(summarize_sample)
    )
    loader = build_loader(dataset, args)

    print(
        f"loader_mode={args.loader_mode} "
        f"num_workers={args.num_workers} "
        "worker_batch_size="
        f"{args.worker_batch_size if args.loader_mode == 'microbatch' else 'n/a'} "
    )
    total_samples = 0
    total_batches = 0
    start_time = time.perf_counter()

    for batch_index, batch in enumerate(loader):
        if not isinstance(batch, list):
            continue
        total_batches += 1
        total_samples += len(batch)

        if batch_index % 10 == 0:
            print(f"batch={batch_index} size={len(batch)} total_samples={total_samples} total_batches={total_batches} ")
        if batch_index + 1 >= args.max_batches:
            break

    elapsed = time.perf_counter() - start_time
    samples_per_sec = total_samples / elapsed if elapsed > 0 else 0.0
    batches_per_sec = total_batches / elapsed if elapsed > 0 else 0.0
    print(
        "read_stats "
        f"loader_mode={args.loader_mode} "
        f"num_workers={args.num_workers} "
        f"batches={total_batches} "
        f"samples={total_samples} "
        f"elapsed_sec={elapsed:.3f} "
        f"samples_per_sec={samples_per_sec:.2f} "
        f"batches_per_sec={batches_per_sec:.2f}"
    )


if __name__ == "__main__":
    main()
