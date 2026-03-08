"""Tar source example with throughput-oriented loader-side shuffle pipeline."""

from __future__ import annotations

import argparse
import io
import time
from collections.abc import Callable
from functools import partial

from mvp_dataset import Dataset, TorchLoader


def identity_collate(batch: list[object]) -> list[object]:
    """Return worker micro-batches as plain lists."""

    return batch


def annotate_sample(sample: object, *, decode_jpg: bool) -> object:
    """Add lightweight markers and optionally decode image payload to PIL."""

    if not isinstance(sample, dict):
        return sample

    annotated = dict(sample)
    jpg_payload = annotated.get("images.png")
    if decode_jpg and isinstance(jpg_payload, (bytes, bytearray)):
        from PIL import Image

        image = Image.open(io.BytesIO(jpg_payload))
        image.load()
        annotated["jpg"] = image
        annotated["metadata"] = {"img_size": image.size}
    annotated["seen_in_example"] = True
    return annotated


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
    """Run the tar loading example from CLI arguments."""

    parser = argparse.ArgumentParser(description="Read tar shards with mvp-dataset")
    parser.add_argument("shards", nargs="+", help="Local tar shard paths or patterns")
    parser.add_argument(
        "--tar-key-dot-level",
        type=int,
        default=1,
        help="How many dot-separated segments are used as sample key (default: 1 => id)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Global seed")
    parser.add_argument("--epoch", type=int, default=0, help="Epoch index")
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
        "--decode-jpg",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Decode image bytes to PIL in map stage (off by default for throughput)",
    )
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=64,
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

    annotate: Callable[[object], object] = partial(annotate_sample, decode_jpg=args.decode_jpg)
    dataset = Dataset.from_tars(args.shards, resample=True).map(annotate)
    loader = build_loader(dataset, args)

    total_samples = 0
    total_batches = 0
    start_time = time.perf_counter()

    print(
        "Reading batches... "
        f"mode={args.loader_mode} "
        f"num_workers={args.num_workers} "
        "worker_batch_size="
        f"{args.worker_batch_size if args.loader_mode == 'microbatch' else 'n/a'} "
        f"decode_jpg={args.decode_jpg}"
    )
    for batch_index, batch in enumerate(loader):
        if not isinstance(batch, list):
            continue

        total_batches += 1
        total_samples += len(batch)

        if batch_index % 10 == 0:
            print(
                f"batch {batch_index} "
                f"batch_size={len(batch)} "
                f"total_samples={total_samples} "
                f"total_batches={total_batches}"
            )

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
