"""Minimal tar-only usage example for mm-loader step 1."""

from __future__ import annotations

import argparse

from mm_loader import Dataset
from mm_loader.core import RuntimeContext


def annotate_sample(sample: object) -> object:
    """Add lightweight derived fields without decoding payloads."""

    if not isinstance(sample, dict):
        return sample

    annotated = dict(sample)
    jpg_payload = annotated.get("jpg")
    if isinstance(jpg_payload, (bytes, bytearray)):
        annotated["jpg_num_bytes"] = len(jpg_payload)
    annotated["seen_in_example"] = True
    return annotated


def main() -> None:
    parser = argparse.ArgumentParser(description="Read tar shards with mm-loader")
    parser.add_argument("shards", nargs="+", help="Local tar shard paths or patterns")
    parser.add_argument("--seed", type=int, default=0, help="Global seed")
    parser.add_argument("--epoch", type=int, default=0, help="Epoch index")
    parser.add_argument("--shuffle-buffer", type=int, default=64, help="Sample shuffle buffer size")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-batches", type=int, default=3, help="Stop after N printed batches")
    args = parser.parse_args()

    # Rank/worker defaults can be overridden via env: RANK/WORLD_SIZE/WORKER/NUM_WORKERS.
    context = RuntimeContext.from_env(seed=args.seed, epoch=args.epoch)

    dataset = (
        Dataset.from_tars(args.shards, context=context)
        .shuffle(buffer_size=args.shuffle_buffer)
        .map(annotate_sample)
        .batch(batch_size=args.batch_size)
    )

    for batch_index, batch in enumerate(dataset):
        if not isinstance(batch, list):
            continue
        keys = [sample.get("__key__") for sample in batch if isinstance(sample, dict)]
        print(f"batch={batch_index} size={len(batch)} keys={keys}")
        if batch_index + 1 >= args.max_batches:
            break


if __name__ == "__main__":
    main()
