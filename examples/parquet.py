"""Example: load parquet samples with the standard iterator pipeline."""

from __future__ import annotations

import argparse
import time

from mvp_dataset import Dataset, TorchLoader


def annotate(sample: object) -> object:
    assert isinstance(sample, dict)
    return {
        **sample,
        "field_count": len([key for key in sample if not (key.startswith("__") and key.endswith("__"))]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read parquet shards with mvp-dataset")
    parser.add_argument("shards", nargs="+", help="Local parquet shard paths or patterns")
    parser.add_argument("--columns", nargs="*", default=None, help="Optional subset of parquet columns to read")
    parser.add_argument("--batch-size", type=int, default=8192, help="PyArrow record batch size")
    parser.add_argument("--max-batches", type=int, default=2, help="Maximum number of output batches")
    args = parser.parse_args()

    dataset = Dataset.from_parquet(
        args.shards,
        resample=True,
        columns=args.columns,
        batch_size=args.batch_size,
    ).map(annotate)
    loader = TorchLoader(dataset, num_workers=0, batch_size=4, collate_fn=lambda batch: batch)

    start_time = time.perf_counter()
    batch_count = 0
    sample_count = 0
    for batch in loader:
        batch_count += 1
        sample_count += len(batch)
        print(batch)
        if batch_count >= args.max_batches:
            break

    elapsed = time.perf_counter() - start_time
    print(f"Read {sample_count} samples in {batch_count} batches over {elapsed:.3f}s")


if __name__ == "__main__":
    main()
