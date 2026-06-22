"""Lance source discovery."""

from __future__ import annotations

from collections.abc import Sequence

import lance

from mvp_dataset.core.types import PathLikeStr

from .types import LanceDatasetSpec, LanceSource


def list_lance_sources(dataset_uris: Sequence[PathLikeStr]) -> list[LanceSource]:
    """Resolve Lance dataset URIs into one merged source."""
    datasets: list[LanceDatasetSpec] = []
    row_offset = 0
    for uri in dataset_uris:
        dataset = lance.dataset(str(uri))
        num_rows = dataset.count_rows()
        datasets.append(
            LanceDatasetSpec(
                uri=str(uri),
                num_rows=num_rows,
                row_offset=row_offset,
            )
        )
        row_offset += num_rows

    if not datasets:
        msg = "[EmptyLanceSource] at least one Lance dataset URI is required"
        raise ValueError(msg)

    return [LanceSource(datasets=tuple(datasets))]
