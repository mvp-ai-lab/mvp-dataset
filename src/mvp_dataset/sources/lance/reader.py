"""Lance batch row reading."""

from __future__ import annotations

from collections.abc import Sequence

import lance
import pyarrow as pa
import pyarrow.dataset as pads

from mvp_dataset.core.types import Sample

from .types import LanceIndexItem, LanceSource


class LanceBatchReader:
    """Read Lance rows for precomputed physical indexes."""

    def __init__(self, source: LanceSource) -> None:
        self.source = source

    def read(self, indexes: Sequence[LanceIndexItem], *, columns: Sequence[str] | None = None) -> list[Sample]:
        """Read a batch of Lance rows and restore input index order."""
        if not indexes:
            return []

        per_dataset_indices: dict[int, list[int]] = {}
        for index_item in indexes:
            per_dataset_indices.setdefault(index_item.dataset_i, []).append(index_item.local_index)

        per_dataset_rows: dict[int, list[Sample]] = {}
        take_indices_type = pa.int64()
        for dataset_i, local_indices in per_dataset_indices.items():
            dataset = self.source.datasets[dataset_i]
            dataset_handle = dataset.handle if dataset.handle is not None else lance.dataset(dataset.uri)

            if isinstance(dataset_handle, pa.Table):
                table = dataset_handle
                if columns is not None:
                    table = table.select(columns)
                table = table.take(pa.array(local_indices, type=take_indices_type))
            elif isinstance(dataset_handle, pads.Dataset):
                table = dataset_handle.take(
                    pa.array(local_indices, type=take_indices_type),
                    columns=columns,
                )
            else:
                table = dataset_handle.take(local_indices, columns=columns)

            per_dataset_rows[dataset_i] = table.to_pylist()

        per_dataset_offsets = {dataset_i: 0 for dataset_i in per_dataset_indices}
        batch: list[Sample] = []
        for index_item in indexes:
            dataset = self.source.datasets[index_item.dataset_i]
            dataset_offset = per_dataset_offsets[index_item.dataset_i]
            sample = dict(per_dataset_rows[index_item.dataset_i][dataset_offset])
            sample["__file__"] = dataset.uri
            sample["__local_index__"] = index_item.local_index
            sample["__global_index__"] = index_item.global_index
            sample["__key__"] = f"{dataset.uri}:{index_item.local_index}"
            batch.append(sample)
            per_dataset_offsets[index_item.dataset_i] += 1

        return batch
