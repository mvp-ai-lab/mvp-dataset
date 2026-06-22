"""Lance ref-column sample resolution."""

from __future__ import annotations

import bisect
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

import lance
import numpy as np

from mvp_dataset.core.context import RuntimeContext
from mvp_dataset.core.resume import ResumeStateError, stable_fingerprint
from mvp_dataset.core.types import Sample

from ..types import (
    LanceIndexItem,
    LanceRefIndexBuildStrategy,
    LanceRefIndexScope,
    LanceSourceSpec,
)
from .index import REF_INDEX_MISSING_ROW, _open_ref_value_source, prepare_ref_indexes
from .read import _read_table_rows


def _read_ref_value_rows(
    value_source: LanceSourceSpec,
    row_indices: Sequence[int],
    *,
    columns: Sequence[str],
) -> list[Sample]:
    """Read referenced values for a batch of keys."""
    if not row_indices:
        return []

    row_offsets = [dataset.row_offset for dataset in value_source.datasets]
    batch_indexes: list[LanceIndexItem] = []
    for row_index in row_indices:
        dataset_i = bisect.bisect_right(row_offsets, row_index) - 1
        dataset = value_source.datasets[dataset_i]
        local_index = int(row_index - dataset.row_offset)
        batch_indexes.append(LanceIndexItem(dataset_i=dataset_i, local_index=local_index, global_index=int(row_index)))

    per_dataset_indices: dict[int, list[int]] = {}
    for index_item in batch_indexes:
        per_dataset_indices.setdefault(index_item.dataset_i, []).append(index_item.local_index)

    per_dataset_rows: dict[int, list[Sample]] = {}
    for dataset_i, local_indices in per_dataset_indices.items():
        dataset = value_source.datasets[dataset_i]
        dataset_handle = dataset.handle if dataset.handle is not None else lance.dataset(dataset.uri)
        per_dataset_rows[dataset_i] = _read_table_rows(dataset_handle, local_indices, columns=columns)

    per_dataset_offsets = {dataset_i: 0 for dataset_i in per_dataset_indices}
    rows: list[Sample] = []
    for index_item in batch_indexes:
        dataset_offset = per_dataset_offsets[index_item.dataset_i]
        rows.append(per_dataset_rows[index_item.dataset_i][dataset_offset])
        per_dataset_offsets[index_item.dataset_i] += 1
    return rows


def _looks_like_lance_index_items(value: object) -> bool:
    """Return whether values look like serialized Lance index items."""
    if value is None:
        return False
    if not isinstance(value, Sequence) or isinstance(value, str):
        return False
    return bool(value) and isinstance(value[0], LanceIndexItem)


def _get_ref_field(sample: object, field: str) -> Any:
    if isinstance(sample, Mapping):
        return sample[field]
    return getattr(sample, field)


def _set_ref_field(sample: object, field: str, value: Any) -> None:
    if isinstance(sample, MutableMapping):
        sample[field] = value
    else:
        setattr(sample, field, value)


def _apply_ref_columns(
    source: LanceSourceSpec,
    batch: list[object],
    columns_or_indexes: Sequence[str] | Sequence[LanceIndexItem] | None = None,
    *,
    columns: Sequence[str] | None = None,
) -> None:
    """Resolve configured Lance reference columns in-place for one sample batch.

    ``prepare_ref_indexes`` builds a CSR-style lookup for each configured ref
    column. For every main-dataset global row, ``offsets[row]:offsets[row + 1]``
    points into ``entries``. The ``entries`` slice contains the row indexes in
    the reference Lance dataset that correspond to the key or keys stored in
    the main sample column.

    This function uses those prepared indexes to replace each requested ref
    column value in ``batch``:

    1. Map every batch sample to its global main-table row index.
    2. For each active ref column, recover the reference row indexes needed by
       each sample.
    3. Fetch each unique reference row once from the reference Lance dataset.
    4. Scatter fetched values back into the original sample positions while
       preserving scalar-vs-multi-value shape.

    The input ``batch`` is mutated in place. Samples are expected to contain the
    internal ``__global_index__`` metadata added by the Lance resolver assembler.
    """

    # Step 1: Restrict work to the requested/projection-visible ref columns.
    # If ``columns`` is None, all configured refs are active. Otherwise a ref is
    # resolved only when its source column is present in the current projection.
    if columns is None:
        columns = None if _looks_like_lance_index_items(columns_or_indexes) else columns_or_indexes

    active_refs = (
        source.ref_columns if columns is None else tuple(ref for ref in source.ref_columns if ref.column in columns)
    )
    if not active_refs:
        return

    # Step 2: Convert each batch sample into the global row index used by the
    # prepared CSR ref index. The index is global across all physical Lance
    # datasets in ``source``, not local to a single dataset file.
    global_indices = np.asarray([_get_ref_field(item, "__global_index__") for item in batch], dtype=np.int64)
    for ref in active_refs:
        # Step 3: Validate that ``prepare_ref_indexes`` has attached the CSR
        # arrays and the prepared reference dataset handle for this ref column.
        if not isinstance(ref.index_handle, dict):
            msg = f"[UnpreparedLanceRefIndex] ref index for {ref.column!r} was not prepared"
            raise RuntimeError(msg)

        # Step 4: Read each sample's CSR range. For sample global row ``i``,
        # ``offsets[i]:offsets[i + 1]`` gives the slice in ``entries`` holding
        # the reference dataset row indexes for that sample's key(s).
        offsets = ref.index_handle["offsets"]
        entries = ref.index_handle["entries"]
        starts = np.asarray(offsets[global_indices], dtype=np.int64)
        ends = np.asarray(offsets[global_indices + 1], dtype=np.int64)

        # Step 5: Build a scatter plan.
        #
        # ``per_sample_row_indices`` keeps each sample's ref row indexes in
        # original key order. ``positions_by_row_index`` groups all output slots
        # by reference dataset row index, so duplicate references across the
        # batch are fetched once and then scattered back to every consumer.
        positions_by_row_index: dict[int, list[tuple[int, int]]] = {}
        per_sample_row_indices: list[np.ndarray] = []
        for sample_position, (start, end) in enumerate(zip(starts, ends, strict=True)):
            ref_row_indices = np.asarray(entries[start:end], dtype=np.int64)
            per_sample_row_indices.append(ref_row_indices)
            for value_position, row_index in enumerate(ref_row_indices):
                if int(row_index) != REF_INDEX_MISSING_ROW:
                    positions_by_row_index.setdefault(int(row_index), []).append((sample_position, value_position))

        # Step 6: Allocate the output slots before reading values. The slot
        # counts mirror the source cardinality exactly, which preserves list
        # lengths and keeps scalar refs distinguishable from empty multi-refs.
        resolved_values: list[list[Any]] = [[] for _ in per_sample_row_indices]
        for sample_position, ref_row_indices in enumerate(per_sample_row_indices):
            resolved_values[sample_position] = [None] * len(ref_row_indices)

        # Step 7: Fetch each unique reference row once, then scatter the fetched
        # ``value_column`` into every sample/value slot recorded in the plan.
        if positions_by_row_index:
            ordered_row_indices = list(positions_by_row_index)
            value_source = ref.index_handle.get("value_source")
            if not isinstance(value_source, LanceSourceSpec):
                value_source = _open_ref_value_source(ref)
            ref_rows = _read_ref_value_rows(value_source, ordered_row_indices, columns=[ref.value_column])
            for row_index, row in zip(ordered_row_indices, ref_rows, strict=True):
                for sample_position, value_position in positions_by_row_index[row_index]:
                    resolved_values[sample_position][value_position] = row[ref.value_column]

        # Step 8: Replace the original key column with resolved values. Empty
        # CSR ranges resolve to ``None`` for scalar refs and ``[]`` for
        # list/tuple/ndarray refs; non-empty ranges preserve scalar-vs-multi
        # shape based on the original value.
        for sample, ref_row_indices, values in zip(batch, per_sample_row_indices, resolved_values, strict=True):
            original_value = _get_ref_field(sample, ref.column)
            is_multi_value = isinstance(original_value, (list, tuple, np.ndarray))
            if len(ref_row_indices) == 0:
                _set_ref_field(sample, ref.column, [] if is_multi_value else None)
                continue
            _set_ref_field(sample, ref.column, values if is_multi_value else values[0])


def validate_ref_names(source: LanceSourceSpec, ref_names: Sequence[str]) -> tuple[str, ...]:
    """Validate that reference fields do not collide with sample fields.

    Args:
        source: Lance source specification.
        ref_names: Reference column names to resolve.

    Returns:
        Validated reference names."""
    if isinstance(ref_names, str):
        raw_ref_names = (ref_names,)
    else:
        raw_ref_names = tuple(ref_names)

    if not raw_ref_names:
        msg = "[InvalidLanceRefNames] at least one ref column name is required"
        raise ValueError(msg)
    if not all(isinstance(ref_name, str) and ref_name for ref_name in raw_ref_names):
        msg = "[InvalidLanceRefNames] ref column names must be non-empty strings"
        raise ValueError(msg)
    normalized = tuple(dict.fromkeys(raw_ref_names))

    available_refs = {ref.column for ref in source.ref_columns}
    missing_refs = [ref_name for ref_name in normalized if ref_name not in available_refs]
    if missing_refs:
        msg = f"[UnknownLanceRefColumn] requested ref column(s) were not configured: {', '.join(missing_refs)}"
        raise ValueError(msg)

    return normalized


def iter_lance_ref_resolver(
    source: LanceSourceSpec,
    sample_stream: Iterable[object],
    ref_names: Sequence[str],
    *,
    batch_size: int = 1024,
    context: RuntimeContext | None = None,
    ref_index_scope: LanceRefIndexScope | None = None,
    ref_index_build_strategy: LanceRefIndexBuildStrategy | None = None,
    ref_index_bucket_count: int | None = None,
):
    """Resolve configured Lance reference columns for already-read samples.

    Args:
        source: Lance source specification.
        sample_stream: Stream of samples whose references should be resolved.
        ref_names: Reference column names to resolve.
        batch_size: Number of samples to group into each batch.
        context: Runtime context used for sharding and deterministic randomness.
        ref_index_scope: Scope that controls where Lance reference indexes are stored.

    Returns:
        An iterator over samples with resolved Lance references."""

    assembler = LanceRefResolverAssembler(
        source=source,
        ref_names=ref_names,
        batch_size=batch_size,
        context=context,
        ref_index_scope=ref_index_scope,
        ref_index_build_strategy=ref_index_build_strategy,
        ref_index_bucket_count=ref_index_bucket_count,
    )
    for sample in sample_stream:
        yield from assembler.push(sample)
    yield from assembler.finish()


@dataclass(frozen=True, slots=True)
class LanceResolveRefFactory:
    """Factory that creates Lance reference resolver assemblers."""

    source: LanceSourceSpec
    ref_names: tuple[str, ...]
    batch_size: int = 1024
    ref_index_scope: LanceRefIndexScope | None = None
    ref_index_build_strategy: LanceRefIndexBuildStrategy | None = None
    ref_index_bucket_count: int | None = None

    def __call__(self, context: RuntimeContext) -> LanceRefResolverAssembler:
        """Apply this callable object."""
        return LanceRefResolverAssembler(
            source=self.source,
            ref_names=self.ref_names,
            batch_size=self.batch_size,
            context=context,
            ref_index_scope=self.ref_index_scope,
            ref_index_build_strategy=self.ref_index_build_strategy,
            ref_index_bucket_count=self.ref_index_bucket_count,
        )


class LanceRefResolverAssembler:
    """Assembler that resolves Lance reference values in batches."""

    def __init__(
        self,
        *,
        source: LanceSourceSpec,
        ref_names: Sequence[str],
        batch_size: int = 1024,
        context: RuntimeContext | None = None,
        ref_index_scope: LanceRefIndexScope | None = None,
        ref_index_build_strategy: LanceRefIndexBuildStrategy | None = None,
        ref_index_bucket_count: int | None = None,
    ) -> None:
        """Initialize the object."""
        if batch_size <= 0:
            msg = f"[InvalidLanceRefBatchSize] batch_size must be > 0, got {batch_size}"
            raise ValueError(msg)
        if ref_index_bucket_count is not None and ref_index_bucket_count <= 0:
            msg = f"[InvalidLanceRefIndexBucketCount] bucket_count must be > 0, got {ref_index_bucket_count}"
            raise ValueError(msg)

        self.ref_names = validate_ref_names(source, ref_names)
        self.source = prepare_ref_indexes(
            source,
            columns=self.ref_names,
            context=context,
            ref_index_scope=ref_index_scope,
            ref_index_build_strategy=ref_index_build_strategy,
            ref_index_bucket_count=ref_index_bucket_count,
        )
        self.batch_size = batch_size
        self.ref_index_scope = ref_index_scope
        self.ref_index_build_strategy = ref_index_build_strategy
        self.ref_index_bucket_count = ref_index_bucket_count
        self.batch: list[object] = []
        self.queue_size = 0

    def state_dict(self) -> dict[str, object]:
        """Return resumable resolver state."""
        return {
            "batch": list(self.batch),
            "queue_size": self.queue_size,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore resolver state."""
        batch = state.get("batch")
        if not isinstance(batch, list):
            msg = "[InvalidResumeState] lance ref resolver batch must be a list"
            raise ResumeStateError(msg)
        queue_size = state.get("queue_size")
        if not isinstance(queue_size, int) or queue_size < 0:
            msg = "[InvalidResumeState] lance ref resolver queue_size must be a non-negative integer"
            raise ResumeStateError(msg)
        self.batch = list(batch)
        self.queue_size = queue_size

    def fingerprint(self) -> str:
        """Return a stable fingerprint for resume compatibility checks."""
        return stable_fingerprint(
            {
                "kind": "lance_ref_resolver",
                "ref_names": list(self.ref_names),
                "batch_size": self.batch_size,
                "ref_index_scope": self.ref_index_scope,
                "ref_index_build_strategy": self.ref_index_build_strategy,
                "ref_index_bucket_count": self.ref_index_bucket_count,
            }
        )

    def _flush(self) -> Iterable[object]:
        """Resolve and emit pending reference samples."""
        if not self.batch:
            return ()

        targets: list[object] = []
        for sample in self.batch:
            if isinstance(sample, list):
                targets.extend(sample)
            else:
                targets.append(sample)
        _apply_ref_columns(self.source, targets, self.ref_names)

        flushed_batch = self.batch
        self.batch = []
        self.queue_size = 0
        return flushed_batch

    def push(self, sample: object) -> Iterable[object]:
        """Handle push for pipeline execution.

        Args:
            sample: Input sample consumed by the assembler.

        Returns:
            Completed outputs produced after consuming the sample."""
        if sample is None:
            msg = "[InvalidLanceSample] expected dict, list, or object sample, got None"
            raise TypeError(msg)
        self.batch.append(sample)
        self.queue_size += len(sample) if isinstance(sample, list) else 1

        if not sample or self.queue_size >= self.batch_size:
            return self._flush()

        return ()

    def finish(self, *, drop_last: bool = False) -> Iterable[object]:
        """Handle finish for pipeline execution.

        Args:
            drop_last: Whether to discard the final incomplete batch.

        Returns:
            Remaining outputs produced during final flush."""
        if drop_last:
            self.batch.clear()
            self.queue_size = 0
            return ()
        return self._flush()
