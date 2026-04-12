"""Lance source implementation for fragment-parallel sample iteration."""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Final

import lance as _lance

from ...core.types import PathLikeStr, Sample

_DEFAULT_BATCH_SIZE: Final[int] = 65536
_FRAGMENT_SUFFIX: Final[str] = "@@frag="


@dataclass(frozen=True, slots=True)
class LanceFragment:
    """One schedulable lance fragment."""

    uri: str
    fragment_id: int
    num_rows: int
    row_offset: int

    @property
    def cache_key(self) -> str:
        """Stable shard identifier used by cache manifests and routing."""
        return f"{self.uri}{_FRAGMENT_SUFFIX}{self.fragment_id}"


def list_lance_fragments(
    dataset_uris: Sequence[PathLikeStr],
    *,
    min_fragments: int = 0,
) -> list[LanceFragment]:
    """Expand lance datasets into schedulable fragments.

    Fragments with fewer than ``MVP_DATASET_LANCE_MIN_ROWS_PER_FRAGMENT``
    rows (default 5000) are merged with subsequent fragments until the
    threshold is reached.  When the resulting count is below *min_fragments*,
    the threshold is lowered to 0 so every physical fragment stands alone.
    """
    min_rows = int(os.environ.get("MVP_DATASET_LANCE_MIN_ROWS_PER_FRAGMENT", "5000"))
    result = _collect_lance_fragments(dataset_uris, min_rows)
    if len(result) < min_fragments:
        result = _collect_lance_fragments(dataset_uris, 0)
    return result


def _collect_lance_fragments(
    dataset_uris: Sequence[PathLikeStr],
    min_rows: int,
) -> list[LanceFragment]:
    result: list[LanceFragment] = []
    for uri in dataset_uris:
        ds = _lance.dataset(str(uri))
        fragments = ds.get_fragments()

        row_offset = 0
        pending_ids: list[int] = []
        pending_rows = 0

        for frag in fragments:
            frag_rows = frag.count_rows()
            pending_ids.append(frag.fragment_id)
            pending_rows += frag_rows

            if pending_rows >= min_rows:
                result.append(
                    LanceFragment(
                        uri=str(uri),
                        fragment_id=pending_ids[0],
                        num_rows=pending_rows,
                        row_offset=row_offset,
                    )
                )
                row_offset += pending_rows
                pending_ids = []
                pending_rows = 0

        if pending_ids:
            result.append(
                LanceFragment(
                    uri=str(uri),
                    fragment_id=pending_ids[0],
                    num_rows=pending_rows,
                    row_offset=row_offset,
                )
            )

    return result


def iter_lance(
    fragment: LanceFragment,
    *,
    columns: Sequence[str] | None = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> Iterator[Sample]:
    """Iterate one lance fragment and yield one sample dict per row."""

    ds = _lance.dataset(fragment.uri)
    lance_fragments = ds.get_fragments()
    target = [f for f in lance_fragments if f.fragment_id == fragment.fragment_id]
    if not target:
        msg = f"[LanceFragmentNotFound] fragment_id={fragment.fragment_id} uri={fragment.uri!r}"
        raise KeyError(msg)

    index_in_file = fragment.row_offset
    for record_batch in target[0].to_batches(
        columns=columns,
        batch_size=batch_size,
    ):
        column_names = record_batch.schema.names
        columns_data = [record_batch.column(i) for i in range(record_batch.num_columns)]
        for batch_row_index in range(record_batch.num_rows):
            sample: Sample = {
                name: columns_data[column_index][batch_row_index].as_py()
                for column_index, name in enumerate(column_names)
            }
            sample["__file__"] = fragment.uri
            sample["__index_in_file__"] = index_in_file
            sample["__key__"] = f"{fragment.uri}:{index_in_file}"
            yield sample
            index_in_file += 1


def iter_lances(
    fragments: Iterator[LanceFragment],
    *,
    columns: Sequence[str] | None = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> Iterator[Sample]:
    """Iterate lance fragments in order and yield row samples."""

    for fragment in fragments:
        yield from iter_lance(
            fragment,
            columns=columns,
            batch_size=batch_size,
        )
