"""Lance source utility package."""

from __future__ import annotations

import lance as lance

from .config import resolve_lance_source_config
from .refs import (
    REF_INDEX_BUILD_BATCH_SIZE,
    REF_INDEX_BUILDER_VERSION,
    REF_INDEX_DIR,
    REF_INDEX_LOCK_POLL_SECONDS,
    REF_INDEX_MANIFEST,
    REF_INDEX_MISSING_ROW,
    REF_INDEX_WAIT_TIMEOUT_SECONDS,
    LanceRefIndexScope,
    LanceRefResolverAssembler,
    LanceResolveRefFactory,
    _apply_ref_columns,
    _iter_table_record_batches,
    _read_table_rows,
    attach_lance_ref_columns,
    iter_lance_ref_resolver,
    parse_lance_ref_columns,
    prepare_ref_indexes,
    validate_ref_names,
)
from .source import _read_batch, list_lance_sources
from .types import (
    LanceDatasetSpec,
    LanceIndexItem,
    LanceRefSpec,
    LanceShuffleMode,
    LanceSourceSpec,
)

__all__ = [
    "REF_INDEX_BUILD_BATCH_SIZE",
    "REF_INDEX_BUILDER_VERSION",
    "REF_INDEX_DIR",
    "REF_INDEX_LOCK_POLL_SECONDS",
    "REF_INDEX_MANIFEST",
    "REF_INDEX_MISSING_ROW",
    "REF_INDEX_WAIT_TIMEOUT_SECONDS",
    "LanceDatasetSpec",
    "LanceIndexItem",
    "LanceRefIndexScope",
    "LanceRefResolverAssembler",
    "LanceResolveRefFactory",
    "LanceRefSpec",
    "LanceShuffleMode",
    "LanceSourceSpec",
    "_apply_ref_columns",
    "_iter_table_record_batches",
    "_read_batch",
    "_read_table_rows",
    "attach_lance_ref_columns",
    "iter_lance_ref_resolver",
    "lance",
    "list_lance_sources",
    "parse_lance_ref_columns",
    "prepare_ref_indexes",
    "resolve_lance_source_config",
    "validate_ref_names",
]
