"""Polars schema for the unified Parquet cache file.

One row per accepted activity carries its metrics plus the fields of the
source-index entry that produced it. Failed and duplicate source outcomes
have null activity fields (no ``start_time``) so change detection and data
quality keep working without a second table. High-frequency FIT record time
series are deliberately absent from the schema: only aggregated, derived
data is persisted.
"""

from __future__ import annotations

import polars as pl

from ...domain.frames import (
    ACTIVITY_COLUMNS as DOMAIN_ACTIVITY_COLUMNS,
)
from ...domain.frames import (
    ACTIVITY_FRAME_SCHEMA,
    SOURCE_INDEX_COLUMNS,
    SOURCE_INDEX_FRAME_SCHEMA,
)

ANALYSIS_VERSION = "analysis_version"
# Cache-invalidation key for derived analytics, owned by the code: bump it
# whenever parsing or metric semantics change so the next sync reprocesses
# every file (hash-based change detection alone would serve stale rows).
ANALYSIS_VERSION_VALUE = "4"
PROCESSED_AT = "processed_at"
# The source-index copy of the analysis version drives incremental change
# detection; the activity-level one records which code produced the metrics.
SOURCE_ANALYSIS_VERSION = "source_analysis_version"
FILE_REPAIR_FLAGS = "file_repair_flags"
INDEX_COLUMNS = SOURCE_INDEX_COLUMNS
ACTIVITY_COLUMNS = DOMAIN_ACTIVITY_COLUMNS

# The adapter persists the canonical domain Polars columns in one physical
# table, with nullable activity facts on source-only outcome rows.
CACHE_SCHEMA: dict[str, pl.DataType] = {
    **SOURCE_INDEX_FRAME_SCHEMA,
    **ACTIVITY_FRAME_SCHEMA,
}

CACHE_FILE = "training_cache.parquet"
METADATA_FILE = "metadata.json"
LOCK_FILE = "cache.lock"
