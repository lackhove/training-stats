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

ANALYSIS_VERSION = "analysis_version"
# Cache-invalidation key for derived analytics, owned by the code: bump it
# whenever parsing or metric semantics change so the next sync reprocesses
# every file (hash-based change detection alone would serve stale rows).
ANALYSIS_VERSION_VALUE = "4"
PROCESSED_AT = "processed_at"
# The source-index copy of the analysis version drives incremental change
# detection; the activity-level one records which code produced the metrics.
SOURCE_ANALYSIS_VERSION = "source_analysis_version"
# File-level repair flags describe the whole FIT file; the per-activity
# ``repair_flags_json`` can differ when a file splits into segments.
FILE_REPAIR_FLAGS = "file_repair_flags_json"
# Shared by ingest (row creation) and sync (aggregate filtering, counts).
STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_DUPLICATE = "duplicate"

# Source-index fields: present on every row, with or without an activity.
INDEX_COLUMNS: tuple[str, ...] = (
    "source_name",
    "source_path",
    "source_hash",
    "file_size",
    "modified_at",
    "activity_id",
    "segment_index",
    "segment_count",
    FILE_REPAIR_FLAGS,
    "discarded_segment_count",
    PROCESSED_AT,
    SOURCE_ANALYSIS_VERSION,
    "status",
    "error_message",
)

# Activity fields: null on rows without an accepted activity (failed reads,
# duplicate activity ids). ``activities_scan`` selects exactly these.
ACTIVITY_COLUMNS: tuple[str, ...] = (
    "activity_id",
    "start_time",
    "sport",
    "sub_sport",
    "duration_s",
    "moving_time_s",
    "paused_time_s",
    "distance_m",
    "elevation_gain_m",
    "avg_hr",
    "max_hr",
    "avg_speed_mps",
    "max_speed_mps",
    "calories",
    "segment_index",
    "segment_count",
    "repair_flags_json",
    "source_name",
    "source_path",
    "hr_zone_1_s",
    "hr_zone_2_s",
    "hr_zone_3_s",
    "hr_zone_4_s",
    "hr_zone_5_s",
    "hr_coverage_ratio",
    "hr_histogram_json",
    "hr_10min_json",
    "hr_drift_bpm_per_hour",
    "load_value",
    "load_method",
    "load_confidence",
    "data_completeness",
    "ef_first_half",
    "ef_second_half",
    "aerobic_decoupling_pct",
    "decoupling_confidence",
    ANALYSIS_VERSION,
)

CACHE_SCHEMA: dict[str, pl.DataType] = {
    "source_name": pl.String(),
    "source_path": pl.String(),
    "source_hash": pl.String(),
    "file_size": pl.UInt64(),
    "modified_at": pl.Datetime("us", "UTC"),
    "status": pl.String(),
    "error_message": pl.String(),
    FILE_REPAIR_FLAGS: pl.String(),
    "discarded_segment_count": pl.UInt32(),
    PROCESSED_AT: pl.Datetime("us", "UTC"),
    SOURCE_ANALYSIS_VERSION: pl.String(),
    "activity_id": pl.String(),
    "segment_index": pl.UInt32(),
    "segment_count": pl.UInt32(),
    "start_time": pl.Datetime("us", "UTC"),
    "sport": pl.String(),
    "sub_sport": pl.String(),
    "duration_s": pl.Float64(),
    "moving_time_s": pl.Float64(),
    "paused_time_s": pl.Float64(),
    "distance_m": pl.Float64(),
    "elevation_gain_m": pl.Float64(),
    "avg_hr": pl.Float64(),
    "max_hr": pl.Float64(),
    "avg_speed_mps": pl.Float64(),
    "max_speed_mps": pl.Float64(),
    "calories": pl.Float64(),
    "repair_flags_json": pl.String(),
    "hr_zone_1_s": pl.Float64(),
    "hr_zone_2_s": pl.Float64(),
    "hr_zone_3_s": pl.Float64(),
    "hr_zone_4_s": pl.Float64(),
    "hr_zone_5_s": pl.Float64(),
    "hr_coverage_ratio": pl.Float64(),
    "hr_histogram_json": pl.String(),
    "hr_10min_json": pl.String(),
    "hr_drift_bpm_per_hour": pl.Float64(),
    "load_value": pl.Float64(),
    "load_method": pl.String(),
    "load_confidence": pl.String(),
    "data_completeness": pl.Float64(),
    "ef_first_half": pl.Float64(),
    "ef_second_half": pl.Float64(),
    "aerobic_decoupling_pct": pl.Float64(),
    "decoupling_confidence": pl.String(),
    ANALYSIS_VERSION: pl.String(),
}

CACHE_FILE = "training_cache.parquet"
METADATA_FILE = "metadata.json"
LOCK_FILE = "cache.lock"
