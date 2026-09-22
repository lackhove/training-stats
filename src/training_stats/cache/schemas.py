"""Polars schemas for the Parquet cache files.

High-frequency FIT record time series are deliberately absent from every
schema: only aggregated, derived data is persisted.
"""

from __future__ import annotations

import polars as pl

ANALYSIS_VERSION = "analysis_version"
# Cache-invalidation key for derived analytics, owned by the code: bump it
# whenever parsing or metric semantics change so the next sync reprocesses
# every file (hash-based change detection alone would serve stale rows).
ANALYSIS_VERSION_VALUE = "4"
PROCESSED_AT = "processed_at"
# source_index.parquet status values, shared by ingest (row creation) and
# sync (aggregate filtering, metadata counts).
STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_DUPLICATE = "duplicate"

ACTIVITY_SCHEMA: dict[str, pl.DataType] = {
    "activity_id": pl.String(),
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
    "segment_index": pl.UInt32(),
    "segment_count": pl.UInt32(),
    "repair_flags_json": pl.String(),
    "source_name": pl.String(),
    "source_path": pl.String(),
}

ACTIVITY_METRICS_SCHEMA: dict[str, pl.DataType] = {
    "activity_id": pl.String(),
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

DAILY_SCHEMA: dict[str, pl.DataType] = {
    "date": pl.String(),
    "sport": pl.String(),
    "activity_count": pl.UInt32(),
    "duration_s": pl.Float64(),
    "distance_m": pl.Float64(),
    "elevation_gain_m": pl.Float64(),
    "hr_zone_1_s": pl.Float64(),
    "hr_zone_2_s": pl.Float64(),
    "hr_zone_3_s": pl.Float64(),
    "hr_zone_4_s": pl.Float64(),
    "hr_zone_5_s": pl.Float64(),
    "load_value": pl.Float64(),
    "hard_activity_count": pl.UInt32(),
    "longest_activity_s": pl.Float64(),
    "longest_activity_id": pl.String(),
}

WEEKLY_SCHEMA: dict[str, pl.DataType] = {
    "iso_year": pl.Int32(),
    "iso_week": pl.Int32(),
    "week_start": pl.String(),
    "sport": pl.String(),
    "activity_count": pl.UInt32(),
    "duration_s": pl.Float64(),
    "distance_m": pl.Float64(),
    "elevation_gain_m": pl.Float64(),
    "hr_zone_1_s": pl.Float64(),
    "hr_zone_2_s": pl.Float64(),
    "hr_zone_3_s": pl.Float64(),
    "hr_zone_4_s": pl.Float64(),
    "hr_zone_5_s": pl.Float64(),
    "load_value": pl.Float64(),
    "hard_activity_count": pl.UInt32(),
    "longest_activity_s": pl.Float64(),
    "longest_activity_id": pl.String(),
    "load_value_prev": pl.Float64(),
    "duration_s_prev": pl.Float64(),
    "distance_m_prev": pl.Float64(),
}

SOURCE_INDEX_SCHEMA: dict[str, pl.DataType] = {
    "source_name": pl.String(),
    "source_path": pl.String(),
    "source_hash": pl.String(),
    "file_size": pl.UInt64(),
    "modified_at": pl.Datetime("us", "UTC"),
    "activity_id": pl.String(),
    "segment_index": pl.UInt32(),
    "segment_count": pl.UInt32(),
    "repair_flags_json": pl.String(),
    "discarded_segment_count": pl.UInt32(),
    PROCESSED_AT: pl.Datetime("us", "UTC"),
    ANALYSIS_VERSION: pl.String(),
    "status": pl.String(),
    "error_message": pl.String(),
}

ACTIVITIES_FILE = "activities.parquet"
ACTIVITY_METRICS_FILE = "activity_metrics.parquet"
DAILY_FILE = "daily_metrics.parquet"
WEEKLY_FILE = "weekly_metrics.parquet"
SOURCE_INDEX_FILE = "source_index.parquet"
METADATA_FILE = "metadata.json"
LOCK_FILE = "sync.lock"
