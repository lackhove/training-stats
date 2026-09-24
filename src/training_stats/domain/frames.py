"""Canonical Polars frame contracts used by domain analytics."""

from __future__ import annotations

from typing import Final, Literal

import polars as pl

SourceStatus = Literal["ok", "failed", "duplicate"]
STATUS_OK: Final[SourceStatus] = "ok"
STATUS_FAILED: Final[SourceStatus] = "failed"
STATUS_DUPLICATE: Final[SourceStatus] = "duplicate"
ANALYSIS_VERSION = "analysis_version"
SOURCE_ANALYSIS_VERSION = "source_analysis_version"
PROCESSED_AT = "processed_at"
FILE_REPAIR_FLAGS = "file_repair_flags"

REPAIR_FLAGS_DTYPE = pl.List(pl.String())
HR_HISTOGRAM_DTYPE = pl.List(pl.Struct({"bin": pl.String(), "seconds": pl.Float64()}))
HR_BUCKETS_DTYPE = pl.List(
    pl.Struct(
        {
            "start_min": pl.Int64(),
            "avg_hr": pl.Float64(),
            "hr_time_s": pl.Float64(),
        }
    )
)

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
    "segment_index",
    "segment_count",
    "repair_flags",
    "source_name",
    "source_path",
    "hr_zone_1_s",
    "hr_zone_2_s",
    "hr_zone_3_s",
    "hr_zone_4_s",
    "hr_zone_5_s",
    "hr_coverage_ratio",
    "hr_histogram",
    "hr_10min_averages",
    "hr_drift_bpm_per_hour",
    "load_value",
    "load_method",
    "load_confidence",
    "data_completeness",
    "ef_first_half",
    "ef_second_half",
    "aerobic_decoupling_pct",
    "decoupling_confidence",
    "analysis_version",
)

ACTIVITY_FRAME_SCHEMA: dict[str, pl.DataType] = {
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
    "segment_index": pl.UInt32(),
    "segment_count": pl.UInt32(),
    "repair_flags": REPAIR_FLAGS_DTYPE,
    "source_name": pl.String(),
    "source_path": pl.String(),
    "hr_zone_1_s": pl.Float64(),
    "hr_zone_2_s": pl.Float64(),
    "hr_zone_3_s": pl.Float64(),
    "hr_zone_4_s": pl.Float64(),
    "hr_zone_5_s": pl.Float64(),
    "hr_coverage_ratio": pl.Float64(),
    "hr_histogram": HR_HISTOGRAM_DTYPE,
    "hr_10min_averages": HR_BUCKETS_DTYPE,
    "hr_drift_bpm_per_hour": pl.Float64(),
    "load_value": pl.Float64(),
    "load_method": pl.String(),
    "load_confidence": pl.String(),
    "data_completeness": pl.Float64(),
    "ef_first_half": pl.Float64(),
    "ef_second_half": pl.Float64(),
    "aerobic_decoupling_pct": pl.Float64(),
    "decoupling_confidence": pl.String(),
    "analysis_version": pl.String(),
}

SOURCE_INDEX_COLUMNS: tuple[str, ...] = (
    "source_name",
    "source_path",
    "source_hash",
    "file_size",
    "modified_at",
    "activity_id",
    "segment_index",
    "segment_count",
    "file_repair_flags",
    "discarded_segment_count",
    "processed_at",
    "source_analysis_version",
    "status",
    "error_message",
)

SOURCE_INDEX_FRAME_SCHEMA: dict[str, pl.DataType] = {
    "source_name": pl.String(),
    "source_path": pl.String(),
    "source_hash": pl.String(),
    "file_size": pl.UInt64(),
    "modified_at": pl.Datetime("us", "UTC"),
    "activity_id": pl.String(),
    "segment_index": pl.UInt32(),
    "segment_count": pl.UInt32(),
    "file_repair_flags": REPAIR_FLAGS_DTYPE,
    "discarded_segment_count": pl.UInt32(),
    "processed_at": pl.Datetime("us", "UTC"),
    "source_analysis_version": pl.String(),
    "status": pl.String(),
    "error_message": pl.String(),
}
