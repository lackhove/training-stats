"""Activity-level derived metrics and cache queries.

``compute_activity_metrics`` derives per-activity analytics during ingest.
The query functions (``activity_summaries``, ``activity_details``,
``data_quality``) are the read-side counterpart: they consume lazy cache
scans, own the user-facing filtering/aggregation, and map rows into domain
models. Pure frame computation — no I/O, no locking.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, tzinfo

import polars as pl

from ..converters import optional_float, optional_str
from ..fit.types import ParsedActivity
from ..frames import (
    ANALYSIS_VERSION,
    FILE_REPAIR_FLAGS,
    STATUS_DUPLICATE,
    STATUS_FAILED,
    STATUS_OK,
)
from ..models import ActivityDetails, ActivitySummary, DataQuality
from ..settings import AthleteConfig, QualityConfig
from .endurance import DecouplingResult, compute_aerobic_decoupling
from .heart_rate import (
    HrBucketAvg,
    ZoneResult,
    build_zone_bounds,
    hr_bucket_averages,
    hr_drift_bpm_per_hour,
    hr_histogram,
    time_in_zones,
)
from .load import LOAD_METHOD, LOAD_VERSION, hr_trimp_load, load_confidence

_COMPLETENESS_FIELDS = 8


@dataclass(frozen=True, slots=True)
class ActivityMetrics:
    """Derived per-activity analytics (no raw time-series data)."""

    activity_id: str
    hr_zone_1_s: float
    hr_zone_2_s: float
    hr_zone_3_s: float
    hr_zone_4_s: float
    hr_zone_5_s: float
    hr_coverage_ratio: float
    hr_histogram_s: dict[str, float]
    hr_10min_averages: tuple[HrBucketAvg, ...]
    hr_drift_bpm_per_hour: float | None
    load_value: float | None
    load_method: str | None
    load_confidence: str | None
    data_completeness: float
    analysis_version: str
    ef_first_half: float | None = None
    ef_second_half: float | None = None
    aerobic_decoupling_pct: float | None = None
    decoupling_confidence: str = "insufficient_duration"


@dataclass(frozen=True, slots=True)
class AnalyzedActivity:
    """Pair of a parsed activity and its derived metrics."""

    activity: ParsedActivity
    metrics: ActivityMetrics


def compute_activity_metrics(
    activity: ParsedActivity,
    athlete: AthleteConfig,
    analysis_version: str,
    quality: QualityConfig | None = None,
) -> ActivityMetrics:
    """Compute zones, load and data-quality metrics for one activity."""
    zones = build_zone_bounds(athlete.heart_rate_zones.as_bounds())
    zone_result: ZoneResult = time_in_zones(activity.records, zones)
    buckets = hr_bucket_averages(activity.records)
    load_value = hr_trimp_load(
        zone_result, zones, max_hr=athlete.max_heart_rate, resting_hr=athlete.resting_heart_rate
    )
    effective_quality = quality or QualityConfig()
    decoupling: DecouplingResult = compute_aerobic_decoupling(
        activity.records,
        moving_speed_threshold_mps=effective_quality.moving_speed_threshold_mps,
        min_duration_s=effective_quality.decoupling_min_duration_s,
        min_coverage=effective_quality.decoupling_min_coverage,
        session_moving_time_s=activity.moving_time_s,
    )
    return ActivityMetrics(
        activity_id=activity.activity_id,
        hr_zone_1_s=zone_result.zone_seconds[0],
        hr_zone_2_s=zone_result.zone_seconds[1],
        hr_zone_3_s=zone_result.zone_seconds[2],
        hr_zone_4_s=zone_result.zone_seconds[3],
        hr_zone_5_s=zone_result.zone_seconds[4],
        hr_coverage_ratio=zone_result.hr_coverage_ratio,
        hr_histogram_s=hr_histogram(activity.records),
        hr_10min_averages=tuple(buckets),
        hr_drift_bpm_per_hour=hr_drift_bpm_per_hour(buckets),
        load_value=load_value,
        load_method=LOAD_METHOD if load_value is not None else None,
        load_confidence=load_confidence(zone_result.hr_coverage_ratio)
        if load_value is not None
        else None,
        data_completeness=_completeness(activity),
        analysis_version=f"{analysis_version}:{LOAD_VERSION}",
        ef_first_half=decoupling.ef_first_half,
        ef_second_half=decoupling.ef_second_half,
        aerobic_decoupling_pct=decoupling.aerobic_decoupling_pct,
        decoupling_confidence=decoupling.confidence,
    )


def _completeness(activity: ParsedActivity) -> float:
    """Fraction of key session fields that are present."""
    present = sum(
        [
            activity.duration_s is not None,
            activity.moving_time_s is not None,
            activity.distance_m is not None,
            activity.elevation_gain_m is not None,
            activity.avg_hr is not None,
            activity.max_hr is not None,
            activity.avg_speed_mps is not None,
            activity.calories is not None,
        ]
    )
    return present / _COMPLETENESS_FIELDS


# -- cache queries: LazyFrame in, domain rows out ---------------------------


_SUMMARY_COLUMNS: tuple[str, ...] = (
    "activity_id",
    "start_time",
    "sport",
    "sub_sport",
    "duration_s",
    "moving_time_s",
    "distance_m",
    "elevation_gain_m",
    "avg_hr",
    "max_hr",
    "calories",
    "load_value",
)


def activity_summaries(
    activities: pl.LazyFrame,
    tz: tzinfo,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    sport: str | None = None,
    limit: int = 50,
) -> list[ActivitySummary]:
    """List compact activity summaries, newest first, filtered by local date."""
    frame = activities.select(_SUMMARY_COLUMNS).with_columns(
        pl.col("start_time").dt.convert_time_zone(str(tz)).dt.date().alias("local_date")
    )
    if start_date is not None:
        frame = frame.filter(pl.col("local_date") >= start_date)
    if end_date is not None:
        frame = frame.filter(pl.col("local_date") <= end_date)
    if sport is not None:
        frame = frame.filter(pl.col("sport").str.to_lowercase() == sport.lower())
    rows = frame.sort("start_time", descending=True).head(limit).collect().to_dicts()
    return [_summary_row(row, tz) for row in rows]


def activity_details(
    activities: pl.LazyFrame, activity_id: str, tz: tzinfo
) -> ActivityDetails | None:
    """Map one activity row into full details; ``None`` for unknown ids."""
    rows = activities.filter(pl.col("activity_id") == activity_id).collect().to_dicts()
    if not rows:
        return None
    return _details_row(rows[0], tz)


def data_quality(activities: pl.LazyFrame, source_index: pl.LazyFrame) -> DataQuality:
    """Aggregate data-quality indicators over activities and source outcomes."""
    metrics = (
        activities.select(
            pl.len().alias("activity_count"),
            pl.col("hr_coverage_ratio").mean().alias("mean_coverage"),
            pl.col("data_completeness").mean().alias("mean_completeness"),
        )
        .collect()
        .to_dicts()[0]
    )
    failed = 0
    duplicates = 0
    repaired_files = 0
    discarded_segments = 0
    files: dict[tuple[str, str], tuple[bool, int]] = {}
    for row in source_index.collect().to_dicts():
        status = str(row.get("status"))
        if status == STATUS_FAILED:
            failed += 1
        elif status == STATUS_DUPLICATE:
            duplicates += 1
        key = (str(row.get("source_name")), str(row.get("source_path")))
        repaired = status == STATUS_OK and bool(row.get(FILE_REPAIR_FLAGS))
        previous = files.get(key, (False, 0))
        files[key] = (previous[0] or repaired, _int(row.get("discarded_segment_count")))
    repaired_files = sum(1 for repaired, _discarded in files.values() if repaired)
    discarded_segments = sum(discarded for _repaired, discarded in files.values())
    return DataQuality(
        activity_count=_int(metrics.get("activity_count")),
        mean_hr_coverage_ratio=optional_float(metrics.get("mean_coverage")),
        mean_data_completeness=optional_float(metrics.get("mean_completeness")),
        failed_file_count=failed,
        repaired_file_count=repaired_files,
        discarded_segment_count=discarded_segments,
        duplicate_activity_count=duplicates,
    )


# -- row mapping helpers ----------------------------------------------------


def _summary_row(row: dict[str, object], tz: tzinfo) -> ActivitySummary:
    start_time = row["start_time"]
    if not isinstance(start_time, datetime):
        msg = "activity frame start_time has unexpected type"
        raise ValueError(msg)
    return ActivitySummary(
        activity_id=str(row["activity_id"]),
        start_time=start_time,
        sport=str(row["sport"]),
        sub_sport=optional_str(row.get("sub_sport")),
        duration_s=optional_float(row.get("duration_s")),
        distance_m=optional_float(row.get("distance_m")),
        elevation_gain_m=optional_float(row.get("elevation_gain_m")),
        avg_hr=optional_float(row.get("avg_hr")),
        max_hr=optional_float(row.get("max_hr")),
        calories=optional_float(row.get("calories")),
        load_value=optional_float(row.get("load_value")),
        moving_time_s=optional_float(row.get("moving_time_s")),
        local_start_time=start_time.astimezone(tz).isoformat(),
    )


def _details_row(row: dict[str, object], tz: tzinfo) -> ActivityDetails:
    summary = _summary_row(row, tz)
    return ActivityDetails(
        activity_id=summary.activity_id,
        start_time=summary.start_time,
        sport=summary.sport,
        sub_sport=summary.sub_sport,
        duration_s=summary.duration_s,
        moving_time_s=summary.moving_time_s,
        distance_m=summary.distance_m,
        elevation_gain_m=summary.elevation_gain_m,
        avg_hr=summary.avg_hr,
        max_hr=summary.max_hr,
        avg_speed_mps=optional_float(row.get("avg_speed_mps")),
        max_speed_mps=optional_float(row.get("max_speed_mps")),
        calories=summary.calories,
        hr_zone_1_s=_float(row.get("hr_zone_1_s")),
        hr_zone_2_s=_float(row.get("hr_zone_2_s")),
        hr_zone_3_s=_float(row.get("hr_zone_3_s")),
        hr_zone_4_s=_float(row.get("hr_zone_4_s")),
        hr_zone_5_s=_float(row.get("hr_zone_5_s")),
        hr_coverage_ratio=_float(row.get("hr_coverage_ratio")),
        hr_histogram_s=_histogram_from_frame(row.get("hr_histogram")),
        hr_10min_averages=_buckets_from_frame(row.get("hr_10min_averages")),
        hr_drift_bpm_per_hour=optional_float(row.get("hr_drift_bpm_per_hour")),
        load_value=summary.load_value,
        load_method=optional_str(row.get("load_method")),
        load_confidence=optional_str(row.get("load_confidence")),
        data_completeness=_float(row.get("data_completeness")),
        analysis_version=str(row.get(ANALYSIS_VERSION, "")),
        source_name=str(row.get("source_name", "")),
        source_path=str(row.get("source_path", "")),
        ef_first_half=optional_float(row.get("ef_first_half")),
        ef_second_half=optional_float(row.get("ef_second_half")),
        aerobic_decoupling_pct=optional_float(row.get("aerobic_decoupling_pct")),
        decoupling_confidence=optional_str(row.get("decoupling_confidence")),
        local_start_time=summary.local_start_time,
        paused_time_s=optional_float(row.get("paused_time_s")),
        repair_flags=_flags_from_frame(row.get("repair_flags")),
    )


def _histogram_from_frame(value: object) -> dict[str, float]:
    if not isinstance(value, list):
        return {}
    histogram: dict[str, float] = {}
    for entry in value:
        if isinstance(entry, dict) and isinstance(entry.get("bin"), str):
            histogram[str(entry["bin"])] = _float(entry.get("seconds"))
    return histogram


def _buckets_from_frame(value: object) -> tuple[HrBucketAvg, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        HrBucketAvg(
            start_min=_int(entry.get("start_min")),
            avg_hr=_float(entry.get("avg_hr")),
            hr_time_s=_float(entry.get("hr_time_s")),
        )
        for entry in value
        if isinstance(entry, dict)
    )


def _flags_from_frame(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(flag) for flag in value)


def _float(value: object) -> float:
    result = optional_float(value)
    return 0.0 if result is None else result


def _int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return int(value)
