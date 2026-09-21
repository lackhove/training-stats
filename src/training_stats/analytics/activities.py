"""Activity-level derived metrics."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import AthleteConfig, QualityConfig
from ..fit.types import ParsedActivity
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
