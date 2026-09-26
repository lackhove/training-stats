"""Pure per-trip metric functions and private activity projections."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta

from ..fit.types import ParsedActivity
from ..metrics import (
    Axis,
    Confidence,
    Metric,
    MetricKind,
    MetricMethod,
    MissingReason,
    Period,
    ScalarValue,
    TimeBasis,
    TripPeriod,
)
from ..models import STATUS_DUPLICATE, STATUS_FAILED, STATUS_OK, ActivityRecord, SourceIndexEntry
from ..settings import AthleteConfig, QualityConfig
from .activity_statistics import compute_activity_statistics
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
from .load import LOAD_COEFFICIENT_SET, LOAD_FORMULA, LOAD_VERSION, hr_trimp_load, load_confidence
from .metric_helpers import scalar_metric, table_metric

_COMPLETENESS_FIELDS = 7


@dataclass(frozen=True, slots=True)
class AnalyzedActivity:
    """Pair of a parsed activity and its immutable metric outputs."""

    activity: ParsedActivity
    metrics: tuple[Metric, ...]


def compute_activity_metrics(
    activity: ParsedActivity,
    athlete: AthleteConfig,
    analysis_version: str,
    quality: QualityConfig | None = None,
) -> tuple[Metric, ...]:
    """Compute per-trip metrics from one transient parsed activity."""
    zones = build_zone_bounds(athlete.heart_rate_zones.as_bounds())
    zone_result: ZoneResult = time_in_zones(
        activity.records, zones, pause_intervals=activity.pause_intervals
    )
    buckets = hr_bucket_averages(activity.records, pause_intervals=activity.pause_intervals)
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
        pause_intervals=activity.pause_intervals,
    )
    period = TripPeriod(
        activity_id=activity.activity_id,
        start=activity.start_time,
        end=(
            activity.start_time + timedelta(seconds=activity.duration_s)
            if activity.duration_s is not None
            else None
        ),
    )
    histogram = hr_histogram(activity.records, pause_intervals=activity.pause_intervals)
    metrics: list[Metric] = []
    metrics.extend(_session_metrics(activity, analysis_version, period))
    metrics.extend(_heart_rate_metrics(period, zone_result, buckets, histogram))
    metrics.extend(_load_metrics(activity, period, zone_result, load_value))
    metrics.extend(_endurance_metrics(period, decoupling))
    metrics.extend(compute_activity_statistics(activity, athlete, effective_quality))
    return tuple(metrics)


def _session_metrics(
    activity: ParsedActivity, analysis_version: str, period: TripPeriod
) -> list[Metric]:
    kind = MetricKind.MEASURED
    return [
        scalar_metric(
            "analysis.version",
            period,
            analysis_version,
            description="Analysis code version used to derive this activity's metrics.",
            kind=MetricKind.INTERPRETED,
        ),
        scalar_metric(
            "activity.sport",
            period,
            activity.sport,
            description="Sport category reported for this activity.",
            kind=MetricKind.INTERPRETED,
        ),
        scalar_metric(
            "activity.sub_sport",
            period,
            activity.sub_sport,
            description="Sub-sport category reported for this activity.",
            kind=MetricKind.INTERPRETED,
        ),
        scalar_metric(
            "activity.duration",
            period,
            activity.duration_s,
            description="Elapsed time from the first to last record, including pauses.",
            unit="s",
            kind=kind,
        ),
        scalar_metric(
            "activity.moving_time",
            period,
            activity.moving_time_s,
            description="Time classified as moving, excluding paused intervals.",
            unit="s",
            kind=kind,
        ),
        scalar_metric(
            "activity.paused_time",
            period,
            activity.paused_time_s,
            description="Time excluded from moving statistics as paused.",
            unit="s",
            kind=kind,
        ),
        scalar_metric(
            "activity.distance",
            period,
            activity.distance_m,
            description="Distance derived from active record intervals.",
            unit="m",
            kind=kind,
        ),
        scalar_metric(
            "activity.elevation_gain",
            period,
            activity.elevation_gain_m,
            description="Positive elevation gain derived from active records.",
            unit="m",
            kind=kind,
        ),
        scalar_metric(
            "activity.average_heart_rate",
            period,
            activity.avg_hr,
            description="Time-weighted average heart rate over valid samples.",
            unit="bpm",
            kind=kind,
        ),
        scalar_metric(
            "activity.maximum_heart_rate",
            period,
            activity.max_hr,
            description="Maximum valid heart-rate sample during the activity.",
            unit="bpm",
            kind=kind,
        ),
        scalar_metric(
            "activity.average_speed",
            period,
            activity.avg_speed_mps,
            description="Average speed derived from distance and moving time or active intervals.",
            unit="m/s",
            kind=kind,
        ),
        scalar_metric(
            "activity.maximum_speed",
            period,
            activity.max_speed_mps,
            description="Maximum valid speed sample during the activity.",
            unit="m/s",
            kind=kind,
        ),
    ]


def _heart_rate_metrics(
    period: TripPeriod,
    zone_result: ZoneResult,
    buckets: list[HrBucketAvg],
    histogram: dict[str, float],
) -> list[Metric]:
    zone_axis = Axis(
        name="heart_rate_zone",
        values=tuple(f"zone_{index}" for index in range(1, 6)),
    )
    metrics = [
        scalar_metric(
            "heart_rate.coverage",
            period,
            zone_result.hr_coverage_ratio,
            description="Share of active time covered by valid heart-rate samples.",
            unit="ratio",
            kind=MetricKind.CALCULATED,
            time_basis=TimeBasis.ACTIVE_TIME,
        ),
        table_metric(
            "heart_rate.zone_time",
            period,
            zone_axis,
            zone_result.zone_seconds,
            description="Active time assigned to each configured heart-rate zone.",
            unit="s",
            missing_reason=(MissingReason.NO_DATA if zone_result.valid_hr_time_s <= 0.0 else None),
            time_basis=TimeBasis.ACTIVE_TIME,
        ),
    ]

    histogram_values = tuple(histogram.values())
    metrics.append(
        table_metric(
            "heart_rate.histogram",
            period,
            Axis(name="heart_rate_bin", values=tuple(histogram)),
            histogram_values,
            description="Active time accumulated in five-beat-per-minute heart-rate bins.",
            unit="s",
            missing_reason=MissingReason.NO_DATA if not histogram else None,
            time_basis=TimeBasis.ACTIVE_TIME,
        )
    )

    bucket_axis = Axis(name="activity_minute", values=tuple(bucket.start_min for bucket in buckets))
    metrics.extend(
        [
            table_metric(
                "heart_rate.average_by_10_minutes",
                period,
                bucket_axis,
                tuple(bucket.avg_hr for bucket in buckets),
                description="Time-weighted average heart rate in consecutive ten-minute windows.",
                unit="bpm",
                missing_reason=MissingReason.NO_DATA if not buckets else None,
                time_basis=TimeBasis.ACTIVE_TIME,
            ),
            table_metric(
                "heart_rate.observed_time_by_10_minutes",
                period,
                bucket_axis,
                tuple(bucket.hr_time_s for bucket in buckets),
                description="Valid heart-rate time contributing to each ten-minute average.",
                unit="s",
                missing_reason=MissingReason.NO_DATA if not buckets else None,
                time_basis=TimeBasis.ACTIVE_TIME,
            ),
        ]
    )

    metrics.append(
        scalar_metric(
            "heart_rate.drift",
            period,
            hr_drift_bpm_per_hour(buckets),
            description="Heart-rate change per hour estimated from ten-minute averages.",
            unit="bpm/h",
            kind=MetricKind.CALCULATED,
            missing_reason=MissingReason.INSUFFICIENT_DATA,
            time_basis=TimeBasis.ACTIVE_TIME,
        )
    )
    return metrics


def _load_metrics(
    activity: ParsedActivity,
    period: TripPeriod,
    zone_result: ZoneResult,
    load_value: float | None,
) -> list[Metric]:
    return [
        scalar_metric(
            "training_load.hr_trimp",
            period,
            load_value,
            description="Banister HR load estimated from time in configured zones.",
            kind=MetricKind.ESTIMATED,
            unit="TRIMP",
            missing_reason=MissingReason.NO_DATA,
            method=MetricMethod.HR_TRIMP,
            algorithm_version=LOAD_VERSION,
            confidence=(
                Confidence(load_confidence(zone_result.hr_coverage_ratio))
                if load_value is not None
                else None
            ),
            formula=LOAD_FORMULA,
            coefficient_set=LOAD_COEFFICIENT_SET,
        ),
        scalar_metric(
            "data.completeness",
            period,
            _completeness(activity),
            description="Fraction of seven key activity fields present in the FIT data.",
            unit="ratio",
            kind=MetricKind.CALCULATED,
        ),
    ]


def _endurance_metrics(period: TripPeriod, decoupling: DecouplingResult) -> list[Metric]:
    metrics: list[Metric] = []
    decoupling_reason = _decoupling_missing_reason(decoupling.confidence)
    decoupling_confidence = Confidence.HIGH if decoupling.confidence == "high" else None
    for name, description, value, unit in (
        (
            "endurance.efficiency_first_half",
            "Speed-to-heart-rate efficiency in the first half of valid moving time.",
            decoupling.ef_first_half,
            "km/h/bpm",
        ),
        (
            "endurance.efficiency_second_half",
            "Speed-to-heart-rate efficiency in the second half of valid moving time.",
            decoupling.ef_second_half,
            "km/h/bpm",
        ),
        (
            "endurance.aerobic_decoupling",
            "Percent change in speed-to-heart-rate efficiency between equal moving-time halves.",
            decoupling.aerobic_decoupling_pct,
            "%",
        ),
    ):
        metrics.append(
            scalar_metric(
                name,
                period,
                value,
                description=description,
                unit=unit,
                kind=MetricKind.CALCULATED,
                missing_reason=decoupling_reason,
                confidence=decoupling_confidence,
                formula="speed_per_heart_rate; compare equal moving-time halves",
                time_basis=TimeBasis.MOVING_TIME,
            )
        )
    return metrics


def _decoupling_missing_reason(confidence: str) -> MissingReason:
    if confidence == "insufficient_coverage":
        return MissingReason.INSUFFICIENT_COVERAGE
    if confidence == "invalid_ef":
        return MissingReason.INVALID_EF
    return MissingReason.INSUFFICIENT_DURATION


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
        ]
    )
    return present / _COMPLETENESS_FIELDS


def data_quality_metrics(
    records: Sequence[ActivityRecord],
    source_index: Sequence[SourceIndexEntry],
    period: Period,
) -> tuple[Metric, ...]:
    """Summarize activity and source outcomes as metrics over ``period``."""
    coverage = [
        value
        for record in records
        if (value := _record_number(record, "heart_rate.coverage")) is not None
    ]
    completeness = [
        value
        for record in records
        if (value := _record_number(record, "data.completeness")) is not None
    ]
    failed_count = sum(entry.status == STATUS_FAILED for entry in source_index)
    duplicate_count = sum(entry.status == STATUS_DUPLICATE for entry in source_index)
    files: dict[tuple[str, str], tuple[bool, int]] = {}
    for entry in source_index:
        key = (entry.source_name, entry.source_path)
        repaired, discarded = files.get(key, (False, 0))
        files[key] = (
            repaired or (entry.status == STATUS_OK and bool(entry.file_repair_flags)),
            max(discarded, entry.discarded_segment_count),
        )
    mean_coverage = sum(coverage) / len(coverage) if coverage else None
    mean_completeness = sum(completeness) / len(completeness) if completeness else None
    values: tuple[tuple[str, str, int | float | None, str], ...] = (
        (
            "activity_count",
            "Number of accepted activities in this cache scope.",
            len(records),
            "activities",
        ),
        (
            "mean_hr_coverage_ratio",
            "Mean share of active time covered by valid heart-rate samples.",
            mean_coverage,
            "ratio",
        ),
        (
            "mean_data_completeness",
            "Mean fraction of key activity fields present in the source data.",
            mean_completeness,
            "ratio",
        ),
        (
            "failed_file_count",
            "Number of source files that failed to parse or read.",
            failed_count,
            "files",
        ),
        (
            "repaired_file_count",
            "Number of source files with one or more repair flags.",
            sum(repaired for repaired, _ in files.values()),
            "files",
        ),
        (
            "discarded_segment_count",
            "Number of short activity segments discarded during ingestion.",
            sum(discarded for _, discarded in files.values()),
            "segments",
        ),
        (
            "duplicate_activity_count",
            "Number of duplicate activity segments excluded from analysis.",
            duplicate_count,
            "activities",
        ),
    )
    return tuple(
        scalar_metric(
            f"data_quality.{name}",
            period,
            value,
            description=description,
            unit=unit,
            kind=MetricKind.CALCULATED,
        )
        for name, description, value, unit in values
    )


def _record_number(record: ActivityRecord, name: str) -> float | None:
    metric = next((item for item in record.metrics if item.name == name), None)
    if metric is None or not isinstance(metric.value, ScalarValue):
        return None
    value = metric.value.value
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)
