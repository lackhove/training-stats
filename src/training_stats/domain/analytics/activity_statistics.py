"""Explicit Metric producers for coarse activity statistics.

Each public producer takes one ``ParsedActivity`` plus only the explicit
athlete or quality configuration it needs, derives its narrow facts from the
record helpers in ``statistics_records`` and constructs its metrics with
literal names, descriptions, units, kinds and missing reasons. Producers may
re-read the records rather than sharing one summary object. The public
functions are grouped per output family; ``compute_activity_statistics`` is
the thin composer over them.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta

from ..fit.types import ParsedActivity
from ..metrics import (
    Axis,
    AxisBin,
    MatrixValue,
    Metric,
    MetricKind,
    MetricScalar,
    MissingReason,
    TimeBasis,
    TripPeriod,
)
from ..settings import AthleteConfig, QualityConfig
from .metric_helpers import scalar_metric, table_metric
from .statistics_records import (
    Interval,
    active_time_s,
    build_intervals,
    elapsed_time_s,
    is_short_activity,
    merged_pauses,
    moving_s,
    moving_time_s,
    timestamp_issues,
)

_STATISTICS_PREFIX = "activity.statistics"
WINDOW_S = 300.0

CADENCE_BANDS: tuple[tuple[str, float, float | None], ...] = (
    ("rpm_1_59", 1.0, 60.0),
    ("rpm_60_69", 60.0, 70.0),
    ("rpm_70_79", 70.0, 80.0),
    ("rpm_80_89", 80.0, 90.0),
    ("rpm_90_99", 90.0, 100.0),
    ("rpm_100_plus", 100.0, None),
)

HR_RESERVE_FORMULA = "(mean HR - resting HR) / (maximum HR - resting HR)"
DRIFT_DESCRIPTION = "heart rate drift bpm per hour for fatigue response during the activity."


@dataclass(frozen=True, slots=True)
class _Window:
    """One five-minute window summary; ``cadence_reason`` is set when masked."""

    start_s: float
    end_s: float
    duration_s: float
    moving_time_s: float
    distance_m: float
    moving_time_ratio: float
    average_speed_kmh: float | None
    median_speed_kmh: float | None
    speed_p25_kmh: float | None
    speed_p75_kmh: float | None
    average_heart_rate_bpm: float | None
    median_heart_rate_bpm: float | None
    heart_rate_p25_bpm: float | None
    heart_rate_p75_bpm: float | None
    heart_rate_valid_time_s: float
    average_pedaling_cadence_rpm: float | None
    median_pedaling_cadence_rpm: float | None
    cadence_coverage_ratio: float
    cadence_reason: str | None


@dataclass(frozen=True, slots=True)
class _EfficiencyFacts:
    """Speed-per-heart-rate efficiency over valid moving time."""

    speed_per_heart_rate: float | None
    speed_per_heart_rate_reserve: float | None
    first_half_average_speed_kmh: float | None
    second_half_average_speed_kmh: float | None
    first_half_average_heart_rate_bpm: float | None
    second_half_average_heart_rate_bpm: float | None
    ef_first_half: float | None
    ef_second_half: float | None
    aerobic_decoupling_pct: float | None
    valid_time_s: float
    coverage_ratio: float
    heart_rate_reserve_formula: str | None
    missing_reason: MissingReason | None


@dataclass(frozen=True, slots=True)
class _Bin:
    """One observed two-sensor bin with its duration and sample count."""

    x_lower: float
    x_upper: float
    y_lower: float
    y_upper: float
    duration_s: float
    observation_count: int


type _WindowColumn = tuple[str, str, Callable[[_Window], MetricScalar | None]]
type _EfficiencyField = tuple[
    str, str, str | None, MetricKind, Callable[[_EfficiencyFacts], MetricScalar | None]
]

_WINDOW_COLUMNS: tuple[_WindowColumn, ...] = (
    ("average_heart_rate_bpm", "average heart rate bpm", lambda w: w.average_heart_rate_bpm),
    (
        "average_pedaling_cadence_rpm",
        "average pedaling cadence rpm",
        lambda w: w.average_pedaling_cadence_rpm,
    ),
    ("average_speed_kmh", "average speed kmh", lambda w: w.average_speed_kmh),
    ("cadence_coverage_ratio", "cadence coverage ratio", lambda w: w.cadence_coverage_ratio),
    ("cadence_reason", "cadence reason", lambda w: w.cadence_reason),
    ("distance_m", "distance m", lambda w: w.distance_m),
    ("duration_s", "duration", lambda w: w.duration_s),
    ("end_s", "end", lambda w: w.end_s),
    ("heart_rate_p25_bpm", "heart rate p25 bpm", lambda w: w.heart_rate_p25_bpm),
    ("heart_rate_p75_bpm", "heart rate p75 bpm", lambda w: w.heart_rate_p75_bpm),
    ("heart_rate_valid_time_s", "heart rate valid time", lambda w: w.heart_rate_valid_time_s),
    (
        "heart_rate_zone_percent_denominator",
        "heart rate zone percent denominator",
        lambda w: "valid_heart_rate_time",
    ),
    ("median_heart_rate_bpm", "median heart rate bpm", lambda w: w.median_heart_rate_bpm),
    (
        "median_pedaling_cadence_rpm",
        "median pedaling cadence rpm",
        lambda w: w.median_pedaling_cadence_rpm,
    ),
    ("median_speed_kmh", "median speed kmh", lambda w: w.median_speed_kmh),
    ("moving_time_ratio", "moving time ratio", lambda w: w.moving_time_ratio),
    ("moving_time_s", "moving time", lambda w: w.moving_time_s),
    ("speed_p25_kmh", "speed p25 kmh", lambda w: w.speed_p25_kmh),
    ("speed_p75_kmh", "speed p75 kmh", lambda w: w.speed_p75_kmh),
)

_EFFICIENCY_FIELDS: tuple[_EfficiencyField, ...] = (
    (
        "speed_per_heart_rate",
        "speed per heart rate for {context} during the activity.",
        "bpm",
        MetricKind.CALCULATED,
        lambda e: e.speed_per_heart_rate,
    ),
    (
        "speed_per_heart_rate_reserve",
        "speed per heart rate reserve for {context} during the activity.",
        "bpm",
        MetricKind.CALCULATED,
        lambda e: e.speed_per_heart_rate_reserve,
    ),
    (
        "first_half_average_speed_kmh",
        "first half average speed kmh for {context} during the activity.",
        "km/h",
        MetricKind.CALCULATED,
        lambda e: e.first_half_average_speed_kmh,
    ),
    (
        "second_half_average_speed_kmh",
        "second half average speed kmh for {context} during the activity.",
        "km/h",
        MetricKind.CALCULATED,
        lambda e: e.second_half_average_speed_kmh,
    ),
    (
        "first_half_average_heart_rate_bpm",
        "first half average heart rate bpm for {context} during the activity.",
        "bpm",
        MetricKind.CALCULATED,
        lambda e: e.first_half_average_heart_rate_bpm,
    ),
    (
        "second_half_average_heart_rate_bpm",
        "second half average heart rate bpm for {context} during the activity.",
        "bpm",
        MetricKind.CALCULATED,
        lambda e: e.second_half_average_heart_rate_bpm,
    ),
    (
        "ef_first_half",
        "ef first half for {context} during the activity.",
        None,
        MetricKind.CALCULATED,
        lambda e: e.ef_first_half,
    ),
    (
        "ef_second_half",
        "ef second half for {context} during the activity.",
        None,
        MetricKind.CALCULATED,
        lambda e: e.ef_second_half,
    ),
    (
        "aerobic_decoupling_pct",
        "aerobic decoupling pct for {context} during the activity.",
        "%",
        MetricKind.CALCULATED,
        lambda e: e.aerobic_decoupling_pct,
    ),
    (
        "valid_time_s",
        "Duration of valid {context} measurements used in the distribution.",
        "s",
        MetricKind.CALCULATED,
        lambda e: e.valid_time_s,
    ),
    (
        "coverage_ratio",
        "Share of {context} time covered by valid measurements.",
        "ratio",
        MetricKind.CALCULATED,
        lambda e: e.coverage_ratio,
    ),
    (
        "heart_rate_reserve_formula",
        "heart rate reserve formula for {context} during the activity.",
        "bpm",
        MetricKind.INTERPRETED,
        lambda e: e.heart_rate_reserve_formula,
    ),
)

_RELATIONSHIP_SPECS: dict[str, tuple[str, str]] = {
    "speed_heart_rate": ("speed heart rate", "bpm"),
    "speed_cadence": ("speed cadence", "rpm"),
    "heart_rate_cadence": ("heart rate cadence", "bpm"),
}

_HEATMAP_SPECS: dict[str, tuple[str, str, str]] = {
    "heart_rate_speed": ("heart rate and speed", "heart rate speed", "bpm"),
    "heart_rate_cadence": ("heart rate and cadence", "heart rate cadence", "bpm"),
    "speed_cadence": ("speed and cadence", "speed cadence", "rpm"),
}


def compute_activity_statistics(
    activity: ParsedActivity,
    athlete: AthleteConfig,
    quality: QualityConfig,
) -> tuple[Metric, ...]:
    """Return detailed distribution, window, relationship, and heatmap metrics."""
    return (
        *speed_metrics(activity, quality),
        *heart_rate_metrics(activity),
        *cadence_metrics(activity, quality),
        *zone_metrics(activity, athlete),
        *time_window_metrics(activity, quality),
        *efficiency_metrics(activity, athlete, quality),
        *fatigue_response_metrics(activity, athlete, quality),
        *relationship_metrics(activity, quality),
        *heatmap_metrics(activity, quality),
        *sensor_quality_metrics(activity, quality),
    )


def speed_metrics(activity: ParsedActivity, quality: QualityConfig) -> tuple[Metric, ...]:
    """Speed distribution, time above configured thresholds and coverage ratios."""
    period = _period(activity)
    prefix = f"{_STATISTICS_PREFIX}.speed_statistics"
    short = is_short_activity(activity)
    intervals = build_intervals(activity)
    threshold_mps = quality.moving_speed_threshold_mps
    samples: list[tuple[float, float]] = []
    valid_samples = 0
    during_movement = 0
    for interval in intervals:
        speed = interval.speed_kmh
        if speed is None:
            continue
        valid_samples += 1
        weight = moving_s(interval, threshold_mps)
        if weight > 0.0:
            during_movement += 1
            samples.append((speed, weight))
    moving = moving_time_s(activity, intervals, threshold_mps)
    metrics = _distribution_metrics(
        prefix,
        samples,
        moving,
        context="speed",
        unit="km/h",
        time_basis=TimeBasis.MOVING_TIME,
        empty_reason=MissingReason.INSUFFICIENT_DATA,
        period=period,
    )
    if short:
        return tuple(metrics)
    threshold_seconds = {float(f"{threshold:g}"): 0.0 for threshold in quality.speed_thresholds_kmh}
    for speed, weight in samples:
        for threshold in quality.speed_thresholds_kmh:
            if speed >= threshold:
                threshold_seconds[float(f"{threshold:g}")] += weight
    metrics.append(
        table_metric(
            f"{prefix}.speed_above_threshold_s.seconds",
            period,
            Axis(
                name="threshold_kmh",
                values=tuple(threshold_seconds),
                unit="km/h",
            ),
            tuple(threshold_seconds.values()),
            description="Time spent at or above each configured speed threshold.",
            unit="s",
            kind=MetricKind.CALCULATED,
            time_basis=TimeBasis.MOVING_TIME,
        )
    )
    metrics.append(
        scalar_metric(
            f"{prefix}.valid_moving_time_ratio",
            period,
            _capped_coverage(sum(weight for _speed, weight in samples), moving),
            description="Share of speed time covered by valid measurements.",
            unit="s",
            kind=MetricKind.CALCULATED,
            time_basis=TimeBasis.MOVING_TIME,
        )
    )
    metrics.append(
        scalar_metric(
            f"{prefix}.valid_measurement_ratio",
            period,
            valid_samples / len(intervals) if intervals else 0.0,
            description="Share of speed time covered by valid measurements.",
            unit="km/h",
            kind=MetricKind.CALCULATED,
            time_basis=TimeBasis.MOVING_TIME,
        )
    )
    metrics.append(
        scalar_metric(
            f"{prefix}.measurements_during_movement_ratio",
            period,
            during_movement / valid_samples if valid_samples > 0 else 0.0,
            description="Share of speed time covered by valid measurements.",
            unit="km/h",
            kind=MetricKind.CALCULATED,
            time_basis=TimeBasis.MOVING_TIME,
        )
    )
    return tuple(metrics)


def heart_rate_metrics(activity: ParsedActivity) -> tuple[Metric, ...]:
    """Heart-rate distribution and its five-beat-per-minute histogram."""
    period = _period(activity)
    prefix = f"{_STATISTICS_PREFIX}.heart_rate_statistics"
    intervals = build_intervals(activity)
    samples = [
        (interval.heart_rate, interval.active_s)
        for interval in intervals
        if interval.heart_rate is not None
    ]
    metrics = _distribution_metrics(
        prefix,
        samples,
        active_time_s(activity),
        context="heart rate",
        unit="bpm",
        time_basis=TimeBasis.ACTIVE_TIME,
        empty_reason=MissingReason.INSUFFICIENT_DATA,
        period=period,
    )
    histogram = _hr_histogram(samples)
    if histogram:
        metrics.append(
            table_metric(
                f"{prefix}.histogram_s.seconds",
                period,
                Axis(name="bin", values=tuple(label for label, _seconds in histogram)),
                tuple(seconds for _label, seconds in histogram),
                description=(
                    "Time-weighted distribution of heart rate histogram seconds across sensor bins."
                ),
                unit="s",
                kind=MetricKind.CALCULATED,
                time_basis=TimeBasis.ACTIVE_TIME,
            )
        )
    return tuple(metrics)


def cadence_metrics(activity: ParsedActivity, quality: QualityConfig) -> tuple[Metric, ...]:
    """Cadence availability, distributions, pedaling time and cadence bands."""
    period = _period(activity)
    prefix = f"{_STATISTICS_PREFIX}.cadence_statistics"
    short = is_short_activity(activity)
    intervals = build_intervals(activity)
    threshold_mps = quality.moving_speed_threshold_mps
    moving = moving_time_s(activity, intervals, threshold_mps)
    samples: list[tuple[float, float]] = []
    pedaling_samples: list[tuple[float, float]] = []
    if short:
        available = False
        reason: MissingReason | None = MissingReason.NO_CADENCE_DATA
        overall_reason = MissingReason.INSUFFICIENT_DATA
        pedaling_reason = MissingReason.INSUFFICIENT_DATA
        pedaling_time_s: float | None = 0.0
        no_pedaling_time_s: float | None = 0.0
        pedaling_time_ratio: float | None = 0.0
        coverage = 0.0
        bands = tuple((band, 0.0) for band, _low, _high in CADENCE_BANDS)
    else:
        coverage = _cadence_coverage(intervals, moving)
        available, reason = _cadence_availability(intervals, coverage, quality)
        samples = _cadence_samples(intervals, pedaling_only=False) if available else []
        pedaling_samples = _cadence_samples(intervals, pedaling_only=True) if available else []
        pedaling_total = sum(weight for _value, weight in pedaling_samples) if available else 0.0
        pedaling_time_s = pedaling_total if available else None
        no_pedaling_time_s = (
            sum(weight for value, weight in samples if value <= 0.0) if available else None
        )
        pedaling_time_ratio = pedaling_total / moving if available and moving > 0.0 else None
        overall_reason = reason or MissingReason.INSUFFICIENT_DATA
        pedaling_reason = (
            (MissingReason.NO_PEDALING_INTERVALS if available and not pedaling_samples else None)
            or reason
            or MissingReason.INSUFFICIENT_DATA
        )
        bands = tuple(_cadence_bands(pedaling_samples)) if available else ()
    missing_reason = reason or MissingReason.NO_DATA
    metrics = [
        scalar_metric(
            f"{prefix}.available",
            period,
            available,
            description="Whether cadence data meets the configured availability requirements.",
            unit="rpm",
            kind=MetricKind.INTERPRETED,
        ),
        *_distribution_metrics(
            f"{prefix}.overall",
            samples,
            moving,
            context="cadence",
            unit="rpm",
            time_basis=TimeBasis.MOVING_TIME,
            empty_reason=overall_reason,
            period=period,
        ),
        *_distribution_metrics(
            f"{prefix}.pedaling",
            pedaling_samples,
            moving,
            context="cadence",
            unit="rpm",
            time_basis=TimeBasis.PEDALING_TIME,
            empty_reason=pedaling_reason,
            period=period,
        ),
        scalar_metric(
            f"{prefix}.pedaling_time_s",
            period,
            pedaling_time_s,
            description="pedaling time for cadence during the activity.",
            unit="s",
            kind=MetricKind.CALCULATED,
            missing_reason=missing_reason,
        ),
        scalar_metric(
            f"{prefix}.no_pedaling_time_s",
            period,
            no_pedaling_time_s,
            description="no pedaling time for cadence during the activity.",
            unit="s",
            kind=MetricKind.CALCULATED,
            missing_reason=missing_reason,
        ),
        scalar_metric(
            f"{prefix}.pedaling_time_ratio",
            period,
            pedaling_time_ratio,
            description="Share of cadence time covered by valid measurements.",
            unit="s",
            kind=MetricKind.CALCULATED,
            missing_reason=missing_reason,
        ),
        scalar_metric(
            f"{prefix}.coverage_ratio",
            period,
            coverage,
            description="Share of cadence time covered by valid measurements.",
            unit="rpm",
            kind=MetricKind.CALCULATED,
        ),
    ]
    if bands:
        metrics.append(
            table_metric(
                f"{prefix}.bands_s.seconds",
                period,
                Axis(name="band", values=tuple(band for band, _seconds in bands)),
                tuple(seconds for _band, seconds in bands),
                description="Time accumulated in each cadence band.",
                unit="s",
                kind=MetricKind.CALCULATED,
            )
        )
    return tuple(metrics)


def zone_metrics(activity: ParsedActivity, athlete: AthleteConfig) -> tuple[Metric, ...]:
    """Activity-level heart-rate zone time, percent and denominator."""
    period = _period(activity)
    prefix = f"{_STATISTICS_PREFIX}.zone_distribution"
    intervals = build_intervals(activity)
    zone_bounds = athlete.heart_rate_zones.as_bounds()
    seconds = [0.0] * 5
    valid_time_s = 0.0
    for interval in intervals:
        heart_rate = interval.heart_rate
        if heart_rate is None:
            continue
        valid_time_s += interval.active_s
        for index, (_zone, low, high) in enumerate(zone_bounds):
            if low <= heart_rate <= high:
                seconds[index] += interval.active_s
                break
    zone_values = tuple(f"zone_{index}" for index in range(1, 6))
    return (
        table_metric(
            f"{prefix}.seconds",
            period,
            Axis(name="seconds", values=zone_values, unit="s"),
            tuple(seconds),
            description="zone distribution seconds values grouped by seconds.",
            unit="s",
            kind=MetricKind.CALCULATED,
        ),
        table_metric(
            f"{prefix}.percent",
            period,
            Axis(name="percent", values=zone_values, unit="%"),
            tuple(value / valid_time_s * 100.0 if valid_time_s > 0.0 else 0.0 for value in seconds),
            description="zone distribution percent values grouped by percent.",
            unit="%",
            kind=MetricKind.CALCULATED,
        ),
        scalar_metric(
            f"{prefix}.valid_heart_rate_time_s",
            period,
            valid_time_s,
            description="valid heart rate time for zone distribution during the activity.",
            unit="s",
            kind=MetricKind.CALCULATED,
        ),
        scalar_metric(
            f"{prefix}.unclassified_time_s",
            period,
            max(0.0, valid_time_s - sum(seconds)),
            description="unclassified time for zone distribution during the activity.",
            unit="s",
            kind=MetricKind.CALCULATED,
        ),
        scalar_metric(
            f"{prefix}.percent_denominator",
            period,
            "valid_heart_rate_time",
            description="percent denominator for zone distribution during the activity.",
            unit="%",
            kind=MetricKind.INTERPRETED,
        ),
    )


def time_window_metrics(activity: ParsedActivity, quality: QualityConfig) -> tuple[Metric, ...]:
    """Five-minute window scalars, per-window tables and the movement variant."""
    period = _period(activity)
    prefix = f"{_STATISTICS_PREFIX}.time_windows"
    metrics: list[Metric] = [
        scalar_metric(
            f"{prefix}.window_s",
            period,
            WINDOW_S,
            description="window for time windows during the activity.",
            unit="s",
            kind=MetricKind.CALCULATED,
            time_basis=TimeBasis.ELAPSED_TIME,
        )
    ]
    if is_short_activity(activity):
        return tuple(metrics)
    windows, movement = _available_windows(activity, quality)
    metrics.extend(_window_tables(f"{prefix}.items", "time windows", windows, period))
    if movement:
        metrics.extend(
            _window_tables(
                f"{prefix}.movement_time_variant",
                "time windows movement time variant",
                movement,
                period,
            )
        )
    else:
        metrics.append(
            scalar_metric(
                f"{prefix}.movement_time_variant",
                period,
                None,
                description="movement time variant for time windows during the activity.",
                unit="s",
                kind=MetricKind.CALCULATED,
                missing_reason=MissingReason.NO_DATA,
                time_basis=TimeBasis.ELAPSED_TIME,
            )
        )
    return tuple(metrics)


def efficiency_metrics(
    activity: ParsedActivity, athlete: AthleteConfig, quality: QualityConfig
) -> tuple[Metric, ...]:
    """Speed-per-heart-rate efficiency and aerobic decoupling."""
    if is_short_activity(activity):
        return ()
    period = _period(activity)
    facts = _efficiency_facts(activity, athlete, quality)
    return tuple(
        _efficiency_metrics(f"{_STATISTICS_PREFIX}.efficiency", "efficiency", facts, period)
    )


def fatigue_response_metrics(
    activity: ParsedActivity, athlete: AthleteConfig, quality: QualityConfig
) -> tuple[Metric, ...]:
    """Heart-rate drift plus the efficiency values mirrored under fatigue response."""
    period = _period(activity)
    prefix = f"{_STATISTICS_PREFIX}.fatigue_response"
    if is_short_activity(activity):
        return (
            scalar_metric(
                f"{prefix}.heart_rate_drift_bpm_per_hour",
                period,
                None,
                description=DRIFT_DESCRIPTION,
                unit="bpm",
                kind=MetricKind.CALCULATED,
                missing_reason=MissingReason.INSUFFICIENT_DATA,
            ),
        )
    facts = _efficiency_facts(activity, athlete, quality)
    drift = scalar_metric(
        f"{prefix}.heart_rate_drift_bpm_per_hour",
        period,
        _heart_rate_drift(activity, quality),
        description=DRIFT_DESCRIPTION,
        unit="bpm",
        kind=MetricKind.CALCULATED,
        missing_reason=facts.missing_reason or MissingReason.NO_DATA,
    )
    return (
        drift,
        *_efficiency_metrics(prefix, "fatigue response", facts, period),
    )


def relationship_metrics(activity: ParsedActivity, quality: QualityConfig) -> tuple[Metric, ...]:
    """Spearman and Pearson correlation for each available window relationship."""
    if is_short_activity(activity):
        return ()
    period = _period(activity)
    prefix = f"{_STATISTICS_PREFIX}.relationships"
    windows = _available_windows(activity, quality)[0]
    pairs: dict[str, list[tuple[float, float]]] = {
        "speed_heart_rate": [],
        "speed_cadence": [],
        "heart_rate_cadence": [],
    }
    for window in windows:
        speed = window.average_speed_kmh
        heart_rate = window.average_heart_rate_bpm
        cadence = window.average_pedaling_cadence_rpm
        if speed is not None and heart_rate is not None:
            pairs["speed_heart_rate"].append((speed, heart_rate))
        if speed is not None and cadence is not None:
            pairs["speed_cadence"].append((speed, cadence))
        if heart_rate is not None and cadence is not None:
            pairs["heart_rate_cadence"].append((heart_rate, cadence))
    metrics: list[Metric] = []
    for name, values in pairs.items():
        if name != "speed_heart_rate" and not values:
            continue
        label, unit = _RELATIONSHIP_SPECS[name]
        spearman = _spearman(values) if len(values) >= 3 else None
        pearson = _pearson(values) if len(values) >= 3 else None
        missing_reason = (
            MissingReason.INSUFFICIENT_WINDOWS
            if len(values) < 3
            else MissingReason.INSUFFICIENT_DATA
            if spearman is None and pearson is None
            else MissingReason.NO_DATA
        )
        for key, value, description in (
            ("spearman", spearman, f"spearman for relationships {label} during the activity."),
            ("pearson", pearson, f"pearson for relationships {label} during the activity."),
            (
                "window_count",
                len(values),
                f"window count for relationships {label} during the activity.",
            ),
        ):
            metrics.append(
                scalar_metric(
                    f"{prefix}.{name}.{key}",
                    period,
                    value,
                    description=description,
                    unit=unit,
                    kind=MetricKind.CALCULATED,
                    missing_reason=missing_reason,
                )
            )
    return tuple(metrics)


def heatmap_metrics(activity: ParsedActivity, quality: QualityConfig) -> tuple[Metric, ...]:
    """Duration and observation matrices for each binned sensor pair."""
    if is_short_activity(activity):
        return ()
    period = _period(activity)
    prefix = f"{_STATISTICS_PREFIX}.heatmap_data"
    intervals = build_intervals(activity)
    threshold_mps = quality.moving_speed_threshold_mps
    cells: dict[str, dict[tuple[int, int], list[float]]] = {
        "heart_rate_speed": defaultdict(lambda: [0.0, 0.0]),
        "heart_rate_cadence": defaultdict(lambda: [0.0, 0.0]),
        "speed_cadence": defaultdict(lambda: [0.0, 0.0]),
    }
    for interval in intervals:
        heart_rate = interval.heart_rate
        speed = interval.speed_kmh
        cadence = interval.cadence_rpm
        moving = moving_s(interval, threshold_mps)
        if heart_rate is not None and speed is not None and moving > 0.0:
            key = (int(heart_rate // 10) * 10, int(speed // 5) * 5)
            cells["heart_rate_speed"][key][0] += moving
            cells["heart_rate_speed"][key][1] += 1.0
        if heart_rate is not None and cadence is not None and cadence > 0.0:
            key = (int(heart_rate // 10) * 10, int(cadence // 10) * 10)
            cells["heart_rate_cadence"][key][0] += interval.active_s
            cells["heart_rate_cadence"][key][1] += 1.0
        if speed is not None and cadence is not None and cadence > 0.0:
            key = (int(speed // 5) * 5, int(cadence // 10) * 10)
            cells["speed_cadence"][key][0] += moving
            cells["speed_cadence"][key][1] += 1.0
    moving = moving_time_s(activity, intervals, threshold_mps)
    available, _reason = _cadence_availability(
        intervals, _cadence_coverage(intervals, moving), quality
    )
    metrics: list[Metric] = []
    for name, x_width, y_width in (
        ("heart_rate_speed", 10, 5),
        ("heart_rate_cadence", 10, 10),
        ("speed_cadence", 5, 10),
    ):
        axes, label, unit = _HEATMAP_SPECS[name]
        values = cells[name] if available or name == "heart_rate_speed" else {}
        valid_observation_count = int(sum(value[1] for value in values.values()))
        base = f"{prefix}.{name}"
        bins = _bins(values, x_width, y_width)
        if bins:
            metrics.append(
                _heatmap_matrix(
                    f"{base}.duration_s",
                    period,
                    bins,
                    f"Duration accumulated in each binned {axes} combination.",
                    "s",
                    lambda bin_value: bin_value.duration_s,
                )
            )
            metrics.append(
                _heatmap_matrix(
                    f"{base}.observation_count",
                    period,
                    bins,
                    f"Count of valid sensor intervals in each binned {axes} combination.",
                    "observations",
                    lambda bin_value: bin_value.observation_count,
                )
            )
        metrics.append(
            scalar_metric(
                f"{base}.valid_observation_count",
                period,
                valid_observation_count,
                description=(
                    f"valid observation count for heatmap data {label} during the activity."
                ),
                unit=unit,
                kind=MetricKind.CALCULATED,
            )
        )
    return tuple(metrics)


def sensor_quality_metrics(activity: ParsedActivity, quality: QualityConfig) -> tuple[Metric, ...]:
    """Record-pass counts, outliers and sensor coverage ratios."""
    period = _period(activity)
    prefix = f"{_STATISTICS_PREFIX}.data_quality"
    short = is_short_activity(activity)
    intervals = build_intervals(activity)
    threshold_mps = quality.moving_speed_threshold_mps
    valid_speed = 0
    during_movement = 0
    valid_heart_rate = 0
    valid_cadence = 0
    speed_time_s = 0.0
    heart_rate_time_s = 0.0
    cadence_time_s = 0.0
    for interval in intervals:
        weight = moving_s(interval, threshold_mps)
        if interval.speed_kmh is not None:
            valid_speed += 1
            if weight > 0.0:
                during_movement += 1
                speed_time_s += weight
        if interval.heart_rate is not None:
            valid_heart_rate += 1
            heart_rate_time_s += interval.active_s
        if interval.cadence_rpm is not None:
            valid_cadence += 1
            cadence_time_s += interval.active_s
    if short:
        duplicates, gaps = 0, 0
    else:
        duplicates, gaps = timestamp_issues(activity)
    outliers = sum(interval.rejected_values for interval in intervals)
    counts: tuple[tuple[str, str, str | None, int | float], ...] = (
        (
            "raw_record_points",
            "raw record points for data quality during the activity.",
            None,
            activity.records.height,
        ),
        (
            "valid_speed_intervals",
            "valid speed intervals for data quality during the activity.",
            "km/h",
            during_movement,
        ),
        (
            "valid_speed_samples",
            "valid speed samples for data quality during the activity.",
            "km/h",
            valid_speed,
        ),
        (
            "speed_samples_during_movement",
            "speed samples during movement for data quality during the activity.",
            "km/h",
            during_movement,
        ),
        (
            "valid_heart_rate_intervals",
            "valid heart rate intervals for data quality during the activity.",
            "bpm",
            valid_heart_rate,
        ),
        (
            "valid_cadence_intervals",
            "valid cadence intervals for data quality during the activity.",
            "rpm",
            valid_cadence,
        ),
        (
            "invalid_speed_intervals",
            "invalid speed intervals for data quality during the activity.",
            "km/h",
            len(intervals) - valid_speed,
        ),
        (
            "invalid_heart_rate_intervals",
            "invalid heart rate intervals for data quality during the activity.",
            "bpm",
            len(intervals) - valid_heart_rate,
        ),
        (
            "invalid_cadence_intervals",
            "invalid cadence intervals for data quality during the activity.",
            "rpm",
            len(intervals) - valid_cadence,
        ),
        (
            "duplicate_timestamps",
            "duplicate timestamps for data quality during the activity.",
            "s",
            duplicates,
        ),
        (
            "gap_intervals_over_30s",
            "gap intervals over 30s for data quality during the activity.",
            "count",
            gaps,
        ),
        (
            "rejected_outlier_values",
            "rejected outlier values for data quality during the activity.",
            None,
            outliers,
        ),
        (
            "pause_intervals",
            "pause intervals for data quality during the activity.",
            "count",
            len(activity.pause_intervals) if short else len(merged_pauses(activity)),
        ),
    )
    metrics = [
        scalar_metric(
            f"{prefix}.{key}",
            period,
            value,
            description=description,
            unit=unit,
            kind=MetricKind.CALCULATED,
        )
        for key, description, unit, value in counts
    ]
    if short:
        return tuple(metrics)
    moving = moving_time_s(activity, intervals, threshold_mps)
    active = active_time_s(activity)
    for key, unit, weight, denominator in (
        ("speed_coverage_ratio", "km/h", speed_time_s, moving),
        ("heart_rate_coverage_ratio", "bpm", heart_rate_time_s, active),
        ("cadence_coverage_ratio", "rpm", cadence_time_s, moving),
    ):
        metrics.append(
            scalar_metric(
                f"{prefix}.{key}",
                period,
                _capped_coverage(weight, denominator),
                description="Share of data quality time covered by valid measurements.",
                unit=unit,
                kind=MetricKind.CALCULATED,
            )
        )
    return tuple(metrics)


def _period(activity: ParsedActivity) -> TripPeriod:
    end = (
        activity.start_time + timedelta(seconds=activity.duration_s)
        if activity.duration_s is not None
        else None
    )
    return TripPeriod(activity_id=activity.activity_id, start=activity.start_time, end=end)


def _capped_coverage(weight_s: float, denominator_s: float) -> float:
    return min(weight_s / denominator_s, 1.0) if denominator_s > 0.0 else 0.0


_DISTRIBUTION_FIELDS: tuple[tuple[str, str, str | None], ...] = (
    ("mean", "Time-weighted mean of valid {context} samples during the activity.", None),
    ("median", "Duration-weighted median of valid {context} samples during the activity.", None),
    (
        "standard_deviation",
        "Duration-weighted variation in valid {context} samples during the activity.",
        None,
    ),
    ("p10", "Duration-weighted 10th percentile of valid {context} samples.", None),
    ("p25", "Duration-weighted 25th percentile of valid {context} samples.", None),
    ("p75", "Duration-weighted 75th percentile of valid {context} samples.", None),
    ("p90", "Duration-weighted 90th percentile of valid {context} samples.", None),
    ("minimum", "minimum for {context} during the activity.", None),
    ("maximum", "maximum for {context} during the activity.", None),
    ("valid_sample_count", "Number of valid {context} samples used in the distribution.", None),
    ("valid_time_s", "Duration of valid {context} measurements used in the distribution.", "s"),
    ("coverage_ratio", "Share of {context} time covered by valid measurements.", None),
)


def _distribution_metrics(
    prefix: str,
    samples: Sequence[tuple[float, float]],
    denominator_s: float,
    *,
    context: str,
    unit: str,
    time_basis: TimeBasis,
    empty_reason: MissingReason,
    period: TripPeriod,
) -> list[Metric]:
    total_weight = sum(weight for _value, weight in samples)
    if not samples or total_weight <= 0.0:
        values: tuple[MetricScalar | None, ...] = (None,) * 9 + (0, 0.0, 0.0)
        reason = empty_reason
    else:
        mean = sum(value * weight for value, weight in samples) / total_weight
        variance = sum(weight * (value - mean) ** 2 for value, weight in samples) / total_weight
        deviation = math.sqrt(max(0.0, variance))
        minimum = min(value for value, _weight in samples)
        maximum = max(value for value, _weight in samples)
        values = (
            mean,
            _weighted_quantile(samples, 0.5),
            deviation,
            _weighted_quantile(samples, 0.1),
            _weighted_quantile(samples, 0.25),
            _weighted_quantile(samples, 0.75),
            _weighted_quantile(samples, 0.9),
            minimum,
            maximum,
            len(samples),
            total_weight,
            _capped_coverage(total_weight, denominator_s),
        )
        reason = None
    missing_reason = reason or MissingReason.NO_DATA
    return [
        scalar_metric(
            f"{prefix}.{key}",
            period,
            value,
            description=description.format(context=context),
            unit=field_unit if field_unit is not None else unit,
            kind=MetricKind.CALCULATED,
            missing_reason=missing_reason,
            time_basis=time_basis,
        )
        for (key, description, field_unit), value in zip(_DISTRIBUTION_FIELDS, values, strict=True)
    ]


def _weighted_quantile(samples: Sequence[tuple[float, float]], quantile: float) -> float:
    ordered = sorted(samples)
    total_weight = sum(weight for _value, weight in ordered)
    target = total_weight * quantile
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= target:
            return value
    return ordered[-1][0]


def _hr_histogram(samples: Sequence[tuple[float, float]]) -> tuple[tuple[str, float], ...]:
    histogram: dict[str, float] = {}
    for value, duration in samples:
        low = int(value // 5) * 5
        histogram[f"{low}-{low + 4}"] = histogram.get(f"{low}-{low + 4}", 0.0) + duration
    return tuple(sorted(histogram.items()))


def _spearman(values: list[tuple[float, float]]) -> float | None:
    return _pearson(
        list(zip(_ranks([x for x, _y in values]), _ranks([y for _x, y in values]), strict=True))
    )


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _pearson(values: list[tuple[float, float]]) -> float | None:
    if len(values) < 2:
        return None
    mean_x = sum(x for x, _y in values) / len(values)
    mean_y = sum(y for _x, y in values) / len(values)
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in values)
    variance_x = sum((x - mean_x) ** 2 for x, _y in values)
    variance_y = sum((y - mean_y) ** 2 for _x, y in values)
    denominator = math.sqrt(variance_x * variance_y)
    return covariance / denominator if denominator > 0.0 else None


def _cadence_samples(
    intervals: Sequence[Interval], *, pedaling_only: bool
) -> list[tuple[float, float]]:
    return [
        (interval.cadence_rpm, interval.active_s)
        for interval in intervals
        if interval.cadence_rpm is not None and (not pedaling_only or interval.cadence_rpm > 0.0)
    ]


def _cadence_bands(pedaling_samples: Sequence[tuple[float, float]]) -> list[tuple[str, float]]:
    seconds = {band: 0.0 for band, _low, _high in CADENCE_BANDS}
    for value, weight in pedaling_samples:
        for band, low, high in CADENCE_BANDS:
            if value >= low and (high is None or value < high):
                seconds[band] += weight
                break
    return list(seconds.items())


def _cadence_coverage(intervals: Sequence[Interval], moving: float) -> float:
    valid_s = sum(interval.active_s for interval in intervals if interval.cadence_rpm is not None)
    return _capped_coverage(valid_s, moving)


def _cadence_availability(
    intervals: Sequence[Interval], coverage: float, quality: QualityConfig
) -> tuple[bool, MissingReason | None]:
    if not any(interval.cadence_rpm is not None for interval in intervals):
        return False, MissingReason.NO_CADENCE_DATA
    if coverage < quality.cadence_min_coverage:
        return False, MissingReason.INSUFFICIENT_COVERAGE
    return True, None


def _build_windows(
    intervals: Sequence[Interval],
    span_s: float,
    *,
    threshold_mps: float,
    moving_basis: bool,
) -> tuple[_Window, ...]:
    groups: dict[int, list[Interval]] = {}
    moving_elapsed_s = 0.0
    for interval in intervals:
        weight = moving_s(interval, threshold_mps)
        axis = moving_elapsed_s if moving_basis else interval.elapsed_s
        if not (moving_basis and weight <= 0.0):
            groups.setdefault(int(axis // WINDOW_S), []).append(interval)
        if moving_basis:
            moving_elapsed_s += weight
    windows: list[_Window] = []
    window_count = math.ceil(span_s / WINDOW_S) if span_s > 0.0 else 0
    for index in range(window_count):
        start_s = index * WINDOW_S
        end_s = min(span_s, start_s + WINDOW_S)
        group = groups.get(index, [])
        speed = [
            (interval.speed_kmh, moving_s(interval, threshold_mps))
            for interval in group
            if interval.speed_kmh is not None
        ]
        heart = [
            (interval.heart_rate, interval.active_s)
            for interval in group
            if interval.heart_rate is not None
        ]
        cadence = [
            (interval.cadence_rpm, interval.active_s)
            for interval in group
            if interval.cadence_rpm is not None and interval.cadence_rpm > 0.0
        ]
        moving = sum(moving_s(interval, threshold_mps) for interval in group)
        heart_rate_valid_s = sum(weight for _value, weight in heart)
        windows.append(
            _Window(
                start_s=start_s,
                end_s=end_s,
                duration_s=end_s - start_s,
                moving_time_s=moving,
                distance_m=sum(interval.distance_m for interval in group),
                moving_time_ratio=moving / (end_s - start_s) if end_s > start_s else 0.0,
                average_speed_kmh=_weighted_mean(speed),
                median_speed_kmh=_weighted_quantile(speed, 0.5) if speed else None,
                speed_p25_kmh=_weighted_quantile(speed, 0.25) if speed else None,
                speed_p75_kmh=_weighted_quantile(speed, 0.75) if speed else None,
                average_heart_rate_bpm=_weighted_mean(heart),
                median_heart_rate_bpm=_weighted_quantile(heart, 0.5) if heart else None,
                heart_rate_p25_bpm=_weighted_quantile(heart, 0.25) if heart else None,
                heart_rate_p75_bpm=_weighted_quantile(heart, 0.75) if heart else None,
                heart_rate_valid_time_s=heart_rate_valid_s,
                average_pedaling_cadence_rpm=_weighted_mean(cadence),
                median_pedaling_cadence_rpm=(_weighted_quantile(cadence, 0.5) if cadence else None),
                cadence_coverage_ratio=(
                    sum(weight for _value, weight in cadence) / moving if moving > 0.0 else 0.0
                ),
                cadence_reason=None,
            )
        )
    return tuple(windows)


def _weighted_mean(samples: Sequence[tuple[float, float]]) -> float | None:
    total = sum(weight for _value, weight in samples)
    return sum(value * weight for value, weight in samples) / total if total > 0.0 else None


def _available_windows(
    activity: ParsedActivity, quality: QualityConfig
) -> tuple[tuple[_Window, ...], tuple[_Window, ...]]:
    """Elapsed-time windows with the optional movement-time variant, cadence-masked."""
    intervals = build_intervals(activity)
    threshold_mps = quality.moving_speed_threshold_mps
    moving = moving_time_s(activity, intervals, threshold_mps)
    span = activity.duration_s or elapsed_time_s(activity)
    windows = _build_windows(intervals, span, threshold_mps=threshold_mps, moving_basis=False)
    movement = (
        _build_windows(intervals, moving, threshold_mps=threshold_mps, moving_basis=True)
        if abs(span - moving) > 1.0
        else ()
    )
    available, reason = _cadence_availability(
        intervals, _cadence_coverage(intervals, moving), quality
    )
    if not available:
        masked = (reason or MissingReason.INSUFFICIENT_COVERAGE).value
        windows = _mask_window_cadence(windows, masked)
        movement = _mask_window_cadence(movement, masked)
    return windows, movement


def _mask_window_cadence(windows: tuple[_Window, ...], reason: str) -> tuple[_Window, ...]:
    return tuple(
        _Window(
            start_s=window.start_s,
            end_s=window.end_s,
            duration_s=window.duration_s,
            moving_time_s=window.moving_time_s,
            distance_m=window.distance_m,
            moving_time_ratio=window.moving_time_ratio,
            average_speed_kmh=window.average_speed_kmh,
            median_speed_kmh=window.median_speed_kmh,
            speed_p25_kmh=window.speed_p25_kmh,
            speed_p75_kmh=window.speed_p75_kmh,
            average_heart_rate_bpm=window.average_heart_rate_bpm,
            median_heart_rate_bpm=window.median_heart_rate_bpm,
            heart_rate_p25_bpm=window.heart_rate_p25_bpm,
            heart_rate_p75_bpm=window.heart_rate_p75_bpm,
            heart_rate_valid_time_s=window.heart_rate_valid_time_s,
            average_pedaling_cadence_rpm=None,
            median_pedaling_cadence_rpm=None,
            cadence_coverage_ratio=window.cadence_coverage_ratio,
            cadence_reason=reason,
        )
        for window in windows
    )


def _window_tables(
    prefix: str,
    context: str,
    windows: tuple[_Window, ...],
    period: TripPeriod,
) -> list[Metric]:
    if not windows:
        return []
    axis = Axis(
        name="start_s",
        values=tuple(window.start_s for window in windows),
        unit="s",
    )
    masked = any(window.cadence_reason is not None for window in windows)
    metrics: list[Metric] = []
    for key, label, extract in _WINDOW_COLUMNS:
        if key == "cadence_reason" and not masked:
            continue
        metrics.append(
            table_metric(
                f"{prefix}.{key}",
                period,
                axis,
                tuple(extract(window) for window in windows),
                description=f"{context} {label} measured in consecutive elapsed-time windows.",
                unit="s",
                kind=MetricKind.CALCULATED,
                time_basis=TimeBasis.ELAPSED_TIME,
            )
        )
    return metrics


def _heart_rate_drift(activity: ParsedActivity, quality: QualityConfig) -> float | None:
    windows = _available_windows(activity, quality)[0]
    points: list[tuple[float, float]] = []
    for window in windows:
        heart_rate = window.average_heart_rate_bpm
        if heart_rate is not None:
            points.append((window.start_s + window.duration_s / 2.0, heart_rate))
    if len(points) < 2:
        return None
    mean_x = sum(x for x, _y in points) / len(points)
    mean_y = sum(y for _x, y in points) / len(points)
    variance = sum((x - mean_x) ** 2 for x, _y in points)
    if variance <= 0.0:
        return None
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in points)
    return covariance / variance * 3600.0


def _efficiency_facts(
    activity: ParsedActivity, athlete: AthleteConfig, quality: QualityConfig
) -> _EfficiencyFacts:
    intervals = build_intervals(activity)
    threshold_mps = quality.moving_speed_threshold_mps
    pairs: list[tuple[float, float, float]] = []
    for interval in intervals:
        speed = interval.speed_kmh
        heart_rate = interval.heart_rate
        weight = moving_s(interval, threshold_mps)
        if speed is None or heart_rate is None or weight <= 0.0:
            continue
        pairs.append((speed, heart_rate, weight))
    valid_s = sum(duration for _speed, _heart_rate, duration in pairs)
    moving = moving_time_s(activity, intervals, threshold_mps)
    coverage = valid_s / moving if moving > 0.0 else 0.0
    if valid_s < quality.decoupling_min_duration_s:
        return _insufficient_efficiency(valid_s, coverage, MissingReason.INSUFFICIENT_DURATION)
    if coverage < quality.decoupling_min_coverage:
        return _insufficient_efficiency(valid_s, coverage, MissingReason.INSUFFICIENT_COVERAGE)

    mean_speed = sum(speed * duration for speed, _heart_rate, duration in pairs) / valid_s
    mean_heart_rate = sum(heart_rate * duration for _speed, heart_rate, duration in pairs) / valid_s
    heart_rate_reserve = athlete.max_heart_rate - athlete.resting_heart_rate
    reserve_ratio = (
        (mean_heart_rate - athlete.resting_heart_rate) / heart_rate_reserve
        if heart_rate_reserve > 0
        else 0.0
    )
    (
        first_half_speed,
        first_half_heart_rate,
        first_half,
        second_half_speed,
        second_half_heart_rate,
        (second_half),
    ) = _half_efficiency(pairs, valid_s)
    decoupling = (
        (first_half - second_half) / first_half * 100.0
        if first_half is not None and second_half is not None and first_half > 0.0
        else None
    )
    return _EfficiencyFacts(
        speed_per_heart_rate=mean_speed / mean_heart_rate if mean_heart_rate > 0.0 else None,
        speed_per_heart_rate_reserve=(mean_speed / reserve_ratio if reserve_ratio > 0.0 else None),
        first_half_average_speed_kmh=first_half_speed,
        second_half_average_speed_kmh=second_half_speed,
        first_half_average_heart_rate_bpm=first_half_heart_rate,
        second_half_average_heart_rate_bpm=second_half_heart_rate,
        ef_first_half=first_half,
        ef_second_half=second_half,
        aerobic_decoupling_pct=decoupling,
        valid_time_s=valid_s,
        coverage_ratio=coverage,
        heart_rate_reserve_formula=HR_RESERVE_FORMULA,
        missing_reason=None,
    )


def _insufficient_efficiency(
    valid_s: float, coverage: float, reason: MissingReason
) -> _EfficiencyFacts:
    return _EfficiencyFacts(
        speed_per_heart_rate=None,
        speed_per_heart_rate_reserve=None,
        first_half_average_speed_kmh=None,
        second_half_average_speed_kmh=None,
        first_half_average_heart_rate_bpm=None,
        second_half_average_heart_rate_bpm=None,
        ef_first_half=None,
        ef_second_half=None,
        aerobic_decoupling_pct=None,
        valid_time_s=valid_s,
        coverage_ratio=coverage,
        heart_rate_reserve_formula=None,
        missing_reason=reason,
    )


def _half_efficiency(
    pairs: Sequence[tuple[float, float, float]], total_s: float
) -> tuple[
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
]:
    half = total_s / 2.0
    values = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    cursor = 0.0
    for speed, heart_rate, duration in pairs:
        end = cursor + duration
        first_s = max(0.0, min(end, half) - cursor)
        second_s = max(0.0, end - max(cursor, half))
        if first_s > 0.0:
            values[0][0] += speed * first_s
            values[0][1] += heart_rate * first_s
            values[0][2] += first_s
        if second_s > 0.0:
            values[1][0] += speed * second_s
            values[1][1] += heart_rate * second_s
            values[1][2] += second_s
        cursor = end
    summaries = [
        (speed / duration, heart_rate / duration, speed / heart_rate)
        if duration > 0.0 and heart_rate > 0.0
        else (None, None, None)
        for speed, heart_rate, duration in values
    ]
    return (*summaries[0], *summaries[1])


def _efficiency_metrics(
    prefix: str,
    context: str,
    facts: _EfficiencyFacts,
    period: TripPeriod,
) -> list[Metric]:
    missing_reason = facts.missing_reason or MissingReason.NO_DATA
    metrics: list[Metric] = []
    for key, description_template, unit, kind, extract in _EFFICIENCY_FIELDS:
        value = extract(facts)
        if key == "heart_rate_reserve_formula" and value is None:
            continue
        metrics.append(
            scalar_metric(
                f"{prefix}.{key}",
                period,
                value,
                description=description_template.format(context=context),
                unit=unit,
                kind=kind,
                missing_reason=missing_reason,
            )
        )
    return metrics


def _bins(
    values: dict[tuple[int, int], list[float]], x_width: int, y_width: int
) -> tuple[_Bin, ...]:
    return tuple(
        _Bin(
            x_lower=x,
            x_upper=x + x_width,
            y_lower=y,
            y_upper=y + y_width,
            duration_s=value[0],
            observation_count=int(value[1]),
        )
        for (x, y), value in sorted(values.items())
    )


def _heatmap_matrix(
    name: str,
    period: TripPeriod,
    bins: tuple[_Bin, ...],
    description: str,
    unit: str,
    measure: Callable[[_Bin], MetricScalar],
) -> Metric:
    x_bounds = sorted({(bin_value.x_lower, bin_value.x_upper) for bin_value in bins})
    y_bounds = sorted({(bin_value.y_lower, bin_value.y_upper) for bin_value in bins})
    x_bins = tuple(AxisBin(f"{low:g}\u2013{high:g}", low, high) for low, high in x_bounds)
    y_bins = tuple(AxisBin(f"{low:g}\u2013{high:g}", low, high) for low, high in y_bounds)
    x_index = {bounds: index for index, bounds in enumerate(x_bounds)}
    y_index = {bounds: index for index, bounds in enumerate(y_bounds)}
    cells: list[list[MetricScalar | None]] = [[0 for _x in x_bins] for _y in y_bins]
    for bin_value in bins:
        x = (bin_value.x_lower, bin_value.x_upper)
        y = (bin_value.y_lower, bin_value.y_upper)
        if x in x_index and y in y_index:
            cells[y_index[y]][x_index[x]] = measure(bin_value)
    return Metric(
        name=name,
        description=description,
        period=period,
        value=MatrixValue(
            x_axis=Axis(name="x", values=x_bins),
            y_axis=Axis(name="y", values=y_bins),
            values=tuple(tuple(row) for row in cells),
        ),
        kind=MetricKind.CALCULATED,
        unit=unit,
    )
