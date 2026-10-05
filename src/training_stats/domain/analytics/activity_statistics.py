"""Explicit Metric producers for coarse activity statistics.

Each public producer takes one ``ParsedActivity`` plus only the explicit
athlete or quality configuration it needs, derives its narrow facts from the
record helpers in ``statistics_records`` and constructs its metrics with
literal names, descriptions, units, kinds and missing reasons. Standalone
calls build their own interval pass; ``compute_activity_statistics`` builds
one per-activity ``_ActivityStatistics`` and shares its interval sequence and
derived facts across families. Standalone family calls create their own
statistics object. The public functions are grouped per output family;
``compute_activity_statistics`` is the thin composer over them.
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
POWER_HISTOGRAM_BIN_W = 25
POWER_HEATMAP_BIN_W = 50

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
    average_power_w: float | None
    power_coverage_ratio: float
    power_reason: str | None


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


@dataclass(slots=True)
class _ActivityStatistics:
    """Activity-local owner of shared interval facts for statistics families."""

    activity: ParsedActivity
    quality: QualityConfig
    athlete: AthleteConfig | None
    intervals: tuple[Interval, ...] | None
    _short: bool | None = None
    _moving_time: float | None = None
    _active_time: float | None = None
    _cadence_coverage: float | None = None
    _power_coverage: float | None = None
    _windows: tuple[tuple[_Window, ...], tuple[_Window, ...]] | None = None
    _efficiency: _EfficiencyFacts | None = None

    @classmethod
    def build(
        cls,
        activity: ParsedActivity,
        quality: QualityConfig,
        athlete: AthleteConfig | None = None,
        intervals: Sequence[Interval] | None = None,
    ) -> _ActivityStatistics:
        shared_intervals = tuple(intervals) if intervals is not None else None
        return cls(activity, quality, athlete, shared_intervals)

    def interval_sequence(self) -> tuple[Interval, ...]:
        if self.intervals is None:
            self.intervals = tuple(
                build_intervals(self.activity, self.quality.max_moving_interval_s)
            )
        return self.intervals

    def is_short(self) -> bool:
        if self._short is None:
            self._short = is_short_activity(self.activity)
        return self._short

    def moving_time_s(self) -> float:
        if self._moving_time is None:
            self._moving_time = moving_time_s(
                self.activity, self.interval_sequence(), self.quality.moving_speed_threshold_mps
            )
        return self._moving_time

    def active_time_s(self) -> float:
        if self._active_time is None:
            self._active_time = active_time_s(self.activity, short_activity=self.is_short())
        return self._active_time

    def cadence_coverage(self) -> float:
        if self._cadence_coverage is None:
            self._cadence_coverage = _cadence_coverage(
                self.interval_sequence(), self.moving_time_s()
            )
        return self._cadence_coverage

    def power_coverage(self) -> float:
        if self._power_coverage is None:
            power_time = sum(
                interval.active_s
                for interval in self.interval_sequence()
                if interval.power_w is not None
            )
            self._power_coverage = _capped_coverage(power_time, self.active_time_s())
        return self._power_coverage

    def power_availability(self) -> tuple[bool, MissingReason | None]:
        return _power_availability(self.interval_sequence(), self.power_coverage(), self.quality)

    def available_windows(self) -> tuple[tuple[_Window, ...], tuple[_Window, ...]]:
        if self._windows is None:
            self._windows = _build_available_windows(self)
        return self._windows

    def efficiency_facts(self) -> _EfficiencyFacts:
        if self._efficiency is None:
            if self.athlete is None:
                msg = "athlete configuration is required for efficiency statistics"
                raise ValueError(msg)
            self._efficiency = _build_efficiency_facts(self)
        return self._efficiency


@dataclass(frozen=True, slots=True)
class _Bin:
    """One observed two-sensor bin with its duration and sample count."""

    x_lower: float
    x_upper: float
    y_lower: float
    y_upper: float
    duration_s: float
    observation_count: int


type _WindowColumn = tuple[str, str, str | None, Callable[[_Window], MetricScalar | None]]
type _EfficiencyField = tuple[
    str, str, str | None, MetricKind, Callable[[_EfficiencyFacts], MetricScalar | None]
]

_WINDOW_COLUMNS: tuple[_WindowColumn, ...] = (
    (
        "average_heart_rate_bpm",
        "average heart rate bpm",
        "bpm",
        lambda w: w.average_heart_rate_bpm,
    ),
    (
        "average_pedaling_cadence_rpm",
        "average pedaling cadence rpm",
        "rpm",
        lambda w: w.average_pedaling_cadence_rpm,
    ),
    ("average_power_w", "average recorded power", "W", lambda w: w.average_power_w),
    ("average_speed_kmh", "average speed kmh", "km/h", lambda w: w.average_speed_kmh),
    (
        "cadence_coverage_ratio",
        "cadence coverage ratio",
        "ratio",
        lambda w: w.cadence_coverage_ratio,
    ),
    ("cadence_reason", "cadence reason", None, lambda w: w.cadence_reason),
    (
        "power_coverage_ratio",
        "recorded power coverage ratio",
        "ratio",
        lambda w: w.power_coverage_ratio,
    ),
    ("power_reason", "recorded power reason", None, lambda w: w.power_reason),
    ("distance_m", "distance m", "m", lambda w: w.distance_m),
    ("duration_s", "duration", "s", lambda w: w.duration_s),
    ("end_s", "end", "s", lambda w: w.end_s),
    ("heart_rate_p25_bpm", "heart rate p25 bpm", "bpm", lambda w: w.heart_rate_p25_bpm),
    ("heart_rate_p75_bpm", "heart rate p75 bpm", "bpm", lambda w: w.heart_rate_p75_bpm),
    (
        "heart_rate_valid_time_s",
        "heart rate valid time",
        "s",
        lambda w: w.heart_rate_valid_time_s,
    ),
    (
        "heart_rate_zone_percent_denominator",
        "heart rate zone percent denominator",
        None,
        lambda w: "valid_heart_rate_time",
    ),
    (
        "median_heart_rate_bpm",
        "median heart rate bpm",
        "bpm",
        lambda w: w.median_heart_rate_bpm,
    ),
    (
        "median_pedaling_cadence_rpm",
        "median pedaling cadence rpm",
        "rpm",
        lambda w: w.median_pedaling_cadence_rpm,
    ),
    ("median_speed_kmh", "median speed kmh", "km/h", lambda w: w.median_speed_kmh),
    ("moving_time_ratio", "moving time ratio", "ratio", lambda w: w.moving_time_ratio),
    ("moving_time_s", "moving time", "s", lambda w: w.moving_time_s),
    ("speed_p25_kmh", "speed p25 kmh", "km/h", lambda w: w.speed_p25_kmh),
    ("speed_p75_kmh", "speed p75 kmh", "km/h", lambda w: w.speed_p75_kmh),
)

_EFFICIENCY_FIELDS: tuple[_EfficiencyField, ...] = (
    (
        "speed_per_heart_rate",
        "speed per heart rate for {context} during the activity.",
        "km/h/bpm",
        MetricKind.CALCULATED,
        lambda e: e.speed_per_heart_rate,
    ),
    (
        "speed_per_heart_rate_reserve",
        "speed per heart rate reserve for {context} during the activity.",
        "km/h",
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
        "Speed-to-heart-rate efficiency in the first half of valid moving time.",
        "km/h/bpm",
        MetricKind.CALCULATED,
        lambda e: e.ef_first_half,
    ),
    (
        "ef_second_half",
        "Speed-to-heart-rate efficiency in the second half of valid moving time.",
        "km/h/bpm",
        MetricKind.CALCULATED,
        lambda e: e.ef_second_half,
    ),
    (
        "aerobic_decoupling_pct",
        "Percent change in speed-to-heart-rate efficiency between equal moving-time halves.",
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
        "Fraction of moving time with valid simultaneous speed and heart-rate measurements.",
        "ratio",
        MetricKind.CALCULATED,
        lambda e: e.coverage_ratio,
    ),
    (
        "heart_rate_reserve_formula",
        "heart rate reserve formula for {context} during the activity.",
        None,
        MetricKind.INTERPRETED,
        lambda e: e.heart_rate_reserve_formula,
    ),
)

_RELATIONSHIP_SPECS: dict[str, str] = {
    "speed_heart_rate": "speed heart rate",
    "speed_cadence": "speed cadence",
    "heart_rate_cadence": "heart rate cadence",
    "power_heart_rate": "recorded power heart rate",
    "power_speed": "recorded power speed",
    "power_cadence": "recorded power cadence",
}

_HEATMAP_SPECS: dict[str, tuple[str, str]] = {
    "heart_rate_speed": ("heart rate and speed", "heart rate speed"),
    "heart_rate_cadence": ("heart rate and cadence", "heart rate cadence"),
    "speed_cadence": ("speed and cadence", "speed cadence"),
    "power_heart_rate": ("recorded power and heart rate", "recorded power heart rate"),
    "power_cadence": ("recorded power and cadence", "recorded power cadence"),
    "power_speed": ("recorded power and speed", "recorded power speed"),
}


def compute_activity_statistics(
    activity: ParsedActivity,
    athlete: AthleteConfig,
    quality: QualityConfig,
) -> tuple[Metric, ...]:
    """Return detailed distribution, window, relationship, and heatmap metrics."""
    statistics = _ActivityStatistics.build(activity, quality, athlete)
    return (
        *_speed_metrics(statistics),
        *_heart_rate_metrics(statistics),
        *_cadence_metrics(statistics),
        *_power_metrics(statistics),
        *_zone_metrics(statistics),
        *_time_window_metrics(statistics),
        *_efficiency_family_metrics(statistics),
        *_fatigue_response_metrics(statistics),
        *_relationship_metrics(statistics),
        *_heatmap_metrics(statistics),
        *_sensor_quality_metrics(statistics),
    )


def speed_metrics(
    activity: ParsedActivity,
    quality: QualityConfig,
    *,
    intervals: Sequence[Interval] | None = None,
) -> tuple[Metric, ...]:
    """Build speed metrics from one activity and its quality settings."""
    return _speed_metrics(_ActivityStatistics.build(activity, quality, intervals=intervals))


def _speed_metrics(statistics: _ActivityStatistics) -> tuple[Metric, ...]:
    """Speed distribution, time above configured thresholds and coverage ratios."""
    activity = statistics.activity
    quality = statistics.quality
    period = _period(activity)
    prefix = f"{_STATISTICS_PREFIX}.speed_statistics"
    short = statistics.is_short()
    intervals = statistics.interval_sequence()
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
    moving = statistics.moving_time_s()
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
    if quality.speed_thresholds_kmh:
        threshold_seconds = {
            float(f"{threshold:g}"): 0.0 for threshold in quality.speed_thresholds_kmh
        }
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
            description="Fraction of moving time represented by valid speed measurements.",
            unit="ratio",
            kind=MetricKind.CALCULATED,
            time_basis=TimeBasis.MOVING_TIME,
        )
    )
    metrics.append(
        scalar_metric(
            f"{prefix}.valid_measurement_ratio",
            period,
            valid_samples / len(intervals) if intervals else 0.0,
            description="Fraction of record intervals containing a valid speed measurement.",
            unit="ratio",
            kind=MetricKind.CALCULATED,
        )
    )
    metrics.append(
        scalar_metric(
            f"{prefix}.measurements_during_movement_ratio",
            period,
            during_movement / valid_samples if valid_samples > 0 else 0.0,
            description="Fraction of valid speed samples recorded during movement.",
            unit="ratio",
            kind=MetricKind.CALCULATED,
        )
    )
    return tuple(metrics)


def heart_rate_metrics(
    activity: ParsedActivity, *, intervals: Sequence[Interval] | None = None
) -> tuple[Metric, ...]:
    """Build heart-rate metrics for one activity using default quality settings."""
    return _heart_rate_metrics(
        _ActivityStatistics.build(activity, QualityConfig(), intervals=intervals)
    )


def _heart_rate_metrics(statistics: _ActivityStatistics) -> tuple[Metric, ...]:
    """Heart-rate distribution and its five-beat-per-minute histogram."""
    activity = statistics.activity
    period = _period(activity)
    prefix = f"{_STATISTICS_PREFIX}.heart_rate_statistics"
    intervals = statistics.interval_sequence()
    samples = [
        (interval.heart_rate, interval.active_s)
        for interval in intervals
        if interval.heart_rate is not None
    ]
    metrics = _distribution_metrics(
        prefix,
        samples,
        active_time_s(activity, short_activity=statistics.is_short()),
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


def cadence_metrics(
    activity: ParsedActivity,
    quality: QualityConfig,
    *,
    intervals: Sequence[Interval] | None = None,
) -> tuple[Metric, ...]:
    """Build cadence metrics for one activity and its quality settings."""
    return _cadence_metrics(_ActivityStatistics.build(activity, quality, intervals=intervals))


def power_metrics(
    activity: ParsedActivity,
    quality: QualityConfig,
    *,
    intervals: Sequence[Interval] | None = None,
) -> tuple[Metric, ...]:
    """Build recorded-power metrics from one activity and its quality settings."""
    return _power_metrics(_ActivityStatistics.build(activity, quality, intervals=intervals))


def _cadence_metrics(statistics: _ActivityStatistics) -> tuple[Metric, ...]:
    """Cadence availability, distributions, pedaling time and cadence bands."""
    activity = statistics.activity
    quality = statistics.quality
    period = _period(activity)
    prefix = f"{_STATISTICS_PREFIX}.cadence_statistics"
    short = statistics.is_short()
    intervals = statistics.interval_sequence()
    moving = statistics.moving_time_s()
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
        coverage = statistics.cadence_coverage()
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
            coverage_time_basis=TimeBasis.MOVING_TIME,
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
            description="Fraction of moving time classified as pedaling from valid cadence data.",
            unit="ratio",
            kind=MetricKind.CALCULATED,
            missing_reason=missing_reason,
        ),
        scalar_metric(
            f"{prefix}.coverage_ratio",
            period,
            coverage,
            description="Fraction of moving time covered by valid cadence measurements.",
            unit="ratio",
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


def _power_metrics(statistics: _ActivityStatistics) -> tuple[Metric, ...]:
    """Recorded-power distribution, coverage, and time-weighted histogram."""
    activity = statistics.activity
    period = _period(activity)
    prefix = f"{_STATISTICS_PREFIX}.power_statistics"
    short = statistics.is_short()
    intervals = statistics.interval_sequence()
    reason: MissingReason | None
    if short:
        available = False
        reason = MissingReason.NO_POWER_DATA
    else:
        available, reason = statistics.power_availability()
    samples = (
        [
            (interval.power_w, interval.active_s)
            for interval in intervals
            if interval.power_w is not None
        ]
        if available
        else []
    )
    overall_reason = reason or MissingReason.INSUFFICIENT_DATA
    metrics = [
        scalar_metric(
            f"{prefix}.available",
            period,
            available,
            description=(
                "Whether recorded power data meets the configured availability requirements."
            ),
            kind=MetricKind.INTERPRETED,
        ),
        *_distribution_metrics(
            f"{prefix}.overall",
            samples,
            statistics.active_time_s(),
            context="recorded power",
            unit="W",
            time_basis=TimeBasis.ACTIVE_TIME,
            empty_reason=overall_reason,
            period=period,
        ),
        scalar_metric(
            f"{prefix}.coverage_ratio",
            period,
            statistics.power_coverage(),
            description="Fraction of active time covered by valid recorded-power measurements.",
            unit="ratio",
            kind=MetricKind.CALCULATED,
            time_basis=TimeBasis.ACTIVE_TIME,
        ),
    ]
    histogram = _power_histogram(samples)
    labels = tuple(label for label, _seconds in histogram)
    metrics.append(
        table_metric(
            f"{prefix}.histogram_s.seconds",
            period,
            Axis(name="power_bin_w", values=labels, unit="W"),
            tuple(seconds for _label, seconds in histogram),
            description="Active time accumulated in 25-watt recorded-power bins.",
            unit="s",
            missing_reason=overall_reason if not histogram else None,
            time_basis=TimeBasis.ACTIVE_TIME,
        )
    )
    return tuple(metrics)


def zone_metrics(
    activity: ParsedActivity,
    athlete: AthleteConfig,
    *,
    intervals: Sequence[Interval] | None = None,
) -> tuple[Metric, ...]:
    """Build configured heart-rate-zone metrics for one activity."""
    return _zone_metrics(
        _ActivityStatistics.build(activity, QualityConfig(), athlete, intervals=intervals)
    )


def _zone_metrics(statistics: _ActivityStatistics) -> tuple[Metric, ...]:
    """Activity-level heart-rate zone time, percent and denominator."""
    activity = statistics.activity
    athlete = statistics.athlete
    if athlete is None:
        msg = "athlete configuration is required for heart-rate-zone statistics"
        raise ValueError(msg)
    period = _period(activity)
    prefix = f"{_STATISTICS_PREFIX}.zone_distribution"
    intervals = statistics.interval_sequence()
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
            description="Time assigned to each configured heart-rate zone.",
            unit="s",
            kind=MetricKind.CALCULATED,
            time_basis=TimeBasis.ACTIVE_TIME,
        ),
        table_metric(
            f"{prefix}.percent",
            period,
            Axis(name="percent", values=zone_values, unit="%"),
            tuple(value / valid_time_s * 100.0 if valid_time_s > 0.0 else 0.0 for value in seconds),
            description="Percent of valid heart-rate time assigned to each configured zone.",
            unit="%",
            kind=MetricKind.CALCULATED,
            time_basis=TimeBasis.VALID_HEART_RATE_TIME,
        ),
        scalar_metric(
            f"{prefix}.valid_heart_rate_time_s",
            period,
            valid_time_s,
            description="valid heart rate time for zone distribution during the activity.",
            unit="s",
            kind=MetricKind.CALCULATED,
            time_basis=TimeBasis.VALID_HEART_RATE_TIME,
        ),
        scalar_metric(
            f"{prefix}.unclassified_time_s",
            period,
            max(0.0, valid_time_s - sum(seconds)),
            description="Valid heart-rate time outside the configured heart-rate zones.",
            unit="s",
            kind=MetricKind.CALCULATED,
            time_basis=TimeBasis.VALID_HEART_RATE_TIME,
        ),
        scalar_metric(
            f"{prefix}.percent_denominator",
            period,
            "valid_heart_rate_time",
            description="Time basis used as the denominator for heart-rate zone percentages.",
            kind=MetricKind.INTERPRETED,
        ),
    )


def time_window_metrics(
    activity: ParsedActivity,
    quality: QualityConfig,
    *,
    intervals: Sequence[Interval] | None = None,
) -> tuple[Metric, ...]:
    """Build time-window metrics for one activity and its quality settings."""
    return _time_window_metrics(_ActivityStatistics.build(activity, quality, intervals=intervals))


def _time_window_metrics(statistics: _ActivityStatistics) -> tuple[Metric, ...]:
    """Five-minute window scalars, per-window tables and the movement variant."""
    activity = statistics.activity
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
    if statistics.is_short():
        return tuple(metrics)
    windows, movement = statistics.available_windows()
    metrics.extend(_window_tables(f"{prefix}.items", "time windows", windows, period))
    if movement:
        metrics.extend(
            _window_tables(
                f"{prefix}.movement_time_variant",
                "time windows movement time variant",
                movement,
                period,
                time_basis=TimeBasis.MOVING_TIME,
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
                time_basis=TimeBasis.MOVING_TIME,
            )
        )
    return tuple(metrics)


def efficiency_metrics(
    activity: ParsedActivity,
    athlete: AthleteConfig,
    quality: QualityConfig,
    *,
    intervals: Sequence[Interval] | None = None,
) -> tuple[Metric, ...]:
    """Build efficiency metrics for one activity and its explicit settings."""
    return _efficiency_family_metrics(
        _ActivityStatistics.build(activity, quality, athlete, intervals=intervals)
    )


def _efficiency_family_metrics(statistics: _ActivityStatistics) -> tuple[Metric, ...]:
    """Speed-per-heart-rate efficiency and aerobic decoupling."""
    activity = statistics.activity
    if statistics.is_short():
        return ()
    period = _period(activity)
    facts = statistics.efficiency_facts()
    return tuple(
        _efficiency_metrics(f"{_STATISTICS_PREFIX}.efficiency", "efficiency", facts, period)
    )


def fatigue_response_metrics(
    activity: ParsedActivity,
    athlete: AthleteConfig,
    quality: QualityConfig,
    *,
    intervals: Sequence[Interval] | None = None,
) -> tuple[Metric, ...]:
    """Build fatigue-response metrics for one activity and its settings."""
    return _fatigue_response_metrics(
        _ActivityStatistics.build(activity, quality, athlete, intervals=intervals)
    )


def _fatigue_response_metrics(statistics: _ActivityStatistics) -> tuple[Metric, ...]:
    """Heart-rate drift over the activity's elapsed-time windows."""
    activity = statistics.activity
    period = _period(activity)
    prefix = f"{_STATISTICS_PREFIX}.fatigue_response"
    if statistics.is_short():
        return (
            scalar_metric(
                f"{prefix}.heart_rate_drift_bpm_per_hour",
                period,
                None,
                description=DRIFT_DESCRIPTION,
                unit="bpm/h",
                kind=MetricKind.CALCULATED,
                missing_reason=MissingReason.INSUFFICIENT_DATA,
            ),
        )
    drift = scalar_metric(
        f"{prefix}.heart_rate_drift_bpm_per_hour",
        period,
        _heart_rate_drift(statistics),
        description=DRIFT_DESCRIPTION,
        unit="bpm/h",
        kind=MetricKind.CALCULATED,
        missing_reason=MissingReason.INSUFFICIENT_DATA,
    )
    return (drift,)


def relationship_metrics(
    activity: ParsedActivity,
    quality: QualityConfig,
    *,
    intervals: Sequence[Interval] | None = None,
) -> tuple[Metric, ...]:
    """Build window relationships for one activity and its quality settings."""
    return _relationship_metrics(_ActivityStatistics.build(activity, quality, intervals=intervals))


def _relationship_metrics(statistics: _ActivityStatistics) -> tuple[Metric, ...]:
    """Spearman and Pearson correlation for each available window relationship."""
    activity = statistics.activity
    if statistics.is_short():
        return ()
    period = _period(activity)
    prefix = f"{_STATISTICS_PREFIX}.relationships"
    windows = statistics.available_windows()[0]
    pairs: dict[str, list[tuple[float, float]]] = {
        "speed_heart_rate": [],
        "speed_cadence": [],
        "heart_rate_cadence": [],
        "power_heart_rate": [],
        "power_speed": [],
        "power_cadence": [],
    }
    power_available, power_reason = statistics.power_availability()
    cadence_available, cadence_reason = _cadence_availability(
        statistics.interval_sequence(), statistics.cadence_coverage(), statistics.quality
    )
    for window in windows:
        speed = window.average_speed_kmh
        heart_rate = window.average_heart_rate_bpm
        cadence = window.average_pedaling_cadence_rpm
        power = window.average_power_w
        if speed is not None and heart_rate is not None:
            pairs["speed_heart_rate"].append((speed, heart_rate))
        if speed is not None and cadence is not None:
            pairs["speed_cadence"].append((speed, cadence))
        if heart_rate is not None and cadence is not None:
            pairs["heart_rate_cadence"].append((heart_rate, cadence))
        if power is not None and heart_rate is not None:
            pairs["power_heart_rate"].append((power, heart_rate))
        if power is not None and speed is not None:
            pairs["power_speed"].append((power, speed))
        if power is not None and cadence is not None:
            pairs["power_cadence"].append((power, cadence))
    metrics: list[Metric] = []
    for name, values in pairs.items():
        if (
            name not in {"speed_heart_rate", "power_heart_rate", "power_speed", "power_cadence"}
            and not values
        ):
            continue
        label = _RELATIONSHIP_SPECS[name]
        spearman = _spearman(values) if len(values) >= 3 else None
        pearson = _pearson(values) if len(values) >= 3 else None
        missing_reason = power_reason if name.startswith("power_") and not power_available else None
        if name == "power_cadence" and power_available and not cadence_available:
            missing_reason = cadence_reason or MissingReason.INSUFFICIENT_COVERAGE
        if missing_reason is None:
            missing_reason = (
                MissingReason.INSUFFICIENT_WINDOWS
                if len(values) < 3
                else MissingReason.INSUFFICIENT_DATA
                if spearman is None and pearson is None
                else MissingReason.NO_DATA
            )
        for key, value, description, unit in (
            (
                "spearman",
                spearman,
                f"Spearman rank correlation between {label} across valid time windows.",
                "correlation_coefficient",
            ),
            (
                "pearson",
                pearson,
                f"Pearson correlation between {label} across valid time windows.",
                "correlation_coefficient",
            ),
            (
                "window_count",
                len(values),
                f"Number of valid time windows used for the {label} relationship.",
                "count",
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


def heatmap_metrics(
    activity: ParsedActivity,
    quality: QualityConfig,
    *,
    intervals: Sequence[Interval] | None = None,
) -> tuple[Metric, ...]:
    """Build sensor heatmaps for one activity and its quality settings."""
    return _heatmap_metrics(_ActivityStatistics.build(activity, quality, intervals=intervals))


def _heatmap_metrics(statistics: _ActivityStatistics) -> tuple[Metric, ...]:
    """Duration and observation matrices for each binned sensor pair."""
    activity = statistics.activity
    quality = statistics.quality
    if statistics.is_short():
        return ()
    period = _period(activity)
    prefix = f"{_STATISTICS_PREFIX}.heatmap_data"
    intervals = statistics.interval_sequence()
    threshold_mps = quality.moving_speed_threshold_mps
    cells: dict[str, dict[tuple[int, int], list[float]]] = {
        "heart_rate_speed": defaultdict(lambda: [0.0, 0.0]),
        "heart_rate_cadence": defaultdict(lambda: [0.0, 0.0]),
        "speed_cadence": defaultdict(lambda: [0.0, 0.0]),
        "power_heart_rate": defaultdict(lambda: [0.0, 0.0]),
        "power_cadence": defaultdict(lambda: [0.0, 0.0]),
        "power_speed": defaultdict(lambda: [0.0, 0.0]),
    }
    for interval in intervals:
        heart_rate = interval.heart_rate
        speed = interval.speed_kmh
        cadence = interval.cadence_rpm
        power = interval.power_w
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
        if power is not None and heart_rate is not None:
            key = (
                int(power // POWER_HEATMAP_BIN_W) * POWER_HEATMAP_BIN_W,
                int(heart_rate // 10) * 10,
            )
            cells["power_heart_rate"][key][0] += interval.active_s
            cells["power_heart_rate"][key][1] += 1.0
        if power is not None and cadence is not None and cadence > 0.0:
            key = (int(power // POWER_HEATMAP_BIN_W) * POWER_HEATMAP_BIN_W, int(cadence // 10) * 10)
            cells["power_cadence"][key][0] += interval.active_s
            cells["power_cadence"][key][1] += 1.0
        if power is not None and speed is not None and moving > 0.0:
            key = (int(power // POWER_HEATMAP_BIN_W) * POWER_HEATMAP_BIN_W, int(speed // 5) * 5)
            cells["power_speed"][key][0] += moving
            cells["power_speed"][key][1] += 1.0
    available, _reason = _cadence_availability(intervals, statistics.cadence_coverage(), quality)
    power_available, _power_reason = statistics.power_availability()
    metrics: list[Metric] = []
    for name, x_width, y_width in (
        ("heart_rate_speed", 10, 5),
        ("heart_rate_cadence", 10, 10),
        ("speed_cadence", 5, 10),
        ("power_heart_rate", POWER_HEATMAP_BIN_W, 10),
        ("power_cadence", POWER_HEATMAP_BIN_W, 10),
        ("power_speed", POWER_HEATMAP_BIN_W, 5),
    ):
        axes, label = _HEATMAP_SPECS[name]
        if name.startswith("power_"):
            if name == "power_cadence":
                values = cells[name] if power_available and available else {}
            else:
                values = cells[name] if power_available else {}
        else:
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
                description=(f"Number of valid observations in the {label} heatmap."),
                unit="observations",
                kind=MetricKind.CALCULATED,
            )
        )
    return tuple(metrics)


def sensor_quality_metrics(
    activity: ParsedActivity,
    quality: QualityConfig,
    *,
    intervals: Sequence[Interval] | None = None,
) -> tuple[Metric, ...]:
    """Build sensor-quality metrics for one activity and its quality settings."""
    return _sensor_quality_metrics(
        _ActivityStatistics.build(activity, quality, intervals=intervals)
    )


def _sensor_quality_metrics(statistics: _ActivityStatistics) -> tuple[Metric, ...]:
    """Record-pass counts, outliers and sensor coverage ratios."""
    activity = statistics.activity
    quality = statistics.quality
    period = _period(activity)
    prefix = f"{_STATISTICS_PREFIX}.data_quality"
    short = statistics.is_short()
    intervals = statistics.interval_sequence()
    threshold_mps = quality.moving_speed_threshold_mps
    valid_speed = 0
    during_movement = 0
    valid_heart_rate = 0
    valid_cadence = 0
    valid_power = 0
    speed_time_s = 0.0
    heart_rate_time_s = 0.0
    cadence_time_s = 0.0
    power_time_s = 0.0
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
        if interval.power_w is not None:
            valid_power += 1
            power_time_s += interval.active_s
    if short:
        duplicates, gaps = 0, 0
    else:
        duplicates, gaps = timestamp_issues(activity, quality.max_moving_interval_s)
    outliers = sum(interval.rejected_values for interval in intervals)
    counts: tuple[tuple[str, str, str | None, int | float], ...] = (
        (
            "raw_record_points",
            "Number of raw record points in the activity.",
            "count",
            activity.records.height,
        ),
        (
            "valid_speed_intervals",
            "Number of valid speed intervals classified as movement.",
            "count",
            during_movement,
        ),
        (
            "valid_speed_samples",
            "Number of intervals containing a valid speed sample.",
            "count",
            valid_speed,
        ),
        (
            "speed_samples_during_movement",
            "Number of valid speed samples recorded during movement.",
            "count",
            during_movement,
        ),
        (
            "valid_heart_rate_intervals",
            "Number of intervals containing a valid heart-rate sample.",
            "count",
            valid_heart_rate,
        ),
        (
            "valid_cadence_intervals",
            "Number of intervals containing a valid cadence sample.",
            "count",
            valid_cadence,
        ),
        (
            "valid_power_intervals",
            "Number of intervals containing a valid recorded-power sample.",
            "count",
            valid_power,
        ),
        (
            "invalid_speed_intervals",
            "Number of intervals without a valid speed sample.",
            "count",
            len(intervals) - valid_speed,
        ),
        (
            "invalid_heart_rate_intervals",
            "Number of intervals without a valid heart-rate sample.",
            "count",
            len(intervals) - valid_heart_rate,
        ),
        (
            "invalid_cadence_intervals",
            "Number of intervals without a valid cadence sample.",
            "count",
            len(intervals) - valid_cadence,
        ),
        (
            "invalid_power_intervals",
            "Number of intervals without a valid recorded-power sample.",
            "count",
            len(intervals) - valid_power,
        ),
        (
            "duplicate_timestamps",
            "Number of duplicate record timestamps.",
            "count",
            duplicates,
        ),
        (
            "gap_intervals_over_30s",
            "Number of record gaps over 30 seconds.",
            "count",
            gaps,
        ),
        (
            "rejected_outlier_values",
            "Number of sensor values rejected as outliers.",
            "count",
            outliers,
        ),
        (
            "pause_intervals",
            "Number of pause intervals in the activity.",
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
    moving = statistics.moving_time_s()
    active = statistics.active_time_s()
    for key, weight, denominator, description, time_basis in (
        (
            "speed_coverage_ratio",
            speed_time_s,
            moving,
            "Fraction of moving time covered by valid speed measurements.",
            TimeBasis.MOVING_TIME,
        ),
        (
            "heart_rate_coverage_ratio",
            heart_rate_time_s,
            active,
            "Fraction of active time covered by valid heart-rate measurements.",
            TimeBasis.ACTIVE_TIME,
        ),
        (
            "cadence_coverage_ratio",
            cadence_time_s,
            moving,
            "Fraction of moving time covered by valid cadence measurements.",
            TimeBasis.MOVING_TIME,
        ),
        (
            "power_coverage_ratio",
            power_time_s,
            active,
            "Fraction of active time covered by valid recorded-power measurements.",
            TimeBasis.ACTIVE_TIME,
        ),
    ):
        metrics.append(
            scalar_metric(
                f"{prefix}.{key}",
                period,
                _capped_coverage(weight, denominator),
                description=description,
                unit="ratio",
                kind=MetricKind.CALCULATED,
                time_basis=time_basis,
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
    (
        "valid_sample_count",
        "Number of valid {context} samples used in the distribution.",
        "count",
    ),
    ("valid_time_s", "Duration of valid {context} measurements used in the distribution.", "s"),
    (
        "coverage_ratio",
        "Fraction of {context} time covered by valid measurements.",
        "ratio",
    ),
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
    coverage_time_basis: TimeBasis | None = None,
) -> list[Metric]:
    coverage_basis = coverage_time_basis or time_basis
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
            description=(
                f"Fraction of {coverage_basis.value.replace('_', ' ')} covered by valid "
                f"{context} measurements."
                if key == "coverage_ratio"
                else description.format(context=context)
            ),
            unit=field_unit if field_unit is not None else unit,
            kind=MetricKind.CALCULATED,
            missing_reason=missing_reason,
            time_basis=(
                None
                if key == "valid_sample_count"
                else coverage_basis
                if key == "coverage_ratio"
                else time_basis
            ),
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


def _power_histogram(samples: Sequence[tuple[float, float]]) -> tuple[tuple[str, float], ...]:
    histogram: dict[str, float] = {}
    for value, duration in samples:
        low = int(value // POWER_HISTOGRAM_BIN_W) * POWER_HISTOGRAM_BIN_W
        label = f"{low}-{low + POWER_HISTOGRAM_BIN_W - 1}"
        histogram[label] = histogram.get(label, 0.0) + duration
    return tuple(sorted(histogram.items(), key=lambda item: int(item[0].split("-", 1)[0])))


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


def _power_availability(
    intervals: Sequence[Interval], coverage: float, quality: QualityConfig
) -> tuple[bool, MissingReason | None]:
    if not any(interval.power_w is not None for interval in intervals):
        return False, MissingReason.NO_POWER_DATA
    if coverage < quality.power_min_coverage:
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
        power = [
            (interval.power_w, interval.active_s)
            for interval in group
            if interval.power_w is not None
        ]
        moving = sum(moving_s(interval, threshold_mps) for interval in group)
        heart_rate_valid_s = sum(weight for _value, weight in heart)
        active_s = sum(interval.active_s for interval in group)
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
                average_power_w=_weighted_mean(power),
                power_coverage_ratio=(
                    sum(weight for _value, weight in power) / active_s if active_s > 0.0 else 0.0
                ),
                power_reason=None,
            )
        )
    return tuple(windows)


def _weighted_mean(samples: Sequence[tuple[float, float]]) -> float | None:
    total = sum(weight for _value, weight in samples)
    return sum(value * weight for value, weight in samples) / total if total > 0.0 else None


def _build_available_windows(
    statistics: _ActivityStatistics,
) -> tuple[tuple[_Window, ...], tuple[_Window, ...]]:
    """Build elapsed-time windows and their optional movement-time variant."""
    activity = statistics.activity
    quality = statistics.quality
    intervals = statistics.interval_sequence()
    threshold_mps = quality.moving_speed_threshold_mps
    moving = statistics.moving_time_s()
    span = activity.duration_s or elapsed_time_s(activity)
    windows = _build_windows(intervals, span, threshold_mps=threshold_mps, moving_basis=False)
    movement = (
        _build_windows(intervals, moving, threshold_mps=threshold_mps, moving_basis=True)
        if abs(span - moving) > 1.0
        else ()
    )
    available, reason = _cadence_availability(intervals, statistics.cadence_coverage(), quality)
    if not available:
        masked = (reason or MissingReason.INSUFFICIENT_COVERAGE).value
        windows = _mask_window_cadence(windows, masked)
        movement = _mask_window_cadence(movement, masked)
    power_available, power_reason = statistics.power_availability()
    if not power_available:
        masked = (power_reason or MissingReason.INSUFFICIENT_COVERAGE).value
        windows = _mask_window_power(windows, masked)
        movement = _mask_window_power(movement, masked)
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
            average_power_w=window.average_power_w,
            power_coverage_ratio=window.power_coverage_ratio,
            power_reason=window.power_reason,
        )
        for window in windows
    )


def _mask_window_power(windows: tuple[_Window, ...], reason: str) -> tuple[_Window, ...]:
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
            average_pedaling_cadence_rpm=window.average_pedaling_cadence_rpm,
            median_pedaling_cadence_rpm=window.median_pedaling_cadence_rpm,
            cadence_coverage_ratio=window.cadence_coverage_ratio,
            cadence_reason=window.cadence_reason,
            average_power_w=None,
            power_coverage_ratio=window.power_coverage_ratio,
            power_reason=reason,
        )
        for window in windows
    )


def _window_tables(
    prefix: str,
    context: str,
    windows: tuple[_Window, ...],
    period: TripPeriod,
    *,
    time_basis: TimeBasis = TimeBasis.ELAPSED_TIME,
) -> list[Metric]:
    if not windows:
        return []
    axis = Axis(
        name="start_s",
        values=tuple(window.start_s for window in windows),
        unit="s",
    )
    cadence_masked = any(window.cadence_reason is not None for window in windows)
    power_masked = any(window.power_reason is not None for window in windows)
    basis_description = "movement-time" if time_basis is TimeBasis.MOVING_TIME else "elapsed-time"
    metrics: list[Metric] = []
    for key, label, unit, extract in _WINDOW_COLUMNS:
        if key == "cadence_reason" and not cadence_masked:
            continue
        if key == "power_reason" and not power_masked:
            continue
        metrics.append(
            table_metric(
                f"{prefix}.{key}",
                period,
                axis,
                tuple(extract(window) for window in windows),
                description=f"{context} {label} for consecutive {basis_description} windows.",
                unit=unit,
                kind=MetricKind.CALCULATED,
                time_basis=time_basis,
            )
        )
    return metrics


def _heart_rate_drift(statistics: _ActivityStatistics) -> float | None:
    windows = statistics.available_windows()[0]
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


def _build_efficiency_facts(statistics: _ActivityStatistics) -> _EfficiencyFacts:
    athlete = statistics.athlete
    if athlete is None:
        msg = "athlete configuration is required for efficiency statistics"
        raise ValueError(msg)
    quality = statistics.quality
    intervals = statistics.interval_sequence()
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
    moving = statistics.moving_time_s()
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
