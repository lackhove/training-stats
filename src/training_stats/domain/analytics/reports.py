"""Public metric-report summaries composed from pure analytics functions."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timedelta, tzinfo

from ..metrics import (
    AllTimePeriod,
    Axis,
    DateRangePeriod,
    MatrixValue,
    Metric,
    MetricKind,
    MetricMethod,
    MetricReport,
    MetricScalar,
    MissingReason,
    SnapshotPeriod,
    TimeBasis,
)
from ..models import ActivityRecord, SourceIndexEntry, TrainingFormResult
from ..settings import AthleteConfig
from .activities import data_quality_metrics
from .daily import DailySummaryRow, select_daily, summarize_daily
from .fitness import training_form_from
from .load import LOAD_COEFFICIENT_SET, LOAD_FORMULA, LOAD_VERSION
from .metric_helpers import scalar_metric, table_metric
from .profile import athlete_profile
from .record_frames import activity_records_frame
from .weekly import WeeklySummaryRow, daily_to_weekly, select_weekly


def activity_report(records: Sequence[ActivityRecord], activity_id: str) -> MetricReport | None:
    """Return the cached per-trip metric collection for one activity."""
    record = next((item for item in records if item.activity_id == activity_id), None)
    if record is None:
        return None
    return MetricReport(metrics=record.metrics)


def activity_list_report(
    records: Sequence[ActivityRecord],
    tz: tzinfo,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    sport: str | None = None,
    limit: int = 50,
) -> MetricReport:
    """Return selected activity metrics, newest trip first."""
    selected = [
        record
        for record in records
        if _in_range(record.start_time.astimezone(tz).date(), start_date, end_date)
        and (sport is None or record.sport.casefold() == sport.casefold())
    ]
    selected.sort(key=lambda record: (record.start_time, record.activity_id), reverse=True)
    return MetricReport(
        metrics=tuple(metric for record in selected[:limit] for metric in record.metrics)
    )


def weekly_summary(
    records: Sequence[ActivityRecord],
    tz: tzinfo,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    weeks: int = 8,
    sport: str | None = None,
    today: date | None = None,
) -> MetricReport:
    """Summarize cached activity metrics into weekly table metrics."""
    daily = summarize_daily(activity_records_frame(list(records)), tz)
    rows = select_weekly(
        daily_to_weekly(daily),
        start_date=start_date,
        end_date=end_date,
        weeks=weeks,
        sport=sport,
    )
    if today is None:
        today = datetime.now(tz).date()
    first = min((date.fromisoformat(row.week_start) for row in rows), default=start_date or today)
    last = max(
        (date.fromisoformat(row.week_start) for row in rows), default=end_date or today
    ) + timedelta(days=6)
    period = DateRangePeriod(start=start_date or first, end=end_date or last, timezone=str(tz))
    return MetricReport(metrics=_weekly_metrics(rows, period))


def daily_summary(
    records: Sequence[ActivityRecord],
    tz: tzinfo,
    *,
    start_date: date,
    end_date: date,
    sport: str | None = None,
) -> MetricReport:
    """Summarize daily activity metrics over an inclusive date range."""
    rows = select_daily(
        summarize_daily(activity_records_frame(list(records)), tz),
        start_date=start_date,
        end_date=end_date,
        sport=sport,
    )
    period = DateRangePeriod(start=start_date, end=end_date, timezone=str(tz))
    return MetricReport(metrics=_daily_metrics(rows, period))


def overview_report(
    records: Sequence[ActivityRecord],
    source_index: Sequence[SourceIndexEntry],
    *,
    athlete: AthleteConfig,
    tz: tzinfo,
    days: int = 28,
    sport: str | None = None,
    today: date | None = None,
) -> MetricReport:
    """Compose current activity, trend, form, profile, and quality metrics."""
    if today is None:
        today = datetime.now(tz).date()
    period_start = today - timedelta(days=max(days, 1) - 1)
    period = DateRangePeriod(start=period_start, end=today, timezone=str(tz))
    frame = activity_records_frame(list(records))
    daily_all = summarize_daily(frame, tz)
    selected_daily = select_daily(daily_all, start_date=period_start, end_date=today, sport=sport)
    selected_weekly = select_weekly(daily_to_weekly(daily_all), weeks=4, sport=sport)
    weekly_period = (
        DateRangePeriod(
            start=min(date.fromisoformat(row.week_start) for row in selected_weekly),
            end=max(date.fromisoformat(row.week_start) for row in selected_weekly)
            + timedelta(days=6),
            timezone=str(tz),
        )
        if selected_weekly
        else period
    )
    form = training_form_from(frame, tz=tz, days=days, sport=sport, today=today)
    latest = sorted(
        (
            record
            for record in records
            if sport is None or record.sport.casefold() == sport.casefold()
        ),
        key=lambda record: (record.start_time, record.activity_id),
        reverse=True,
    )
    recent = activity_list_report(
        records,
        tz,
        start_date=period_start,
        end_date=today,
        sport=sport,
        limit=5,
    )

    metrics: list[Metric] = []
    metrics.extend(athlete_profile(athlete))
    metrics.extend(_daily_metrics(selected_daily, period))
    metrics.extend(_weekly_metrics(selected_weekly, weekly_period))
    metrics.extend(_form_metrics(form, today, str(tz)))
    metrics.extend(data_quality_metrics(records, source_index, AllTimePeriod()))
    zone_values = _zone_distribution(selected_daily)
    metrics.append(
        table_metric(
            "heart_rate.zone_time_overview",
            period,
            Axis(name="heart_rate_zone", values=tuple(zone_values)),
            tuple(zone_values.values()),
            unit="s",
            time_basis=TimeBasis.ACTIVE_TIME,
            kind=MetricKind.CALCULATED,
        )
    )
    metrics.extend(recent.metrics)
    if latest:
        metrics.extend(latest[0].metrics)
    return MetricReport(metrics=tuple(metrics))


def _weekly_metrics(
    rows: Sequence[WeeklySummaryRow], period: DateRangePeriod
) -> tuple[Metric, ...]:
    axis = Axis(name="week_start", values=tuple(row.week_start for row in rows))
    specs: tuple[tuple[str, str, tuple[MetricScalar | None, ...], MetricKind], ...] = (
        (
            "activity.count_by_week",
            "activities",
            tuple(row.activity_count for row in rows),
            MetricKind.CALCULATED,
        ),
        (
            "activity.duration_by_week",
            "s",
            tuple(row.duration_s for row in rows),
            MetricKind.CALCULATED,
        ),
        (
            "activity.distance_by_week",
            "m",
            tuple(row.distance_m for row in rows),
            MetricKind.CALCULATED,
        ),
        (
            "activity.elevation_gain_by_week",
            "m",
            tuple(row.elevation_gain_m for row in rows),
            MetricKind.CALCULATED,
        ),
        (
            "training_load.hr_trimp_by_week",
            "TRIMP",
            tuple(row.load_value for row in rows),
            MetricKind.ESTIMATED,
        ),
        (
            "activity.hard_count_by_week",
            "activities",
            tuple(row.hard_activity_count for row in rows),
            MetricKind.CALCULATED,
        ),
        (
            "activity.longest_duration_by_week",
            "s",
            tuple(row.longest_activity_s for row in rows),
            MetricKind.CALCULATED,
        ),
    )
    metrics = []
    for name, unit, values, kind in specs:
        load_provenance = name == "training_load.hr_trimp_by_week"
        metrics.append(
            table_metric(
                name,
                period,
                axis,
                values,
                unit=unit,
                kind=kind,
                missing_reason=MissingReason.NO_DATA if not rows else None,
                method=MetricMethod.HR_TRIMP if load_provenance else None,
                algorithm_version=LOAD_VERSION if load_provenance else None,
                formula=LOAD_FORMULA if load_provenance else None,
                coefficient_set=LOAD_COEFFICIENT_SET if load_provenance else None,
            )
        )
    for zone in range(1, 6):
        metrics.append(
            table_metric(
                f"heart_rate.zone_{zone}_time_by_week",
                period,
                axis,
                tuple(row.zone_seconds()[f"zone_{zone}"] for row in rows),
                unit="s",
                kind=MetricKind.CALCULATED,
                missing_reason=MissingReason.NO_DATA if not rows else None,
                time_basis=TimeBasis.ACTIVE_TIME,
            )
        )
    metrics.extend(
        [
            table_metric(
                "activity.distance_change_vs_previous_week",
                period,
                axis,
                tuple(
                    row.distance_m - row.distance_m_prev
                    if row.distance_m_prev is not None
                    else None
                    for row in rows
                ),
                unit="m",
                missing_reason=MissingReason.NO_DATA if not rows else None,
            ),
            table_metric(
                "training_load.change_vs_previous_week",
                period,
                axis,
                tuple(
                    row.load_value - row.load_value_prev
                    if row.load_value_prev is not None
                    else None
                    for row in rows
                ),
                unit="TRIMP",
                kind=MetricKind.ESTIMATED,
                missing_reason=MissingReason.NO_DATA if not rows else None,
            ),
        ]
    )
    return tuple(metrics)


def _daily_metrics(rows: Sequence[DailySummaryRow], period: DateRangePeriod) -> tuple[Metric, ...]:
    axis = Axis(name="date", values=tuple(row.date for row in rows))
    specs: tuple[tuple[str, str, tuple[MetricScalar | None, ...], MetricKind], ...] = (
        (
            "activity.count_by_day",
            "activities",
            tuple(row.activity_count for row in rows),
            MetricKind.CALCULATED,
        ),
        (
            "activity.duration_by_day",
            "s",
            tuple(row.duration_s for row in rows),
            MetricKind.CALCULATED,
        ),
        (
            "activity.distance_by_day",
            "m",
            tuple(row.distance_m for row in rows),
            MetricKind.CALCULATED,
        ),
        (
            "activity.elevation_gain_by_day",
            "m",
            tuple(row.elevation_gain_m for row in rows),
            MetricKind.CALCULATED,
        ),
        (
            "training_load.hr_trimp_by_day",
            "TRIMP",
            tuple(row.load_value for row in rows),
            MetricKind.ESTIMATED,
        ),
    )
    metrics = []
    for name, unit, values, kind in specs:
        load_provenance = name == "training_load.hr_trimp_by_day"
        metrics.append(
            table_metric(
                name,
                period,
                axis,
                values,
                unit=unit,
                kind=kind,
                missing_reason=MissingReason.NO_DATA if not rows else None,
                method=MetricMethod.HR_TRIMP if load_provenance else None,
                algorithm_version=LOAD_VERSION if load_provenance else None,
                formula=LOAD_FORMULA if load_provenance else None,
                coefficient_set=LOAD_COEFFICIENT_SET if load_provenance else None,
            )
        )
    zone_axis = Axis(name="heart_rate_zone", values=tuple(f"zone_{i}" for i in range(1, 6)))
    matrix_values = tuple(
        tuple(row.zone_seconds()[f"zone_{zone}"] for row in rows) for zone in range(1, 6)
    )
    metrics.append(
        Metric(
            name="heart_rate.zone_matrix_by_day",
            period=period,
            value=None,
            kind=MetricKind.CALCULATED,
            unit="s",
            missing_reason=MissingReason.NO_DATA,
            time_basis=TimeBasis.ACTIVE_TIME,
        )
        if not rows
        else _matrix_metric("heart_rate.zone_matrix_by_day", period, axis, zone_axis, matrix_values)
    )
    return tuple(metrics)


def _matrix_metric(
    name: str,
    period: DateRangePeriod,
    x_axis: Axis,
    y_axis: Axis,
    values: tuple[tuple[MetricScalar | None, ...], ...],
) -> Metric:
    return Metric(
        name=name,
        period=period,
        value=MatrixValue(x_axis=x_axis, y_axis=y_axis, values=values),
        kind=MetricKind.CALCULATED,
        unit="s",
        time_basis=TimeBasis.ACTIVE_TIME,
    )


def _form_metrics(result: TrainingFormResult, today: date, timezone: str) -> tuple[Metric, ...]:
    snapshot = SnapshotPeriod(at=today)
    values: tuple[tuple[str, MetricScalar, str], ...] = (
        ("training_form.ctl", result.current_ctl, "TRIMP"),
        ("training_form.atl", result.current_atl, "TRIMP"),
        ("training_form.tsb", result.current_tsb, "TRIMP"),
        ("training_form.classification", result.form_classification, ""),
        ("training_form.is_warmup", result.is_warmup, ""),
        ("training_form.load_confidence_ratio", result.load_confidence_ratio, "ratio"),
        ("tour_readiness.rolling_7d_distance", result.rolling_7d_distance_km, "km"),
        ("tour_readiness.rolling_28d_distance", result.rolling_28d_distance_km, "km"),
    )
    metrics = [
        scalar_metric(
            name,
            snapshot,
            value,
            unit=unit,
            kind=(
                MetricKind.INTERPRETED
                if name in {"training_form.classification", "training_form.is_warmup"}
                else MetricKind.CALCULATED
            ),
        )
        for name, value, unit in values
    ]
    metrics.append(
        scalar_metric(
            "tour_readiness.days_since_60k_ride",
            snapshot,
            result.days_since_60k_ride,
            unit="days",
            kind=MetricKind.CALCULATED,
            missing_reason=MissingReason.NO_DATA,
        )
    )
    if result.trend_points:
        period = DateRangePeriod(
            start=date.fromisoformat(result.trend_points[0].date),
            end=today,
            timezone=timezone,
        )
        axis = Axis(name="date", values=tuple(point.date for point in result.trend_points))
        for name, attr in (
            ("training_form.ctl_by_day", "ctl"),
            ("training_form.atl_by_day", "atl"),
            ("training_form.tsb_by_day", "tsb"),
            ("training_load.daily_hr_trimp", "daily_load"),
        ):
            load_provenance = name == "training_load.daily_hr_trimp"
            metrics.append(
                table_metric(
                    name,
                    period,
                    axis,
                    tuple(getattr(point, attr) for point in result.trend_points),
                    unit="TRIMP",
                    kind=MetricKind.CALCULATED,
                    method=MetricMethod.HR_TRIMP if load_provenance else None,
                    algorithm_version=LOAD_VERSION if load_provenance else None,
                    formula=LOAD_FORMULA if load_provenance else None,
                    coefficient_set=LOAD_COEFFICIENT_SET if load_provenance else None,
                )
            )
    return tuple(metrics)


def _zone_distribution(rows: Sequence[DailySummaryRow]) -> dict[str, float]:
    zones = {f"zone_{index}": 0.0 for index in range(1, 6)}
    for row in rows:
        for name, seconds in row.zone_seconds().items():
            zones[name] += seconds
    return zones


def _in_range(value: date, start: date | None, end: date | None) -> bool:
    return (start is None or value >= start) and (end is None or value <= end)
