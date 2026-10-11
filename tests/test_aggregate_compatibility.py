"""Golden aggregate outputs captured from the pre-refactor Polars implementation.

The fixtures lock public weekly/daily/overview/training-form values, metric
order, periods, and missing reasons so the record-based aggregation emits the
same JSON for identical inputs.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

import pytest

from tests.conftest import make_athlete
from training_stats.domain.analytics.daily import summarize_daily
from training_stats.domain.analytics.fitness import training_form_from
from training_stats.domain.analytics.reports import overview_report, weekly_summary
from training_stats.domain.metrics import (
    AllTimePeriod,
    Axis,
    Confidence,
    DateRangePeriod,
    MatrixValue,
    Metric,
    MetricKind,
    MetricMethod,
    MetricReport,
    MissingReason,
    ScalarValue,
    SnapshotPeriod,
    TableValue,
    TimeBasis,
    TripPeriod,
)
from training_stats.domain.models import ActivityRecord, SourceIndexEntry

TZ = ZoneInfo("Europe/Berlin")
TODAY = date(2026, 6, 15)
_ABSENT = object()

ZONE_AXIS = Axis(name="heart_rate_zone", values=tuple(f"zone_{i}" for i in range(1, 6)))


def _cached_scalar(name: str, period: TripPeriod, value: object, unit: str | None) -> Metric:
    return Metric(
        name=name,
        description=f"Cached {name} value for aggregate compatibility tests.",
        period=period,
        value=ScalarValue(value) if value is not None else None,  # type: ignore[arg-type]
        kind=MetricKind.CALCULATED,
        unit=unit,
        missing_reason=None if value is not None else MissingReason.NO_DATA,
    )


def _record(
    activity_id: str,
    start: datetime,
    sport: str,
    *,
    duration: object = _ABSENT,
    distance: object = _ABSENT,
    elevation: object = _ABSENT,
    load: object = _ABSENT,
    confidence: str | None = None,
    zones: tuple[float | None, ...] | None = None,
    coverage: float | None = None,
    completeness: float | None = None,
    present_quantities: tuple[str, ...] | None = None,
) -> ActivityRecord:
    """Build one metric-bearing cached record."""
    period = TripPeriod(activity_id=activity_id, start=start)
    metrics: list[Metric] = []
    for name, value, unit in (
        ("activity.duration", duration, "s"),
        ("activity.distance", distance, "m"),
        ("activity.elevation_gain", elevation, "m"),
    ):
        if value is not _ABSENT:
            metrics.append(_cached_scalar(name, period, value, unit))
    if load is not _ABSENT:
        metrics.append(
            Metric(
                name="training_load.hr_trimp",
                description="Estimated Banister heart-rate load for compatibility tests.",
                period=period,
                value=ScalarValue(load) if load is not None else None,  # type: ignore[arg-type]
                kind=MetricKind.ESTIMATED,
                unit="TRIMP",
                missing_reason=None if load is not None else MissingReason.NO_DATA,
                method=MetricMethod.HR_TRIMP,
                confidence=Confidence(confidence) if confidence is not None else None,
            )
        )
    if zones is not None:
        axis = Axis(
            name="heart_rate_zone",
            values=tuple(f"zone_{index}" for index in range(1, len(zones) + 1)),
        )
        metrics.append(
            Metric(
                name="heart_rate.zone_time",
                description="Active time assigned to each configured heart-rate zone.",
                period=period,
                value=TableValue(axis=axis, values=zones),
                kind=MetricKind.CALCULATED,
                unit="s",
                time_basis=TimeBasis.ACTIVE_TIME,
            )
        )
    if coverage is not None:
        metrics.append(_cached_scalar("heart_rate.coverage", period, coverage, "ratio"))
    if completeness is not None:
        metrics.append(_cached_scalar("data.completeness", period, completeness, "ratio"))
    if present_quantities is None:
        quantities = set()
        if distance is not _ABSENT:
            quantities.add("distance")
        if elevation is not _ABSENT:
            quantities.add("elevation")
        if load is not _ABSENT or zones is not None:
            quantities.add("heart_rate")
        present_quantities = tuple(sorted(quantities))
    return ActivityRecord(
        activity_id=activity_id,
        start_time=start,
        sport=sport,
        sub_sport=None,
        segment_index=1,
        segment_count=1,
        repair_flags=(),
        source_name="local",
        source_path=f"{activity_id}.fit",
        analysis_version="8",
        metrics=tuple(metrics),
        present_quantities=present_quantities,
    )


def _fixture_records() -> tuple[ActivityRecord, ...]:
    """Deterministic mixed-history fixture (deliberately unsorted input)."""
    return (
        _record(
            "act-a",
            datetime(2026, 6, 2, 9, 0, tzinfo=TZ),
            "cycling",
            duration=3600.0,
            distance=60_000.0,
            elevation=50.0,
            load=150.0,
            confidence="high",
            zones=(50.0, 50.0, 50.0, 50.0, 50.0),
            coverage=0.7,
            completeness=1.0,
        ),
        _record(
            "act-b",
            datetime(2026, 6, 1, 23, 40, tzinfo=UTC),
            "cycling",
            duration=3600.0,
            distance=50_000.0,
            elevation=100.0,
            load=160.0,
            confidence="high",
            zones=(100.0, 200.0, 300.0, 400.0, 500.0),
            coverage=0.8,
            completeness=1.0,
        ),
        _record(
            "act-c",
            datetime(2026, 6, 2, 12, 0, tzinfo=UTC),
            "running",
            duration=1800.0,
            distance=10_000.0,
            elevation=20.0,
            load=149.0,
            confidence="low",
            zones=(10.0, 20.0, 30.0),
            coverage=0.6,
            completeness=1.0,
        ),
        _record(
            "act-d",
            datetime(2026, 6, 2, 15, 0, tzinfo=UTC),
            "cycling",
            coverage=0.5,
            completeness=0.5,
        ),
        _record(
            "act-g",
            datetime(2026, 6, 2, 18, 0, tzinfo=UTC),
            "cycling",
            duration="not-a-number",
            distance=30_000.0,
            elevation=True,
            load=None,
            zones=(),
        ),
        _record(
            "act-e",
            datetime(2026, 5, 28, 10, 0, tzinfo=UTC),
            "cycling",
            duration=7200.0,
            distance=90_000.0,
            elevation=300.0,
            load=200.0,
            confidence="high",
            zones=(0.0, 100.0, 200.0, 300.0, 400.0),
            coverage=1.0,
            completeness=1.0,
        ),
        _record(
            "act-f",
            datetime(2026, 4, 1, 8, 0, tzinfo=UTC),
            "cycling",
            duration=5400.0,
            distance=45_000.0,
            elevation=200.0,
            load=100.0,
            confidence="low",
            zones=(100.0, 200.0, 300.0, 400.0, 500.0),
            coverage=0.9,
            completeness=1.0,
        ),
        _record(
            "act-h",
            datetime(2026, 6, 12, 7, 0, tzinfo=UTC),
            "cycling",
            duration=7200.0,
            distance=80_000.0,
            elevation=120.0,
            load=220.0,
            confidence="high",
            zones=(0.0, 0.0, 1000.0, 2000.0, 1000.0),
            coverage=1.0,
            completeness=1.0,
        ),
    )


def _source_index() -> tuple[SourceIndexEntry, ...]:
    def entry(path: str, status: str, flags: tuple[str, ...] = ()) -> SourceIndexEntry:
        return SourceIndexEntry(
            source_name="local",
            source_path=path,
            source_hash="hash",
            file_size=10,
            modified_at=datetime(2026, 6, 1, tzinfo=UTC),
            activity_id="act-a",
            segment_index=1,
            segment_count=1,
            file_repair_flags=flags,
            discarded_segment_count=1 if status == "failed" else 0,
            processed_at=datetime(2026, 6, 2, tzinfo=UTC),
            source_analysis_version="8",
            status=status,  # type: ignore[arg-type]
            error_message="boom" if status == "failed" else None,
        )

    return (
        entry("a.fit", "ok", ("auto_pause_fix",)),
        entry("b.fit", "failed"),
        entry("c.fit", "duplicate"),
    )


RECORDS = _fixture_records()
SOURCE_INDEX = _source_index()
ATHLETE = make_athlete()


def _names(report: MetricReport) -> list[str]:
    return [metric.name for metric in report.metrics]


def _metric(report: MetricReport, name: str) -> Metric:
    matches = [metric for metric in report.metrics if metric.name == name]
    assert len(matches) == 1, name
    return matches[0]


def _table(report: MetricReport, name: str) -> tuple[tuple[object, ...], tuple[object, ...]]:
    metric = _metric(report, name)
    assert isinstance(metric.value, TableValue)
    return metric.value.axis.values, metric.value.values


def _matrix(
    report: MetricReport, name: str
) -> tuple[tuple[object, ...], tuple[object, ...], tuple[tuple[object, ...], ...]]:
    metric = _metric(report, name)
    value = metric.value
    assert isinstance(value, MatrixValue)
    return (
        value.x_axis.values,
        value.y_axis.values,
        tuple(tuple(row) for row in value.values),
    )


def _scalar(report: MetricReport, name: str) -> object:
    metric = _metric(report, name)
    if metric.value is None:
        return None
    assert isinstance(metric.value, ScalarValue)
    return metric.value.value


WEEKLY_NAMES = [
    "activity.count_by_week",
    "activity.duration_by_week",
    "activity.distance_by_week",
    "activity.elevation_gain_by_week",
    "training_load.hr_trimp_by_week",
    "activity.hard_count_by_week",
    "activity.longest_duration_by_week",
    "heart_rate.zone_1_time_by_week",
    "heart_rate.zone_2_time_by_week",
    "heart_rate.zone_3_time_by_week",
    "heart_rate.zone_4_time_by_week",
    "heart_rate.zone_5_time_by_week",
    "activity.distance_change_vs_previous_week",
    "training_load.change_vs_previous_week",
]

DAILY_NAMES = [
    "activity.count_by_day",
    "activity.duration_by_day",
    "activity.distance_by_day",
    "activity.elevation_gain_by_day",
    "training_load.hr_trimp_by_day",
    "heart_rate.zone_matrix_by_day",
]

WEEKLY_VALUES: dict[str, tuple[object, ...]] = {
    "activity.count_by_week": (1, 5, 1, 1),
    "activity.duration_by_week": (7200.0, 9000.0, 7200.0, 5400.0),
    "activity.distance_by_week": (80000.0, 150000.0, 90000.0, 45000.0),
    "activity.elevation_gain_by_week": (120.0, 170.0, 300.0, 200.0),
    "training_load.hr_trimp_by_week": (220.0, 459.0, 200.0, 100.0),
    "activity.hard_count_by_week": (1, 2, 1, 0),
    "activity.longest_duration_by_week": (7200.0, 3600.0, 7200.0, 5400.0),
    "heart_rate.zone_1_time_by_week": (0.0, 160.0, 0.0, 100.0),
    "heart_rate.zone_2_time_by_week": (0.0, 270.0, 100.0, 200.0),
    "heart_rate.zone_3_time_by_week": (1000.0, 380.0, 200.0, 300.0),
    "heart_rate.zone_4_time_by_week": (2000.0, 450.0, 300.0, 400.0),
    "heart_rate.zone_5_time_by_week": (1000.0, 550.0, 400.0, 500.0),
    "activity.distance_change_vs_previous_week": (-60000.0, 60000.0, None, None),
    "training_load.change_vs_previous_week": (-90.0, 259.0, None, None),
}

WEEKLY_AXIS = ("2026-06-08", "2026-06-01", "2026-05-25", "2026-03-30")


class TestDailyRowParity:
    """Exact daily grouping values from the previous Polars aggregation."""

    def test_exact_rows(self) -> None:
        rows = summarize_daily(RECORDS, TZ)
        assert [row.date for row in rows] == [
            date(2026, 4, 1),
            date(2026, 5, 28),
            date(2026, 6, 2),
            date(2026, 6, 2),
            date(2026, 6, 12),
        ]
        assert [(row.date, row.sport) for row in rows] == [
            (date(2026, 4, 1), "cycling"),
            (date(2026, 5, 28), "cycling"),
            (date(2026, 6, 2), "cycling"),
            (date(2026, 6, 2), "running"),
            (date(2026, 6, 12), "cycling"),
        ]
        cycling = rows[2]
        # act-d has no aggregate metrics at all; act-g carries a non-numeric
        # duration, a boolean elevation, and a null load.
        assert cycling.activity_count == 4
        assert cycling.duration_s == 7200.0
        assert cycling.distance_m == 140_000.0
        assert cycling.elevation_gain_m == 150.0
        assert cycling.load_value == 310.0
        assert cycling.hard_activity_count == 2  # 160.0 and the inclusive 150.0
        assert (
            cycling.hr_zone_1_s,
            cycling.hr_zone_2_s,
            cycling.hr_zone_3_s,
            cycling.hr_zone_4_s,
            cycling.hr_zone_5_s,
        ) == (150.0, 250.0, 350.0, 450.0, 550.0)
        # act-b and act-a tie at 3600 s; act-b starts first in sorted order.
        assert cycling.longest_activity_s == 3600.0
        assert cycling.longest_activity_id == "act-b"

        running = rows[3]
        assert running.activity_count == 1
        assert running.load_value == 149.0
        assert running.hard_activity_count == 0
        # The truncated zone table leaves zones 4 and 5 empty.
        assert (running.hr_zone_4_s, running.hr_zone_5_s) == (0.0, 0.0)

    def test_timezone_bucketing(self) -> None:
        rows = summarize_daily(RECORDS, TZ)
        # act-b starts 2026-06-01 23:40 UTC, which is 2026-06-02 in Berlin.
        dates = {row.date for row in rows}
        assert date(2026, 6, 1) not in dates

    def test_input_order_is_irrelevant(self) -> None:
        assert summarize_daily(RECORDS, TZ) == summarize_daily(list(reversed(RECORDS)), TZ)

    def test_empty(self) -> None:
        assert summarize_daily([], TZ) == []


class TestWeeklyReportParity:
    def _report(self) -> MetricReport:
        return weekly_summary(RECORDS, TZ, weeks=8, today=TODAY)

    def test_metric_order(self) -> None:
        assert _names(self._report()) == WEEKLY_NAMES

    def test_values_and_axis(self) -> None:
        report = self._report()
        for name, values in WEEKLY_VALUES.items():
            assert _table(report, name) == (WEEKLY_AXIS, values), name

    def test_period(self) -> None:
        period = DateRangePeriod(start=date(2026, 3, 30), end=date(2026, 6, 14), timezone=str(TZ))
        assert all(metric.period == period for metric in self._report().metrics)

    def test_load_provenance(self) -> None:
        load = _metric(self._report(), "training_load.hr_trimp_by_week")
        assert load.method == MetricMethod.HR_TRIMP
        assert load.kind is MetricKind.ESTIMATED
        assert load.algorithm_version is not None
        assert load.formula is not None
        assert load.coefficient_set is not None

    def test_week_bounds_keep_week_granularity(self) -> None:
        report = weekly_summary(
            RECORDS,
            TZ,
            start_date=date(2026, 5, 1),
            end_date=date(2026, 6, 15),
            weeks=8,
            today=TODAY,
        )
        axis, values = _table(report, "activity.count_by_week")
        assert axis == ("2026-06-08", "2026-06-01", "2026-05-25")
        assert values == (1, 5, 1)
        assert _table(report, "activity.distance_change_vs_previous_week")[1] == (
            -60000.0,
            60000.0,
            None,
        )
        period = DateRangePeriod(start=date(2026, 5, 1), end=date(2026, 6, 15), timezone=str(TZ))
        assert all(metric.period == period for metric in report.metrics)

    def test_previous_week_comes_from_unfiltered_history(self) -> None:
        # The bounds exclude the previous week itself but its totals still
        # drive the change metrics.
        report = weekly_summary(
            RECORDS,
            TZ,
            start_date=date(2026, 6, 1),
            end_date=date(2026, 6, 15),
            weeks=8,
            today=TODAY,
        )
        axis, values = _table(report, "activity.count_by_week")
        assert axis == ("2026-06-08", "2026-06-01")
        assert values == (1, 5)
        assert _table(report, "activity.distance_change_vs_previous_week")[1] == (
            -60000.0,
            60000.0,
        )
        assert _table(report, "training_load.change_vs_previous_week")[1] == (-90.0, 259.0)
        period = DateRangePeriod(start=date(2026, 6, 1), end=date(2026, 6, 15), timezone=str(TZ))
        assert all(metric.period == period for metric in report.metrics)

    def test_sport_filter_is_case_insensitive(self) -> None:
        report = weekly_summary(RECORDS, TZ, weeks=8, sport="cycling", today=TODAY)
        assert _table(report, "activity.count_by_week")[1] == (1, 4, 1, 1)
        assert _table(report, "activity.distance_by_week")[1] == (
            80_000.0,
            140_000.0,
            90_000.0,
            45_000.0,
        )
        upper = weekly_summary(RECORDS, TZ, weeks=4, sport="CYCLING", today=TODAY)
        assert _names(upper) == WEEKLY_NAMES
        assert _table(upper, "activity.count_by_week") == _table(report, "activity.count_by_week")

    def test_empty_report(self) -> None:
        report = weekly_summary((), TZ, weeks=8, today=TODAY)
        assert _names(report) == ["activity.count_by_week"]
        count = _metric(report, "activity.count_by_week")
        assert count.value is None
        assert count.missing_reason is MissingReason.NO_DATA
        assert all(
            metric.period
            == DateRangePeriod(start=date(2026, 6, 15), end=date(2026, 6, 21), timezone=str(TZ))
            for metric in report.metrics
        )

    def test_metrics_use_only_activities_with_the_required_quantities(self) -> None:
        report = weekly_summary(
            (
                _record(
                    "measured-zeroes",
                    datetime(2026, 6, 2, 9, 0, tzinfo=TZ),
                    "cycling",
                    duration=100.0,
                    distance=0.0,
                    elevation=0.0,
                    load=0.0,
                    zones=(0.0, None, 0.0, 0.0, 0.0),
                ),
                _record(
                    "without-sensors",
                    datetime(2026, 6, 2, 10, 0, tzinfo=TZ),
                    "cycling",
                    duration=200.0,
                ),
                _record(
                    "sensor-free-week",
                    datetime(2026, 6, 9, 9, 0, tzinfo=TZ),
                    "cycling",
                    duration=600.0,
                ),
            ),
            TZ,
            weeks=8,
            today=TODAY,
        )

        assert _table(report, "activity.count_by_week") == (
            ("2026-06-08", "2026-06-01"),
            (1, 2),
        )
        assert _table(report, "activity.duration_by_week")[1] == (600.0, 300.0)
        assert _table(report, "activity.distance_by_week")[1] == (None, 0.0)
        assert _table(report, "activity.elevation_gain_by_week")[1] == (None, 0.0)
        assert _table(report, "training_load.hr_trimp_by_week")[1] == (None, 0.0)
        assert _table(report, "activity.hard_count_by_week")[1] == (None, 0)
        assert _table(report, "heart_rate.zone_1_time_by_week")[1] == (None, 0.0)
        assert _table(report, "heart_rate.zone_2_time_by_week")[1] == (None, None)

    def test_metrics_are_omitted_when_no_selected_activity_has_the_quantity(self) -> None:
        no_sensors = _record(
            "no-sensors",
            datetime(2026, 6, 2, 9, 0, tzinfo=TZ),
            "running",
            duration=600.0,
        )
        report = weekly_summary((no_sensors,), TZ, sport="running", today=TODAY)

        assert "activity.count_by_week" in _names(report)
        assert "activity.duration_by_week" in _names(report)
        assert not {
            "activity.distance_by_week",
            "activity.elevation_gain_by_week",
            "training_load.hr_trimp_by_week",
            "activity.hard_count_by_week",
            "heart_rate.zone_1_time_by_week",
            "activity.distance_change_vs_previous_week",
            "training_load.change_vs_previous_week",
        } & set(_names(report))


OVERVIEW_NAMES = [
    "athlete_profile.max_heart_rate",
    "athlete_profile.resting_heart_rate",
    "athlete_profile.primary_goal",
    "athlete_profile.weight",
    "athlete_profile.functional_threshold_power",
    "athlete_profile.heart_rate_zone_minimum",
    "athlete_profile.heart_rate_zone_maximum",
    *DAILY_NAMES,
    *WEEKLY_NAMES,
    "training_form.ctl",
    "training_form.atl",
    "training_form.tsb",
    "training_form.classification",
    "training_form.is_warmup",
    "training_form.load_confidence_ratio",
    "tour_readiness.rolling_7d_distance",
    "tour_readiness.rolling_28d_distance",
    "tour_readiness.days_since_60k_ride",
    "training_form.ctl_by_day",
    "training_form.atl_by_day",
    "training_form.tsb_by_day",
    "training_load.daily_hr_trimp",
    "data_quality.activity_count",
    "data_quality.mean_hr_coverage_ratio",
    "data_quality.mean_data_completeness",
    "data_quality.failed_file_count",
    "data_quality.repaired_file_count",
    "data_quality.discarded_segment_count",
    "data_quality.duplicate_activity_count",
    "heart_rate.zone_time_overview",
]


class TestOverviewReportParity:
    def _report(self) -> MetricReport:
        return overview_report(RECORDS, SOURCE_INDEX, athlete=ATHLETE, tz=TZ, days=28, today=TODAY)

    def test_metric_order(self) -> None:
        assert _names(self._report()) == OVERVIEW_NAMES

    def test_periods(self) -> None:
        report = self._report()
        day_period = DateRangePeriod(start=date(2026, 5, 19), end=TODAY, timezone=str(TZ))
        week_period = DateRangePeriod(
            start=date(2026, 3, 30), end=date(2026, 6, 14), timezone=str(TZ)
        )
        for name in DAILY_NAMES:
            assert _metric(report, name).period == day_period, name
        for name in WEEKLY_NAMES:
            assert _metric(report, name).period == week_period, name
        assert _metric(report, "heart_rate.zone_time_overview").period == day_period
        assert _metric(report, "training_form.ctl_by_day").period == day_period
        assert _metric(report, "training_form.ctl").period == SnapshotPeriod(at=TODAY)
        assert _metric(report, "data_quality.activity_count").period == AllTimePeriod()

    def test_weekly_and_daily_values(self) -> None:
        report = self._report()
        assert _table(report, "activity.count_by_day") == (
            ("2026-05-28", "2026-06-02", "2026-06-12"),
            (1, 5, 1),
        )
        for name, values in WEEKLY_VALUES.items():
            assert _table(report, name) == (WEEKLY_AXIS, values), name
        assert _table(report, "heart_rate.zone_time_overview") == (
            ("zone_1", "zone_2", "zone_3", "zone_4", "zone_5"),
            (160.0, 370.0, 1580.0, 2750.0, 1950.0),
        )

    def test_aggregates_use_only_activities_with_the_required_quantities(self) -> None:
        records = (
            _record(
                "measured",
                datetime(2026, 6, 2, 9, 0, tzinfo=TZ),
                "cycling",
                duration=100.0,
                distance=10_000.0,
                elevation=0.0,
                load=100.0,
                zones=(60.0, 0.0, 0.0, 0.0, 0.0),
                coverage=0.6,
                completeness=1.0,
            ),
            _record(
                "without-heart-rate",
                datetime(2026, 6, 2, 10, 0, tzinfo=TZ),
                "cycling",
                duration=200.0,
                distance=20_000.0,
                coverage=0.0,
                completeness=0.5,
            ),
            _record(
                "heart-rate-free-week",
                datetime(2026, 6, 10, 9, 0, tzinfo=TZ),
                "cycling",
                duration=300.0,
                coverage=0.0,
                completeness=0.5,
            ),
        )
        report = overview_report(records, (), athlete=ATHLETE, tz=TZ, days=14, today=TODAY)

        assert _table(report, "activity.count_by_day") == (
            ("2026-06-02", "2026-06-10"),
            (2, 1),
        )
        assert _table(report, "activity.duration_by_day")[1] == (300.0, 300.0)
        assert _table(report, "activity.distance_by_day")[1] == (30_000.0, None)
        assert _table(report, "activity.elevation_gain_by_day")[1] == (0.0, None)
        assert _table(report, "training_load.hr_trimp_by_day")[1] == (100.0, None)
        assert _matrix(report, "heart_rate.zone_matrix_by_day")[2] == (
            (60.0, None),
            (0.0, None),
            (0.0, None),
            (0.0, None),
            (0.0, None),
        )
        assert _table(report, "activity.distance_by_week")[1] == (None, 30_000.0)
        assert _table(report, "training_load.hr_trimp_by_week")[1] == (None, 100.0)
        assert _table(report, "heart_rate.zone_1_time_by_week")[1] == (None, 60.0)
        assert _table(report, "heart_rate.zone_time_overview")[1] == (
            60.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        load_axis, load_values = _table(report, "training_load.daily_hr_trimp")
        assert load_values[load_axis.index("2026-06-10")] is None
        assert load_values[load_axis.index("2026-06-11")] == 0.0
        ctl_axis, ctl_values = _table(report, "training_form.ctl_by_day")
        assert ctl_values[ctl_axis.index("2026-06-10")] is not None
        assert _scalar(report, "data_quality.activity_count") == 3
        assert _scalar(report, "data_quality.mean_hr_coverage_ratio") == pytest.approx(0.2)
        assert _scalar(report, "data_quality.mean_data_completeness") == pytest.approx(2 / 3)

    def test_quantity_dependent_metrics_are_omitted_when_no_activity_has_that_quantity(
        self,
    ) -> None:
        no_heart_rate = _record(
            "no-heart-rate",
            datetime(2026, 6, 2, 9, 0, tzinfo=TZ),
            "running",
            duration=600.0,
            distance=10_000.0,
            coverage=0.0,
            completeness=0.5,
        )
        report = overview_report((no_heart_rate,), (), athlete=ATHLETE, tz=TZ, days=14, today=TODAY)
        names = set(_names(report))

        assert (
            not {
                "training_load.hr_trimp_by_day",
                "heart_rate.zone_matrix_by_day",
                "heart_rate.zone_time_overview",
                "training_load.hr_trimp_by_week",
                "activity.hard_count_by_week",
                "heart_rate.zone_1_time_by_week",
                "training_form.ctl",
                "training_form.ctl_by_day",
                "training_load.daily_hr_trimp",
            }
            & names
        )
        assert {
            "activity.distance_by_day",
            "activity.distance_by_week",
            "tour_readiness.rolling_7d_distance",
        } <= names
        assert _scalar(report, "data_quality.activity_count") == 1
        assert _scalar(report, "data_quality.mean_hr_coverage_ratio") == 0.0

    def test_zone_cells_without_contributing_measurements_are_null(self) -> None:
        report = overview_report(
            RECORDS,
            SOURCE_INDEX,
            athlete=ATHLETE,
            tz=TZ,
            days=28,
            sport="running",
            today=TODAY,
        )

        assert _table(report, "heart_rate.zone_time_overview")[1] == (
            10.0,
            20.0,
            30.0,
            None,
            None,
        )

    def test_training_form_scalars(self) -> None:
        report = self._report()
        assert _scalar(report, "training_form.ctl") == pytest.approx(16.203992919332396)
        assert _scalar(report, "training_form.atl") == pytest.approx(30.652995043209273)
        assert _scalar(report, "training_form.tsb") == pytest.approx(-14.449002123876877)
        assert _scalar(report, "training_form.classification") == "high_fatigue"
        assert _scalar(report, "training_form.is_warmup") is False
        assert _scalar(report, "training_form.load_confidence_ratio") == 0.7457
        assert _scalar(report, "tour_readiness.rolling_7d_distance") == 80.0
        assert _scalar(report, "tour_readiness.rolling_28d_distance") == 320.0
        assert _scalar(report, "tour_readiness.days_since_60k_ride") == 3
        trend = _table(report, "training_load.daily_hr_trimp")
        assert trend[0][0] == "2026-05-19"
        assert trend[0][-1] == "2026-06-15"
        assert len(trend[1]) == 28
        assert [value for value in trend[1] if value] == [200.0, 459.0, 220.0]

    def test_data_quality_values(self) -> None:
        report = self._report()
        assert _scalar(report, "data_quality.activity_count") == 8
        assert _scalar(report, "data_quality.mean_hr_coverage_ratio") == pytest.approx(
            0.7857142857142857
        )
        assert _scalar(report, "data_quality.mean_data_completeness") == pytest.approx(
            0.9285714285714286
        )
        assert _scalar(report, "data_quality.failed_file_count") == 1
        assert _scalar(report, "data_quality.repaired_file_count") == 1
        assert _scalar(report, "data_quality.discarded_segment_count") == 1
        assert _scalar(report, "data_quality.duplicate_activity_count") == 1

    def test_sport_and_window_selection(self) -> None:
        report = overview_report(
            RECORDS, SOURCE_INDEX, athlete=ATHLETE, tz=TZ, days=7, sport="running", today=TODAY
        )
        names = set(_names(report))
        assert (
            not {
                "activity.duration_by_day",
                "activity.distance_by_day",
                "activity.elevation_gain_by_day",
                "training_load.hr_trimp_by_day",
                "heart_rate.zone_matrix_by_day",
                "heart_rate.zone_time_overview",
            }
            & names
        )
        assert _metric(report, "activity.count_by_day").value is None
        assert _metric(report, "activity.count_by_day").missing_reason is MissingReason.NO_DATA
        assert _table(report, "activity.count_by_week") == (("2026-06-01",), (1,))
        assert _scalar(report, "training_form.load_confidence_ratio") == 0.0
        assert _scalar(report, "tour_readiness.rolling_7d_distance") == 0.0
        assert _scalar(report, "tour_readiness.rolling_28d_distance") == 10.0
        assert _scalar(report, "tour_readiness.days_since_60k_ride") is None
        assert (
            _metric(report, "tour_readiness.days_since_60k_ride").missing_reason
            is MissingReason.NO_DATA
        )

    def test_empty_report(self) -> None:
        report = overview_report((), (), athlete=ATHLETE, tz=TZ, days=28, today=TODAY)
        expected = [
            name
            for name in OVERVIEW_NAMES
            if name.startswith(("athlete_profile.", "data_quality."))
            or name in {"activity.count_by_day", "activity.count_by_week"}
        ]
        assert _names(report) == expected
        assert _scalar(report, "data_quality.activity_count") == 0
        assert all(
            metric.value is None
            for metric in report.metrics
            if metric.name in DAILY_NAMES + WEEKLY_NAMES
        )
        assert all(
            metric.missing_reason is MissingReason.NO_DATA
            for metric in report.metrics
            if metric.name in DAILY_NAMES + WEEKLY_NAMES
        )


class TestTrainingFormParity:
    def test_default_values(self) -> None:
        result = training_form_from(RECORDS, tz=TZ, days=28, today=TODAY)
        assert result.current_ctl == pytest.approx(16.203992919332396)
        assert result.current_atl == pytest.approx(30.652995043209273)
        assert result.current_tsb == pytest.approx(-14.449002123876877)
        assert result.form_classification == "high_fatigue"
        assert result.is_warmup is False
        assert result.history_days == 76  # full history from 2026-04-01, not date-trimmed
        assert result.load_confidence_ratio == 0.7457
        assert result.rolling_7d_distance_km == 80.0
        assert result.rolling_28d_distance_km == 320.0
        assert result.days_since_60k_ride == 3
        assert len(result.trend_points) == 28
        assert result.trend_points[0].date == date(2026, 5, 19)
        assert result.trend_points[-1].date == date(2026, 6, 15)
        assert result.trend_points[-1].ctl == pytest.approx(result.current_ctl)

    def test_days_slices_trend_only(self) -> None:
        result = training_form_from(RECORDS, tz=TZ, days=7, today=TODAY)
        assert result.history_days == 76
        assert len(result.trend_points) == 7
        assert result.trend_points[0].date == date(2026, 6, 9)
        assert result.current_ctl == pytest.approx(16.203992919332396)

    def test_sport_filter_is_case_insensitive(self) -> None:
        result = training_form_from(RECORDS, tz=TZ, days=28, sport="CYCLING", today=TODAY)
        assert result.current_ctl == pytest.approx(13.631509517164096)
        assert result.load_confidence_ratio == 0.8795
        assert result.rolling_28d_distance_km == 310.0
        assert result.rolling_7d_distance_km == 80.0
        assert result.days_since_60k_ride == 3

        running = training_form_from(RECORDS, tz=TZ, days=28, sport="running", today=TODAY)
        assert running.history_days == 14
        assert running.load_confidence_ratio == 0.0
        assert running.rolling_7d_distance_km == 0.0
        assert running.rolling_28d_distance_km == 10.0
        assert running.days_since_60k_ride is None
        assert running.form_classification == "optimal"

    def test_empty(self) -> None:
        result = training_form_from([], tz=TZ, days=28, today=TODAY)
        assert result.current_ctl == 0.0
        assert result.current_atl == 0.0
        assert result.current_tsb == 0.0
        assert result.form_classification == "optimal"
        assert result.is_warmup is True
        assert result.history_days == 0
        assert result.trend_points == ()
        assert result.load_confidence_ratio == 1.0
        assert result.days_since_60k_ride is None
