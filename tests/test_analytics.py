"""Tests for activity metrics and daily/weekly aggregation."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from tests.conftest import StubParser, make_app_config, make_record_frame, stub_activity_content
from tests.test_repository import T0, build_repository, two_files
from training_stats.domain.analytics.activities import (
    compute_activity_metrics,
    data_quality_metrics,
)
from training_stats.domain.analytics.daily import (
    HARD_LOAD_THRESHOLD,
    DailySummaryRow,
    select_daily,
    summarize_daily,
)
from training_stats.domain.analytics.metric_helpers import table_metric
from training_stats.domain.analytics.reports import activity_list_report, activity_report
from training_stats.domain.analytics.weekly import (
    WeeklySummaryRow,
    aggregate_weekly,
    select_weekly,
)
from training_stats.domain.fit.types import RECORD_SCHEMA, ParsedActivity, empty_records
from training_stats.domain.metrics import (
    AllTimePeriod,
    Axis,
    Confidence,
    Metric,
    MetricKind,
    MetricMethod,
    MissingReason,
    ScalarValue,
    TableValue,
    TimeBasis,
    TripPeriod,
)
from training_stats.domain.models import ActivityRecord


def _parsed(
    start: datetime,
    sport: str = "cycling",
    heart_rates: list[int] | None = None,
    duration_s: float = 600.0,
    activity_id: str = "a1",
) -> ParsedActivity:
    if heart_rates is not None:
        records = make_record_frame(start, heart_rates, interval_s=1)
    else:
        records = empty_records()
    return ParsedActivity(
        activity_id=activity_id,
        start_time=start,
        sport=sport,
        sub_sport=None,
        duration_s=duration_s,
        moving_time_s=None,
        distance_m=1000.0,
        elevation_gain_m=None,
        avg_hr=None,
        max_hr=None,
        avg_speed_mps=None,
        max_speed_mps=None,
        records=records,
        source_path="x",
    )


# Cached metric facts the aggregate analytics read back from ActivityRecords.
_ABSENT = object()


def _cached_scalar(name: str, period: TripPeriod, value: object, unit: str | None) -> Metric:
    return Metric(
        name=name,
        description=f"Cached {name} value for aggregate analytics tests.",
        period=period,
        value=ScalarValue(value) if value is not None else None,  # type: ignore[arg-type]
        kind=MetricKind.CALCULATED,
        unit=unit,
        missing_reason=None if value is not None else MissingReason.NO_DATA,
    )


def _record(
    activity_id: str,
    start: datetime,
    *,
    sport: str = "cycling",
    duration_s: object = 600.0,
    distance_m: object = 1000.0,
    elevation_gain_m: object = 0.0,
    zones: tuple[float, ...] | None = (0.0, 10.0, 0.0, 0.0, 0.0),
    load_value: object = None,
    load_confidence: str | None = None,
    coverage: float | None = None,
) -> ActivityRecord:
    """Build a metric-bearing cached record for aggregate analytics tests."""
    period = TripPeriod(activity_id=activity_id, start=start)
    metrics: list[Metric] = []
    for name, value, unit in (
        ("activity.duration", duration_s, "s"),
        ("activity.distance", distance_m, "m"),
        ("activity.elevation_gain", elevation_gain_m, "m"),
    ):
        if value is not _ABSENT:
            metrics.append(_cached_scalar(name, period, value, unit))
    if load_value is not _ABSENT:
        metrics.append(
            Metric(
                name="training_load.hr_trimp",
                description="Estimated Banister heart-rate load for aggregate tests.",
                period=period,
                value=ScalarValue(load_value) if load_value is not None else None,  # type: ignore[arg-type]
                kind=MetricKind.ESTIMATED,
                unit="TRIMP",
                missing_reason=None if load_value is not None else MissingReason.NO_DATA,
                method=MetricMethod.HR_TRIMP,
                confidence=Confidence(load_confidence) if load_confidence is not None else None,
            )
        )
    if zones is not None:
        metrics.append(
            table_metric(
                "heart_rate.zone_time",
                period,
                Axis(
                    name="heart_rate_zone",
                    values=tuple(f"zone_{index}" for index in range(1, len(zones) + 1)),
                ),
                zones,
                description="Active time assigned to each configured heart-rate zone.",
                unit="s",
                time_basis=TimeBasis.ACTIVE_TIME,
            )
        )
    if coverage is not None:
        metrics.append(
            _cached_scalar("heart_rate.coverage", period, coverage, "ratio"),
        )
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
        analysis_version="1",
        metrics=tuple(metrics),
    )


def _scalar(metrics: tuple[Metric, ...], name: str) -> object:
    metric = next(item for item in metrics if item.name == name)
    value = metric.value
    return value.value if isinstance(value, ScalarValue) else None


def _table(metrics: tuple[Metric, ...], name: str) -> TableValue | None:
    metric = next(item for item in metrics if item.name == name)
    value = metric.value
    return value if isinstance(value, TableValue) else None


class TestActivityMetrics:
    def test_zones_and_load(self) -> None:
        config = make_app_config(Path("/tmp/x1"), Path("/tmp/x2"))
        start = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
        # One second apart -> no interval cap applied: 300 intervals in zone 2,
        # 299 in zone 4 (each interval belongs to the zone of its start record).
        activity = _parsed(start, heart_rates=[120] * 300 + [150] * 300)
        metrics = compute_activity_metrics(activity, config.athlete, "1")
        assert all(metric.description.strip() for metric in metrics)
        zones = _table(metrics, "heart_rate.zone_time")
        assert zones is not None
        assert zones.values[1] == pytest.approx(300.0)
        assert zones.values[3] == pytest.approx(299.0)
        load = next(metric for metric in metrics if metric.name == "training_load.hr_trimp")
        assert _scalar(metrics, "training_load.hr_trimp") is not None
        assert float(_scalar(metrics, "training_load.hr_trimp")) > 0
        assert load.method == "hr_trimp"
        assert load.confidence == "high"
        assert _scalar(metrics, "heart_rate.coverage") == pytest.approx(1.0)

    def test_no_hr_data(self) -> None:
        config = make_app_config(Path("/tmp/x1"), Path("/tmp/x2"))
        activity = _parsed(datetime(2026, 6, 1, tzinfo=UTC))
        metrics = compute_activity_metrics(activity, config.athlete, "1")
        load = next(metric for metric in metrics if metric.name == "training_load.hr_trimp")
        histogram = next(metric for metric in metrics if metric.name == "heart_rate.histogram")
        assert load.value is None
        assert load.method == "hr_trimp"
        assert _scalar(metrics, "heart_rate.coverage") == 0.0
        assert histogram.value is None
        assert _scalar(metrics, "heart_rate.drift") is None

    def test_histogram_and_buckets_present(self) -> None:
        config = make_app_config(Path("/tmp/x1"), Path("/tmp/x2"))
        start = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
        activity = _parsed(start, heart_rates=[130] * 7200)  # 2 hours at 1 s
        metrics = compute_activity_metrics(activity, config.athlete, "1")
        histogram = _table(metrics, "heart_rate.histogram")
        buckets = _table(metrics, "heart_rate.average_by_10_minutes")
        assert histogram is not None
        assert histogram.axis.values == ("130-134",)
        assert histogram.values == pytest.approx((7199.0,))
        assert buckets is not None
        assert buckets.axis.values == tuple(range(0, 120, 10))
        drift = _scalar(metrics, "heart_rate.drift")
        assert drift is not None
        assert abs(float(drift)) < 1e-9  # constant HR: no drift

    def test_analysis_version_included(self) -> None:
        config = make_app_config(Path("/tmp/x1"), Path("/tmp/x2"))
        activity = _parsed(datetime(2026, 6, 1, tzinfo=UTC))
        metrics = compute_activity_metrics(activity, config.athlete, "3")
        assert _scalar(metrics, "analysis.version") == "3"

    def test_decoupling_computed(self) -> None:
        config = make_app_config(Path("/tmp/x1"), Path("/tmp/x2"))
        start = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
        # 1400 s with speed and HR
        rows = []
        t = start
        for _ in range(1400):
            rows.append({"timestamp": t, "heart_rate": 130, "speed_mps": 5.0})
            t += timedelta(seconds=1)
        records = pl.DataFrame(rows, schema=RECORD_SCHEMA)
        activity = ParsedActivity(
            activity_id="a-decoupling",
            start_time=start,
            sport="cycling",
            sub_sport=None,
            duration_s=1400.0,
            moving_time_s=1400.0,
            distance_m=7000.0,
            elevation_gain_m=0.0,
            avg_hr=130.0,
            max_hr=130.0,
            avg_speed_mps=5.0,
            max_speed_mps=5.0,
            records=records,
            source_path="x",
        )
        metrics = compute_activity_metrics(activity, config.athlete, "1")
        assert _scalar(metrics, "endurance.efficiency_first_half") is not None
        assert _scalar(metrics, "endurance.efficiency_second_half") is not None
        assert _scalar(metrics, "endurance.aerobic_decoupling") == pytest.approx(0.0)


class TestSummarizeDaily:
    """Golden semantics of the cached-record daily grouping."""

    TZ = ZoneInfo("Europe/Berlin")

    def test_local_date_conversion(self) -> None:
        # 23:40 UTC is already the next day in Berlin (UTC+2 in summer).
        start = datetime(2026, 6, 1, 23, 40, tzinfo=UTC)
        rows = summarize_daily([_record("a1", start)], self.TZ)
        assert rows[0].date == "2026-06-02"

    def test_grouping_by_sport(self) -> None:
        start = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
        rows = summarize_daily(
            [
                _record("a1", start, sport="cycling", load_value=100.0),
                _record("a2", start, sport="running", load_value=200.0),
            ],
            self.TZ,
        )
        assert [(row.date, row.sport) for row in rows] == [
            ("2026-06-01", "cycling"),
            ("2026-06-01", "running"),
        ]

    def test_grouping_is_case_sensitive(self) -> None:
        start = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
        rows = summarize_daily(
            [
                _record("a1", start, sport="Cycling"),
                _record("a2", start, sport="cycling"),
            ],
            self.TZ,
        )
        assert [(row.date, row.sport) for row in rows] == [
            ("2026-06-01", "Cycling"),
            ("2026-06-01", "cycling"),
        ]

    def test_hard_activity_threshold(self) -> None:
        start = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
        rows = summarize_daily(
            [
                _record("a1", start, load_value=HARD_LOAD_THRESHOLD),
                _record("a2", start + timedelta(hours=2), load_value=HARD_LOAD_THRESHOLD - 1),
            ],
            self.TZ,
        )
        assert rows[0].hard_activity_count == 1

    def test_longest_activity(self) -> None:
        start = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
        rows = summarize_daily(
            [
                _record("a1", start),
                _record("a2", start + timedelta(hours=2), duration_s=3600.0),
            ],
            self.TZ,
        )
        assert rows[0].longest_activity_id == "a2"
        assert rows[0].longest_activity_s == 3600.0

    def test_longest_activity_tie_breaks_to_sorted_first(self) -> None:
        start = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
        # Input order differs from (start_time, activity_id) order; the
        # first-seen rule resolves ties deterministically.
        rows = summarize_daily(
            [
                _record("b2", start),
                _record("a1", start),
            ],
            self.TZ,
        )
        assert rows[0].longest_activity_id == "a1"

    def test_zero_duration_has_no_longest(self) -> None:
        start = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
        rows = summarize_daily([_record("a1", start, duration_s=0.0)], self.TZ)
        assert rows[0].longest_activity_id is None
        assert rows[0].longest_activity_s == 0.0

    def test_null_load_excluded(self) -> None:
        start = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
        rows = summarize_daily(
            [
                _record("a1", start, load_value=None),
                _record("a2", start, load_value=200.0),
            ],
            self.TZ,
        )
        assert rows[0].load_value == 200.0
        assert rows[0].hard_activity_count == 1
        assert rows[0].activity_count == 2

    def test_missing_metrics_are_counted_as_zero(self) -> None:
        start = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
        rows = summarize_daily(
            [
                _record(
                    "a1",
                    start,
                    duration_s=_ABSENT,
                    distance_m=_ABSENT,
                    elevation_gain_m=_ABSENT,
                    load_value=_ABSENT,
                    zones=None,
                    coverage=0.5,
                ),
                _record("a2", start, duration_s=600.0),
            ],
            self.TZ,
        )
        assert rows[0].activity_count == 2
        assert rows[0].duration_s == 600.0
        assert rows[0].distance_m == 1000.0
        assert rows[0].elevation_gain_m == 0.0
        assert rows[0].load_value == 0.0
        assert rows[0].hard_activity_count == 0
        assert rows[0].hr_zone_1_s == 0.0

    def test_non_numeric_metric_values_rejected(self) -> None:
        start = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
        rows = summarize_daily(
            [
                _record(
                    "a1",
                    start,
                    duration_s="600",
                    elevation_gain_m=True,
                    load_value="150",
                    zones=(True, 10.0),
                ),
            ],
            self.TZ,
        )
        assert rows[0].duration_s == 0.0
        assert rows[0].elevation_gain_m == 0.0
        assert rows[0].load_value == 0.0
        assert rows[0].hard_activity_count == 0
        assert rows[0].hr_zone_1_s == 0.0
        assert rows[0].hr_zone_2_s == 10.0
        assert rows[0].longest_activity_id is None

    def test_zone_cells_missing_from_table_default_to_zero(self) -> None:
        start = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
        rows = summarize_daily([_record("a1", start, zones=(42.0,))], self.TZ)
        assert (
            rows[0].hr_zone_1_s,
            rows[0].hr_zone_2_s,
            rows[0].hr_zone_3_s,
            rows[0].hr_zone_4_s,
            rows[0].hr_zone_5_s,
        ) == (42.0, 0.0, 0.0, 0.0, 0.0)

    def test_input_order_does_not_change_rows(self) -> None:
        start = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
        records = [
            _record("b2", start, duration_s=900.0),
            _record("a1", start + timedelta(hours=1), duration_s=900.0),
            _record("c3", start, sport="running", duration_s=60.0),
        ]
        assert summarize_daily(records, self.TZ) == summarize_daily(
            list(reversed(records)), self.TZ
        )

    def test_empty(self) -> None:
        assert summarize_daily([], self.TZ) == []
        assert summarize_daily((), self.TZ) == []


class TestSelectDaily:
    TZ = ZoneInfo("Europe/Berlin")
    START = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)

    def _rows(self) -> list[DailySummaryRow]:
        return summarize_daily(
            [
                _record("a1", self.START, sport="cycling", load_value=100.0),
                _record("a2", self.START, sport="running", load_value=50.0),
                _record("a3", self.START + timedelta(days=1), sport="cycling", load_value=10.0),
            ],
            self.TZ,
        )

    def test_sport_filter_keeps_per_sport_rows(self) -> None:
        rows = select_daily(self._rows(), sport="CYCLING")
        assert [(row.date, row.sport) for row in rows] == [
            ("2026-06-01", "cycling"),
            ("2026-06-02", "cycling"),
        ]

    def test_no_sport_merges_to_all(self) -> None:
        rows = select_daily(self._rows())
        assert [(row.date, row.sport) for row in rows] == [
            ("2026-06-01", "all"),
            ("2026-06-02", "all"),
        ]
        first = rows[0]
        assert first.activity_count == 2
        assert first.load_value == 150.0

    def test_merge_picks_longest(self) -> None:
        rows = select_daily(self._rows())
        day_one = next(row for row in rows if row.date == "2026-06-01")
        # 600 s running vs 600 s cycling: max of equal values keeps the first
        # (deterministic) longest id, which is a1 in sorted order.
        assert day_one.longest_activity_s == 600.0

    def test_date_range(self) -> None:
        rows = select_daily(self._rows(), start_date=date(2026, 6, 2))
        assert [row.date for row in rows] == ["2026-06-02"]


class TestSelectWeekly:
    def _weekly(self) -> list[WeeklySummaryRow]:
        return [
            _weekly("2026-05-25", 22, "cycling", 100.0, prev=None),
            _weekly("2026-06-01", 23, "cycling", 200.0, prev=100.0),
            _weekly("2026-05-25", 22, "running", 50.0, prev=None),
            _weekly("2026-06-08", 24, "cycling", 300.0, prev=200.0),
        ]

    def test_weeks_truncation_newest_first(self) -> None:
        rows = select_weekly(self._weekly(), weeks=2)
        assert [(row.iso_year, row.iso_week) for row in rows] == [(2026, 24), (2026, 23)]
        # sport=None merges the two week-22 sports into sport="all".
        assert rows[0].sport == "all"

    def test_weeks_truncation_ignored_with_date_bounds(self) -> None:
        rows = select_weekly(
            self._weekly(), start_date=date(2026, 1, 1), end_date=date(2026, 12, 31), weeks=1
        )
        assert len(rows) == 3

    def test_sport_filter(self) -> None:
        rows = select_weekly(self._weekly(), sport="cycling", weeks=4)
        assert all(row.sport == "cycling" for row in rows)
        assert [(row.iso_year, row.iso_week) for row in rows] == [
            (2026, 24),
            (2026, 23),
            (2026, 22),
        ]


class TestAggregateWeekly:
    def test_aggregate_weekly_merges_sports_and_prev(self) -> None:
        rows = [
            _weekly("2026-05-25", 22, "cycling", 100.0, prev=10.0),
            _weekly("2026-05-25", 22, "running", 50.0, prev=None),
        ]
        merged = aggregate_weekly(rows)
        assert len(merged) == 1
        row = merged[0]
        assert row.sport == "all"
        assert row.load_value == 150.0
        assert row.load_value_prev == 10.0
        assert row.week_start == "2026-05-25"


def _weekly(
    week_start: str, iso_week: int, sport: str, load: float, prev: float | None
) -> WeeklySummaryRow:
    return WeeklySummaryRow(
        iso_year=2026,
        iso_week=iso_week,
        week_start=week_start,
        sport=sport,
        activity_count=1,
        duration_s=1800.0,
        distance_m=10000.0,
        elevation_gain_m=50.0,
        hr_zone_1_s=0.0,
        hr_zone_2_s=600.0,
        hr_zone_3_s=0.0,
        hr_zone_4_s=0.0,
        hr_zone_5_s=0.0,
        load_value=load,
        hard_activity_count=0,
        longest_activity_s=1800.0,
        longest_activity_id=None,
        load_value_prev=prev,
    )


def _daily(date: str, load: float) -> DailySummaryRow:
    return DailySummaryRow(
        date=date,
        sport="cycling",
        activity_count=1,
        duration_s=1800.0,
        distance_m=10000.0,
        elevation_gain_m=50.0,
        hr_zone_1_s=0.0,
        hr_zone_2_s=600.0,
        hr_zone_3_s=1200.0,
        hr_zone_4_s=0.0,
        hr_zone_5_s=0.0,
        load_value=load,
        hard_activity_count=0,
        longest_activity_s=1800.0,
        longest_activity_id=None,
    )


def test_stub_parser_roundtrip() -> None:
    """The conftest StubParser is deterministic."""
    parser = StubParser()
    from training_stats.domain.models import SourceFile

    content = stub_activity_content(
        datetime(2026, 6, 1, 10, 0, tzinfo=UTC), "cycling", 600.0, 10000.0, [120] * 5
    )
    first = parser.parse(content, SourceFile(source_name="s", path="a.fit"))
    second = parser.parse(content, SourceFile(source_name="s", path="a.fit"))
    assert first.frame.equals(second.frame)
    assert first.frame.filter(pl.col("message_type") == "record").height > 5


CACHE_TZ = ZoneInfo("Europe/Berlin")


class TestActivityMetricReports:
    def test_newest_first(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, two_files())
        records = tuple(repository.load_sync_state().activities.values())
        report = activity_list_report(records, CACHE_TZ)
        trip_ids = [
            metric.period.activity_id
            for metric in report.metrics
            if metric.name == "activity.sport" and isinstance(metric.period, TripPeriod)
        ]
        assert [
            next(row.sport for row in records if row.activity_id == item) for item in trip_ids
        ] == [
            "running",
            "cycling",
        ]

    def test_limit(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, two_files())
        records = tuple(repository.load_sync_state().activities.values())
        report = activity_list_report(records, CACHE_TZ, limit=1)
        assert len({metric.period.activity_id for metric in report.metrics}) == 1

    def test_sport_filter_case_insensitive(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, two_files())
        records = tuple(repository.load_sync_state().activities.values())
        report = activity_list_report(records, CACHE_TZ, sport="RUNNING")
        assert len({metric.period.activity_id for metric in report.metrics}) == 1
        assert _scalar(report.metrics, "activity.sport") == "running"

    def test_date_filtering(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, two_files())
        records = tuple(repository.load_sync_state().activities.values())
        day = datetime(2026, 6, 3, tzinfo=UTC).date()
        report = activity_list_report(records, CACHE_TZ, start_date=day, end_date=day)
        assert len({metric.period.activity_id for metric in report.metrics}) == 1
        assert _scalar(report.metrics, "activity.sport") == "running"

    def test_activity_report_contains_cached_metrics(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, two_files())
        records = tuple(repository.load_sync_state().activities.values())
        report = activity_report(records, records[0].activity_id)
        assert report is not None
        assert all(metric.period.activity_id == records[0].activity_id for metric in report.metrics)
        assert _scalar(report.metrics, "training_load.hr_trimp") is not None
        assert activity_report(records, "deadbeef") is None


class TestDataQuality:
    def test_values(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, two_files())
        state = repository.load_sync_state()
        metrics = data_quality_metrics(
            tuple(state.activities.values()), tuple(state.index_rows), AllTimePeriod()
        )
        assert _scalar(metrics, "data_quality.activity_count") == 2
        assert _scalar(metrics, "data_quality.failed_file_count") == 0
        assert _scalar(metrics, "data_quality.duplicate_activity_count") == 0
        assert _scalar(metrics, "data_quality.mean_hr_coverage_ratio") is not None

    def test_duplicate_activity_counted(self, tmp_path: Path) -> None:
        content = stub_activity_content(T0, "cycling", 1800.0, 20000.0)
        repository = build_repository(tmp_path, {"one.fit": content, "two.fit": content})
        state = repository.load_sync_state()
        metrics = data_quality_metrics(
            tuple(state.activities.values()), tuple(state.index_rows), AllTimePeriod()
        )
        assert _scalar(metrics, "data_quality.duplicate_activity_count") == 1
        assert _scalar(metrics, "data_quality.activity_count") == 1

    def test_failed_file_counted(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, {"broken.fit": b"garbage"})
        state = repository.load_sync_state()
        metrics = data_quality_metrics(
            tuple(state.activities.values()), tuple(state.index_rows), AllTimePeriod()
        )
        assert _scalar(metrics, "data_quality.activity_count") == 0
        assert _scalar(metrics, "data_quality.failed_file_count") == 1
        assert _scalar(metrics, "data_quality.mean_hr_coverage_ratio") is None
