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
from training_stats.domain.analytics.reports import activity_list_report, activity_report
from training_stats.domain.analytics.weekly import (
    WeeklySummaryRow,
    aggregate_weekly,
    select_weekly,
)
from training_stats.domain.fit.types import RECORD_SCHEMA, ParsedActivity, empty_records
from training_stats.domain.metrics import (
    AllTimePeriod,
    Metric,
    ScalarValue,
    TableValue,
    TripPeriod,
)

# Columns summarize_daily and its selection helpers consume.
ACTIVITY_FRAME_SCHEMA: dict[str, pl.DataType] = {
    "activity_id": pl.String(),
    "start_time": pl.Datetime("us", "UTC"),
    "sport": pl.String(),
    "duration_s": pl.Float64(),
    "distance_m": pl.Float64(),
    "elevation_gain_m": pl.Float64(),
    "hr_zone_1_s": pl.Float64(),
    "hr_zone_2_s": pl.Float64(),
    "hr_zone_3_s": pl.Float64(),
    "hr_zone_4_s": pl.Float64(),
    "hr_zone_5_s": pl.Float64(),
    "load_value": pl.Float64(),
}


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


def _activity_row(
    activity_id: str,
    start: datetime,
    *,
    sport: str = "cycling",
    duration_s: float = 600.0,
    distance_m: float = 1000.0,
    elevation_gain_m: float = 0.0,
    zones: tuple[float, float, float, float, float] = (0.0, 10.0, 0.0, 0.0, 0.0),
    load_value: float | None = None,
) -> dict[str, object]:
    return {
        "activity_id": activity_id,
        "start_time": start,
        "sport": sport,
        "duration_s": duration_s,
        "distance_m": distance_m,
        "elevation_gain_m": elevation_gain_m,
        "hr_zone_1_s": zones[0],
        "hr_zone_2_s": zones[1],
        "hr_zone_3_s": zones[2],
        "hr_zone_4_s": zones[3],
        "hr_zone_5_s": zones[4],
        "load_value": load_value,
    }


def _activity_frame(rows: list[dict[str, object]]) -> pl.LazyFrame:
    return pl.DataFrame(rows, schema=ACTIVITY_FRAME_SCHEMA).lazy()


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
    """Golden semantics of the vectorized daily grouping."""

    TZ = ZoneInfo("Europe/Berlin")

    def test_local_date_conversion(self) -> None:
        # 23:40 UTC is already the next day in Berlin (UTC+2 in summer).
        start = datetime(2026, 6, 1, 23, 40, tzinfo=UTC)
        rows = summarize_daily(_activity_frame([_activity_row("a1", start)]), self.TZ)
        assert rows[0].date == "2026-06-02"

    def test_grouping_by_sport(self) -> None:
        start = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
        rows = summarize_daily(
            _activity_frame(
                [
                    _activity_row("a1", start, sport="cycling", load_value=100.0),
                    _activity_row("a2", start, sport="running", load_value=200.0),
                ]
            ),
            self.TZ,
        )
        assert [(row.date, row.sport) for row in rows] == [
            ("2026-06-01", "cycling"),
            ("2026-06-01", "running"),
        ]

    def test_hard_activity_threshold(self) -> None:
        start = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
        rows = summarize_daily(
            _activity_frame(
                [
                    _activity_row("a1", start, load_value=HARD_LOAD_THRESHOLD),
                    _activity_row(
                        "a2", start + timedelta(hours=2), load_value=HARD_LOAD_THRESHOLD - 1
                    ),
                ]
            ),
            self.TZ,
        )
        assert rows[0].hard_activity_count == 1

    def test_longest_activity(self) -> None:
        start = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
        rows = summarize_daily(
            _activity_frame(
                [
                    _activity_row("a1", start),
                    _activity_row("a2", start + timedelta(hours=2), duration_s=3600.0),
                ]
            ),
            self.TZ,
        )
        assert rows[0].longest_activity_id == "a2"
        assert rows[0].longest_activity_s == 3600.0

    def test_longest_activity_tie_breaks_to_sorted_first(self) -> None:
        start = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
        # Input order differs from (start_time, activity_id) order; the
        # first-seen rule resolves ties deterministically.
        rows = summarize_daily(
            _activity_frame(
                [
                    _activity_row("b2", start),
                    _activity_row("a1", start),
                ]
            ),
            self.TZ,
        )
        assert rows[0].longest_activity_id == "a1"

    def test_zero_duration_has_no_longest(self) -> None:
        start = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
        rows = summarize_daily(
            _activity_frame([_activity_row("a1", start, duration_s=0.0)]), self.TZ
        )
        assert rows[0].longest_activity_id is None
        assert rows[0].longest_activity_s == 0.0

    def test_null_load_excluded(self) -> None:
        start = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
        rows = summarize_daily(
            _activity_frame(
                [
                    _activity_row("a1", start, load_value=None),
                    _activity_row("a2", start, load_value=200.0),
                ]
            ),
            self.TZ,
        )
        assert rows[0].load_value == 200.0
        assert rows[0].hard_activity_count == 1
        assert rows[0].activity_count == 2

    def test_empty(self) -> None:
        assert summarize_daily(_activity_frame([]), self.TZ) == []


class TestSelectDaily:
    TZ = ZoneInfo("Europe/Berlin")
    START = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)

    def _rows(self) -> list[DailySummaryRow]:
        return summarize_daily(
            _activity_frame(
                [
                    _activity_row("a1", self.START, sport="cycling", load_value=100.0),
                    _activity_row("a2", self.START, sport="running", load_value=50.0),
                    _activity_row(
                        "a3", self.START + timedelta(days=1), sport="cycling", load_value=10.0
                    ),
                ]
            ),
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
