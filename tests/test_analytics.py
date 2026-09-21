"""Tests for activity metrics and daily/weekly aggregation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from tests.conftest import StubParser, make_app_config, make_record_frame, stub_activity_content
from training_coach_mcp.analytics.activities import (
    ActivityMetrics,
    AnalyzedActivity,
    compute_activity_metrics,
)
from training_coach_mcp.analytics.daily import (
    HARD_LOAD_THRESHOLD,
    DailySummaryRow,
    summarize_daily,
)
from training_coach_mcp.analytics.weekly import daily_to_weekly
from training_coach_mcp.fit.types import RECORD_SCHEMA, ParsedActivity, empty_records


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
        calories=None,
        records=records,
        source_path="x",
        source_hash="h",
    )


class TestActivityMetrics:
    def test_zones_and_load(self) -> None:
        config = make_app_config(Path("/tmp/x1"), Path("/tmp/x2"))
        start = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
        # One second apart -> no interval cap applied: 300 intervals in zone 2,
        # 299 in zone 4 (each interval belongs to the zone of its start record).
        activity = _parsed(start, heart_rates=[120] * 300 + [150] * 300)
        metrics = compute_activity_metrics(activity, config.athlete, "1")
        assert metrics.hr_zone_2_s == pytest.approx(300.0)
        assert metrics.hr_zone_4_s == pytest.approx(299.0)
        assert metrics.load_value is not None
        assert metrics.load_value > 0
        assert metrics.load_method == "hr_trimp"
        assert metrics.load_confidence == "high"
        assert metrics.hr_coverage_ratio == pytest.approx(1.0)

    def test_no_hr_data(self) -> None:
        config = make_app_config(Path("/tmp/x1"), Path("/tmp/x2"))
        activity = _parsed(datetime(2026, 6, 1, tzinfo=UTC))
        metrics = compute_activity_metrics(activity, config.athlete, "1")
        assert metrics.load_value is None
        assert metrics.load_method is None
        assert metrics.hr_coverage_ratio == 0.0
        assert metrics.hr_histogram_s == {}
        assert metrics.hr_10min_averages == ()
        assert metrics.hr_drift_bpm_per_hour is None

    def test_histogram_and_buckets_present(self) -> None:
        config = make_app_config(Path("/tmp/x1"), Path("/tmp/x2"))
        start = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
        activity = _parsed(start, heart_rates=[130] * 7200)  # 2 hours at 1 s
        metrics = compute_activity_metrics(activity, config.athlete, "1")
        assert metrics.hr_histogram_s == {"130-134": 7199.0}
        assert [bucket.start_min for bucket in metrics.hr_10min_averages] == list(range(0, 120, 10))
        assert metrics.hr_drift_bpm_per_hour is not None
        assert abs(metrics.hr_drift_bpm_per_hour) < 1e-9  # constant HR: no drift

    def test_analysis_version_included(self) -> None:
        config = make_app_config(Path("/tmp/x1"), Path("/tmp/x2"))
        activity = _parsed(datetime(2026, 6, 1, tzinfo=UTC))
        metrics = compute_activity_metrics(activity, config.athlete, "3")
        assert metrics.analysis_version == "3:1"

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
            calories=200.0,
            records=records,
            source_path="x",
            source_hash="h",
        )
        metrics = compute_activity_metrics(activity, config.athlete, "1")
        assert metrics.decoupling_confidence == "high"
        assert metrics.ef_first_half is not None
        assert metrics.ef_second_half is not None
        assert metrics.aerobic_decoupling_pct == pytest.approx(0.0)


class TestDailyAggregation:
    TZ = ZoneInfo("Europe/Berlin")

    def test_local_date_conversion(self) -> None:
        # 23:40 UTC is already the next day in Berlin (UTC+2 in summer).
        start = datetime(2026, 6, 1, 23, 40, tzinfo=UTC)
        items = [AnalyzedActivity(activity=_parsed(start), metrics=_metrics("a1"))]
        rows = summarize_daily(items, self.TZ)
        assert rows[0].date == "2026-06-02"

    def test_grouping_by_sport(self) -> None:
        start = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
        items = [
            AnalyzedActivity(
                activity=_parsed(start, "cycling"), metrics=_metrics("a1", load=100.0)
            ),
            AnalyzedActivity(
                activity=_parsed(start, "running"), metrics=_metrics("a2", load=200.0)
            ),
        ]
        rows = summarize_daily(items, self.TZ)
        assert [(row.date, row.sport) for row in rows] == [
            ("2026-06-01", "cycling"),
            ("2026-06-01", "running"),
        ]

    def test_hard_activity_threshold(self) -> None:
        start = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
        items = [
            AnalyzedActivity(
                activity=_parsed(start), metrics=_metrics("a1", load=HARD_LOAD_THRESHOLD)
            ),
            AnalyzedActivity(
                activity=_parsed(start + timedelta(hours=2)),
                metrics=_metrics("a2", load=HARD_LOAD_THRESHOLD - 1),
            ),
        ]
        rows = summarize_daily(items, self.TZ)
        assert rows[0].hard_activity_count == 1

    def test_longest_activity(self) -> None:
        start = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
        items = [
            AnalyzedActivity(activity=_parsed(start), metrics=_metrics("a1")),
            AnalyzedActivity(
                activity=_parsed(start + timedelta(hours=2), duration_s=3600.0, activity_id="a2"),
                metrics=_metrics("a2"),
            ),
        ]
        rows = summarize_daily(items, self.TZ)
        assert rows[0].longest_activity_id == "a2"
        assert rows[0].longest_activity_s == 3600.0

    def test_empty(self) -> None:
        assert summarize_daily([], self.TZ) == []


class TestWeeklyAggregation:
    TZ = ZoneInfo("Europe/Berlin")

    def test_daily_to_weekly_year_boundary(self) -> None:
        # 2026-12-28 is Monday of ISO week 53 of 2026; 2027-01-04 starts
        # ISO week 1 of 2027, whose previous week is 2026-W53.
        rows = [
            _daily("2026-12-28", 100.0),
            _daily("2026-12-29", 50.0),
            _daily("2027-01-04", 200.0),
            _daily("2027-01-05", 60.0),
        ]
        weekly = daily_to_weekly(rows)
        assert [(row.iso_year, row.iso_week) for row in weekly] == [(2026, 53), (2027, 1)]
        first_week = weekly[0]
        assert first_week.load_value == 150.0
        assert first_week.load_value_prev is None
        # Week 1 of 2027 follows week 53 of 2026 across the year boundary.
        second_week = weekly[1]
        assert second_week.load_value == 260.0
        assert second_week.load_value_prev == 150.0

    def test_week_start_is_monday(self) -> None:
        weekly = daily_to_weekly([_daily("2026-06-03", 50.0)])  # Wednesday
        assert weekly[0].week_start == "2026-06-01"


def _metrics(activity_id: str, load: float | None = None) -> ActivityMetrics:
    return ActivityMetrics(
        activity_id=activity_id,
        hr_zone_1_s=0.0,
        hr_zone_2_s=10.0,
        hr_zone_3_s=0.0,
        hr_zone_4_s=0.0,
        hr_zone_5_s=0.0,
        hr_coverage_ratio=1.0,
        hr_histogram_s={},
        hr_10min_averages=(),
        hr_drift_bpm_per_hour=None,
        load_value=load,
        load_method="hr_trimp" if load else None,
        load_confidence="high" if load else None,
        data_completeness=1.0,
        analysis_version="1:1",
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
    from training_coach_mcp.models import SourceFile

    content = stub_activity_content(
        datetime(2026, 6, 1, 10, 0, tzinfo=UTC), "cycling", 600.0, 10000.0, [120] * 5
    )
    first = parser.parse(content, SourceFile(source_name="s", path="a.fit"))
    second = parser.parse(content, SourceFile(source_name="s", path="a.fit"))
    assert first.activities[0].activity_id == second.activities[0].activity_id
    assert first.activities[0].records.height == 5
