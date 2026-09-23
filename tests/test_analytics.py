"""Tests for activity metrics and daily/weekly aggregation."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from tests.conftest import StubParser, make_app_config, make_record_frame, stub_activity_content
from tests.test_repository import T0, build_repository, two_files
from training_stats.analytics.activities import (
    ActivityMetrics,
    activity_details,
    activity_summaries,
    compute_activity_metrics,
    data_quality,
)
from training_stats.analytics.daily import (
    HARD_LOAD_THRESHOLD,
    DailySummaryRow,
    select_daily,
    summarize_daily,
)
from training_stats.analytics.weekly import (
    WeeklySummaryRow,
    aggregate_weekly,
    select_weekly,
)
from training_stats.fit.types import RECORD_SCHEMA, ParsedActivity, empty_records

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
        calories=None,
        records=records,
        source_path="x",
        source_hash="h",
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
    from training_stats.models import SourceFile

    content = stub_activity_content(
        datetime(2026, 6, 1, 10, 0, tzinfo=UTC), "cycling", 600.0, 10000.0, [120] * 5
    )
    first = parser.parse(content, SourceFile(source_name="s", path="a.fit"))
    second = parser.parse(content, SourceFile(source_name="s", path="a.fit"))
    assert first.activities[0].activity_id == second.activities[0].activity_id
    assert first.activities[0].records.height == 5


# -- cache queries: LazyFrame in, domain rows out ---------------------------


CACHE_TZ = ZoneInfo("Europe/Berlin")


class TestActivitySummaries:
    def test_newest_first(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, two_files())
        summaries = activity_summaries(repository.activities_scan(), CACHE_TZ)
        assert [item.sport for item in summaries] == ["running", "cycling"]

    def test_limit(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, two_files())
        assert len(activity_summaries(repository.activities_scan(), CACHE_TZ, limit=1)) == 1

    def test_sport_filter_case_insensitive(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, two_files())
        summaries = activity_summaries(repository.activities_scan(), CACHE_TZ, sport="RUNNING")
        assert [item.sport for item in summaries] == ["running"]

    def test_date_filtering(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, two_files())
        day = datetime(2026, 6, 3, tzinfo=UTC).date()
        summaries = activity_summaries(
            repository.activities_scan(), CACHE_TZ, start_date=day, end_date=day
        )
        assert [item.sport for item in summaries] == ["running"]

    def test_local_start_time_rendered(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, two_files())
        summaries = activity_summaries(repository.activities_scan(), CACHE_TZ)
        assert all(item.local_start_time != "" for item in summaries)


class TestActivityDetails:
    def test_found(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, two_files())
        summaries = activity_summaries(repository.activities_scan(), CACHE_TZ)
        details = activity_details(repository.activities_scan(), summaries[0].activity_id, CACHE_TZ)
        assert details is not None
        assert details.sport == summaries[0].sport
        assert details.load_method == "hr_trimp"
        assert details.source_name == "stub"
        assert details.hr_zone_1_s >= 0.0

    def test_not_found(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, two_files())
        assert activity_details(repository.activities_scan(), "deadbeef", CACHE_TZ) is None


class TestDataQuality:
    def test_values(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, two_files())
        quality = data_quality(repository.activities_scan(), repository.source_index_scan())
        assert quality.activity_count == 2
        assert quality.failed_file_count == 0
        assert quality.repaired_file_count == 0
        assert quality.discarded_segment_count == 0
        assert quality.duplicate_activity_count == 0
        assert quality.mean_hr_coverage_ratio is not None

    def test_duplicate_activity_counted(self, tmp_path: Path) -> None:
        content = stub_activity_content(T0, "cycling", 1800.0, 20000.0)
        repository = build_repository(tmp_path, {"one.fit": content, "two.fit": content})
        quality = data_quality(repository.activities_scan(), repository.source_index_scan())
        assert quality.duplicate_activity_count == 1
        assert quality.activity_count == 1

    def test_failed_file_counted(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, {"broken.fit": b"garbage"})
        quality = data_quality(repository.activities_scan(), repository.source_index_scan())
        assert quality.activity_count == 0
        assert quality.failed_file_count == 1
        assert quality.mean_hr_coverage_ratio is None
