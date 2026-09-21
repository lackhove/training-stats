"""Tests for the Parquet-backed repository."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from tests.conftest import StubParser, StubSource, make_app_config, stub_activity_content
from training_coach_mcp.cache.repository import ParquetTrainingRepository
from training_coach_mcp.cache.sync import SyncService

T0 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)


def build_repository(tmp_path: Path, files: dict[str, bytes]) -> ParquetTrainingRepository:
    source = StubSource(files=files)
    cache_dir = tmp_path / "cache"
    config = make_app_config(cache_dir, tmp_path / "src")
    service = SyncService(
        sources=[source],
        parser=StubParser(),
        cache_dir=cache_dir,
        athlete=config.athlete,
        timezone=config.application.zoneinfo,
        analysis_version="1",
    )
    service.sync()
    return ParquetTrainingRepository(
        cache_dir=cache_dir, timezone=ZoneInfo("Europe/Berlin"), athlete=config.athlete
    )


def two_files() -> dict[str, bytes]:
    return {
        "easy.fit": stub_activity_content(T0, "cycling", 3600.0, 30000.0, heart_rates=[120] * 60),
        "intervals.fit": stub_activity_content(
            T0 + timedelta(days=2), "running", 1800.0, 8000.0, heart_rates=[150] * 30
        ),
    }


class TestEmptyCache:
    def test_all_queries_return_empty(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, {})
        assert repository.list_activities() == []
        assert repository.get_activity("nope") is None
        assert repository.daily_summary() == []
        assert repository.weekly_summary() == []
        form = repository.training_form()
        assert form.current_ctl == 0.0
        assert form.days_since_60k_ride is None
        quality = repository.data_quality()
        assert quality.activity_count == 0
        overview = repository.overview()
        assert overview.daily_series == ()


class TestListActivities:
    def test_newest_first(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, two_files())
        activities = repository.list_activities()
        assert [item.sport for item in activities] == ["running", "cycling"]

    def test_limit(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, two_files())
        assert len(repository.list_activities(limit=1)) == 1

    def test_sport_filter_case_insensitive(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, two_files())
        activities = repository.list_activities(sport="RUNNING")
        assert [item.sport for item in activities] == ["running"]

    def test_date_filtering(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, two_files())
        activities = repository.list_activities(
            start_date=date(2026, 6, 3), end_date=date(2026, 6, 3)
        )
        assert [item.sport for item in activities] == ["running"]

    def test_load_joined(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, two_files())
        activities = repository.list_activities()
        assert all(item.load_value is not None and item.load_value > 0 for item in activities)


class TestGetActivity:
    def test_found(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, two_files())
        activities = repository.list_activities()
        details = repository.get_activity(activities[0].activity_id)
        assert details is not None
        assert details.sport == activities[0].sport
        assert details.load_method == "hr_trimp"
        assert details.source_name == "stub"
        assert details.hr_zone_1_s >= 0.0

    def test_not_found(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, two_files())
        assert repository.get_activity("deadbeef") is None


class TestSummaries:
    def test_daily_rows(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, two_files())
        rows = repository.daily_summary()
        assert len(rows) == 2  # two different days
        assert rows[0].date == "2026-06-01"
        assert rows[0].activity_count == 1

    def test_daily_sport_filter(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, two_files())
        rows = repository.daily_summary(sport="cycling")
        assert [row.sport for row in rows] == ["cycling"]

    def test_weekly_rows(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, two_files())
        rows = repository.weekly_summary()
        assert rows[0].iso_year == 2026
        assert rows[0].iso_week == 23

    def test_weekly_sport_filter(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, two_files())
        rows = repository.weekly_summary(sport="cycling")
        assert [row.sport for row in rows] == ["cycling"]


class TestDataQuality:
    def test_values(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, two_files())
        quality = repository.data_quality()
        assert quality.activity_count == 2
        assert quality.failed_file_count == 0
        assert quality.repaired_file_count == 0
        assert quality.discarded_segment_count == 0
        assert quality.duplicate_activity_count == 0
        assert quality.mean_hr_coverage_ratio is not None

    def test_duplicate_activity_counted(self, tmp_path: Path) -> None:
        content = stub_activity_content(T0, "cycling", 1800.0, 20000.0)
        repository = build_repository(tmp_path, {"one.fit": content, "two.fit": content})
        quality = repository.data_quality()
        assert quality.duplicate_activity_count == 1
        assert quality.activity_count == 1


class TestTrainingForm:
    def test_form_values_and_readiness(self, tmp_path: Path) -> None:
        today = datetime.now(ZoneInfo("Europe/Berlin")).date()
        days_back = (today - date(2026, 6, 1)).days
        files = two_files()
        # Add a 65 km ride
        files["long_ride.fit"] = stub_activity_content(
            T0 + timedelta(days=1), "cycling", 7200.0, 65000.0, heart_rates=[130] * 120
        )
        repository = build_repository(tmp_path, files)

        form = repository.training_form(days=days_back + 2)
        assert form.current_ctl > 0.0
        assert form.current_atl > 0.0
        assert form.form_classification in (
            "fresh",
            "optimal",
            "high_fatigue",
            "overtraining_risk",
        )
        assert form.days_since_60k_ride is not None
        # Activity was on 2026-06-02
        expected_days_since = (today - date(2026, 6, 2)).days
        assert form.days_since_60k_ride == expected_days_since

    def test_sport_filter(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, two_files())
        form_run = repository.training_form(sport="running")
        assert form_run.current_ctl > 0.0
        assert form_run.days_since_60k_ride is None


def test_t0_is_utc() -> None:
    assert T0.tzinfo is UTC
