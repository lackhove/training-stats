"""Tests for the Parquet-backed repository."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from tests.conftest import StubSource, make_app_config, make_processor, stub_activity_content
from training_stats.cache.repository import ParquetTrainingRepository
from training_stats.cache.sync import SyncService

T0 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)


def build_repository(tmp_path: Path, files: dict[str, bytes]) -> ParquetTrainingRepository:
    source = StubSource(files=files)
    cache_dir = tmp_path / "cache"
    config = make_app_config(cache_dir, tmp_path / "src")
    service = SyncService(
        sources=[source],
        processor=make_processor(athlete=config.athlete, analysis_version="1"),
        repository=ParquetTrainingRepository(
            cache_dir=cache_dir,
            timezone=config.application.zoneinfo,
        ),
        analysis_version="1",
    )
    service.sync()
    return ParquetTrainingRepository(cache_dir=cache_dir, timezone=ZoneInfo("Europe/Berlin"))


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
        assert repository.activities_scan().collect().is_empty()
        quality = repository.data_quality()
        assert quality.activity_count == 0
        assert quality.mean_hr_coverage_ratio is None


class TestActivitiesScan:
    def test_scan_has_merged_columns(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, two_files())
        frame = repository.activities_scan().collect()
        assert frame.height == 2
        # The merged table carries session facts and derived metrics side by
        # side: no join is needed to read load or zones.
        assert "load_value" in frame.columns
        assert "hr_zone_1_s" in frame.columns
        assert "analysis_version" in frame.columns

    def test_load_values_present(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, two_files())
        rows = repository.activities_scan().select("sport", "load_value").collect().to_dicts()
        loads = {str(row["sport"]): row["load_value"] for row in rows}
        assert loads["cycling"] is not None
        assert loads["running"] is not None


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
            start_date=datetime(2026, 6, 3, tzinfo=UTC).date(),
            end_date=datetime(2026, 6, 3, tzinfo=UTC).date(),
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


def test_t0_is_utc() -> None:
    assert T0.tzinfo is UTC
