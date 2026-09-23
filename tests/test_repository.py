"""Tests for the Parquet-backed repository: scans and sync-state I/O."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from tests.conftest import StubSource, make_app_config, make_processor, stub_activity_content
from training_stats.adapters.cache.repository import ParquetTrainingRepository
from training_stats.services.sync import SyncService

T0 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)


def build_repository(tmp_path: Path, files: dict[str, bytes]) -> ParquetTrainingRepository:
    source = StubSource(files=files)
    cache_dir = tmp_path / "cache"
    config = make_app_config(cache_dir, tmp_path / "src")
    service = SyncService(
        sources=[source],
        processor=make_processor(athlete=config.athlete, analysis_version="1"),
        repository=ParquetTrainingRepository(cache_dir=cache_dir),
        analysis_version="1",
    )
    service.sync()
    return ParquetTrainingRepository(cache_dir=cache_dir)


def two_files() -> dict[str, bytes]:
    return {
        "easy.fit": stub_activity_content(T0, "cycling", 3600.0, 30000.0, heart_rates=[120] * 60),
        "intervals.fit": stub_activity_content(
            T0 + timedelta(days=2), "running", 1800.0, 8000.0, heart_rates=[150] * 30
        ),
    }


class TestEmptyCache:
    def test_scans_return_empty_after_successful_empty_sync(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, {})
        assert repository.activities_scan().collect().is_empty()
        assert repository.source_index_scan().collect().is_empty()
        # A successful listing with zero files still stamps the throttle.
        assert repository.last_sync_stamp() is not None


class TestActivitiesScan:
    def test_scan_has_merged_columns(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, two_files())
        frame = repository.activities_scan().collect()
        assert frame.height == 2
        # The merged table carries session facts, derived metrics and the
        # source-index fields side by side: no join is needed to read them.
        assert "load_value" in frame.columns
        assert "hr_zone_1_s" in frame.columns
        assert "analysis_version" in frame.columns

    def test_load_values_present(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, two_files())
        rows = repository.activities_scan().select("sport", "load_value").collect().to_dicts()
        loads = {str(row["sport"]): row["load_value"] for row in rows}
        assert loads["cycling"] is not None
        assert loads["running"] is not None

    def test_activity_rows_enriched_with_source_fields(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, two_files())
        rows = (
            repository.cache_scan().filter(pl.col("start_time").is_not_null()).collect().to_dicts()
        )
        assert {str(row["sport"]) for row in rows} == {"cycling", "running"}
        for row in rows:
            assert row["status"] == "ok"
            assert row["source_name"] == "stub"
            assert row["source_hash"] is not None
            assert row["file_size"] is not None
            assert row["modified_at"] is not None
            assert row["processed_at"] is not None
            # Two distinct versions: the source-index copy drives change
            # detection, the activity-level one names the metric-producing code.
            assert row["source_analysis_version"] == "1"
            assert str(row["analysis_version"]).startswith("1:")


class TestSourceIndexScan:
    def test_failed_files_kept_without_activity_fields(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, {"broken.fit": b"garbage"})
        rows = repository.source_index_scan().collect().to_dicts()
        assert len(rows) == 1
        assert rows[0]["status"] == "failed"
        assert rows[0]["error_message"] is not None
        assert rows[0]["activity_id"] is None
        # The failed file is kept as a source-only row: no activity fields.
        cache_rows = repository.cache_scan().collect().to_dicts()
        assert cache_rows[0]["start_time"] is None
        assert repository.activities_scan().collect().is_empty()

    def test_duplicate_rows_kept_without_activity_fields(self, tmp_path: Path) -> None:
        content = stub_activity_content(T0, "cycling", 1800.0, 20000.0)
        repository = build_repository(tmp_path, {"one.fit": content, "two.fit": content})
        rows = repository.source_index_scan().collect().to_dicts()
        assert sorted(str(row["status"]) for row in rows) == ["duplicate", "ok"]
        activities = repository.activities_scan().collect()
        assert activities.height == 1


class TestMultiSegmentRows:
    def _two_segments(self) -> bytes:
        first = stub_activity_content(T0, "cycling", 1800.0, 20000.0)
        second = stub_activity_content(T0 + timedelta(hours=2), "cycling", 1200.0, 8000.0)
        return first + b"\n" + second

    def test_file_level_fields_repeat_across_segments(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, {"ride.fit": self._two_segments()})
        rows = (
            repository.cache_scan().filter(pl.col("start_time").is_not_null()).collect().to_dicts()
        )
        assert len(rows) == 2
        for field in ("file_size", "modified_at", "source_hash", "discarded_segment_count"):
            assert rows[0][field] == rows[1][field]
        assert rows[0]["segment_index"] != rows[1]["segment_index"]
        assert rows[0]["segment_count"] == rows[1]["segment_count"] == 2


class TestSyncState:
    def test_load_round_trips_rows(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, two_files())
        state = repository.load_sync_state()
        assert len(state.index_rows) == 2
        scan_ids = {
            str(row["activity_id"]) for row in repository.activities_scan().collect().to_dicts()
        }
        assert set(state.activities) == scan_ids

    def test_source_only_rows_not_in_activity_store(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, {"broken.fit": b"garbage"})
        state = repository.load_sync_state()
        assert len(state.index_rows) == 1
        assert state.activities == {}

    def test_last_sync_stamp(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, two_files())
        stamp = repository.last_sync_stamp()
        assert stamp is not None
        assert stamp.tzinfo is not None


def test_t0_is_utc() -> None:
    assert T0.tzinfo is UTC
