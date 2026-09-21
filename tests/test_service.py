"""Tests for the service operations in server.py."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tests.conftest import StubParser, StubSource, make_app_config, stub_activity_content
from training_stats.cache import schemas
from training_stats.server import TrainingCoachServer, create_server

# Fixture activities anchored to "today" so overview windows match.
T0 = (datetime.now(UTC) - timedelta(days=8)).replace(hour=10, minute=0, second=0, microsecond=0)


def _stub_files() -> dict[str, bytes]:
    return {
        "easy.fit": stub_activity_content(T0, "cycling", 3600.0, 30000.0, heart_rates=[120] * 60),
        "run.fit": stub_activity_content(
            T0 + timedelta(days=7), "running", 1800.0, 8000.0, heart_rates=[150] * 30
        ),
    }


def _make_server(tmp_path: Path, files: dict[str, bytes]) -> TrainingCoachServer:
    """Build a server backed by the deterministic stub source/parser.

    The sync throttle is disabled so tests observe individual syncs; the
    throttle itself is exercised in ``TestSyncThrottle``.
    """
    config = make_app_config(tmp_path / "cache", tmp_path / "src")
    config.cache.sync_min_interval_seconds = 0.0
    server = create_server(config)
    # Swap in the deterministic stub source/parser for cache independence.
    server.sync_service.sources = [StubSource(files=files)]
    server.sync_service.parser = StubParser()
    return server


def _make_throttled_server(tmp_path: Path, interval_s: float = 10.0) -> TrainingCoachServer:
    server = _make_server(tmp_path, _stub_files())
    server.config.cache.sync_min_interval_seconds = interval_s
    return server


def _write_metadata(cache_dir: Path, age_seconds: float) -> None:
    stamp = datetime.now(UTC) - timedelta(seconds=age_seconds)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / schemas.METADATA_FILE).write_text(json.dumps({"last_sync": stamp.isoformat()}))


@pytest.fixture
def server(tmp_path: Path) -> TrainingCoachServer:
    server = _make_server(tmp_path, _stub_files())
    server.sync_service.sync()
    return server


@pytest.fixture
def unsynced_server(tmp_path: Path) -> TrainingCoachServer:
    """Server whose cache is empty; sync only happens via the code under test."""
    return _make_server(tmp_path, _stub_files())


class TestServiceOperations:
    def test_sync_activities(self, server: TrainingCoachServer) -> None:
        result = server.sync_activities()
        assert result["files_processed"] == 0
        assert result["files_skipped"] == 2

    def test_list_activities_syncs_first(self, unsynced_server: TrainingCoachServer) -> None:
        assert unsynced_server.repository.list_activities(limit=10) == []
        activities = unsynced_server.list_activities()
        assert len(activities) == 2
        assert unsynced_server.repository.list_activities(limit=10) != []

    def test_list_activities_without_sync(self, unsynced_server: TrainingCoachServer) -> None:
        activities = unsynced_server.list_activities(sync_first=False)
        assert activities == []

    def test_list_activities_picks_up_new_files(self, server: TrainingCoachServer) -> None:
        source = server.sync_service.sources[0]
        assert isinstance(source, StubSource)
        source.files["swim.fit"] = stub_activity_content(T0 + timedelta(days=3), "swimming", 1200.0)
        activities = server.list_activities()
        assert len(activities) == 3
        assert {item["sport"] for item in activities} == {"cycling", "running", "swimming"}

    def test_list_activities_serves_cache_when_sync_fails(
        self,
        server: TrainingCoachServer,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        def broken_sync() -> object:
            raise RuntimeError("disk gone")

        monkeypatch.setattr(server.sync_service, "sync", broken_sync)
        with caplog.at_level(logging.WARNING, logger="training_stats.server"):
            activities = server.list_activities()
        assert len(activities) == 2
        assert any("activity sync failed" in record.message for record in caplog.records)

    def test_startup_sync_populates_cache(self, unsynced_server: TrainingCoachServer) -> None:
        from training_stats.cli import _startup_sync

        assert unsynced_server.repository.list_activities(limit=10) == []
        _startup_sync(unsynced_server)
        assert len(unsynced_server.repository.list_activities(limit=10)) == 2

    def test_weekly_summary_syncs_first(self, unsynced_server: TrainingCoachServer) -> None:
        assert unsynced_server.repository.list_activities(limit=10) == []
        rows = unsynced_server.weekly_summary(weeks=4)
        assert len(rows) == 2

    def test_weekly_summary_without_sync(self, unsynced_server: TrainingCoachServer) -> None:
        assert unsynced_server.weekly_summary(weeks=4, sync_first=False) == []

    def test_overview_syncs_first(self, unsynced_server: TrainingCoachServer) -> None:
        overview = unsynced_server.overview(days=14)
        assert overview["data_quality"]["activity_count"] == 2
        assert len(overview["daily_series"]) == 2

    def test_overview_without_sync(self, unsynced_server: TrainingCoachServer) -> None:
        overview = unsynced_server.overview(days=14, sync_first=False)
        assert overview["data_quality"]["activity_count"] == 0

    def test_get_activity_does_not_sync(self, unsynced_server: TrainingCoachServer) -> None:
        with pytest.raises(LookupError, match="not found"):
            unsynced_server.get_activity("does-not-exist")
        # Cache-only: no sync happened, so the store is still empty.
        assert unsynced_server.repository.list_activities(limit=10) == []

    def test_list_activities(self, server: TrainingCoachServer) -> None:
        activities = server.list_activities()
        assert len(activities) == 2
        assert set(activities[0]) >= {
            "activity_id",
            "start_time",
            "local_start_time",
            "sport",
            "moving_time_s",
            "load_value",
        }
        assert activities[0]["local_start_time"] != ""
        # No time-series data may leak into responses.
        assert all("records" not in item for item in activities)

    def test_list_activities_sport_filter(self, server: TrainingCoachServer) -> None:
        activities = server.list_activities(sport="running")
        assert [item["sport"] for item in activities] == ["running"]

    def test_list_activities_date_filter(self, server: TrainingCoachServer) -> None:
        run_day = (T0 + timedelta(days=7)).date()
        activities = server.list_activities(start_date=run_day, end_date=run_day)
        assert [item["sport"] for item in activities] == ["running"]

    def test_list_activities_invalid_limit(self, server: TrainingCoachServer) -> None:
        with pytest.raises(ValueError, match="limit"):
            server.list_activities(limit=0)

    def test_get_activity(self, server: TrainingCoachServer) -> None:
        activities = server.list_activities(limit=1)
        details = server.get_activity(str(activities[0]["activity_id"]))
        assert details["sport"] == activities[0]["sport"]
        assert details["hr_zones_s"]["zone_1"] >= 0.0
        assert "records" not in details
        assert details["load"]["method"] == "hr_trimp"
        assert details["local_start_time"] != ""
        assert "paused_time_s" in details
        assert "data_hygiene_flags" in details
        assert "decoupling" in details
        if details["decoupling"] is not None:
            assert {
                "ef_first_half",
                "ef_second_half",
                "aerobic_decoupling_pct",
                "confidence",
            } <= set(details["decoupling"])

    def test_get_activity_not_found(self, server: TrainingCoachServer) -> None:
        with pytest.raises(LookupError, match="not found"):
            server.get_activity("does-not-exist")

    def test_weekly_summary(self, server: TrainingCoachServer) -> None:
        rows = server.weekly_summary(weeks=4)
        assert len(rows) == 2
        assert rows[0]["comparison_vs_prev_week"]["load_value"] is None

    def test_weekly_summary_sport(self, server: TrainingCoachServer) -> None:
        rows = server.weekly_summary(weeks=4, sport="cycling")
        assert [row["sport"] for row in rows] == ["cycling"]

    def test_weekly_summary_invalid_weeks(self, server: TrainingCoachServer) -> None:
        with pytest.raises(ValueError, match="weeks"):
            server.weekly_summary(weeks=0)

    def test_overview(self, server: TrainingCoachServer) -> None:
        overview = server.overview(days=14)
        assert overview["athlete_profile"]["max_heart_rate"] == 180
        assert len(overview["recent_activities"]) == 2
        assert overview["latest_activity"] is not None
        assert overview["data_quality"]["activity_count"] == 2
        assert {"start", "end"} <= set(overview["period"])
        # The daily series carries every metric; any trend is derivable.
        assert len(overview["daily_series"]) == 2
        assert {"date", "sport", "duration_s", "distance_m", "load_value"} <= set(
            overview["daily_series"][0]
        )
        assert "hr_zones_s" in overview["daily_series"][0]
        # Form snapshot, form history and tour readiness.
        assert {"ctl", "atl", "tsb", "classification", "is_warmup", "load_confidence_ratio"} <= set(
            overview["form"]
        )
        assert isinstance(overview["form_history"], list)
        assert {"date", "ctl", "atl", "tsb", "daily_load"} <= set(overview["form_history"][0])
        assert {"rolling_7d_km", "rolling_28d_km", "days_since_60k"} <= set(
            overview["tour_readiness"]
        )
        assert "records" not in str(overview)

    def test_overview_sport_filter(self, server: TrainingCoachServer) -> None:
        overview = server.overview(days=14, sport="cycling")
        assert {row["sport"] for row in overview["daily_series"]} == {"cycling"}

    def test_overview_invalid_days(self, server: TrainingCoachServer) -> None:
        with pytest.raises(ValueError, match="days"):
            server.overview(days=-1)


class TestSyncThrottle:
    def test_throttles_within_interval(self, tmp_path: Path) -> None:
        server = _make_throttled_server(tmp_path)
        first = server.sync_if_due()
        assert first.get("throttled") is not True  # fresh cache: real sync
        assert first["files_processed"] == 2

        second = server.sync_if_due()
        assert second["throttled"] is True
        assert second["seconds_since_last_sync"] >= 0

    def test_throttle_expires(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import training_stats.server as server_module

        server = _make_throttled_server(tmp_path)
        server.sync_if_due()
        real_monotonic = server_module.time.monotonic
        monkeypatch.setattr(server_module.time, "monotonic", lambda: real_monotonic() + 60.0)
        result = server.sync_if_due()
        assert result.get("throttled") is not True

    def test_zero_interval_always_syncs(self, tmp_path: Path) -> None:
        server = _make_throttled_server(tmp_path, interval_s=0.0)
        server.sync_if_due()
        result = server.sync_if_due()
        assert result.get("throttled") is not True

    def test_failed_sync_does_not_stamp(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server = _make_throttled_server(tmp_path)

        def broken_sync() -> object:
            raise RuntimeError("disk gone")

        monkeypatch.setattr(server.sync_service, "sync", broken_sync)
        failed = server.sync_if_due()
        assert failed["cache_updated_at"] is None
        monkeypatch.undo()
        result = server.sync_if_due()
        assert result.get("throttled") is not True  # failure must not block retries
        assert result["files_processed"] == 2

    def test_persisted_last_sync_within_window(self, tmp_path: Path) -> None:
        server = _make_throttled_server(tmp_path)
        _write_metadata(tmp_path / "cache", age_seconds=5.0)
        result = server.sync_if_due()
        assert result["throttled"] is True
        assert result["seconds_since_last_sync"] <= 10.0

    def test_persisted_last_sync_expired(self, tmp_path: Path) -> None:
        server = _make_throttled_server(tmp_path)
        _write_metadata(tmp_path / "cache", age_seconds=60.0)
        result = server.sync_if_due()
        assert result.get("throttled") is not True
        assert result["files_processed"] == 2

    def test_persisted_invalid_metadata_syncs(self, tmp_path: Path) -> None:
        server = _make_throttled_server(tmp_path)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / schemas.METADATA_FILE).write_text("not json at all")
        result = server.sync_if_due()
        assert result.get("throttled") is not True


class TestCreateServer:
    def test_local_source_built(self, tmp_path: Path) -> None:
        config = make_app_config(tmp_path / "cache", tmp_path / "activities")
        server = create_server(config)
        assert [source.name for source in server.sources] == ["local"]

    def test_webdav_source_built(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from training_stats.config import (
            WebDAVSourceConfig,
        )

        monkeypatch.setenv("NC_USER", "u")
        monkeypatch.setenv("NC_PASS", "p")
        config = make_app_config(tmp_path / "cache", tmp_path / "activities")
        config.sources.append(
            WebDAVSourceConfig(
                name="cloud",
                base_url="https://example.com/dav/",
                username_env="NC_USER",
                password_env="NC_PASS",
            )
        )
        server = create_server(config)
        assert [source.name for source in server.sources] == ["local", "cloud"]
