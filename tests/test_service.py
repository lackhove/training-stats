"""Tests for the service operations in services.py."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tests.conftest import StubSource, make_app_config, make_processor, stub_activity_content
from training_stats.adapters.bootstrap import create_service
from training_stats.adapters.cache import schemas
from training_stats.adapters.cache.repository import JsonTrainingRepository
from training_stats.domain.metrics import Metric, MetricReport, ScalarValue, TableValue, TripPeriod
from training_stats.domain.models import ActivitySummary
from training_stats.services.results import SyncResult, ThrottledSync
from training_stats.services.state import SyncState
from training_stats.services.training import TrainingStatsService

# Fixture activities anchored to "today" so overview windows match.
T0 = (datetime.now(UTC) - timedelta(days=8)).replace(hour=10, minute=0, second=0, microsecond=0)


def _stub_files() -> dict[str, bytes]:
    return {
        "easy.fit": stub_activity_content(T0, "cycling", 3600.0, 30000.0, heart_rates=[120] * 60),
        "run.fit": stub_activity_content(
            T0 + timedelta(days=7), "running", 1800.0, 8000.0, heart_rates=[150] * 30
        ),
    }


def _make_server(
    tmp_path: Path, files: dict[str, bytes], *, interval_s: float = 0.0
) -> TrainingStatsService:
    """Build a server backed by the deterministic stub source/parser.

    The sync throttle is disabled so tests observe individual syncs; the
    throttle itself is exercised in ``TestSyncThrottle``.
    """
    config = make_app_config(tmp_path / "cache", tmp_path / "src")
    config.cache.sync_min_interval_seconds = interval_s
    server = create_service(config)
    # Swap in the deterministic stub source/parser for cache independence.
    server.sync_service.sources = [StubSource(files=files)]
    server.sync_service.processor = make_processor(
        athlete=config.athlete, analysis_version=server.sync_service.analysis_version
    )
    return server


def _make_throttled_server(tmp_path: Path, interval_s: float = 10.0) -> TrainingStatsService:
    return _make_server(tmp_path, _stub_files(), interval_s=interval_s)


def _write_metadata(cache_dir: Path, age_seconds: float) -> None:
    stamp = datetime.now(UTC) - timedelta(seconds=age_seconds)
    JsonTrainingRepository(cache_dir).save_sync_snapshot(SyncState([], {}), last_sync=stamp)


def _write_valid_snapshot(cache_dir: Path) -> None:
    """An empty current-format snapshot, so throttling depends on metadata alone."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / schemas.CACHE_FILE).write_text(
        json.dumps(
            {
                "format_version": schemas.CACHE_FORMAT_VERSION,
                "activities": [],
                "source_index": [],
            }
        )
    )


def _activity_ids(activities: tuple[ActivitySummary, ...]) -> set[str]:
    return {activity.activity_id for activity in activities}


def _metric(report: MetricReport, name: str) -> Metric:
    return next(metric for metric in report.metrics if metric.name == name)


def _scalar(report: MetricReport, name: str) -> object:
    value = _metric(report, name).value
    return value.value if isinstance(value, ScalarValue) else None


@pytest.fixture
def server(tmp_path: Path) -> TrainingStatsService:
    server = _make_server(tmp_path, _stub_files())
    server.sync_service.sync()
    return server


@pytest.fixture
def unsynced_server(tmp_path: Path) -> TrainingStatsService:
    """Server whose cache is empty; sync only happens via the code under test."""
    return _make_server(tmp_path, _stub_files())


class TestServiceOperations:
    def test_sync_activities(self, server: TrainingStatsService) -> None:
        result = server.sync_activities()
        assert result.files_processed == 0
        assert result.files_skipped == 2

    def test_list_activities_syncs_first(self, unsynced_server: TrainingStatsService) -> None:
        assert unsynced_server.repository.load_sync_state().activities == {}
        activities = unsynced_server.list_activities()
        assert len(_activity_ids(activities)) == 2
        assert len(unsynced_server.repository.load_sync_state().activities) == 2

    def test_list_activities_without_sync(self, unsynced_server: TrainingStatsService) -> None:
        activities = unsynced_server.list_activities(sync_first=False)
        assert activities == ()

    def test_list_activities_picks_up_new_files(self, server: TrainingStatsService) -> None:
        source = server.sync_service.sources[0]
        assert isinstance(source, StubSource)
        source.files["swim.fit"] = stub_activity_content(T0 + timedelta(days=3), "swimming", 1200.0)
        activities = server.list_activities()
        assert len(_activity_ids(activities)) == 3
        sports = {activity.sport for activity in activities}
        assert sports == {"cycling", "running", "swimming"}

    def test_list_activities_serves_cache_when_sync_fails(
        self,
        server: TrainingStatsService,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        def broken_sync() -> object:
            raise RuntimeError("disk gone")

        monkeypatch.setattr(server.sync_service, "sync", broken_sync)
        with caplog.at_level(logging.WARNING, logger="training_stats.services.training"):
            activities = server.list_activities()
        assert len(_activity_ids(activities)) == 2
        assert any("activity sync failed" in record.message for record in caplog.records)

    def test_startup_sync_populates_cache(self, unsynced_server: TrainingStatsService) -> None:
        from training_stats.adapters.cli.app import _startup_sync

        assert unsynced_server.repository.load_sync_state().activities == {}
        _startup_sync(unsynced_server)
        assert len(unsynced_server.repository.load_sync_state().activities) == 2

    def test_weekly_summary_syncs_first(self, unsynced_server: TrainingStatsService) -> None:
        assert unsynced_server.repository.load_sync_state().activities == {}
        report = unsynced_server.weekly_summary(weeks=4)
        assert report.metrics

    def test_weekly_summary_without_sync(self, unsynced_server: TrainingStatsService) -> None:
        report = unsynced_server.weekly_summary(weeks=4, sync_first=False)
        assert all(metric.value is None for metric in report.metrics)

    def test_overview_syncs_first(self, unsynced_server: TrainingStatsService) -> None:
        overview = unsynced_server.overview(days=14)
        assert _scalar(overview, "data_quality.activity_count") == 2
        assert all(not isinstance(metric.period, TripPeriod) for metric in overview.metrics)

    def test_overview_without_sync(self, unsynced_server: TrainingStatsService) -> None:
        overview = unsynced_server.overview(days=14, sync_first=False)
        assert _scalar(overview, "data_quality.activity_count") == 0

    def test_get_activity_does_not_sync(self, unsynced_server: TrainingStatsService) -> None:
        with pytest.raises(LookupError, match="not found"):
            unsynced_server.get_activity("does-not-exist")
        # Cache-only: no sync happened, so the store is still empty.
        assert unsynced_server.repository.load_sync_state().activities == {}

    def test_list_activities(self, server: TrainingStatsService) -> None:
        activities = server.list_activities()
        assert len(_activity_ids(activities)) == 2
        assert activities[0].sport == "running"
        assert all(not hasattr(activity, "metrics") for activity in activities)
        assert all(not hasattr(activity, "records") for activity in activities)

    def test_list_activities_sport_filter(self, server: TrainingStatsService) -> None:
        activities = server.list_activities(sport="running")
        assert len(_activity_ids(activities)) == 1
        assert activities[0].sport == "running"

    def test_list_activities_date_filter(self, server: TrainingStatsService) -> None:
        run_day = (T0 + timedelta(days=7)).date()
        activities = server.list_activities(start_date=run_day, end_date=run_day)
        assert len(_activity_ids(activities)) == 1
        assert activities[0].sport == "running"

    def test_list_activities_invalid_limit(self, server: TrainingStatsService) -> None:
        with pytest.raises(ValueError, match="limit"):
            server.list_activities(limit=0)

    def test_get_activity(self, server: TrainingStatsService) -> None:
        activities = server.list_activities(limit=1)
        activity_id = next(iter(_activity_ids(activities)))
        details = server.get_activity(activity_id)
        assert details.activity_id == activity_id
        assert details.sport
        assert details.metrics
        load = next(metric for metric in details.metrics if metric.name == "training_load.hr_trimp")
        assert load.method == "hr_trimp"
        zones = next(metric for metric in details.metrics if metric.name == "heart_rate.zone_time")
        assert isinstance(zones.value, TableValue)

    def test_get_activity_not_found(self, server: TrainingStatsService) -> None:
        with pytest.raises(LookupError, match="not found"):
            server.get_activity("does-not-exist")

    def test_weekly_summary(self, server: TrainingStatsService) -> None:
        report = server.weekly_summary(weeks=4)
        assert _metric(report, "activity.distance_by_week").value is not None
        assert _metric(report, "training_load.hr_trimp_by_week").kind.value == "estimated"

    def test_weekly_summary_sport(self, server: TrainingStatsService) -> None:
        rows = server.weekly_summary(weeks=4, sport="cycling")
        distances = _metric(rows, "activity.distance_by_week")
        assert isinstance(distances.value, TableValue)
        assert len(distances.value.axis.values) == 1

    def test_weekly_summary_invalid_weeks(self, server: TrainingStatsService) -> None:
        with pytest.raises(ValueError, match="weeks"):
            server.weekly_summary(weeks=0)

    def test_overview(self, server: TrainingStatsService) -> None:
        overview = server.overview(days=14)
        assert _scalar(overview, "athlete_profile.max_heart_rate") == 180
        assert all(not isinstance(metric.period, TripPeriod) for metric in overview.metrics)
        assert _scalar(overview, "data_quality.activity_count") == 2
        assert _scalar(overview, "training_form.ctl") is not None
        assert _scalar(overview, "tour_readiness.rolling_7d_distance") >= 0.0
        assert not hasattr(overview, "records")

    def test_overview_sport_filter(self, server: TrainingStatsService) -> None:
        overview = server.overview(days=14, sport="cycling")
        distance = _metric(overview, "activity.distance_by_day")
        assert isinstance(distance.value, TableValue)
        assert len(distance.value.axis.values) == 1
        assert all(not isinstance(metric.period, TripPeriod) for metric in overview.metrics)

    def test_overview_invalid_days(self, server: TrainingStatsService) -> None:
        with pytest.raises(ValueError, match="days"):
            server.overview(days=-1)


class TestSyncThrottle:
    def test_throttles_within_interval(self, tmp_path: Path) -> None:
        server = _make_throttled_server(tmp_path)
        first = server.sync_if_due()
        assert isinstance(first, SyncResult)  # fresh cache: real sync
        assert first.files_processed == 2

        second = server.sync_if_due()
        assert isinstance(second, ThrottledSync)
        assert second.seconds_since_last_sync >= 0

    def test_throttle_expires(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import training_stats.services.training as services_module

        server = _make_throttled_server(tmp_path)
        server.sync_if_due()
        real_monotonic = services_module.time.monotonic
        monkeypatch.setattr(services_module.time, "monotonic", lambda: real_monotonic() + 60.0)
        result = server.sync_if_due()
        assert isinstance(result, SyncResult)

    def test_zero_interval_always_syncs(self, tmp_path: Path) -> None:
        server = _make_throttled_server(tmp_path, interval_s=0.0)
        server.sync_if_due()
        result = server.sync_if_due()
        assert isinstance(result, SyncResult)

    def test_failed_sync_does_not_stamp(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server = _make_throttled_server(tmp_path)

        def broken_sync() -> object:
            raise RuntimeError("disk gone")

        monkeypatch.setattr(server.sync_service, "sync", broken_sync)
        failed = server.sync_if_due()
        assert isinstance(failed, SyncResult)
        assert failed.cache_updated_at is None
        monkeypatch.undo()
        result = server.sync_if_due()
        assert isinstance(result, SyncResult)  # failure must not block retries
        assert result.files_processed == 2

    def test_persisted_last_sync_within_window(self, tmp_path: Path) -> None:
        server = _make_throttled_server(tmp_path)
        _write_metadata(tmp_path / "cache", age_seconds=5.0)
        result = server.sync_if_due()
        assert isinstance(result, ThrottledSync)
        assert result.seconds_since_last_sync <= 10.0

    def test_persisted_last_sync_expired(self, tmp_path: Path) -> None:
        server = _make_throttled_server(tmp_path)
        _write_metadata(tmp_path / "cache", age_seconds=60.0)
        result = server.sync_if_due()
        assert isinstance(result, SyncResult)
        assert result.files_processed == 2

    def test_persisted_invalid_metadata_syncs(self, tmp_path: Path) -> None:
        server = _make_throttled_server(tmp_path)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / schemas.METADATA_FILE).write_text("not json at all")
        result = server.sync_if_due()
        assert isinstance(result, SyncResult)

    def test_legacy_metadata_without_version_syncs(self, tmp_path: Path) -> None:
        server = _make_throttled_server(tmp_path)
        cache_dir = tmp_path / "cache"
        _write_valid_snapshot(cache_dir)
        (cache_dir / schemas.METADATA_FILE).write_text(
            json.dumps({"last_sync": datetime.now(UTC).isoformat()})
        )
        result = server.sync_if_due()
        assert isinstance(result, SyncResult)
        assert result.files_processed == 2

    def test_metadata_format_version_mismatch_syncs(self, tmp_path: Path) -> None:
        server = _make_throttled_server(tmp_path)
        cache_dir = tmp_path / "cache"
        _write_valid_snapshot(cache_dir)
        (cache_dir / schemas.METADATA_FILE).write_text(
            json.dumps(
                {
                    "last_sync": datetime.now(UTC).isoformat(),
                    "format_version": schemas.CACHE_FORMAT_VERSION + 1,
                }
            )
        )
        result = server.sync_if_due()
        assert isinstance(result, SyncResult)
        assert result.files_processed == 2


class TestCreateServer:
    def test_local_source_built(self, tmp_path: Path) -> None:
        config = make_app_config(tmp_path / "cache", tmp_path / "activities")
        server = create_service(config)
        assert [source.name for source in server.sync_service.sources] == ["local"]

    def test_webdav_source_built(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from training_stats.adapters.config import (
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
        server = create_service(config)
        assert [source.name for source in server.sync_service.sources] == ["local", "cloud"]
