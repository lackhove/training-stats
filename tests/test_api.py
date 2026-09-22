"""Tests for the FastAPI REST application."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from tests.conftest import StubSource, make_app_config, make_processor, stub_activity_content
from training_stats.api import build_app
from training_stats.services import TrainingStatsService, create_service

# Fixture activities anchored to "today" so overview windows match.
T0 = (datetime.now(UTC) - timedelta(days=8)).replace(hour=10, minute=0, second=0, microsecond=0)

EXPECTED_PATHS = {"/activities", "/activities/{activity_id}", "/weekly", "/overview"}


def _stub_files() -> dict[str, bytes]:
    return {
        "easy.fit": stub_activity_content(T0, "cycling", 3600.0, 30000.0, heart_rates=[120] * 60),
        "run.fit": stub_activity_content(
            T0 + timedelta(days=7), "running", 1800.0, 8000.0, heart_rates=[150] * 30
        ),
    }


def _make_server(tmp_path: Path, files: dict[str, bytes]) -> TrainingStatsService:
    config = make_app_config(tmp_path / "cache", tmp_path / "src")
    config.cache.sync_min_interval_seconds = 0.0
    server = create_service(config)
    server.sync_service.sources = [StubSource(files=files)]
    server.sync_service.processor = make_processor(
        athlete=config.athlete, analysis_version=server.sync_service.analysis_version
    )
    return server


def _client(server: TrainingStatsService) -> TestClient:
    return TestClient(build_app(server))


class TestActivitiesEndpoint:
    def test_list_syncs_first(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, _stub_files())
        with _client(server) as client:
            response = client.get("/activities")
        assert response.status_code == 200
        activities = response.json()
        assert len(activities) == 2
        assert {item["sport"] for item in activities} == {"cycling", "running"}
        assert server.repository.list_activities(limit=10) != []

    def test_list_filters(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, _stub_files())
        server.sync_service.sync()
        with _client(server) as client:
            response = client.get("/activities", params={"sport": "running"})
            assert [item["sport"] for item in response.json()] == ["running"]
            run_day = (T0 + timedelta(days=7)).date()
            response = client.get(
                "/activities",
                params={"start_date": run_day.isoformat(), "end_date": run_day.isoformat()},
            )
            assert [item["sport"] for item in response.json()] == ["running"]

    def test_get_activity(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, _stub_files())
        server.sync_service.sync()
        with _client(server) as client:
            activity_id = client.get("/activities", params={"limit": 1}).json()[0]["activity_id"]
            response = client.get(f"/activities/{activity_id}")
        assert response.status_code == 200
        details = response.json()
        assert details["load"]["method"] == "hr_trimp"
        assert set(details["hr_zones_s"]) == {f"zone_{i}" for i in range(1, 6)}
        assert "records" not in details

    def test_get_activity_not_found(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, _stub_files())
        server.sync_service.sync()
        with _client(server) as client:
            response = client.get("/activities/does-not-exist")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]


class TestValidation:
    def test_invalid_date_rejected(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, _stub_files())
        server.sync_service.sync()
        with _client(server) as client:
            assert client.get("/activities", params={"start_date": "06/02/2026"}).status_code == 422
            assert client.get("/weekly", params={"start_date": "not-a-date"}).status_code == 422

    def test_out_of_range_int_rejected(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, _stub_files())
        server.sync_service.sync()
        with _client(server) as client:
            assert client.get("/activities", params={"limit": 0}).status_code == 422
            assert client.get("/activities", params={"limit": 501}).status_code == 422
            assert client.get("/weekly", params={"weeks": 0}).status_code == 422
            assert client.get("/overview", params={"days": 0}).status_code == 422


class TestWeeklyEndpoint:
    def test_weekly_rows(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, _stub_files())
        server.sync_service.sync()
        with _client(server) as client:
            response = client.get("/weekly", params={"weeks": 4})
        assert response.status_code == 200
        rows = response.json()
        assert len(rows) == 2
        assert rows[0]["comparison_vs_prev_week"]["load_value"] is None

    def test_weekly_sport_filter(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, _stub_files())
        server.sync_service.sync()
        with _client(server) as client:
            response = client.get("/weekly", params={"weeks": 4, "sport": "cycling"})
        assert [row["sport"] for row in response.json()] == ["cycling"]


class TestOverviewEndpoint:
    def test_overview_sections(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, _stub_files())
        server.sync_service.sync()
        with _client(server) as client:
            response = client.get("/overview", params={"days": 14})
        assert response.status_code == 200
        overview = response.json()
        assert overview["athlete_profile"]["max_heart_rate"] == 180
        assert len(overview["recent_activities"]) == 2
        assert overview["data_quality"]["activity_count"] == 2
        assert len(overview["daily_series"]) == 2
        assert {"date", "sport", "duration_s", "load_value"} <= set(overview["daily_series"][0])
        assert overview["form"]["classification"] in (
            "fresh",
            "optimal",
            "high_fatigue",
            "overtraining_risk",
        )
        assert {"ctl", "atl", "tsb", "daily_load"} <= set(overview["form_history"][0])
        assert {"rolling_7d_km", "rolling_28d_km", "days_since_60k"} <= set(
            overview["tour_readiness"]
        )

    def test_overview_sport_filter(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, _stub_files())
        server.sync_service.sync()
        with _client(server) as client:
            response = client.get("/overview", params={"days": 14, "sport": "cycling"})
        sports = {row["sport"] for row in response.json()["daily_series"]}
        assert sports == {"cycling"}


class TestSyncOnEndpoints:
    def test_weekly_and_overview_sync_first(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, _stub_files())
        with _client(server) as client:
            weekly = client.get("/weekly", params={"weeks": 4})
            overview = client.get("/overview", params={"days": 14})
        assert weekly.status_code == 200
        assert len(weekly.json()) == 2
        assert overview.json()["data_quality"]["activity_count"] == 2

    def test_get_activity_serves_cache_only(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, _stub_files())
        with _client(server) as client:
            response = client.get("/activities/does-not-exist")
        assert response.status_code == 404
        assert server.repository.list_activities(limit=10) == []


class TestOpenApi:
    def test_paths_and_schemas(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, _stub_files())
        server.sync_service.sync()
        with _client(server) as client:
            schema = client.get("/openapi.json").json()
        assert set(schema["paths"]) == EXPECTED_PATHS
        schemas = schema["components"]["schemas"]
        assert {
            "ActivitySummary",
            "ActivityDetails",
            "WeeklyRow",
            "Overview",
        } <= set(schemas)
        # Typed response payloads instead of loose objects.
        details = schemas["ActivityDetails"]["properties"]
        assert details["hr_zones_s"]["$ref"].endswith("HrZones")
        assert details["load"]["$ref"].endswith("LoadInfo")
