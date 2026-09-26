"""Tests for generic MetricReport HTTP responses and OpenAPI schema."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from tests.conftest import StubSource, make_app_config, make_processor, stub_activity_content
from training_stats.adapters.bootstrap import create_service
from training_stats.adapters.http.api import build_app
from training_stats.services.training import TrainingStatsService

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


def _metrics(report: dict[str, Any]) -> list[dict[str, Any]]:
    value = report.get("metrics")
    assert isinstance(value, list)
    return value


def _named(report: dict[str, Any], name: str) -> dict[str, Any]:
    return next(metric for metric in _metrics(report) if metric["name"] == name)


def _trip_ids(report: dict[str, Any]) -> set[str]:
    return {
        metric["period"]["activity_id"]
        for metric in _metrics(report)
        if metric["period"]["kind"] == "trip"
    }


def _activity_ids(activities: list[dict[str, Any]]) -> set[str]:
    return {activity["activity_id"] for activity in activities}


class TestActivitiesEndpoint:
    def test_list_syncs_first_and_returns_activity_summaries(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, _stub_files())
        with _client(server) as client:
            response = client.get("/activities")
        assert response.status_code == 200
        activities = response.json()
        assert len(_activity_ids(activities)) == 2
        assert all("metrics" not in activity for activity in activities)
        assert set(activities[0]) == {
            "activity_id",
            "start_time",
            "sport",
            "sub_sport",
            "segment_index",
            "segment_count",
            "repair_flags",
            "source_name",
            "source_path",
            "analysis_version",
        }
        sports = {activity["sport"] for activity in activities}
        assert sports == {"cycling", "running"}
        assert len(server.repository.load_sync_state().activities) == 2

    def test_empty_list_is_an_empty_array(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, {})
        with _client(server) as client:
            response = client.get("/activities")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_filters(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, _stub_files())
        server.sync_service.sync()
        with _client(server) as client:
            response = client.get("/activities", params={"sport": "running"})
            assert {item["sport"] for item in response.json()} == {"running"}
            run_day = (T0 + timedelta(days=7)).date()
            response = client.get(
                "/activities",
                params={"start_date": run_day.isoformat(), "end_date": run_day.isoformat()},
            )
            assert len(_activity_ids(response.json())) == 1

    def test_get_activity_and_not_found(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, _stub_files())
        server.sync_service.sync()
        with _client(server) as client:
            activities = client.get("/activities", params={"limit": 1}).json()
            activity_id = next(iter(_activity_ids(activities)))
            response = client.get(f"/activities/{activity_id}")
            missing = client.get("/activities/does-not-exist")
        assert response.status_code == 200
        assert _trip_ids(response.json()) == {activity_id}
        load = _named(response.json(), "training_load.hr_trimp")
        assert load["method"] == "hr_trimp"
        assert load["algorithm_version"] == "1"
        assert _named(response.json(), "heart_rate.zone_time")["value"]["shape"] == "table"
        assert missing.status_code == 404


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


class TestSummaryEndpoints:
    def test_weekly_returns_generic_tables(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, _stub_files())
        server.sync_service.sync()
        with _client(server) as client:
            response = client.get("/weekly", params={"weeks": 4})
        assert response.status_code == 200
        distance = _named(response.json(), "activity.distance_by_week")
        assert distance["value"]["shape"] == "table"
        assert len(distance["value"]["axis"]["values"]) == 2

    def test_weekly_sport_filter(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, _stub_files())
        server.sync_service.sync()
        with _client(server) as client:
            response = client.get("/weekly", params={"weeks": 4, "sport": "cycling"})
        distance = _named(response.json(), "activity.distance_by_week")
        assert len(distance["value"]["axis"]["values"]) == 1

    def test_overview_is_generic_metric_report(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, _stub_files())
        server.sync_service.sync()
        with _client(server) as client:
            response = client.get("/overview", params={"days": 14})
        report = response.json()
        assert response.status_code == 200
        assert len(_trip_ids(report)) == 2
        assert _named(report, "athlete_profile.max_heart_rate")["value"]["value"] == 180
        assert _named(report, "data_quality.activity_count")["value"]["value"] == 2
        assert _named(report, "training_form.ctl")["period"]["kind"] == "snapshot"

    def test_weekly_and_overview_sync_first(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, _stub_files())
        with _client(server) as client:
            weekly = client.get("/weekly", params={"weeks": 4})
            overview = client.get("/overview", params={"days": 14})
        assert _named(weekly.json(), "activity.distance_by_week")["value"] is not None
        assert _named(overview.json(), "data_quality.activity_count")["value"]["value"] == 2

    def test_get_activity_serves_cache_only(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, _stub_files())
        with _client(server) as client:
            response = client.get("/activities/does-not-exist")
        assert response.status_code == 404
        assert server.repository.load_sync_state().activities == {}


class TestOpenApi:
    def test_generic_paths_and_metric_shape_schemas(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, _stub_files())
        with _client(server) as client:
            schema = client.get("/openapi.json").json()
        assert set(schema["paths"]) == EXPECTED_PATHS
        components = schema["components"]["schemas"]
        assert "MetricReportWire" in components
        assert "MetricWire" in components
        assert "ScalarValueWire" in components
        assert "TableValueWire" in components
        assert "MatrixValueWire" in components
        assert "ActivitySummaryWire" in components
        assert (
            schema["paths"]["/activities"]["get"]["responses"]["200"]["content"][
                "application/json"
            ]["schema"]["items"]["$ref"]
            == "#/components/schemas/ActivitySummaryWire"
        )
        assert "description" in components["MetricWire"]["required"]
