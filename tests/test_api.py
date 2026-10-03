"""Tests for generic MetricReport HTTP responses and OpenAPI schema."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.conftest import StubSource, make_app_config, make_processor, stub_activity_content
from training_stats.adapters.bootstrap import create_service
from training_stats.adapters.http.api import build_app
from training_stats.domain.errors import TrainingStatsError
from training_stats.services.results import SyncResult
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


def _spy_sync_calls(server: TrainingStatsService) -> Callable[[], int]:
    """Count how often the underlying sync actually runs."""
    calls: list[bool] = []
    original = server.sync_service.sync

    def counting_sync() -> SyncResult:
        calls.append(True)
        return original()

    server.sync_service.sync = counting_sync
    return lambda: len(calls)


def _metrics(report: dict[str, Any]) -> list[dict[str, Any]]:
    value = report.get("metrics")
    assert isinstance(value, list)
    return value


def _named(report: dict[str, Any], name: str) -> dict[str, Any]:
    return next(metric for metric in _metrics(report) if metric["name"] == name)


def _activity_ids(activities: list[dict[str, Any]]) -> set[str]:
    return {activity["activity_id"] for activity in activities}


def _assert_no_none_fields(value: Any) -> None:
    if isinstance(value, dict):
        assert all(item is not None for item in value.values())
        for item in value.values():
            _assert_no_none_fields(item)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict | list):
                _assert_no_none_fields(item)


class TestActivitiesEndpoint:
    def test_list_syncs_first_and_returns_activity_summaries(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, _stub_files())
        with _client(server) as client:
            response = client.get("/activities")
        assert response.status_code == 200
        activities = response.json()
        _assert_no_none_fields(activities)
        assert len(_activity_ids(activities)) == 2
        assert all("metrics" not in activity for activity in activities)
        assert set(activities[0]) == {
            "activity_id",
            "start_time",
            "sport",
        }
        assert all("sub_sport" not in activity for activity in activities)
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
            _assert_no_none_fields(response.json())
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
        report = response.json()
        _assert_no_none_fields(report)
        assert report["activity_id"] == activity_id
        assert set(report) == {
            "activity_id",
            "start_time",
            "sport",
            "segment_index",
            "segment_count",
            "repair_flags",
            "source_name",
            "source_path",
            "analysis_version",
            "metrics",
        }
        assert all("period" not in metric for metric in report["metrics"])
        load = _named(report, "training_load.hr_trimp")
        assert load["method"] == "hr_trimp"
        assert load["algorithm_version"] == "1"
        assert _named(report, "heart_rate.zone_time")["value"]["shape"] == "table"
        assert missing.status_code == 404
        _assert_no_none_fields(missing.json())


class TestValidation:
    def test_invalid_date_rejected(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, _stub_files())
        server.sync_service.sync()
        with _client(server) as client:
            invalid_date = client.get("/activities", params={"start_date": "06/02/2026"})
            assert invalid_date.status_code == 422
            _assert_no_none_fields(invalid_date.json())
            assert client.get("/weekly", params={"start_date": "not-a-date"}).status_code == 422

    def test_out_of_range_int_rejected(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, _stub_files())
        server.sync_service.sync()
        with _client(server) as client:
            invalid_limit = client.get("/activities", params={"limit": 0})
            assert invalid_limit.status_code == 422
            _assert_no_none_fields(invalid_limit.json())
            assert client.get("/activities", params={"limit": 501}).status_code == 422
            assert client.get("/weekly", params={"weeks": 0}).status_code == 422
            assert client.get("/overview", params={"days": 0}).status_code == 422


class TestErrorHandlers:
    @pytest.mark.parametrize(
        ("error", "status"),
        [(ValueError, 400), (LookupError, 404), (TrainingStatsError, 500)],
    )
    def test_registered_handler_keeps_status_and_detail_body(
        self, tmp_path: Path, error: type[Exception], status: int
    ) -> None:
        server = _make_server(tmp_path, _stub_files())

        def raising(*args: object, **kwargs: object) -> None:
            raise error("handler failure")

        server.overview = raising  # type: ignore[method-assign]
        with _client(server) as client:
            response = client.get("/overview")
        assert response.status_code == status
        assert response.headers["content-type"].startswith("application/json")
        assert response.json() == {"detail": "handler failure"}


class TestSummaryEndpoints:
    def test_weekly_returns_generic_tables(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, _stub_files())
        server.sync_service.sync()
        with _client(server) as client:
            response = client.get("/weekly", params={"weeks": 4})
        assert response.status_code == 200
        report = response.json()
        _assert_no_none_fields(report)
        assert report["period"]["kind"] == "date_range"
        assert all("period" not in metric for metric in report["metrics"])
        distance = _named(report, "activity.distance_by_week")
        assert distance["value"]["shape"] == "table"
        assert len(distance["value"]["axis"]["values"]) == 2

    def test_weekly_sport_filter(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, _stub_files())
        server.sync_service.sync()
        with _client(server) as client:
            response = client.get("/weekly", params={"weeks": 4, "sport": "cycling"})
        report = response.json()
        _assert_no_none_fields(report)
        assert report["period"]["kind"] == "date_range"
        assert all("period" not in metric for metric in report["metrics"])
        distance = _named(report, "activity.distance_by_week")
        assert len(distance["value"]["axis"]["values"]) == 1

    def test_overview_is_generic_metric_report(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, _stub_files())
        server.sync_service.sync()
        with _client(server) as client:
            response = client.get("/overview", params={"days": 14})
        report = response.json()
        _assert_no_none_fields(report)
        assert response.status_code == 200
        assert all(metric["period"]["kind"] != "trip" for metric in report["metrics"])
        assert _named(report, "athlete_profile.max_heart_rate")["value"]["value"] == 180
        assert _named(report, "data_quality.activity_count")["value"]["value"] == 2
        assert _named(report, "training_form.ctl")["period"]["kind"] == "snapshot"

    def test_weekly_and_overview_sync_first(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, _stub_files())
        with _client(server) as client:
            weekly = client.get("/weekly", params={"weeks": 4})
            overview = client.get("/overview", params={"days": 14})
        _assert_no_none_fields(weekly.json())
        _assert_no_none_fields(overview.json())
        assert weekly.json()["period"]["kind"] == "date_range"
        assert all("period" not in metric for metric in weekly.json()["metrics"])
        assert all(metric["period"]["kind"] != "trip" for metric in overview.json()["metrics"])
        assert _named(weekly.json(), "activity.distance_by_week")["value"] is not None
        assert _named(overview.json(), "data_quality.activity_count")["value"]["value"] == 2

    def test_get_activity_serves_cache_only(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, _stub_files())
        with _client(server) as client:
            response = client.get("/activities/does-not-exist")
        assert response.status_code == 404
        _assert_no_none_fields(response.json())
        assert server.repository.load_sync_state().activities == {}


class TestQuerySyncBoundary:
    def test_each_query_endpoint_syncs_exactly_once(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, _stub_files())
        sync_calls = _spy_sync_calls(server)
        with _client(server) as client:
            assert client.get("/activities").status_code == 200
            assert sync_calls() == 1
            assert client.get("/weekly", params={"weeks": 4}).status_code == 200
            assert sync_calls() == 2
            assert client.get("/overview", params={"days": 14}).status_code == 200
            assert sync_calls() == 3
            assert client.get("/activities/does-not-exist").status_code == 404
            assert sync_calls() == 3

    def test_invalid_query_is_rejected_before_any_sync(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, _stub_files())
        sync_calls = _spy_sync_calls(server)
        with _client(server) as client:
            assert client.get("/activities", params={"limit": 0}).status_code == 422
            assert client.get("/weekly", params={"weeks": 0}).status_code == 422
        assert sync_calls() == 0


class TestOpenApi:
    def test_generic_paths_and_metric_shape_schemas(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, _stub_files())
        with _client(server) as client:
            schema = client.get("/openapi.json").json()
        assert set(schema["paths"]) == EXPECTED_PATHS
        components = schema["components"]["schemas"]
        assert "MetricReportWire" in components
        assert "MetricWire" in components
        assert "MetricWithoutPeriodWire" in components
        assert "WeeklyReportWire" in components
        assert "ScalarValueWire" in components
        assert "TableValueWire" in components
        assert "MatrixValueWire" in components
        assert "ActivitySummaryWire" in components
        assert "ActivityRecordWire" in components
        assert set(components["ActivitySummaryWire"]["properties"]) == {
            "activity_id",
            "start_time",
            "sport",
            "sub_sport",
        }
        assert (
            schema["paths"]["/activities"]["get"]["responses"]["200"]["content"][
                "application/json"
            ]["schema"]["items"]["$ref"]
            == "#/components/schemas/ActivitySummaryWire"
        )
        assert (
            schema["paths"]["/activities/{activity_id}"]["get"]["responses"]["200"]["content"][
                "application/json"
            ]["schema"]["$ref"]
            == "#/components/schemas/ActivityRecordWire"
        )
        assert set(components["ActivityRecordWire"]["properties"]) == {
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
            "metrics",
        }
        assert set(components["MetricWithoutPeriodWire"]["properties"]) == set(
            components["MetricWire"]["properties"]
        ) - {"period"}
        assert (
            components["ActivityRecordWire"]["properties"]["metrics"]["items"]["$ref"]
            == "#/components/schemas/MetricWithoutPeriodWire"
        )
        assert (
            schema["paths"]["/weekly"]["get"]["responses"]["200"]["content"]["application/json"][
                "schema"
            ]["$ref"]
            == "#/components/schemas/WeeklyReportWire"
        )
        assert (
            components["WeeklyReportWire"]["properties"]["period"]["$ref"]
            == "#/components/schemas/DateRangePeriodWire"
        )
        assert (
            components["WeeklyReportWire"]["properties"]["metrics"]["items"]["$ref"]
            == "#/components/schemas/MetricWithoutPeriodWire"
        )
        assert "description" in components["MetricWire"]["required"]
        assert set(components["ActivitySummaryWire"]["required"]) == {
            "activity_id",
            "start_time",
            "sport",
        }
        assert set(components["MetricWire"]["required"]) == {
            "name",
            "description",
            "period",
            "kind",
        }
