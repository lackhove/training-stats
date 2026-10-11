"""Tests for generic MetricReport HTTP responses and OpenAPI schema."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

from tests.conftest import (
    StubParser,
    assert_no_none_fields,
    json_activity_ids,
    json_metric,
    make_server,
    spy_sync_calls,
    stub_activity_content,
)
from training_stats.adapters.http.api import build_app
from training_stats.domain.errors import TrainingStatsError
from training_stats.domain.fit.types import FitData
from training_stats.domain.models import SourceFile
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


def _client(server: TrainingStatsService) -> TestClient:
    return TestClient(build_app(server))


class TestActivitiesEndpoint:
    def test_list_syncs_first_and_returns_activity_summaries(self, tmp_path: Path) -> None:
        server = make_server(tmp_path, _stub_files())
        with _client(server) as client:
            response = client.get("/activities")
        assert response.status_code == 200
        activities = response.json()
        assert_no_none_fields(activities)
        assert len(json_activity_ids(activities)) == 2
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
        server = make_server(tmp_path, {})
        with _client(server) as client:
            response = client.get("/activities")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_filters(self, tmp_path: Path) -> None:
        server = make_server(tmp_path, _stub_files())
        server.sync_service.sync()
        with _client(server) as client:
            response = client.get("/activities", params={"sport": "running"})
            assert_no_none_fields(response.json())
            assert {item["sport"] for item in response.json()} == {"running"}
            run_day = (T0 + timedelta(days=7)).date()
            response = client.get(
                "/activities",
                params={"start_date": run_day.isoformat(), "end_date": run_day.isoformat()},
            )
            assert len(json_activity_ids(response.json())) == 1

    def test_get_activity_and_not_found(self, tmp_path: Path) -> None:
        server = make_server(tmp_path, _stub_files())
        server.sync_service.sync()
        with _client(server) as client:
            activities = client.get("/activities", params={"limit": 1}).json()
            activity_id = next(iter(json_activity_ids(activities)))
            response = client.get(f"/activities/{activity_id}")
            missing = client.get("/activities/does-not-exist")
        assert response.status_code == 200
        report = response.json()
        assert_no_none_fields(report)
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
        load = json_metric(report, "training_load.hr_trimp")
        assert load["method"] == "hr_trimp"
        assert load["algorithm_version"] == "1"
        assert json_metric(report, "heart_rate.zone_time")["value"]["shape"] == "table"
        assert missing.status_code == 404
        assert_no_none_fields(missing.json())

    def test_activity_detail_omits_analysis_for_unmeasured_quantities(self, tmp_path: Path) -> None:
        server = make_server(
            tmp_path,
            {"no-sensors.fit": stub_activity_content(T0, "running", 1800.0, 8000.0)},
        )
        server.sync_service.sync()
        activity_id = next(iter(server.repository.load_sync_state().activities))

        with _client(server) as client:
            report = client.get(f"/activities/{activity_id}").json()

        names = {metric["name"] for metric in report["metrics"]}
        assert "activity.average_heart_rate" not in names
        assert "activity.maximum_heart_rate" not in names
        assert "activity.elevation_gain" not in names
        assert "heart_rate.zone_time" not in names
        assert "heart_rate.histogram" not in names
        assert "training_load.hr_trimp" not in names
        assert "activity.statistics.heart_rate_statistics.mean" not in names
        assert "activity.statistics.cadence_statistics.overall.mean" not in names
        assert "activity.statistics.power_statistics.overall.mean" not in names
        assert "activity.statistics.efficiency.speed_per_heart_rate" not in names
        assert "activity.statistics.relationships.speed_heart_rate.window_count" not in names
        assert "activity.statistics.relationships.power_cadence.window_count" not in names
        assert (
            "activity.statistics.heatmap_data.heart_rate_speed.valid_observation_count" not in names
        )
        assert "activity.statistics.heatmap_data.power_cadence.valid_observation_count" not in names

        assert json_metric(report, "heart_rate.coverage")["value"]["value"] == 0.0
        assert (
            json_metric(report, "activity.statistics.heart_rate_statistics.coverage_ratio")[
                "value"
            ]["value"]
            == 0.0
        )
        assert (
            json_metric(report, "activity.statistics.cadence_statistics.available")["value"][
                "value"
            ]
            is False
        )
        assert (
            json_metric(report, "activity.statistics.cadence_statistics.coverage_ratio")["value"][
                "value"
            ]
            == 0.0
        )
        assert (
            json_metric(report, "activity.statistics.power_statistics.available")["value"]["value"]
            is False
        )
        assert (
            json_metric(report, "activity.statistics.power_statistics.coverage_ratio")["value"][
                "value"
            ]
            == 0.0
        )
        assert "data.completeness" in names
        assert "activity.statistics.data_quality.heart_rate_coverage_ratio" in names
        assert "present_quantities" not in report
        assert "records" not in report

    def test_activity_detail_omits_movement_analysis_without_speed_or_distance(
        self, tmp_path: Path
    ) -> None:
        class NoMovementSensorsParser(StubParser):
            def parse(self, content: bytes, source_file: SourceFile) -> FitData:
                fit_data = super().parse(content, source_file)
                return FitData(
                    frame=fit_data.frame.with_columns(
                        pl.lit(None, dtype=pl.Float64).alias("speed_mps"),
                        pl.lit(None, dtype=pl.Float64).alias("distance_m"),
                    )
                )

        server = make_server(
            tmp_path,
            {"no-movement-sensors.fit": stub_activity_content(T0, "running", 1800.0, 8000.0)},
        )
        server.sync_service.processor.parser = NoMovementSensorsParser()
        server.sync_service.sync()
        activity_id = next(iter(server.repository.load_sync_state().activities))

        with _client(server) as client:
            report = client.get(f"/activities/{activity_id}").json()

        names = {metric["name"] for metric in report["metrics"]}
        assert "activity.distance" not in names
        assert "activity.average_speed" not in names
        assert "activity.maximum_speed" not in names
        assert "activity.statistics.speed_statistics.mean" not in names
        assert "activity.statistics.time_windows.items.distance_m" not in names
        assert "activity.statistics.time_windows.items.average_speed_kmh" not in names
        assert (
            json_metric(report, "activity.statistics.speed_statistics.coverage_ratio")["value"][
                "value"
            ]
            == 0.0
        )
        assert (
            json_metric(report, "activity.statistics.speed_statistics.valid_moving_time_ratio")[
                "value"
            ]["value"]
            == 0.0
        )
        assert (
            json_metric(report, "activity.statistics.data_quality.speed_coverage_ratio")["value"][
                "value"
            ]
            == 0.0
        )

    def test_zero_heart_rate_sample_counts_as_measured(self, tmp_path: Path) -> None:
        server = make_server(
            tmp_path,
            {
                "zero-hr.fit": stub_activity_content(
                    T0, "cycling", 1800.0, 8000.0, heart_rates=[0] * 31
                )
            },
        )
        server.sync_service.sync()
        activity_id = next(iter(server.repository.load_sync_state().activities))

        with _client(server) as client:
            report = client.get(f"/activities/{activity_id}").json()

        names = {metric["name"] for metric in report["metrics"]}
        assert "heart_rate.zone_time" in names
        assert "activity.statistics.heart_rate_statistics.mean" in names
        assert (
            json_metric(report, "activity.statistics.heart_rate_statistics.mean")["missing_reason"]
            == "insufficient_data"
        )


class TestValidation:
    def test_invalid_date_rejected(self, tmp_path: Path) -> None:
        server = make_server(tmp_path, _stub_files())
        server.sync_service.sync()
        with _client(server) as client:
            invalid_date = client.get("/activities", params={"start_date": "06/02/2026"})
            assert invalid_date.status_code == 422
            assert_no_none_fields(invalid_date.json())
            assert client.get("/weekly", params={"start_date": "not-a-date"}).status_code == 422

    def test_out_of_range_int_rejected(self, tmp_path: Path) -> None:
        server = make_server(tmp_path, _stub_files())
        server.sync_service.sync()
        with _client(server) as client:
            invalid_limit = client.get("/activities", params={"limit": 0})
            assert invalid_limit.status_code == 422
            assert_no_none_fields(invalid_limit.json())
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
        server = make_server(tmp_path, _stub_files())

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
        server = make_server(tmp_path, _stub_files())
        server.sync_service.sync()
        with _client(server) as client:
            response = client.get("/weekly", params={"weeks": 4})
        assert response.status_code == 200
        report = response.json()
        assert_no_none_fields(report)
        assert report["period"]["kind"] == "date_range"
        assert all("period" not in metric for metric in report["metrics"])
        distance = json_metric(report, "activity.distance_by_week")
        assert distance["value"]["shape"] == "table"
        assert len(distance["value"]["axis"]["values"]) == 2

    def test_weekly_sport_filter(self, tmp_path: Path) -> None:
        server = make_server(tmp_path, _stub_files())
        server.sync_service.sync()
        with _client(server) as client:
            response = client.get("/weekly", params={"weeks": 4, "sport": "cycling"})
        report = response.json()
        assert_no_none_fields(report)
        assert report["period"]["kind"] == "date_range"
        assert all("period" not in metric for metric in report["metrics"])
        distance = json_metric(report, "activity.distance_by_week")
        assert len(distance["value"]["axis"]["values"]) == 1

    def test_weekly_quantity_aware_metrics_preserve_full_activity_counts(
        self, tmp_path: Path
    ) -> None:
        server = make_server(
            tmp_path,
            {
                "measured.fit": stub_activity_content(
                    T0, "cycling", 1800.0, 10000.0, heart_rates=[120] * 31
                ),
                "no-hr-same-week.fit": stub_activity_content(
                    T0 + timedelta(days=1), "cycling", 1800.0, 20000.0
                ),
                "no-hr-next-week.fit": stub_activity_content(
                    T0 + timedelta(days=7), "running", 1800.0, 30000.0
                ),
            },
        )
        server.sync_service.sync()

        with _client(server) as client:
            report = client.get("/weekly", params={"weeks": 4}).json()
            running = client.get("/weekly", params={"weeks": 4, "sport": "running"}).json()
            activities = server.repository.load_sync_state().activities.values()
            measured = next(
                item
                for item in activities
                if "training_load.hr_trimp" in {metric.name for metric in item.metrics}
            )
            detail = client.get(f"/activities/{measured.activity_id}").json()

        assert json_metric(report, "activity.count_by_week")["value"]["values"] == [1, 2]
        weekly_load = json_metric(report, "training_load.hr_trimp_by_week")
        assert weekly_load["value"]["values"][0] is None
        assert weekly_load["method"] == "hr_trimp"
        assert weekly_load["algorithm_version"] == "1"
        assert (
            weekly_load["value"]["values"][1]
            == json_metric(detail, "training_load.hr_trimp")["value"]["value"]
        )
        assert json_metric(report, "heart_rate.zone_1_time_by_week")["value"]["values"][0] is None
        assert "training_load.hr_trimp_by_week" not in {
            metric["name"] for metric in running["metrics"]
        }
        assert "heart_rate.zone_1_time_by_week" not in {
            metric["name"] for metric in running["metrics"]
        }

    def test_overview_is_generic_metric_report(self, tmp_path: Path) -> None:
        server = make_server(tmp_path, _stub_files())
        server.sync_service.sync()
        with _client(server) as client:
            response = client.get("/overview", params={"days": 14})
        report = response.json()
        assert_no_none_fields(report)
        assert response.status_code == 200
        assert all(metric["period"]["kind"] != "trip" for metric in report["metrics"])
        assert json_metric(report, "athlete_profile.max_heart_rate")["value"]["value"] == 180
        assert json_metric(report, "data_quality.activity_count")["value"]["value"] == 2
        assert json_metric(report, "training_form.ctl")["period"]["kind"] == "snapshot"

    def test_weekly_and_overview_sync_first(self, tmp_path: Path) -> None:
        server = make_server(tmp_path, _stub_files())
        with _client(server) as client:
            weekly = client.get("/weekly", params={"weeks": 4})
            overview = client.get("/overview", params={"days": 14})
        assert_no_none_fields(weekly.json())
        assert_no_none_fields(overview.json())
        assert weekly.json()["period"]["kind"] == "date_range"
        assert all("period" not in metric for metric in weekly.json()["metrics"])
        assert all(metric["period"]["kind"] != "trip" for metric in overview.json()["metrics"])
        assert json_metric(weekly.json(), "activity.distance_by_week")["value"] is not None
        assert json_metric(overview.json(), "data_quality.activity_count")["value"]["value"] == 2

    def test_get_activity_serves_cache_only(self, tmp_path: Path) -> None:
        server = make_server(tmp_path, _stub_files())
        with _client(server) as client:
            response = client.get("/activities/does-not-exist")
        assert response.status_code == 404
        assert_no_none_fields(response.json())
        assert server.repository.load_sync_state().activities == {}


class TestQuerySyncBoundary:
    def test_each_query_endpoint_syncs_exactly_once(self, tmp_path: Path) -> None:
        server = make_server(tmp_path, _stub_files())
        sync_calls = spy_sync_calls(server)
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
        server = make_server(tmp_path, _stub_files())
        sync_calls = spy_sync_calls(server)
        with _client(server) as client:
            assert client.get("/activities", params={"limit": 0}).status_code == 422
            assert client.get("/weekly", params={"weeks": 0}).status_code == 422
        assert sync_calls() == 0


class TestOpenApi:
    def test_generic_paths_and_metric_shape_schemas(self, tmp_path: Path) -> None:
        server = make_server(tmp_path, _stub_files())
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
