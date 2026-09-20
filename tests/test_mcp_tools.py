"""Tests for MCP tool implementations and the FastMCP wiring."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tests.conftest import StubParser, StubSource, make_app_config, stub_activity_content
from training_coach_mcp.server import TrainingCoachServer, create_server

# Fixture activities anchored to "today" so trends/current-week windows match.
T0 = (datetime.now(UTC) - timedelta(days=8)).replace(hour=10, minute=0, second=0, microsecond=0)


def _stub_files() -> dict[str, bytes]:
    return {
        "easy.fit": stub_activity_content(T0, "cycling", 3600.0, 30000.0, heart_rates=[120] * 60),
        "run.fit": stub_activity_content(
            T0 + timedelta(days=7), "running", 1800.0, 8000.0, heart_rates=[150] * 30
        ),
    }


def _make_server(tmp_path: Path, files: dict[str, bytes]) -> TrainingCoachServer:
    """Build a server backed by the deterministic stub source/parser."""
    config = make_app_config(tmp_path / "cache", tmp_path / "src")
    server = create_server(config)
    # Swap in the deterministic stub source/parser for cache independence.
    server.sync_service.sources = [StubSource(files=files)]
    server.sync_service.parser = StubParser()
    return server


@pytest.fixture
def server(tmp_path: Path) -> TrainingCoachServer:
    server = _make_server(tmp_path, _stub_files())
    server.sync_service.sync()
    return server


@pytest.fixture
def unsynced_server(tmp_path: Path) -> TrainingCoachServer:
    """Server whose cache is empty; sync only happens via the code under test."""
    return _make_server(tmp_path, _stub_files())


class TestToolImplementations:
    def test_sync_activities(self, server: TrainingCoachServer) -> None:
        result = server.sync_activities()
        assert result["files_processed"] == 0
        assert result["files_skipped"] == 2

    def test_list_activities_syncs_first(self, unsynced_server: TrainingCoachServer) -> None:
        assert unsynced_server.repository.list_activities(limit=10) == []
        activities = unsynced_server.list_activities()
        assert len(activities) == 2
        assert unsynced_server.repository.list_activities(limit=10) != []

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
        with caplog.at_level(logging.WARNING, logger="training_coach_mcp.server"):
            activities = server.list_activities()
        assert len(activities) == 2
        assert any("activity sync failed" in record.message for record in caplog.records)

    def test_startup_sync_populates_cache(self, unsynced_server: TrainingCoachServer) -> None:
        from training_coach_mcp.__main__ import _startup_sync

        assert unsynced_server.repository.list_activities(limit=10) == []
        _startup_sync(unsynced_server)
        assert len(unsynced_server.repository.list_activities(limit=10)) == 2

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
        run_date = (T0 + timedelta(days=7)).date().isoformat()
        activities = server.list_activities(start_date=run_date, end_date=run_date)
        assert [item["sport"] for item in activities] == ["running"]

    def test_list_activities_invalid_date(self, server: TrainingCoachServer) -> None:
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            server.list_activities(start_date="06/02/2026")

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

    def test_get_activity_histogram_and_drift(self, server: TrainingCoachServer) -> None:
        activities = server.list_activities(limit=1)
        details = server.get_activity(str(activities[0]["activity_id"]))
        histogram = details["hr_histogram_s"]
        assert isinstance(histogram, dict) and histogram
        # Histogram covers all valid HR time.
        zones_total = sum(details["hr_zones_s"].values())
        assert sum(histogram.values()) >= zones_total
        buckets = details["hr_10min_averages"]
        assert isinstance(buckets, list) and buckets
        assert {"start_min", "avg_hr", "hr_time_s"} <= set(buckets[0])
        assert "hr_drift_bpm_per_hour" in details

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

    def test_training_trends(self, server: TrainingCoachServer) -> None:
        trend = server.training_trends("load", days=14)
        assert trend["metric"] == "load"
        assert len(trend["points"]) == 2

    def test_training_trends_invalid_metric(self, server: TrainingCoachServer) -> None:
        with pytest.raises(ValueError, match="unknown metric"):
            server.training_trends("watts", days=14)

    def test_training_trends_invalid_days(self, server: TrainingCoachServer) -> None:
        with pytest.raises(ValueError, match="days"):
            server.training_trends("load", days=0)

    def test_training_form(self, server: TrainingCoachServer) -> None:
        form = server.training_form(days=14)
        assert "current_ctl" in form
        assert "current_atl" in form
        assert "current_tsb" in form
        assert form["form_classification"] in (
            "fresh",
            "optimal",
            "high_fatigue",
            "overtraining_risk",
        )
        assert isinstance(form["is_warmup"], bool)
        assert isinstance(form["history_days"], int)
        assert "rolling_7d_distance_km" in form
        assert "rolling_28d_distance_km" in form
        assert "days_since_60k_ride" in form
        assert isinstance(form["trend_points"], list)
        assert len(form["trend_points"]) <= 14
        assert "load_confidence_ratio" in form

    def test_training_form_invalid_days(self, server: TrainingCoachServer) -> None:
        with pytest.raises(ValueError, match="days"):
            server.training_form(days=0)

    def test_get_coach_context(self, server: TrainingCoachServer) -> None:
        context = server.get_coach_context(days=14)
        assert context["athlete_profile"]["max_heart_rate"] == 180
        assert len(context["recent_activities"]) == 2
        assert context["data_quality"]["activity_count"] == 2
        assert context["latest_activity"] is not None
        assert "form" in context
        assert {"ctl", "atl", "tsb", "classification", "is_warmup"} <= set(context["form"])
        assert "tour_readiness" in context
        assert {"rolling_7d_km", "rolling_28d_km", "days_since_60k"} <= set(
            context["tour_readiness"]
        )
        assert "decoupling" in context["latest_activity"]

    def test_get_coach_context_invalid_days(self, server: TrainingCoachServer) -> None:
        with pytest.raises(ValueError, match="days"):
            server.get_coach_context(days=-1)


class TestResources:
    def test_athlete_profile(self, server: TrainingCoachServer) -> None:
        profile = server.athlete_profile_resource()
        assert profile["max_heart_rate"] == 180
        assert len(profile["heart_rate_zones"]) == 5

    def test_zones(self, server: TrainingCoachServer) -> None:
        zones = server.zones_resource()
        assert zones["zones"][0]["min_bpm"] == 90

    def test_data_quality(self, server: TrainingCoachServer) -> None:
        quality = server.data_quality_resource()
        assert quality["activity_count"] == 2

    def test_current_week(self, server: TrainingCoachServer) -> None:
        current = server.current_week_resource()
        assert current["activity_count"] >= 1

    def test_training_form_resource(self, server: TrainingCoachServer) -> None:
        form = server.training_form_resource()
        assert "current_ctl" in form
        assert "form_classification" in form


class TestFastMCPWiring:
    def test_tools_and_resources_registered(self, server: TrainingCoachServer) -> None:
        async def scenario() -> tuple[list[str], list[str]]:
            from fastmcp import Client

            mcp = server.build_mcp()
            async with Client(mcp) as client:
                tools = await client.list_tools()
                resources = await client.list_resources()
                return (
                    [tool.name for tool in tools],
                    [str(resource.uri) for resource in resources],
                )

        tool_names, resource_uris = asyncio.run(scenario())
        assert set(tool_names) == {
            "list_activities",
            "get_activity",
            "weekly_summary",
            "training_trends",
            "training_form",
            "get_coach_context",
        }
        assert set(resource_uris) == {
            "athlete://profile",
            "training://zones",
            "training://data-quality",
            "training://current-week",
            "training://form",
        }

    def test_tool_call_via_client(self, server: TrainingCoachServer) -> None:
        async def scenario() -> dict[str, object]:
            from fastmcp import Client

            mcp = server.build_mcp()
            async with Client(mcp) as client:
                result = await client.call_tool("list_activities", {"limit": 5})
                assert result.data is not None
                return result.data[0]

        first = asyncio.run(scenario())
        assert first["sport"] in ("cycling", "running")

    def test_resource_read_via_client(self, server: TrainingCoachServer) -> None:
        async def scenario() -> str:
            from fastmcp import Client

            mcp = server.build_mcp()
            async with Client(mcp) as client:
                blocks = await client.read_resource("training://zones")
                return str(blocks[0].text)

        text = asyncio.run(scenario())
        assert "min_bpm" in text

    def test_training_form_tool_via_client(self, server: TrainingCoachServer) -> None:
        async def scenario() -> dict[str, object]:
            from fastmcp import Client

            mcp = server.build_mcp()
            async with Client(mcp) as client:
                result = await client.call_tool("training_form", {"days": 14})
                assert result.data is not None
                return result.data

        form = asyncio.run(scenario())
        assert "current_ctl" in form
        assert "form_classification" in form

    def test_training_form_resource_via_client(self, server: TrainingCoachServer) -> None:
        async def scenario() -> str:
            from fastmcp import Client

            mcp = server.build_mcp()
            async with Client(mcp) as client:
                blocks = await client.read_resource("training://form")
                return str(blocks[0].text)

        text = asyncio.run(scenario())
        assert "current_ctl" in text

    def test_tool_error_via_client(self, server: TrainingCoachServer) -> None:
        async def scenario() -> bool:
            from fastmcp import Client
            from fastmcp.exceptions import ToolError

            mcp = server.build_mcp()
            async with Client(mcp) as client:
                try:
                    await client.call_tool("training_trends", {"metric": "watts", "days": 30})
                except ToolError:
                    return True
                return False

        assert asyncio.run(scenario())


class TestCreateServer:
    def test_local_source_built(self, tmp_path: Path) -> None:
        config = make_app_config(tmp_path / "cache", tmp_path / "activities")
        server = create_server(config)
        assert [source.name for source in server.sources] == ["local"]

    def test_webdav_source_built(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from training_coach_mcp.config import (
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
