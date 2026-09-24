"""Tests for the Rich rendering adapter."""

from __future__ import annotations

import io
from collections.abc import Callable

from rich.console import Console

from training_stats.adapters.cli.output import (
    format_delta,
    format_distance,
    format_duration,
    format_start,
    render_list,
    render_overview,
    render_show,
    render_weekly,
)

Renderer = Callable[[object, Console], None]


def _render(renderer: Renderer, payload: object) -> str:
    console = Console(record=True, force_terminal=False, width=140, file=io.StringIO())
    renderer(payload, console)
    return console.export_text()


def _summary(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "activity_id": "35d3194d618850d3",
        "start_time": "2024-03-20T10:00:00+00:00",
        "local_start_time": "2024-03-20T11:00:00+01:00",
        "sport": "cycling",
        "sub_sport": "generic",
        "duration_s": 3498.0,
        "moving_time_s": 3371.0,
        "distance_m": 16740.0,
        "elevation_gain_m": 229.0,
        "avg_hr": 125.0,
        "max_hr": 142.0,
        "load_value": 109.0,
    }
    base.update(overrides)
    return base


def _details(**overrides: object) -> dict[str, object]:
    base = _summary(
        avg_speed_mps=4.78,
        max_speed_mps=7.03,
        paused_time_s=127.0,
        hr_zones_s={
            "zone_1": 190.0,
            "zone_2": 1812.0,
            "zone_3": 1496.0,
            "zone_4": 0.0,
            "zone_5": 0.0,
        },
        hr_coverage_ratio=1.0,
        hr_histogram_s={"90-94": 15.0, "120-124": 956.0, "125-129": 942.0},
        hr_10min_averages=[{"start_min": 0, "avg_hr": 120.0, "hr_time_s": 600.0}],
        hr_drift_bpm_per_hour=1.2,
        decoupling={
            "ef_first_half": 0.043,
            "ef_second_half": 0.036,
            "aerobic_decoupling_pct": 15.5,
            "confidence": "high",
        },
        load={"value": 109.0, "method": "hr_trimp", "confidence": "high"},
        data_completeness=1.0,
        analysis_version="5:1",
        data_hygiene_flags=["segments_discarded"],
        source={"name": "local", "path": "cycling.fit"},
    )
    base.update(overrides)
    return base


def _weekly_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "iso_year": 2024,
        "iso_week": 12,
        "week_start": "2024-03-18",
        "sport": "all",
        "activity_count": 1,
        "duration_s": 3498.0,
        "distance_m": 16740.0,
        "elevation_gain_m": 229.0,
        "load_value": 109.0,
        "comparison_vs_prev_week": {
            "load_value": 104.0,
            "duration_s": 3441.0,
            "distance_m": 16040.0,
        },
    }
    base.update(overrides)
    return base


def _overview() -> dict[str, object]:
    return {
        "athlete_profile": {
            "max_heart_rate": 180,
            "resting_heart_rate": 55,
            "primary_goal": "gran fondo",
            "weight_kg": 72.5,
            "ftp_watts": 250,
            "heart_rate_zones": [
                {"zone": zone, "min_bpm": 90 + 19 * (zone - 1), "max_bpm": 108 + 19 * (zone - 1)}
                for zone in range(1, 6)
            ],
        },
        "period": {"start": "2026-08-25", "end": "2026-09-21"},
        "recent_activities": [_summary()],
        "weekly_summaries": [_weekly_row()],
        "daily_series": [],
        "zone_distribution_s": {
            "zone_1": 190.0,
            "zone_2": 1812.0,
            "zone_3": 1496.0,
            "zone_4": 0.0,
            "zone_5": 0.0,
        },
        "form": {
            "ctl": 42.0,
            "atl": 38.0,
            "tsb": 4.0,
            "classification": "optimal",
            "is_warmup": False,
            "load_confidence_ratio": 0.97,
        },
        "form_history": [
            {
                "date": f"2026-09-{day:02d}",
                "ctl": 40.0 + day * 0.1,
                "atl": 38.0,
                "tsb": 2.0 + day * 0.1,
                "daily_load": 50.0,
            }
            for day in range(1, 11)
        ],
        "tour_readiness": {
            "rolling_7d_km": 120.5,
            "rolling_28d_km": 480.2,
            "days_since_60k": 3,
        },
        "latest_activity": _summary() | {"decoupling": {"aerobic_decoupling_pct": 15.5}},
        "data_quality": {
            "activity_count": 2,
            "mean_hr_coverage_ratio": 1.0,
            "mean_data_completeness": 1.0,
            "failed_file_count": 0,
            "repaired_file_count": 0,
            "discarded_segment_count": 0,
            "duplicate_activity_count": 0,
        },
    }


class TestFormatHelpers:
    def test_duration(self) -> None:
        assert format_duration(None) == "—"
        assert format_duration(0.0) == "0:00"
        assert format_duration(65.4) == "1:05"
        assert format_duration(3661.0) == "1:01:01"

    def test_distance(self) -> None:
        assert format_distance(None) == "—"
        assert format_distance(850.0) == "850 m"
        assert format_distance(16740.0) == "16.74 km"

    def test_start_prefers_local_time(self) -> None:
        assert format_start(_summary()) == "2024-03-20 11:00"
        assert format_start({"start_time": "2024-03-20T10:00:00+00:00"}) == (
            "2024-03-20T10:00:00+00:00"
        )
        assert format_start({"local_start_time": "not-a-date"}) == "not-a-date"
        assert format_start({}) == "—"

    def test_delta(self) -> None:
        assert format_delta(None, 100.0).plain == "—"
        assert format_delta(120.0, 100.0).plain == "+20"
        assert format_delta(120.0, 100.0).style == "green"
        assert format_delta(80.0, 100.0).style == "red"
        assert format_delta(100.0, 100.0).style == "dim"


class TestListRendering:
    def test_empty(self) -> None:
        assert "No activities found." in _render(render_list, [])

    def test_rows(self) -> None:
        out = _render(render_list, [_summary()])
        assert "2024-03-20 11:00" in out
        assert "cycling (generic)" in out
        assert "58:18" in out
        assert "16.74 km" in out
        assert "125" in out
        assert "35d3194d618850d3" in out


class TestWeeklyRendering:
    def test_empty(self) -> None:
        assert "No weekly data found." in _render(render_weekly, [])

    def test_row_with_comparison(self) -> None:
        out = _render(render_weekly, [_weekly_row()])
        assert "2024-W12" in out
        assert "2024-03-18" in out
        assert "109" in out
        assert "+5" in out  # load delta
        assert "+0.7 km" in out  # distance delta


class TestShowRendering:
    def test_sections(self) -> None:
        out = _render(render_show, _details())
        assert "cycling (generic)" in out
        assert "Metrics" in out
        assert "Heart rate zones" in out
        assert "HR histogram" in out
        assert "10-minute averages" in out
        assert "hr_trimp" in out
        assert "Decoupling" in out
        assert "segments_discarded" in out
        assert "local: cycling.fit" in out

    def test_optional_sections_absent(self) -> None:
        out = _render(render_show, _summary())
        assert "Heart rate zones" in out
        assert "HR histogram" not in out
        assert "Decoupling" not in out


class TestOverviewRendering:
    def test_sections(self) -> None:
        out = _render(render_overview, _overview())
        assert "Training overview" in out
        assert "2026-08-25 → 2026-09-21" in out
        assert "gran fondo" in out
        assert "CTL (fitness)" in out
        assert "TSB trend" in out
        assert "Tour readiness" in out
        assert "Weekly summaries" in out
        assert "Recent activities" in out
        assert "Zone distribution" in out
        assert "Data quality" in out
        assert "Latest activity" in out
