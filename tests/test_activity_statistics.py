"""Regression tests for coarse multi-sensor activity statistics."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from tests.conftest import make_athlete
from training_stats.domain.analytics.activity_statistics import (
    _compute_activity_statistics_data,
    compute_activity_statistics,
)
from training_stats.domain.fit.types import RECORD_SCHEMA, ParsedActivity, PauseInterval
from training_stats.domain.metrics import MatrixValue, TableValue
from training_stats.domain.settings import QualityConfig


def _activity(
    *, cadence: bool = True, sparse_cadence: bool = False, pause: bool = False
) -> ParsedActivity:
    start = datetime(2026, 6, 1, 10, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    for index in range(31):
        speed = 5.0 if index < 10 else 6.0 if index < 20 else 7.0
        heart_rate = 120 if index < 10 else 145 if index < 20 else 160
        cadence_value: int | None = None
        if cadence:
            cadence_value = (
                None
                if sparse_cadence and index % 5 != 0
                else 0
                if index < 5
                else 80
                if index < 20
                else 95
            )
        rows.append(
            {
                "timestamp": start + timedelta(seconds=index * 30),
                "heart_rate": heart_rate,
                "cadence": cadence_value,
                "fractional_cadence": 0,
                "distance_m": index * 180.0,
                "altitude_m": None,
                "speed_mps": speed,
            }
        )
    return ParsedActivity(
        activity_id="activity-statistics",
        start_time=start,
        sport="cycling",
        sub_sport=None,
        duration_s=900.0,
        moving_time_s=900.0 if not pause else 600.0,
        distance_m=5400.0,
        elevation_gain_m=None,
        avg_hr=140.0,
        max_hr=160,
        avg_speed_mps=6.0,
        max_speed_mps=7.0,
        records=pl.DataFrame(rows, schema=RECORD_SCHEMA),
        source_path="synthetic.fit",
        paused_time_s=0.0 if not pause else 300.0,
        pause_intervals=(
            (PauseInterval(start + timedelta(seconds=300), start + timedelta(seconds=600)),)
            if pause
            else ()
        ),
    )


class TestDetailedActivityStatistics:
    def test_speed_hr_cadence_and_zone_statistics(self) -> None:
        result = _compute_activity_statistics_data(
            _activity(),
            make_athlete(),
            QualityConfig(decoupling_min_duration_s=600.0),
        )

        speed = result["speed_statistics"]
        assert isinstance(speed, dict)
        assert speed["mean"] == pytest.approx(21.6)
        assert next(
            item["seconds"]
            for item in speed["speed_above_threshold_s"]
            if item["threshold_kmh"] == 20.0
        ) == pytest.approx(600.0)
        assert speed["valid_moving_time_ratio"] == pytest.approx(1.0)

        cadence = result["cadence_statistics"]
        assert isinstance(cadence, dict)
        assert cadence["available"] is True
        assert cadence["pedaling_time_s"] == pytest.approx(750.0)
        assert cadence["no_pedaling_time_s"] == pytest.approx(150.0)
        cadence_bands = {item["band"]: item["seconds"] for item in cadence["bands_s"]}
        assert cadence_bands["rpm_80_89"] == pytest.approx(450.0)
        assert cadence_bands["rpm_90_99"] == pytest.approx(300.0)

        zones = result["zone_distribution"]
        assert isinstance(zones, dict)
        assert zones["seconds"]["zone_2"] == pytest.approx(300.0)
        assert zones["seconds"]["zone_4"] == pytest.approx(600.0)
        assert zones["percent"]["zone_2"] == pytest.approx(100.0 / 3.0)

    def test_windows_correlations_and_heatmaps_use_coarse_intervals(self) -> None:
        result = _compute_activity_statistics_data(
            _activity(), make_athlete(), QualityConfig(decoupling_min_duration_s=600.0)
        )
        windows = result["time_windows"]
        assert isinstance(windows, dict)
        assert windows["time_basis"] == "elapsed_time"
        assert len(windows["items"]) == 3
        assert windows["items"][0]["start_s"] == 0.0
        assert windows["items"][0]["end_s"] == 300.0
        assert windows["movement_time_variant"] is None

        relationships = result["relationships"]
        assert isinstance(relationships, dict)
        assert relationships["speed_heart_rate"]["window_count"] == 3
        assert relationships["speed_heart_rate"]["spearman"] == pytest.approx(1.0)

        heatmaps = result["heatmap_data"]
        assert isinstance(heatmaps, dict)
        assert heatmaps["heart_rate_speed"]["valid_observation_count"] == 30
        assert heatmaps["heart_rate_speed"]["bins"]

        efficiency = result["efficiency"]
        assert isinstance(efficiency, dict)
        assert efficiency["first_half_average_speed_kmh"] == pytest.approx(19.2)
        assert efficiency["second_half_average_speed_kmh"] == pytest.approx(24.0)
        assert efficiency["first_half_average_heart_rate_bpm"] == pytest.approx(128.333333)
        assert efficiency["second_half_average_heart_rate_bpm"] == pytest.approx(155.0)
        assert efficiency["aerobic_decoupling_pct"] is not None

    def test_absent_cadence_is_never_inferred(self) -> None:
        result = _compute_activity_statistics_data(
            _activity(cadence=False), make_athlete(), QualityConfig()
        )
        cadence = result["cadence_statistics"]
        assert isinstance(cadence, dict)
        assert cadence["available"] is False
        assert cadence["reason"] == "no_cadence_data"
        assert cadence["pedaling"]["mean"] is None

    def test_incomplete_cadence_is_reported_but_not_summarized(self) -> None:
        result = _compute_activity_statistics_data(
            _activity(sparse_cadence=True), make_athlete(), QualityConfig()
        )
        cadence = result["cadence_statistics"]
        assert isinstance(cadence, dict)
        assert cadence["available"] is False
        assert cadence["reason"] == "insufficient_coverage"
        assert cadence["coverage_ratio"] < 0.8
        assert cadence["pedaling"]["mean"] is None
        windows = result["time_windows"]
        assert isinstance(windows, dict)
        assert windows["items"][0]["average_pedaling_cadence_rpm"] is None
        heatmaps = result["heatmap_data"]
        assert isinstance(heatmaps, dict)
        assert heatmaps["heart_rate_cadence"]["valid_observation_count"] == 0

    def test_paused_time_is_excluded_from_sensor_statistics(self) -> None:
        result = _compute_activity_statistics_data(
            _activity(pause=True), make_athlete(), QualityConfig()
        )
        quality = result["data_quality"]
        assert isinstance(quality, dict)
        assert quality["pause_intervals"] == 1
        assert quality["heart_rate_coverage_ratio"] <= 1.0

    def test_public_function_returns_metrics_including_heatmap(self) -> None:
        metrics = compute_activity_statistics(
            _activity(), make_athlete(), QualityConfig(decoupling_min_duration_s=600.0)
        )
        assert all(metric.period.activity_id == "activity-statistics" for metric in metrics)
        assert any(metric.name == "activity.statistics.speed_statistics.mean" for metric in metrics)
        heatmap = next(
            metric
            for metric in metrics
            if metric.name == "activity.statistics.heatmap_data.heart_rate_speed.duration_s"
        )
        assert isinstance(heatmap.value, MatrixValue)
        histogram = next(
            metric
            for metric in metrics
            if metric.name == "activity.statistics.heart_rate_statistics.histogram_s.seconds"
        )
        assert isinstance(histogram.value, TableValue)
