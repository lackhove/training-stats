"""Pause-aware statistics and active-time segment retention tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from tests.conftest import make_athlete
from training_stats.domain.analytics.segments import SegmentAnalysis, analyze_segments
from training_stats.domain.fit.repair import repair_timeline
from training_stats.domain.fit.types import FitData
from training_stats.domain.metrics import MissingReason, ScalarValue, TableValue
from training_stats.domain.settings import QualityConfig

START = datetime(2026, 9, 17, 8, 0, tzinfo=UTC)


def _fit_data(
    heart_rates: list[int],
    speeds: list[float | None],
    distances: list[float],
    *,
    timer_events: tuple[tuple[int, str], ...] = (),
    cadences: list[int | None] | None = None,
    fractional_cadences: list[int | None] | None = None,
    interval_s: int = 30,
) -> FitData:
    cadence_values = cadences if cadences is not None else [None] * len(heart_rates)
    fractional_cadence_values = (
        fractional_cadences if fractional_cadences is not None else [None] * len(heart_rates)
    )
    rows: list[dict[str, object]] = [
        {
            "message_type": "session",
            "timestamp": None,
            "heart_rate": None,
            "cadence": None,
            "fractional_cadence": None,
            "distance_m": None,
            "altitude_m": None,
            "speed_mps": None,
            "sport": "cycling",
            "sub_sport": "generic",
            "event": None,
            "event_type": None,
        }
    ]
    for index, (hr, speed, distance, cadence, fractional_cadence) in enumerate(
        zip(
            heart_rates,
            speeds,
            distances,
            cadence_values,
            fractional_cadence_values,
            strict=True,
        )
    ):
        rows.append(
            {
                "message_type": "record",
                "timestamp": START + timedelta(seconds=interval_s * index),
                "heart_rate": hr,
                "cadence": cadence,
                "fractional_cadence": fractional_cadence,
                "distance_m": distance,
                "altitude_m": 100.0,
                "speed_mps": speed,
                "sport": None,
                "sub_sport": None,
                "event": None,
                "event_type": None,
            }
        )
    for seconds, event_type in timer_events:
        rows.append(
            {
                "message_type": "event",
                "timestamp": START + timedelta(seconds=seconds),
                "heart_rate": None,
                "cadence": None,
                "fractional_cadence": None,
                "distance_m": None,
                "altitude_m": None,
                "speed_mps": None,
                "sport": None,
                "sub_sport": None,
                "event": "timer",
                "event_type": event_type,
            }
        )
    return FitData(frame=pl.DataFrame(rows, schema=FitData.SCHEMA))


def _analyze(
    data: FitData, *, minimum_s: float = 1.0, quality: QualityConfig | None = None
) -> SegmentAnalysis:
    effective_quality = quality or QualityConfig(minimum_segment_duration_s=minimum_s)
    segments = repair_timeline(
        data,
        gap_threshold_s=effective_quality.activity_gap_threshold_s,
        distance_gap_threshold_m=effective_quality.activity_distance_gap_threshold_m,
    )
    return analyze_segments(
        segments,
        source_path="test.fit",
        athlete=make_athlete(),
        analysis_version="test",
        quality=effective_quality,
    )


def _paused_activity(timer_events: bool) -> FitData:
    return _fit_data(
        heart_rates=[100, 100, 100, 100, 190, 190, 190, 190, 120, 120, 120, 120, 120],
        speeds=[
            5.0,
            5.0,
            5.0,
            5.0,
            *([10.0] * 4 if timer_events else [0.0] * 4),
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
        ],
        distances=[
            0.0,
            150.0,
            300.0,
            450.0,
            600.0,
            600.0,
            600.0,
            600.0,
            600.0,
            750.0,
            900.0,
            1050.0,
            1200.0,
        ],
        timer_events=((120, "stop"), (240, "start")) if timer_events else (),
    )


class TestPauseAwareStatistics:
    def test_speed_coverage_controls_speed_based_pause_inference(self) -> None:
        data = _fit_data(
            heart_rates=[120] * 5,
            speeds=[0.0, 0.0, None, None, 0.0],
            distances=[0.0] * 5,
        )

        default = _analyze(data).activities[0].activity
        strict = (
            _analyze(
                data,
                quality=QualityConfig(minimum_segment_duration_s=1.0, speed_min_coverage=0.75),
            )
            .activities[0]
            .activity
        )

        assert default.moving_time_s == pytest.approx(60.0)
        assert strict.moving_time_s == pytest.approx(120.0)

    def test_max_moving_interval_caps_segment_and_activity_statistics(self) -> None:
        data = _fit_data(
            heart_rates=[120, 120, 120],
            speeds=[5.0, 5.0, 5.0],
            distances=[0.0, 150.0, 300.0],
        )
        result = _analyze(
            data,
            quality=QualityConfig(minimum_segment_duration_s=1.0, max_moving_interval_s=10.0),
        )
        metrics = result.activities[0].metrics
        zone_time = next(metric for metric in metrics if metric.name == "heart_rate.zone_time")
        statistics_hr_time = next(
            metric
            for metric in metrics
            if metric.name == "activity.statistics.zone_distribution.valid_heart_rate_time_s"
        )

        assert isinstance(zone_time.value, TableValue)
        assert sum(zone_time.value.values) == pytest.approx(20.0)
        assert isinstance(statistics_hr_time.value, ScalarValue)
        assert statistics_hr_time.value.value == pytest.approx(20.0)

    def test_max_moving_interval_is_applied_to_endurance_metrics(self) -> None:
        data = _fit_data(
            heart_rates=[120] * 5,
            speeds=[5.0] * 5,
            distances=[0.0, 150.0, 300.0, 450.0, 600.0],
        )
        baseline = _analyze(
            data,
            quality=QualityConfig(minimum_segment_duration_s=1.0, decoupling_min_duration_s=60.0),
        ).activities[0]
        capped = _analyze(
            data,
            quality=QualityConfig(
                minimum_segment_duration_s=1.0,
                decoupling_min_duration_s=60.0,
                max_moving_interval_s=10.0,
            ),
        ).activities[0]
        baseline_metric = next(
            metric for metric in baseline.metrics if metric.name == "endurance.aerobic_decoupling"
        )
        capped_metric = next(
            metric for metric in capped.metrics if metric.name == "endurance.aerobic_decoupling"
        )

        assert isinstance(baseline_metric.value, ScalarValue)
        assert baseline_metric.value.value == pytest.approx(0.0)
        assert capped_metric.value is None
        assert capped_metric.missing_reason == MissingReason.INSUFFICIENT_DURATION

    def test_forgotten_stop_check_duration_is_configurable(self) -> None:
        data = _fit_data(
            heart_rates=[120] * 13,
            speeds=[0.0] * 10 + [5.0] * 3,
            distances=[0.0] * 13,
            interval_s=750,
        )

        default = _analyze(data).activities[0].activity
        delayed_check = (
            _analyze(data, quality=QualityConfig(forgotten_stop_check_duration_s=10000.0))
            .activities[0]
            .activity
        )

        assert "forgotten_stop" in default.repair_flags
        assert "forgotten_stop" not in delayed_check.repair_flags

    def test_cadence_fields_reach_analysis_records_unchanged(self) -> None:
        data = _fit_data(
            heart_rates=[120, 121, 122],
            speeds=[5.0, 5.0, 5.0],
            distances=[0.0, 150.0, 300.0],
            cadences=[80, 81, 82],
            fractional_cadences=[4, 5, 6],
        )

        result = _analyze(data)

        records = result.activities[0].activity.records
        assert records.select("cadence", "fractional_cadence").rows() == [
            (80, 4),
            (81, 5),
            (82, 6),
        ]

    def test_missing_cadence_reaches_analysis_as_null(self) -> None:
        result = _analyze(
            _fit_data(
                heart_rates=[120, 121, 122],
                speeds=[5.0, 5.0, 5.0],
                distances=[0.0, 150.0, 300.0],
            )
        )

        records = result.activities[0].activity.records
        assert records["cadence"].null_count() == records.height
        assert records["fractional_cadence"].null_count() == records.height

    def test_timer_pause_is_in_elapsed_time_not_statistics(self) -> None:
        result = _analyze(_paused_activity(timer_events=True))
        assert len(result.activities) == 1
        activity = result.activities[0].activity
        metrics = result.activities[0].metrics

        assert activity.duration_s == pytest.approx(360.0)
        assert activity.moving_time_s == pytest.approx(240.0)
        assert activity.paused_time_s == pytest.approx(120.0)
        assert activity.avg_hr == pytest.approx(110.0)
        assert activity.max_hr == pytest.approx(120.0)
        assert activity.avg_speed_mps == pytest.approx(5.0)
        assert activity.max_speed_mps == pytest.approx(5.0)
        assert len(activity.pause_intervals) == 1
        zones = next(metric for metric in metrics if metric.name == "heart_rate.zone_time")
        coverage = next(metric for metric in metrics if metric.name == "heart_rate.coverage")
        assert isinstance(zones.value, TableValue)
        assert zones.value.values[0] == pytest.approx(120.0)
        assert zones.value.values[1] == pytest.approx(120.0)
        assert zones.value.values[4] == 0.0
        assert isinstance(coverage.value, ScalarValue)
        assert coverage.value.value == pytest.approx(1.0)

    def test_speed_fallback_detects_pause_when_timer_events_are_absent(self) -> None:
        result = _analyze(_paused_activity(timer_events=False))
        activity = result.activities[0].activity
        assert activity.duration_s == pytest.approx(360.0)
        assert activity.moving_time_s == pytest.approx(240.0)
        assert activity.paused_time_s == pytest.approx(120.0)
        assert activity.avg_hr == pytest.approx(110.0)
        assert activity.max_hr == pytest.approx(120.0)

    def test_timer_events_are_preferred_over_speed_fallback(self) -> None:
        data = _paused_activity(timer_events=True)
        rows = data.frame.with_columns(
            pl.when(pl.col("message_type") == "record")
            .then(pl.lit(0.0))
            .otherwise(pl.col("speed_mps"))
            .alias("speed_mps")
        )
        result = _analyze(FitData(frame=rows))
        assert result.activities[0].activity.moving_time_s == pytest.approx(240.0)
        assert len(result.activities[0].activity.pause_intervals) == 1

    def test_segment_minimum_uses_moving_time_and_discards_all_short_segments(self) -> None:
        result = _analyze(_paused_activity(timer_events=True), minimum_s=300.0)
        assert result.activities == ()
        assert result.discarded_segment_count == 1

    def test_missing_speed_and_timer_events_do_not_infer_pauses(self) -> None:
        data = _fit_data(
            heart_rates=[100, 100, 100, 100, 190, 190, 190, 190, 120, 120, 120, 120, 120],
            speeds=[None] * 13,
            distances=[
                0.0,
                150.0,
                300.0,
                450.0,
                600.0,
                600.0,
                600.0,
                600.0,
                600.0,
                750.0,
                900.0,
                1050.0,
                1200.0,
            ],
        )
        activity = _analyze(data).activities[0].activity
        assert activity.moving_time_s == pytest.approx(360.0)
        assert activity.paused_time_s == 0.0
        assert activity.pause_intervals == ()
