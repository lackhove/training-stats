"""Regression tests for coarse multi-sensor activity statistics."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from tests.conftest import make_athlete
from training_stats.adapters.wire import activity_record
from training_stats.domain.analytics import activity_statistics
from training_stats.domain.analytics.activities import compute_activity_metrics
from training_stats.domain.analytics.activity_statistics import (
    cadence_metrics,
    compute_activity_statistics,
    efficiency_metrics,
    fatigue_response_metrics,
    heart_rate_metrics,
    heatmap_metrics,
    relationship_metrics,
    sensor_quality_metrics,
    speed_metrics,
    time_window_metrics,
    zone_metrics,
)
from training_stats.domain.analytics.statistics_records import Interval, build_intervals
from training_stats.domain.fit.types import RECORD_SCHEMA, ParsedActivity, PauseInterval
from training_stats.domain.metrics import (
    MatrixValue,
    Metric,
    MetricScalar,
    MissingReason,
    ScalarValue,
    TableValue,
)
from training_stats.domain.models import ActivityRecord
from training_stats.domain.settings import QualityConfig

_DECOUPLING_QUALITY = QualityConfig(decoupling_min_duration_s=600.0)
_ANALYSIS_VERSION = "test-version"
_THRESHOLD_METRIC = "activity.statistics.speed_statistics.speed_above_threshold_s.seconds"

VARIANTS: dict[str, tuple[dict[str, bool], QualityConfig]] = {
    "normal": ({"cadence": True}, _DECOUPLING_QUALITY),
    "no_cadence": ({"cadence": False}, QualityConfig()),
    "sparse_cadence": ({"sparse_cadence": True}, QualityConfig()),
    "paused": ({"pause": True}, QualityConfig()),
    "empty": ({"empty": True}, QualityConfig()),
    "empty_thresholds": ({}, replace(_DECOUPLING_QUALITY, speed_thresholds_kmh=())),
}

EXPECTED_STATISTICS_COUNTS = {
    "normal": 143,
    "no_cadence": 131,
    "sparse_cadence": 131,
    "paused": 158,
    "empty": 74,
    "empty_thresholds": 142,
}

EXPECTED_WIRE_DIGESTS = {
    "normal": "6021f4022d0d2bec8e0219aab78e28284752215d498509be6f0d7b2cbaaafc60",
    "no_cadence": "348580fd4712f3a58085b006e39282e326c07147b817aefa86a67d1218a883da",
    "sparse_cadence": "261a8aba62931d95d62ea371d84cf3774f6e61b2a974d149337c1b9e7413103d",
    "paused": "efb3a2f958d15333e49d027bf9db16b67ccdbb84947750164c37905eef74bb92",
    "empty": "e71f0b978ce8dea88946ab262ecf84abb47f0cdb87d2e54cea8aa3764ff25244",
    "empty_thresholds": "3d2bb2020a9a7cc1576e723798129e9aa86bc0b16764d9b36d52b28237aaf774",
}

EXPECTED_NORMAL_NAMES = (
    "activity.statistics.speed_statistics.mean",
    "activity.statistics.speed_statistics.median",
    "activity.statistics.speed_statistics.standard_deviation",
    "activity.statistics.speed_statistics.p10",
    "activity.statistics.speed_statistics.p25",
    "activity.statistics.speed_statistics.p75",
    "activity.statistics.speed_statistics.p90",
    "activity.statistics.speed_statistics.minimum",
    "activity.statistics.speed_statistics.maximum",
    "activity.statistics.speed_statistics.valid_sample_count",
    "activity.statistics.speed_statistics.valid_time_s",
    "activity.statistics.speed_statistics.coverage_ratio",
    "activity.statistics.speed_statistics.speed_above_threshold_s.seconds",
    "activity.statistics.speed_statistics.valid_moving_time_ratio",
    "activity.statistics.speed_statistics.valid_measurement_ratio",
    "activity.statistics.speed_statistics.measurements_during_movement_ratio",
    "activity.statistics.heart_rate_statistics.mean",
    "activity.statistics.heart_rate_statistics.median",
    "activity.statistics.heart_rate_statistics.standard_deviation",
    "activity.statistics.heart_rate_statistics.p10",
    "activity.statistics.heart_rate_statistics.p25",
    "activity.statistics.heart_rate_statistics.p75",
    "activity.statistics.heart_rate_statistics.p90",
    "activity.statistics.heart_rate_statistics.minimum",
    "activity.statistics.heart_rate_statistics.maximum",
    "activity.statistics.heart_rate_statistics.valid_sample_count",
    "activity.statistics.heart_rate_statistics.valid_time_s",
    "activity.statistics.heart_rate_statistics.coverage_ratio",
    "activity.statistics.heart_rate_statistics.histogram_s.seconds",
    "activity.statistics.cadence_statistics.available",
    "activity.statistics.cadence_statistics.overall.mean",
    "activity.statistics.cadence_statistics.overall.median",
    "activity.statistics.cadence_statistics.overall.standard_deviation",
    "activity.statistics.cadence_statistics.overall.p10",
    "activity.statistics.cadence_statistics.overall.p25",
    "activity.statistics.cadence_statistics.overall.p75",
    "activity.statistics.cadence_statistics.overall.p90",
    "activity.statistics.cadence_statistics.overall.minimum",
    "activity.statistics.cadence_statistics.overall.maximum",
    "activity.statistics.cadence_statistics.overall.valid_sample_count",
    "activity.statistics.cadence_statistics.overall.valid_time_s",
    "activity.statistics.cadence_statistics.overall.coverage_ratio",
    "activity.statistics.cadence_statistics.pedaling.mean",
    "activity.statistics.cadence_statistics.pedaling.median",
    "activity.statistics.cadence_statistics.pedaling.standard_deviation",
    "activity.statistics.cadence_statistics.pedaling.p10",
    "activity.statistics.cadence_statistics.pedaling.p25",
    "activity.statistics.cadence_statistics.pedaling.p75",
    "activity.statistics.cadence_statistics.pedaling.p90",
    "activity.statistics.cadence_statistics.pedaling.minimum",
    "activity.statistics.cadence_statistics.pedaling.maximum",
    "activity.statistics.cadence_statistics.pedaling.valid_sample_count",
    "activity.statistics.cadence_statistics.pedaling.valid_time_s",
    "activity.statistics.cadence_statistics.pedaling.coverage_ratio",
    "activity.statistics.cadence_statistics.pedaling_time_s",
    "activity.statistics.cadence_statistics.no_pedaling_time_s",
    "activity.statistics.cadence_statistics.pedaling_time_ratio",
    "activity.statistics.cadence_statistics.coverage_ratio",
    "activity.statistics.cadence_statistics.bands_s.seconds",
    "activity.statistics.zone_distribution.seconds",
    "activity.statistics.zone_distribution.percent",
    "activity.statistics.zone_distribution.valid_heart_rate_time_s",
    "activity.statistics.zone_distribution.unclassified_time_s",
    "activity.statistics.zone_distribution.percent_denominator",
    "activity.statistics.time_windows.window_s",
    "activity.statistics.time_windows.items.average_heart_rate_bpm",
    "activity.statistics.time_windows.items.average_pedaling_cadence_rpm",
    "activity.statistics.time_windows.items.average_speed_kmh",
    "activity.statistics.time_windows.items.cadence_coverage_ratio",
    "activity.statistics.time_windows.items.distance_m",
    "activity.statistics.time_windows.items.duration_s",
    "activity.statistics.time_windows.items.end_s",
    "activity.statistics.time_windows.items.heart_rate_p25_bpm",
    "activity.statistics.time_windows.items.heart_rate_p75_bpm",
    "activity.statistics.time_windows.items.heart_rate_valid_time_s",
    "activity.statistics.time_windows.items.heart_rate_zone_percent_denominator",
    "activity.statistics.time_windows.items.median_heart_rate_bpm",
    "activity.statistics.time_windows.items.median_pedaling_cadence_rpm",
    "activity.statistics.time_windows.items.median_speed_kmh",
    "activity.statistics.time_windows.items.moving_time_ratio",
    "activity.statistics.time_windows.items.moving_time_s",
    "activity.statistics.time_windows.items.speed_p25_kmh",
    "activity.statistics.time_windows.items.speed_p75_kmh",
    "activity.statistics.time_windows.movement_time_variant",
    "activity.statistics.efficiency.speed_per_heart_rate",
    "activity.statistics.efficiency.speed_per_heart_rate_reserve",
    "activity.statistics.efficiency.first_half_average_speed_kmh",
    "activity.statistics.efficiency.second_half_average_speed_kmh",
    "activity.statistics.efficiency.first_half_average_heart_rate_bpm",
    "activity.statistics.efficiency.second_half_average_heart_rate_bpm",
    "activity.statistics.efficiency.ef_first_half",
    "activity.statistics.efficiency.ef_second_half",
    "activity.statistics.efficiency.aerobic_decoupling_pct",
    "activity.statistics.efficiency.valid_time_s",
    "activity.statistics.efficiency.coverage_ratio",
    "activity.statistics.efficiency.heart_rate_reserve_formula",
    "activity.statistics.fatigue_response.heart_rate_drift_bpm_per_hour",
    "activity.statistics.fatigue_response.speed_per_heart_rate",
    "activity.statistics.fatigue_response.speed_per_heart_rate_reserve",
    "activity.statistics.fatigue_response.first_half_average_speed_kmh",
    "activity.statistics.fatigue_response.second_half_average_speed_kmh",
    "activity.statistics.fatigue_response.first_half_average_heart_rate_bpm",
    "activity.statistics.fatigue_response.second_half_average_heart_rate_bpm",
    "activity.statistics.fatigue_response.ef_first_half",
    "activity.statistics.fatigue_response.ef_second_half",
    "activity.statistics.fatigue_response.aerobic_decoupling_pct",
    "activity.statistics.fatigue_response.valid_time_s",
    "activity.statistics.fatigue_response.coverage_ratio",
    "activity.statistics.fatigue_response.heart_rate_reserve_formula",
    "activity.statistics.relationships.speed_heart_rate.spearman",
    "activity.statistics.relationships.speed_heart_rate.pearson",
    "activity.statistics.relationships.speed_heart_rate.window_count",
    "activity.statistics.relationships.speed_cadence.spearman",
    "activity.statistics.relationships.speed_cadence.pearson",
    "activity.statistics.relationships.speed_cadence.window_count",
    "activity.statistics.relationships.heart_rate_cadence.spearman",
    "activity.statistics.relationships.heart_rate_cadence.pearson",
    "activity.statistics.relationships.heart_rate_cadence.window_count",
    "activity.statistics.heatmap_data.heart_rate_speed.duration_s",
    "activity.statistics.heatmap_data.heart_rate_speed.observation_count",
    "activity.statistics.heatmap_data.heart_rate_speed.valid_observation_count",
    "activity.statistics.heatmap_data.heart_rate_cadence.duration_s",
    "activity.statistics.heatmap_data.heart_rate_cadence.observation_count",
    "activity.statistics.heatmap_data.heart_rate_cadence.valid_observation_count",
    "activity.statistics.heatmap_data.speed_cadence.duration_s",
    "activity.statistics.heatmap_data.speed_cadence.observation_count",
    "activity.statistics.heatmap_data.speed_cadence.valid_observation_count",
    "activity.statistics.data_quality.raw_record_points",
    "activity.statistics.data_quality.valid_speed_intervals",
    "activity.statistics.data_quality.valid_speed_samples",
    "activity.statistics.data_quality.speed_samples_during_movement",
    "activity.statistics.data_quality.valid_heart_rate_intervals",
    "activity.statistics.data_quality.valid_cadence_intervals",
    "activity.statistics.data_quality.invalid_speed_intervals",
    "activity.statistics.data_quality.invalid_heart_rate_intervals",
    "activity.statistics.data_quality.invalid_cadence_intervals",
    "activity.statistics.data_quality.duplicate_timestamps",
    "activity.statistics.data_quality.gap_intervals_over_30s",
    "activity.statistics.data_quality.rejected_outlier_values",
    "activity.statistics.data_quality.pause_intervals",
    "activity.statistics.data_quality.speed_coverage_ratio",
    "activity.statistics.data_quality.heart_rate_coverage_ratio",
    "activity.statistics.data_quality.cadence_coverage_ratio",
)


def _activity(
    *, cadence: bool = True, sparse_cadence: bool = False, pause: bool = False, empty: bool = False
) -> ParsedActivity:
    start = datetime(2026, 6, 1, 10, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    for index in range(1 if empty else 31):
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


def _statistics(quality: QualityConfig = _DECOUPLING_QUALITY, **kwargs: bool) -> tuple[Metric, ...]:
    return compute_activity_statistics(_activity(**kwargs), make_athlete(), quality)


def _metric(metrics: tuple[Metric, ...], name: str) -> Metric:
    for metric in metrics:
        if metric.name == name:
            return metric
    raise AssertionError(f"missing metric {name}")


def _scalar(metrics: tuple[Metric, ...], name: str) -> MetricScalar:
    value = _metric(metrics, name).value
    assert isinstance(value, ScalarValue)
    return value.value


def _table(metrics: tuple[Metric, ...], name: str) -> TableValue:
    value = _metric(metrics, name).value
    assert isinstance(value, TableValue)
    return value


def _missing(metrics: tuple[Metric, ...], name: str) -> MissingReason | None:
    return _metric(metrics, name).missing_reason


def _cell(table: TableValue, label: object) -> MetricScalar | None:
    return table.values[table.axis.values.index(label)]  # type: ignore[arg-type]


def _count_interval_builds(monkeypatch: pytest.MonkeyPatch) -> Callable[[], int]:
    builds = 0
    original = activity_statistics.build_intervals

    def counting(activity: ParsedActivity, max_interval_s: float = 30.0) -> list[Interval]:
        nonlocal builds
        builds += 1
        return original(activity, max_interval_s)

    monkeypatch.setattr(activity_statistics, "build_intervals", counting)
    return lambda: builds


class TestDetailedActivityStatistics:
    def test_speed_hr_cadence_and_zone_statistics(self) -> None:
        metrics = _statistics()
        assert _scalar(metrics, "activity.statistics.speed_statistics.mean") == pytest.approx(21.6)
        thresholds = _table(
            metrics, "activity.statistics.speed_statistics.speed_above_threshold_s.seconds"
        )
        assert _cell(thresholds, 20.0) == pytest.approx(600.0)
        assert _scalar(
            metrics, "activity.statistics.speed_statistics.valid_moving_time_ratio"
        ) == pytest.approx(1.0)

        assert _scalar(metrics, "activity.statistics.cadence_statistics.available") is True
        assert _scalar(metrics, "activity.statistics.cadence_statistics.pedaling_time_s") == (
            pytest.approx(750.0)
        )
        assert _scalar(metrics, "activity.statistics.cadence_statistics.no_pedaling_time_s") == (
            pytest.approx(150.0)
        )
        bands = _table(metrics, "activity.statistics.cadence_statistics.bands_s.seconds")
        assert _cell(bands, "rpm_80_89") == pytest.approx(450.0)
        assert _cell(bands, "rpm_90_99") == pytest.approx(300.0)

        seconds = _table(metrics, "activity.statistics.zone_distribution.seconds")
        assert _cell(seconds, "zone_2") == pytest.approx(300.0)
        assert _cell(seconds, "zone_4") == pytest.approx(600.0)
        percent = _table(metrics, "activity.statistics.zone_distribution.percent")
        assert _cell(percent, "zone_2") == pytest.approx(100.0 / 3.0)

    def test_windows_correlations_and_heatmaps_use_coarse_intervals(self) -> None:
        metrics = _statistics()
        assert _scalar(metrics, "activity.statistics.time_windows.window_s") == pytest.approx(300.0)
        windowed_hr = _table(
            metrics, "activity.statistics.time_windows.items.average_heart_rate_bpm"
        )
        assert windowed_hr.axis.values == (0.0, 300.0, 600.0)
        assert windowed_hr.values == (120.0, 145.0, 160.0)
        assert _missing(metrics, "activity.statistics.time_windows.movement_time_variant") == (
            "no_data"
        )

        assert (
            _scalar(metrics, "activity.statistics.relationships.speed_heart_rate.window_count") == 3
        )
        assert _scalar(
            metrics, "activity.statistics.relationships.speed_heart_rate.spearman"
        ) == pytest.approx(1.0)

        heatmap = _metric(
            metrics, "activity.statistics.heatmap_data.heart_rate_speed.duration_s"
        ).value
        assert isinstance(heatmap, MatrixValue)
        assert heatmap.x_axis.values[0].label.startswith("120")  # type: ignore[union-attr]
        assert (
            _scalar(
                metrics, "activity.statistics.heatmap_data.heart_rate_speed.valid_observation_count"
            )
            == 30
        )

        assert _scalar(
            metrics, "activity.statistics.efficiency.first_half_average_speed_kmh"
        ) == pytest.approx(19.2)
        assert _scalar(
            metrics, "activity.statistics.efficiency.second_half_average_speed_kmh"
        ) == pytest.approx(24.0)
        assert _scalar(
            metrics, "activity.statistics.efficiency.first_half_average_heart_rate_bpm"
        ) == pytest.approx(128.333333)
        assert _scalar(
            metrics, "activity.statistics.efficiency.second_half_average_heart_rate_bpm"
        ) == pytest.approx(155.0)
        assert _metric(metrics, "activity.statistics.efficiency.aerobic_decoupling_pct").value

    def test_absent_cadence_is_never_inferred(self) -> None:
        metrics = _statistics(QualityConfig(), cadence=False)
        assert _scalar(metrics, "activity.statistics.cadence_statistics.available") is False
        assert _missing(metrics, "activity.statistics.cadence_statistics.overall.mean") == (
            "no_cadence_data"
        )
        assert _missing(metrics, "activity.statistics.cadence_statistics.pedaling.mean") == (
            "no_cadence_data"
        )
        assert _missing(
            metrics, "activity.statistics.heart_rate_statistics.histogram_s.seconds"
        ) is (None)

    def test_incomplete_cadence_is_reported_but_not_summarized(self) -> None:
        metrics = _statistics(QualityConfig(), sparse_cadence=True)
        assert _scalar(metrics, "activity.statistics.cadence_statistics.available") is False
        assert _scalar(metrics, "activity.statistics.cadence_statistics.coverage_ratio") < 0.8
        assert _missing(metrics, "activity.statistics.cadence_statistics.overall.mean") == (
            "insufficient_coverage"
        )
        windowed_cadence = _table(
            metrics, "activity.statistics.time_windows.items.average_pedaling_cadence_rpm"
        )
        assert windowed_cadence.values == (None, None, None)
        assert (
            _scalar(
                metrics,
                "activity.statistics.heatmap_data.heart_rate_cadence.valid_observation_count",
            )
            == 0
        )

    def test_paused_time_is_excluded_from_sensor_statistics(self) -> None:
        metrics = _statistics(QualityConfig(), pause=True)
        assert _scalar(metrics, "activity.statistics.data_quality.pause_intervals") == 1
        assert _scalar(metrics, "activity.statistics.data_quality.heart_rate_coverage_ratio") <= 1.0

    def test_public_function_returns_metrics_including_heatmap(self) -> None:
        metrics = _statistics()
        assert all(metric.period.activity_id == "activity-statistics" for metric in metrics)
        assert all(metric.description.strip() for metric in metrics)
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

    def test_empty_activity_keeps_coarse_schema_but_omits_intervals(self) -> None:
        metrics = _statistics(QualityConfig(), empty=True)
        assert len(metrics) == EXPECTED_STATISTICS_COUNTS["empty"]
        assert _metric(metrics, "activity.statistics.zone_distribution.seconds").value
        assert _metric(metrics, "activity.statistics.speed_statistics.mean").value is None
        for name in (
            "activity.statistics.efficiency.aerobic_decoupling_pct",
            "activity.statistics.relationships.speed_heart_rate.spearman",
            "activity.statistics.heatmap_data.heart_rate_speed.duration_s",
            "activity.statistics.time_windows.items.average_pedaling_cadence_rpm",
        ):
            assert all(metric.name != name for metric in metrics)


class TestStatisticsParity:
    @pytest.mark.parametrize("variant", VARIANTS)
    def test_metric_count_matches_locked_schema(self, variant: str) -> None:
        kwargs, quality = VARIANTS[variant]
        metrics = compute_activity_statistics(_activity(**kwargs), make_athlete(), quality)
        assert len(metrics) == EXPECTED_STATISTICS_COUNTS[variant]

    def test_normal_variant_metric_names_in_order(self) -> None:
        kwargs, quality = VARIANTS["normal"]
        metrics = compute_activity_statistics(_activity(**kwargs), make_athlete(), quality)
        assert tuple(metric.name for metric in metrics) == EXPECTED_NORMAL_NAMES

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_wire_payload_digest_is_stable(self, variant: str) -> None:
        kwargs, quality = VARIANTS[variant]
        activity = _activity(**kwargs)
        metrics = compute_activity_metrics(activity, make_athlete(), _ANALYSIS_VERSION, quality)
        record = ActivityRecord(
            activity_id=activity.activity_id,
            start_time=activity.start_time,
            sport=activity.sport,
            sub_sport=activity.sub_sport,
            segment_index=activity.segment_index,
            segment_count=activity.segment_count,
            repair_flags=activity.repair_flags,
            source_name="synthetic",
            source_path=activity.source_path,
            analysis_version=_ANALYSIS_VERSION,
            metrics=metrics,
        )
        payload = activity_record(record).model_dump(mode="json", exclude_none=True)
        canonical = json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
        assert hashlib.sha256(canonical).hexdigest() == EXPECTED_WIRE_DIGESTS[variant]


class TestFamilyProducers:
    def test_producers_take_the_activity_and_explicit_configuration(self) -> None:
        activity = _activity()
        athlete = make_athlete()
        quality = _DECOUPLING_QUALITY
        families: tuple[tuple[tuple[Metric, ...], str], ...] = (
            (speed_metrics(activity, quality), "activity.statistics.speed_statistics."),
            (heart_rate_metrics(activity), "activity.statistics.heart_rate_statistics."),
            (cadence_metrics(activity, quality), "activity.statistics.cadence_statistics."),
            (zone_metrics(activity, athlete), "activity.statistics.zone_distribution."),
            (time_window_metrics(activity, quality), "activity.statistics.time_windows."),
            (efficiency_metrics(activity, athlete, quality), "activity.statistics.efficiency."),
            (
                fatigue_response_metrics(activity, athlete, quality),
                "activity.statistics.fatigue_response.",
            ),
            (relationship_metrics(activity, quality), "activity.statistics.relationships."),
            (heatmap_metrics(activity, quality), "activity.statistics.heatmap_data."),
            (sensor_quality_metrics(activity, quality), "activity.statistics.data_quality."),
        )
        for metrics, prefix in families:
            assert metrics, prefix
            assert all(metric.name.startswith(prefix) for metric in metrics), prefix

    def test_composer_is_the_ordered_concatenation_of_families(self) -> None:
        activity = _activity()
        athlete = make_athlete()
        quality = _DECOUPLING_QUALITY
        families = (
            speed_metrics(activity, quality),
            heart_rate_metrics(activity),
            cadence_metrics(activity, quality),
            zone_metrics(activity, athlete),
            time_window_metrics(activity, quality),
            efficiency_metrics(activity, athlete, quality),
            fatigue_response_metrics(activity, athlete, quality),
            relationship_metrics(activity, quality),
            heatmap_metrics(activity, quality),
            sensor_quality_metrics(activity, quality),
        )
        assert compute_activity_statistics(activity, athlete, quality) == tuple(
            metric for family in families for metric in family
        )

    def test_configuration_is_passed_explicitly(self) -> None:
        activity = _activity(sparse_cadence=True)
        assert (
            _scalar(
                cadence_metrics(activity, QualityConfig(cadence_min_coverage=0.0)),
                "activity.statistics.cadence_statistics.available",
            )
            is True
        )
        assert (
            _scalar(
                cadence_metrics(activity, QualityConfig(cadence_min_coverage=1.0)),
                "activity.statistics.cadence_statistics.available",
            )
            is False
        )
        zones = zone_metrics(activity, make_athlete())
        assert _table(zones, "activity.statistics.zone_distribution.seconds").values


class TestSharedIntervalPass:
    def test_composed_statistics_build_intervals_exactly_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        builds = _count_interval_builds(monkeypatch)
        compute_activity_statistics(_activity(), make_athlete(), _DECOUPLING_QUALITY)
        assert builds() == 1

    def test_standalone_family_producers_build_their_own_intervals(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        builds = _count_interval_builds(monkeypatch)
        activity = _activity()
        athlete = make_athlete()
        quality = _DECOUPLING_QUALITY
        producers: tuple[tuple[Callable[[], tuple[Metric, ...]], int], ...] = (
            (lambda: speed_metrics(activity, quality), 1),
            (lambda: heart_rate_metrics(activity), 1),
            (lambda: cadence_metrics(activity, quality), 1),
            (lambda: zone_metrics(activity, athlete), 1),
            (lambda: time_window_metrics(activity, quality), 1),
            (lambda: efficiency_metrics(activity, athlete, quality), 1),
            (lambda: fatigue_response_metrics(activity, athlete, quality), 1),
            (lambda: relationship_metrics(activity, quality), 1),
            (lambda: heatmap_metrics(activity, quality), 1),
            (lambda: sensor_quality_metrics(activity, quality), 1),
        )
        for produce, expected_builds in producers:
            before = builds()
            produce()
            assert builds() - before == expected_builds
        assert builds() == 10

    def test_activity_statistics_reuses_derived_facts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"windows": 0, "efficiency": 0, "cadence": 0, "short": 0, "moving": 0}
        original_windows = activity_statistics._build_available_windows
        original_efficiency = activity_statistics._build_efficiency_facts
        original_cadence = activity_statistics._cadence_coverage
        original_short = activity_statistics.is_short_activity
        original_moving = activity_statistics.moving_time_s

        def count_windows(
            statistics: activity_statistics._ActivityStatistics,
        ) -> tuple[
            tuple[activity_statistics._Window, ...], tuple[activity_statistics._Window, ...]
        ]:
            calls["windows"] += 1
            return original_windows(statistics)

        def count_efficiency(
            statistics: activity_statistics._ActivityStatistics,
        ) -> activity_statistics._EfficiencyFacts:
            calls["efficiency"] += 1
            return original_efficiency(statistics)

        def count_cadence(intervals: Sequence[Interval], moving: float) -> float:
            calls["cadence"] += 1
            return original_cadence(intervals, moving)

        def count_short(activity: ParsedActivity) -> bool:
            calls["short"] += 1
            return original_short(activity)

        def count_moving(
            activity: ParsedActivity, intervals: Sequence[Interval], threshold_mps: float
        ) -> float:
            calls["moving"] += 1
            return original_moving(activity, intervals, threshold_mps)

        monkeypatch.setattr(activity_statistics, "_build_available_windows", count_windows)
        monkeypatch.setattr(activity_statistics, "_build_efficiency_facts", count_efficiency)
        monkeypatch.setattr(activity_statistics, "_cadence_coverage", count_cadence)
        monkeypatch.setattr(activity_statistics, "is_short_activity", count_short)
        monkeypatch.setattr(activity_statistics, "moving_time_s", count_moving)

        compute_activity_statistics(_activity(), make_athlete(), _DECOUPLING_QUALITY)

        assert calls == {"windows": 1, "efficiency": 1, "cadence": 1, "short": 1, "moving": 1}

    def test_shared_statistics_object_suppresses_standalone_interval_builds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        builds = _count_interval_builds(monkeypatch)
        activity = _activity()
        athlete = make_athlete()
        quality = _DECOUPLING_QUALITY
        shared = build_intervals(activity)
        statistics = activity_statistics._ActivityStatistics.build(
            activity, quality, athlete, intervals=shared
        )
        activity_statistics._speed_metrics(statistics)
        activity_statistics._heart_rate_metrics(statistics)
        activity_statistics._cadence_metrics(statistics)
        activity_statistics._zone_metrics(statistics)
        activity_statistics._time_window_metrics(statistics)
        activity_statistics._efficiency_family_metrics(statistics)
        activity_statistics._fatigue_response_metrics(statistics)
        activity_statistics._relationship_metrics(statistics)
        activity_statistics._heatmap_metrics(statistics)
        activity_statistics._sensor_quality_metrics(statistics)
        assert builds() == 0


class TestEmptySpeedThresholds:
    def test_empty_threshold_list_omits_the_threshold_table(self) -> None:
        quality = replace(_DECOUPLING_QUALITY, speed_thresholds_kmh=())
        assert quality.speed_thresholds_kmh == ()
        metrics = compute_activity_statistics(_activity(), make_athlete(), quality)
        assert all(metric.name != _THRESHOLD_METRIC for metric in metrics)
        baseline = compute_activity_statistics(_activity(), make_athlete(), _DECOUPLING_QUALITY)
        assert any(metric.name == _THRESHOLD_METRIC for metric in baseline)
        assert metrics == tuple(metric for metric in baseline if metric.name != _THRESHOLD_METRIC)
