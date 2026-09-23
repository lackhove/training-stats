"""Tests for the estimated HR-based load (TRIMP method)."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from tests.conftest import ZONE_BOUNDS, make_record_frame
from training_stats.domain.analytics.heart_rate import (
    ZoneResult,
    build_zone_bounds,
    time_in_zones,
)
from training_stats.domain.analytics.load import (
    LOAD_METHOD,
    LOAD_VERSION,
    hr_trimp_load,
    load_confidence,
    zone_trimp_weight,
)

ZONES = build_zone_bounds(ZONE_BOUNDS)
MAX_HR = 180
RESTING_HR = 55


def _zone_result(
    zone_seconds: tuple[float, float, float, float, float], coverage: float
) -> ZoneResult:
    valid_time = sum(zone_seconds)
    return ZoneResult(
        zone_seconds=zone_seconds, valid_hr_time_s=valid_time, hr_coverage_ratio=coverage
    )


class TestZoneWeight:
    def test_zone3_weight_matches_formula(self) -> None:
        zone3 = ZONES[2]
        midpoint = (zone3.min_bpm + zone3.max_bpm) / 2
        ratio = (midpoint - RESTING_HR) / (MAX_HR - RESTING_HR)
        expected = 0.64 * math.exp(1.92 * ratio)
        assert zone_trimp_weight(zone3, MAX_HR, RESTING_HR) == pytest.approx(expected)

    def test_weights_increase_with_zone(self) -> None:
        weights = [zone_trimp_weight(zone, MAX_HR, RESTING_HR) for zone in ZONES]
        assert weights == sorted(weights)
        assert all(weight > 0 for weight in weights)


class TestHrTrimpLoad:
    def test_single_zone_hand_calculated(self) -> None:
        # 30 minutes (as uncapped 60 s intervals count fully) in zone 3.
        zone3 = ZONES[2]
        expected = 30.0 * zone_trimp_weight(zone3, MAX_HR, RESTING_HR)
        result = hr_trimp_load(
            _zone_result((0.0, 0.0, 1800.0, 0.0, 0.0), 1.0), ZONES, MAX_HR, RESTING_HR
        )
        assert result == pytest.approx(expected)

    def test_mixed_zones(self) -> None:
        expected = 600.0 / 60 * zone_trimp_weight(
            ZONES[0], MAX_HR, RESTING_HR
        ) + 600.0 / 60 * zone_trimp_weight(ZONES[4], MAX_HR, RESTING_HR)
        result = hr_trimp_load(
            _zone_result((600.0, 0.0, 0.0, 0.0, 600.0), 1.0), ZONES, MAX_HR, RESTING_HR
        )
        assert result == pytest.approx(expected)

    def test_no_hr_data_returns_none(self) -> None:
        empty = ZoneResult(
            zone_seconds=(0.0, 0.0, 0.0, 0.0, 0.0), valid_hr_time_s=0.0, hr_coverage_ratio=0.0
        )
        assert hr_trimp_load(empty, ZONES, MAX_HR, RESTING_HR) is None

    def test_method_is_documented(self) -> None:
        assert LOAD_METHOD == "hr_trimp"
        assert LOAD_VERSION == "1"


class TestLoadConfidence:
    def test_high(self) -> None:
        assert load_confidence(0.95) == "high"

    def test_medium(self) -> None:
        assert load_confidence(0.7) == "medium"

    def test_low(self) -> None:
        assert load_confidence(0.2) == "low"


class TestIntegration:
    def test_capped_intervals_reduce_load(self) -> None:
        """Records one minute apart are capped at 30 s per interval."""
        start = datetime(2026, 6, 1, tzinfo=UTC)
        records = make_record_frame(start, [135] * 11, interval_s=60)
        zone_result = time_in_zones(records, ZONES)
        load = hr_trimp_load(zone_result, ZONES, MAX_HR, RESTING_HR)
        assert load is not None
        # 10 intervals x 30 s = 5 counted minutes in zone 3.
        assert load == pytest.approx(5.0 * zone_trimp_weight(ZONES[2], MAX_HR, RESTING_HR))
