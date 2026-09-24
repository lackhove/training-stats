"""Tests for aerobic decoupling (EF drift) computation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from tests.conftest import FIXTURES
from training_stats.adapters.fit.parser import FitDecodeParser
from training_stats.domain.analytics.endurance import (
    CONFIDENCE_HIGH,
    CONFIDENCE_INSUFFICIENT_COVERAGE,
    CONFIDENCE_INSUFFICIENT_DURATION,
    CONFIDENCE_INVALID_EF,
    compute_aerobic_decoupling,
)
from training_stats.domain.fit.types import RECORD_COLUMNS, RECORD_SCHEMA
from training_stats.domain.models import SourceFile

START = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)


def _make_records(
    samples: list[tuple[float, int | None, float | None]],
    start: datetime = START,
) -> pl.DataFrame:
    """Build record frame from (dt_s, heart_rate, speed_mps) tuples."""
    rows = []
    current_time = start
    for dt, hr, speed in samples:
        rows.append(
            {
                "timestamp": current_time,
                "heart_rate": hr,
                "speed_mps": speed,
            }
        )
        current_time += timedelta(seconds=dt)
    return pl.DataFrame(rows, schema=RECORD_SCHEMA)


class TestAerobicDecoupling:
    def test_known_cardiac_drift(self) -> None:
        # First half: 1000 s at 5.0 m/s and 125 bpm -> EF1 = 5.0 / 125 = 0.040
        # Second half: 1000 s at 5.0 m/s and 135 bpm -> EF2 = 5.0 / 135 = 0.037037...
        # Decoupling = (0.04 - 5.0/135) / 0.04 * 100 = (1 - 125/135) * 100 = 7.4074...%
        half1 = [(1.0, 125, 5.0)] * 1000
        half2 = [(1.0, 135, 5.0)] * 1000
        # Final point to complete the last interval
        final = [(0.0, 135, 5.0)]
        records = _make_records(half1 + half2 + final)

        result = compute_aerobic_decoupling(records)
        assert result.confidence == CONFIDENCE_HIGH
        assert result.ef_first_half == pytest.approx(0.040)
        assert result.ef_second_half == pytest.approx(5.0 / 135.0)
        assert result.aerobic_decoupling_pct == pytest.approx((1.0 - 125.0 / 135.0) * 100.0)

    def test_zero_decoupling(self) -> None:
        # Constant speed and constant heart rate -> 0% decoupling
        samples = [(1.0, 140, 6.0)] * 2000 + [(0.0, 140, 6.0)]
        records = _make_records(samples)

        result = compute_aerobic_decoupling(records)
        assert result.confidence == CONFIDENCE_HIGH
        assert result.ef_first_half == pytest.approx(6.0 / 140.0)
        assert result.ef_second_half == pytest.approx(6.0 / 140.0)
        assert result.aerobic_decoupling_pct == pytest.approx(0.0)

    def test_speed_drop_decoupling(self) -> None:
        # First half: 600 s at 5.0 m/s, 125 bpm -> EF1 = 0.040
        # Second half: 600 s at 4.75 m/s, 125 bpm -> EF2 = 0.038
        # Decoupling = (0.040 - 0.038) / 0.040 * 100 = 5.0%
        half1 = [(1.0, 125, 5.0)] * 600
        half2 = [(1.0, 125, 4.75)] * 600
        records = _make_records(half1 + half2 + [(0.0, 125, 4.75)])

        result = compute_aerobic_decoupling(records, min_duration_s=1200.0)
        assert result.confidence == CONFIDENCE_HIGH
        assert result.aerobic_decoupling_pct == pytest.approx(5.0)

    def test_insufficient_duration(self) -> None:
        # 1000 s is below default 1200 s (20 min)
        samples = [(1.0, 140, 5.0)] * 1000 + [(0.0, 140, 5.0)]
        records = _make_records(samples)

        result = compute_aerobic_decoupling(records, min_duration_s=1200.0)
        assert result.confidence == CONFIDENCE_INSUFFICIENT_DURATION
        assert result.ef_first_half is None
        assert result.ef_second_half is None
        assert result.aerobic_decoupling_pct is None

    def test_insufficient_coverage(self) -> None:
        # 2000 s total moving time, but 600 s missing HR (only 1400 s valid)
        # Coverage = 1400 / 2000 = 70% < 80%
        valid = [(1.0, 140, 5.0)] * 1400
        no_hr = [(1.0, None, 5.0)] * 600
        records = _make_records(valid + no_hr + [(0.0, None, 5.0)])

        result = compute_aerobic_decoupling(records, min_duration_s=1200.0, min_coverage=0.80)
        assert result.confidence == CONFIDENCE_INSUFFICIENT_COVERAGE
        assert result.ef_first_half is None
        assert result.aerobic_decoupling_pct is None

    def test_session_moving_time_coverage(self) -> None:
        # Records have 1300 s valid, but session indicates 2000 s moving time
        # Coverage = 1300 / 2000 = 65% < 80%
        samples = [(1.0, 140, 5.0)] * 1300 + [(0.0, 140, 5.0)]
        records = _make_records(samples)

        result = compute_aerobic_decoupling(
            records,
            min_duration_s=1200.0,
            min_coverage=0.80,
            session_moving_time_s=2000.0,
        )
        assert result.confidence == CONFIDENCE_INSUFFICIENT_COVERAGE
        assert result.ef_first_half is None

    def test_stopped_intervals_excluded(self) -> None:
        # Moving speed threshold = 1.0 m/s
        # 1500 s moving at 5.0 m/s, 500 s stopped at 0.2 m/s
        moving = [(1.0, 130, 5.0)] * 1500
        stopped = [(1.0, 100, 0.2)] * 500
        records = _make_records(moving + stopped + [(0.0, 100, 0.2)])

        result = compute_aerobic_decoupling(records, moving_speed_threshold_mps=1.0)
        assert result.confidence == CONFIDENCE_HIGH
        # Only moving intervals counted; HR=130 and speed=5.0 throughout
        assert result.aerobic_decoupling_pct == pytest.approx(0.0)

    def test_capped_gap_excluded(self) -> None:
        # A 5-minute pause (dt = 300 s > 30 s cap) should not count as moving time
        records = _make_records(
            [(1.0, 130, 5.0)] * 600
            + [(300.0, 130, 5.0)]
            + [(1.0, 130, 5.0)] * 600
            + [(0.0, 130, 5.0)]
        )
        result = compute_aerobic_decoupling(records, min_duration_s=1200.0)
        # 600 + 600 = 1200 s valid moving time (the 300 s gap is excluded)
        assert result.confidence == CONFIDENCE_HIGH

    def test_zero_or_negative_protection(self) -> None:
        # Empty records
        empty = pl.DataFrame(schema=RECORD_SCHEMA)
        assert compute_aerobic_decoupling(empty).confidence == CONFIDENCE_INSUFFICIENT_DURATION

        # All zero speeds
        zero_speed = _make_records([(1.0, 130, 0.0)] * 2000 + [(0.0, 130, 0.0)])
        assert compute_aerobic_decoupling(zero_speed).confidence == CONFIDENCE_INSUFFICIENT_DURATION

        # All zero HR
        zero_hr = _make_records([(1.0, 0, 5.0)] * 2000 + [(0.0, 0, 5.0)])
        assert compute_aerobic_decoupling(zero_hr).confidence == CONFIDENCE_INSUFFICIENT_DURATION

    def test_invalid_ef(self) -> None:
        # Negative speed somehow present in records
        negative_speed = _make_records([(1.0, 130, -5.0)] * 2000 + [(0.0, 130, -5.0)])
        assert (
            compute_aerobic_decoupling(negative_speed, moving_speed_threshold_mps=-10.0).confidence
            == CONFIDENCE_INVALID_EF
        )

    def test_real_cycling_fixture(self) -> None:
        parser = FitDecodeParser()
        content = (FIXTURES / "cycling.fit").read_bytes()
        decoded = parser.parse(content, SourceFile("test", "cycling.fit"))
        records = (
            decoded.frame.filter(pl.col("message_type") == "record")
            .select(RECORD_COLUMNS)
            .cast(pl.Schema(RECORD_SCHEMA))
        )

        result = compute_aerobic_decoupling(records)
        assert result.confidence == CONFIDENCE_HIGH
        assert result.ef_first_half is not None
        assert result.ef_second_half is not None
        assert result.aerobic_decoupling_pct is not None
        # Realistic values: speed ~4.7 m/s, HR ~123 bpm -> EF ~0.038
        assert 0.02 < result.ef_first_half < 0.06
        assert 0.02 < result.ef_second_half < 0.06
        assert -20.0 < result.aerobic_decoupling_pct < 30.0

    def test_real_activity_2_fixture(self) -> None:
        parser = FitDecodeParser()
        content = (FIXTURES / "activity_2.fit").read_bytes()
        decoded = parser.parse(content, SourceFile("test", "activity_2.fit"))
        records = (
            decoded.frame.filter(pl.col("message_type") == "record")
            .select(RECORD_COLUMNS)
            .cast(pl.Schema(RECORD_SCHEMA))
        )

        result = compute_aerobic_decoupling(records)
        assert result.confidence == CONFIDENCE_HIGH
        assert result.ef_first_half is not None
        assert result.ef_second_half is not None
        assert result.aerobic_decoupling_pct is not None

    def test_running_support(self) -> None:
        # Typical running pace: ~3.0 m/s with cardiac drift from 150 to 160 bpm
        half1 = [(1.0, 150, 3.0)] * 1000
        half2 = [(1.0, 160, 3.0)] * 1000
        records = _make_records(half1 + half2 + [(0.0, 160, 3.0)])

        result = compute_aerobic_decoupling(records)
        assert result.confidence == CONFIDENCE_HIGH
        assert result.ef_first_half == pytest.approx(3.0 / 150.0)
        assert result.ef_second_half == pytest.approx(3.0 / 160.0)
        assert result.aerobic_decoupling_pct == pytest.approx((1.0 - 150.0 / 160.0) * 100.0)
