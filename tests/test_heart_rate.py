"""Tests for heart-rate zone computation with hand-calculated fixtures."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from tests.conftest import ZONE_BOUNDS, make_record_frame
from training_coach_mcp.analytics.heart_rate import (
    MAX_RECORD_INTERVAL_S,
    build_zone_bounds,
    hr_bucket_averages,
    hr_drift_bpm_per_hour,
    hr_histogram,
    time_in_zones,
)
from training_coach_mcp.fit.types import RECORD_SCHEMA, empty_records

ZONES = build_zone_bounds(ZONE_BOUNDS)
START = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)


def _irregular_frame() -> pl.DataFrame:
    """Intervals of 5 s, 15 s, 40 s (capped), a null-HR gap and 20 s."""
    rows = []
    timestamp = START
    for delta_s, heart_rate in [(5, 120), (15, 140), (40, 160), (30, None), (20, 121)]:
        rows.append({"timestamp": timestamp, "heart_rate": heart_rate})
        timestamp = timestamp + timedelta(seconds=delta_s)
    return pl.DataFrame(rows, schema=RECORD_SCHEMA)


class TestTimeInZones:
    # 20 s intervals stay below the 30 s cap so expectations are exact.
    INTERVAL_S = 20

    def test_single_zone(self) -> None:
        # 3 intervals (4 samples) in zone 3 (hr=135).
        records = make_record_frame(START, [135, 135, 135, 135], interval_s=self.INTERVAL_S)
        result = time_in_zones(records, ZONES)
        assert result.zone_seconds[2] == 3 * self.INTERVAL_S
        assert sum(result.zone_seconds) == 60.0
        assert result.valid_hr_time_s == 60.0
        assert result.hr_coverage_ratio == 1.0

    def test_multiple_zones(self) -> None:
        records = make_record_frame(START, [100, 120, 150, 170, 135], interval_s=self.INTERVAL_S)
        result = time_in_zones(records, ZONES)
        # Interval starts: z1, z2, z4, z5 -> 20 s each.
        assert result.zone_seconds[0] == 20.0
        assert result.zone_seconds[1] == 20.0
        assert result.zone_seconds[2] == 0.0
        assert result.zone_seconds[3] == 20.0
        assert result.zone_seconds[4] == 20.0

    def test_interval_cap(self) -> None:
        # Two samples 10 minutes apart count as max_interval only.
        records = make_record_frame(START, [120, 120], interval_s=600)
        result = time_in_zones(records, ZONES)
        assert result.zone_seconds[1] == MAX_RECORD_INTERVAL_S
        assert result.valid_hr_time_s == MAX_RECORD_INTERVAL_S

    def test_negative_interval_ignored(self) -> None:
        out_of_order = pl.DataFrame(
            [
                {"timestamp": START, "heart_rate": 120},
                {"timestamp": START - timedelta(seconds=60), "heart_rate": 120},
            ],
            schema=RECORD_SCHEMA,
        )
        result = time_in_zones(out_of_order, ZONES)
        assert result.valid_hr_time_s == 0.0

    def test_missing_hr_samples(self) -> None:
        records = make_record_frame(START, [None, 120, None, None, 140], interval_s=self.INTERVAL_S)
        result = time_in_zones(records, ZONES)
        # Only the interval starting at the 120 bpm record (t2 -> t3) has a
        # starting HR value; every other interval start lacks HR.
        assert result.zone_seconds[1] == self.INTERVAL_S
        assert result.zone_seconds[2] == 0.0
        assert result.valid_hr_time_s == self.INTERVAL_S
        assert result.hr_coverage_ratio == 0.25

    def test_no_records(self) -> None:
        result = time_in_zones(empty_records(), ZONES)
        assert result.zone_seconds == (0.0, 0.0, 0.0, 0.0, 0.0)
        assert result.hr_coverage_ratio == 0.0

    def test_single_record(self) -> None:
        result = time_in_zones(make_record_frame(START, [120]), ZONES)
        assert result.valid_hr_time_s == 0.0
        assert result.hr_coverage_ratio == 0.0

    def test_hr_outside_zones_not_counted(self) -> None:
        records = make_record_frame(START, [60, 60, 200, 200], interval_s=self.INTERVAL_S)
        result = time_in_zones(records, ZONES)
        assert sum(result.zone_seconds) == 0.0
        assert result.valid_hr_time_s == 3 * self.INTERVAL_S
        assert result.hr_coverage_ratio == 1.0


class TestHrHistogram:
    def test_bins_are_time_weighted(self) -> None:
        histogram = hr_histogram(_irregular_frame())
        # 5 s at 120 ("120-124"); 15 s at 140; 40 s at 160 capped to 30 s;
        # null-HR interval excluded; the final 121 record has no following
        # interval and therefore contributes no time.
        assert histogram == {"120-124": 5.0, "140-144": 15.0, "160-164": 30.0}

    def test_covers_hr_outside_zones(self) -> None:
        records = make_record_frame(START, [60, 61, 200], interval_s=20)
        histogram = hr_histogram(records)
        # Intervals start at the two 60-64 bpm records; the final 200 record
        # has no following interval.
        assert histogram == {"60-64": 40.0}

    def test_total_equals_valid_hr_time(self) -> None:
        records = _irregular_frame()
        histogram = hr_histogram(records)
        zones_result = time_in_zones(records, ZONES)
        assert sum(histogram.values()) == zones_result.valid_hr_time_s

    def test_zero_heart_rate_excluded(self) -> None:
        rows = [
            {"timestamp": START, "heart_rate": 0},
            {"timestamp": START + timedelta(seconds=10), "heart_rate": 130},
            {"timestamp": START + timedelta(seconds=20), "heart_rate": 131},
        ]
        records = pl.DataFrame(rows, schema=RECORD_SCHEMA)
        assert hr_histogram(records) == {"130-134": 10.0}

    def test_empty(self) -> None:
        assert hr_histogram(empty_records()) == {}


class TestHrBucketAverages:
    def test_weighted_mean_ignores_interval_lengths(self) -> None:
        buckets = hr_bucket_averages(_irregular_frame())
        # All intervals fall into the first 600 s bucket.
        assert len(buckets) == 1
        assert buckets[0].start_min == 0
        # (5*120 + 15*140 + 30*160) / 50 = 150.0
        assert buckets[0].avg_hr == pytest.approx(150.0)
        assert buckets[0].hr_time_s == pytest.approx(50.0)

    def test_buckets_aligned_to_first_record(self) -> None:
        start = START + timedelta(minutes=4)  # not on a wall-clock boundary
        records = make_record_frame(start, [130] * 13, interval_s=60)
        buckets = hr_bucket_averages(records)
        assert [bucket.start_min for bucket in buckets] == [0, 10]
        # Bucket 0 holds 10 capped 30 s intervals; bucket 1 the remaining two
        # (the final record has no following interval).
        assert buckets[0].hr_time_s == pytest.approx(300.0)
        assert buckets[1].hr_time_s == pytest.approx(60.0)

    def test_gap_buckets_omitted(self) -> None:
        rows = [
            {"timestamp": START, "heart_rate": 120},
            {"timestamp": START + timedelta(seconds=10), "heart_rate": 130},
            # 20 minute gap: next interval lands in bucket 2.
            {"timestamp": START + timedelta(minutes=21), "heart_rate": 150},
            {"timestamp": START + timedelta(minutes=21, seconds=10), "heart_rate": 151},
        ]
        records = pl.DataFrame(rows, schema=RECORD_SCHEMA)
        buckets = hr_bucket_averages(records)
        assert [bucket.start_min for bucket in buckets] == [0, 20]

    def test_empty(self) -> None:
        assert hr_bucket_averages(empty_records()) == []


class TestHrDrift:
    def test_rising_hr_positive_slope(self) -> None:
        buckets = [
            *hr_bucket_averages(make_record_frame(START, [120] * 21, interval_s=60)),
        ]
        rising = hr_bucket_averages(
            make_record_frame(START, [120] * 11 + [150] * 11, interval_s=60)
        )
        slope = hr_drift_bpm_per_hour(rising)
        assert slope is not None and slope > 0
        assert hr_drift_bpm_per_hour(buckets) is not None

    def test_constant_hr_zero_slope(self) -> None:
        buckets = hr_bucket_averages(make_record_frame(START, [130] * 21, interval_s=60))
        slope = hr_drift_bpm_per_hour(buckets)
        assert slope is not None and abs(slope) < 1e-9

    def test_insufficient_data(self) -> None:
        single = hr_bucket_averages(make_record_frame(START, [130] * 5, interval_s=60))
        assert hr_drift_bpm_per_hour(single) is None
        assert hr_drift_bpm_per_hour([]) is None
