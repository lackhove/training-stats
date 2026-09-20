"""Tests for the transient FIT timeline repair layer."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from tests.conftest import make_record_frame
from training_coach_mcp.fit.repair import (
    RepairResult,
    moving_time_from_records,
    repair_timeline,
)

START = datetime(2026, 9, 17, 8, 0, tzinfo=UTC)
GAP = 120.0
JUMP = 3600.0
MIN_SEGMENT = 600.0
MAX_PRE_MOVING = 60.0
THRESHOLD = 1.0


def repair(records: pl.DataFrame) -> RepairResult:
    return repair_timeline(
        records,
        gap_threshold_s=GAP,
        clock_jump_min_gap_s=JUMP,
        minimum_segment_duration_s=MIN_SEGMENT,
        clock_shift_max_pre_moving_s=MAX_PRE_MOVING,
        moving_speed_threshold_mps=THRESHOLD,
    )


def _spaced(
    start: datetime, count: int, seconds: float = 1.0, speed: float | None = 5.0
) -> pl.DataFrame:
    timestamps = [start + timedelta(seconds=i * seconds) for i in range(count)]
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "heart_rate": [110] * count,
            "distance_m": [float(i * 5) for i in range(count)],
            "altitude_m": [None] * count,
            "speed_mps": [speed] * count,
        },
        schema=make_record_frame(START, [110]).schema,
    )


def _static(start: datetime, count: int) -> pl.DataFrame:
    return _spaced(start, count, speed=0.0)


def _span(result: RepairResult, index: int = 0) -> float:
    ts = result.segments[index].records["timestamp"]
    return (ts[-1] - ts[0]).total_seconds()


class TestPassthrough:
    def test_monotonic_records_single_segment(self) -> None:
        records = _spaced(START, 700)
        result = repair(records)
        assert len(result.segments) == 1
        assert result.flags == ()
        assert result.discarded_segment_count == 0
        assert result.segments[0].records.equals(records)
        assert result.segments[0].flags == ()

    def test_unsorted_records_flagged_and_sorted(self) -> None:
        records = pl.concat([_spaced(START, 3), _static(START - timedelta(seconds=10), 2)])
        result = repair(records)
        assert "non_monotonic_timestamps" in result.flags
        timestamps = result.segments[0].records["timestamp"].to_list()
        assert timestamps == sorted(timestamps)


class TestClockShift:
    def test_epoch_style_jump_shifts_static_prefix(self) -> None:
        """Static pre-GPS recording on a wrong clock merges onto the ride timeline."""
        wrong_clock_start = datetime(2021, 1, 1, 0, 3, 32, tzinfo=UTC)
        records = pl.concat([_static(wrong_clock_start, 120), _spaced(START, 400)])
        result = repair(records)
        assert len(result.segments) == 1
        assert result.segments[0].flags == ("clock_corrected",)
        assert "clock_corrected" in result.flags
        first = result.segments[0].records["timestamp"][0]
        assert first == START - timedelta(seconds=119)
        # coherent merged timeline: no jump left in the span
        assert _span(result) == pytest.approx(518.0)

    def test_appended_resume_blip_shifts(self) -> None:
        """A seconds-long static blip days before the ride is shifted forward."""
        blip_start = START - timedelta(days=2)
        records = pl.concat([_static(blip_start, 12), _spaced(START, 400)])
        result = repair(records)
        assert len(result.segments) == 1
        assert result.segments[0].flags == ("clock_corrected",)
        first = result.segments[0].records["timestamp"][0]
        assert first == START - timedelta(seconds=11)

    def test_single_record_prefix_shifts(self) -> None:
        records = pl.concat([_static(START - timedelta(days=1), 1), _spaced(START, 400)])
        result = repair(records)
        assert len(result.segments) == 1
        assert result.segments[0].flags == ("clock_corrected",)
        assert _span(result) == pytest.approx(399.0)

    def test_moving_prefix_is_not_shifted(self) -> None:
        """A real ride leg before a long break keeps its own segment."""
        records = pl.concat([_spaced(START, 700), _spaced(START + timedelta(hours=2), 700)])
        result = repair(records)
        assert len(result.segments) == 2
        assert result.flags == ()
        starts = [segment.records["timestamp"][0] for segment in result.segments]
        assert starts == [START, START + timedelta(hours=2)]


class TestSegmentation:
    def test_gap_splits_segments(self) -> None:
        records = pl.concat([_spaced(START, 700), _spaced(START + timedelta(seconds=999), 700)])
        result = repair(records)
        assert len(result.segments) == 2
        assert result.flags == ()

    def test_small_gap_does_not_split(self) -> None:
        records = pl.concat([_spaced(START, 700), _spaced(START + timedelta(seconds=789), 700)])
        result = repair(records)
        assert len(result.segments) == 1

    def test_short_segment_discarded_when_longer_exists(self) -> None:
        records = pl.concat(
            [
                _spaced(START, 300),
                _spaced(START + timedelta(minutes=30), 700),
            ]
        )
        result = repair(records)
        assert len(result.segments) == 1
        assert result.discarded_segment_count == 1
        assert result.flags == ("segments_discarded",)
        assert result.segments[0].records["timestamp"][0] == START + timedelta(minutes=30)

    def test_all_short_segments_keep_longest(self) -> None:
        records = pl.concat(
            [
                _spaced(START, 300),
                _spaced(START + timedelta(minutes=30), 400),
            ]
        )
        result = repair(records)
        assert len(result.segments) == 1
        assert result.discarded_segment_count == 1
        assert result.segments[0].records["timestamp"][0] == START + timedelta(minutes=30)

    def test_single_short_segment_is_retained(self) -> None:
        records = _spaced(START, 300)
        result = repair(records)
        assert len(result.segments) == 1
        assert result.discarded_segment_count == 0
        assert result.flags == ()


class TestMovingTime:
    def test_speed_gated_intervals_are_summed(self) -> None:
        records = make_record_frame(START, [110] * 5, interval_s=60)
        records = records.with_columns(
            pl.Series("speed_mps", [5.0, 0.0, 5.0, 5.0, 5.0], dtype=pl.Float64)
        )
        moving = moving_time_from_records(records, threshold_mps=1.0)
        # three speed-gated intervals, each capped at 30s
        assert moving == pytest.approx(90.0)

    def test_intervals_capped_at_max(self) -> None:
        records = make_record_frame(START, [110] * 3, interval_s=600)
        records = records.with_columns(pl.Series("speed_mps", [5.0, 5.0, 5.0], dtype=pl.Float64))
        moving = moving_time_from_records(records, threshold_mps=1.0)
        assert moving == pytest.approx(60.0)

    def test_low_speed_coverage_falls_back(self) -> None:
        records = make_record_frame(START, [110] * 5, interval_s=60)
        records = records.with_columns(
            pl.Series("speed_mps", [5.0, None, None, None, None], dtype=pl.Float64)
        )
        assert moving_time_from_records(records, threshold_mps=1.0) is None

    def test_all_null_speeds_falls_back(self) -> None:
        records = make_record_frame(START, [110] * 5, interval_s=60)
        records = records.with_columns(pl.Series("speed_mps", [None] * 5, dtype=pl.Float64))
        assert moving_time_from_records(records, threshold_mps=1.0) is None

    def test_too_few_records(self) -> None:
        records = make_record_frame(START, [110], interval_s=60)
        assert moving_time_from_records(records, threshold_mps=1.0) is None
