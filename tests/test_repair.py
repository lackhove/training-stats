"""Tests for pure FIT activity-gap segmentation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from tests.conftest import make_record_frame
from training_stats.domain.fit.repair import repair_timeline
from training_stats.domain.fit.types import RECORD_COLUMNS, FitData, TimerEvent

START = datetime(2026, 9, 17, 8, 0, tzinfo=UTC)
GAP = 900.0
DISTANCE_GAP = 500.0


def _fit_data(records: pl.DataFrame, events: tuple[TimerEvent, ...] = ()) -> FitData:
    rows: list[dict[str, object]] = []
    for record in records.iter_rows(named=True):
        rows.append(
            {
                **record,
                "message_type": "record",
                "sport": None,
                "sub_sport": None,
                "event": None,
                "event_type": None,
            }
        )
    rows.extend(
        {
            "message_type": "event",
            "timestamp": event.timestamp,
            "heart_rate": None,
            "distance_m": None,
            "altitude_m": None,
            "speed_mps": None,
            "sport": None,
            "sub_sport": None,
            "event": event.event,
            "event_type": event.event_type,
        }
        for event in events
    )
    return FitData(frame=pl.DataFrame(rows, schema=FitData.SCHEMA))


def _spaced(
    start: datetime,
    count: int,
    seconds: float = 1.0,
    distance_step: float = 5.0,
) -> pl.DataFrame:
    timestamps = [start + timedelta(seconds=i * seconds) for i in range(count)]
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "heart_rate": [110] * count,
            "distance_m": [float(i * distance_step) for i in range(count)],
            "altitude_m": [None] * count,
            "speed_mps": [5.0] * count,
        },
        schema=make_record_frame(START, [110]).schema,
    )


def _repair(records: pl.DataFrame, events: tuple[TimerEvent, ...] = ()) -> list[FitData]:
    return repair_timeline(
        _fit_data(records, events),
        gap_threshold_s=GAP,
        distance_gap_threshold_m=DISTANCE_GAP,
    )


def _segment_records(segment: FitData) -> pl.DataFrame:
    return segment.frame.filter(pl.col("message_type") == "record").select(RECORD_COLUMNS)


class TestRepair:
    def test_monotonic_records_return_same_type(self) -> None:
        records = _spaced(START, 700)
        result = _repair(records)
        assert len(result) == 1
        assert isinstance(result[0], FitData)
        assert _segment_records(result[0]).equals(records)

    def test_unsorted_records_are_sorted(self) -> None:
        records = pl.concat([_spaced(START, 3), _spaced(START - timedelta(seconds=10), 2)])
        result = _repair(records)
        timestamps = _segment_records(result[0])["timestamp"].to_list()
        assert timestamps == sorted(timestamps)

    def test_time_gap_over_fifteen_minutes_splits(self) -> None:
        records = pl.concat(
            [
                _spaced(START, 4),
                _spaced(START + timedelta(seconds=904), 4),
            ]
        )
        result = _repair(records)
        assert len(result) == 2
        assert _segment_records(result[1])["timestamp"][0] == START + timedelta(seconds=904)

    def test_pause_under_fifteen_minutes_stays_in_segment(self) -> None:
        records = pl.concat(
            [
                _spaced(START, 4),
                _spaced(START + timedelta(seconds=899), 4),
            ]
        )
        assert len(_repair(records)) == 1

    def test_exact_fifteen_minute_interval_does_not_split(self) -> None:
        records = _spaced(START, 2, seconds=GAP)
        assert len(_repair(records)) == 1

    def test_distance_change_over_five_hundred_metres_splits(self) -> None:
        records = _spaced(START, 4).with_columns(
            pl.Series("distance_m", [0.0, 10.0, 610.0, 620.0], dtype=pl.Float64)
        )
        result = _repair(records)
        assert len(result) == 2
        assert _segment_records(result[1])["timestamp"][0] == START + timedelta(seconds=2)

    def test_exactly_five_hundred_metres_does_not_split(self) -> None:
        records = _spaced(START, 2, distance_step=DISTANCE_GAP)
        assert len(_repair(records)) == 1

    def test_timer_events_are_preserved_on_each_segment(self) -> None:
        events = (
            TimerEvent(START + timedelta(seconds=5), "timer", "stop"),
            TimerEvent(START + timedelta(seconds=10), "timer", "start"),
        )
        records = pl.concat(
            [
                _spaced(START, 2),
                _spaced(START + timedelta(minutes=20), 2),
            ]
        )
        result = _repair(records, events)
        assert len(result) == 2
        assert all(
            result_item.frame.filter(pl.col("message_type") == "event").height == 2
            for result_item in result
        )

    def test_no_record_rows_return_no_segments(self) -> None:
        empty = FitData(frame=pl.DataFrame(schema=FitData.SCHEMA))
        assert (
            repair_timeline(empty, gap_threshold_s=GAP, distance_gap_threshold_m=DISTANCE_GAP) == []
        )
