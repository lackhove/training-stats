"""Time weights for transient record intervals, excluding known pauses."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from ..fit.types import PauseInterval


def merge_pause_intervals(
    intervals: tuple[PauseInterval, ...], start: datetime, end: datetime
) -> tuple[PauseInterval, ...]:
    """Clip and merge pause intervals to one segment's record span."""
    clipped = sorted(
        (
            PauseInterval(max(interval.start, start), min(interval.end, end))
            for interval in intervals
            if interval.end > start and interval.start < end
        ),
        key=lambda interval: interval.start,
    )
    merged: list[PauseInterval] = []
    for interval in clipped:
        if interval.end <= interval.start:
            continue
        if merged and interval.start <= merged[-1].end:
            previous = merged[-1]
            merged[-1] = PauseInterval(previous.start, max(previous.end, interval.end))
        else:
            merged.append(interval)
    return tuple(merged)


def overlap_seconds(start: datetime, end: datetime, pauses: tuple[PauseInterval, ...]) -> float:
    """Return the duration shared by an interval and sorted, merged pauses."""
    overlap = 0.0
    for pause in pauses:
        if pause.end <= start:
            continue
        if pause.start >= end:
            break
        overlap += (min(end, pause.end) - max(start, pause.start)).total_seconds()
    return overlap


def is_paused(timestamp: datetime, pauses: tuple[PauseInterval, ...]) -> bool:
    """Whether a sample timestamp falls inside a half-open pause interval."""
    return any(pause.start <= timestamp < pause.end for pause in pauses)


def interval_frame(
    records: pl.DataFrame,
    pauses: tuple[PauseInterval, ...] = (),
    *,
    max_interval_s: float,
) -> pl.DataFrame:
    """Build per-record interval weights on an active-time axis.

    Each sample owns the interval until the next sample. The statistical
    weight is capped, then paused overlap is removed. ``active_elapsed_s`` is
    cumulative active time before the interval and supports pause-free buckets.
    """
    rows: list[dict[str, object]] = []
    active_elapsed_s = 0.0
    for index in range(max(records.height - 1, 0)):
        timestamp = records["timestamp"][index]
        next_timestamp = records["timestamp"][index + 1]
        if not isinstance(timestamp, datetime) or not isinstance(next_timestamp, datetime):
            continue
        dt_s = max(0.0, (next_timestamp - timestamp).total_seconds())
        counted_s = min(dt_s, max_interval_s)
        counted_end = timestamp + timedelta(seconds=counted_s)
        active_s = max(0.0, counted_s - overlap_seconds(timestamp, counted_end, pauses))
        active_duration_s = max(0.0, dt_s - overlap_seconds(timestamp, next_timestamp, pauses))
        rows.append(
            {
                "timestamp": timestamp,
                "heart_rate": records["heart_rate"][index],
                "speed_mps": records["speed_mps"][index],
                "dt_s": dt_s,
                "counted_s": counted_s,
                "active_s": active_s,
                "active_duration_s": active_duration_s,
                "active_elapsed_s": active_elapsed_s,
            }
        )
        active_elapsed_s += max(0.0, dt_s - overlap_seconds(timestamp, next_timestamp, pauses))
    return pl.DataFrame(
        rows,
        schema={
            "timestamp": pl.Datetime("us", "UTC"),
            "heart_rate": pl.Int32(),
            "speed_mps": pl.Float64(),
            "dt_s": pl.Float64(),
            "counted_s": pl.Float64(),
            "active_s": pl.Float64(),
            "active_duration_s": pl.Float64(),
            "active_elapsed_s": pl.Float64(),
        },
    )
