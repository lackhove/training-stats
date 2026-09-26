"""Focused record-pass helpers shared by the activity statistic producers.

One validated ``Interval`` record plus the time-base, pause and validation
helpers the families share. Nothing here accumulates cross-family summaries:
each producer derives its own narrow facts from these records.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..fit.types import ParsedActivity, PauseInterval
from .intervals import merge_pause_intervals, overlap_seconds

MAX_INTERVAL_S = 30.0

SPEED_BOUNDS_MPS = (0.0, 40.0)
HEART_RATE_BOUNDS_BPM = (25.0, 240.0)
CADENCE_BOUNDS_RPM = (0.0, 220.0)


@dataclass(frozen=True, slots=True)
class Interval:
    """One capped record interval with validated sensor samples.

    ``elapsed_s`` is the cumulative elapsed time before the interval;
    ``rejected_values`` counts samples discarded as outliers in it.
    """

    elapsed_s: float
    active_s: float
    speed_kmh: float | None
    heart_rate: float | None
    cadence_rpm: float | None
    distance_m: float
    rejected_values: int


def number(value: object) -> float | None:
    """Coerce a record cell to a finite float, or ``None`` when unusable."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def accept(value: float | None, low: float, high: float) -> tuple[float | None, bool]:
    """Return the in-range value and whether it was rejected as an outlier."""
    if value is None:
        return None, False
    if value < low or value > high:
        return None, True
    return value, False


def is_short_activity(activity: ParsedActivity) -> bool:
    """Whether records are too few, or badly stamped, for interval statistics."""
    records = activity.records
    if records.height < 2:
        return True
    timestamps = records["timestamp"].to_list()
    return not isinstance(timestamps[0], datetime) or not isinstance(timestamps[-1], datetime)


def merged_pauses(activity: ParsedActivity) -> tuple[PauseInterval, ...]:
    """Clip and merge the activity's pauses to its record span."""
    if is_short_activity(activity):
        return ()
    timestamps = activity.records["timestamp"].to_list()
    first = timestamps[0]
    last = timestamps[-1]
    if not isinstance(first, datetime) or not isinstance(last, datetime):
        return ()
    return merge_pause_intervals(activity.pause_intervals, first, last)


def elapsed_time_s(activity: ParsedActivity) -> float:
    """Sum of positive sample-to-sample gaps over the record span."""
    timestamps = activity.records["timestamp"].to_list()
    total = 0.0
    for index in range(len(timestamps) - 1):
        start = timestamps[index]
        end = timestamps[index + 1]
        if not isinstance(start, datetime) or not isinstance(end, datetime):
            continue
        dt_s = (end - start).total_seconds()
        if dt_s > 0.0:
            total += dt_s
    return total


def moving_s(interval: Interval, threshold_mps: float) -> float:
    """Active time of an interval counted as moving at the configured speed."""
    speed = interval.speed_kmh
    if speed is None or speed / 3.6 < threshold_mps:
        return 0.0
    return interval.active_s


def moving_time_s(
    activity: ParsedActivity, intervals: Sequence[Interval], threshold_mps: float
) -> float:
    """Configured moving time, falling back to the summed interval weights."""
    return activity.moving_time_s or sum(
        moving_s(interval, threshold_mps) for interval in intervals
    )


def active_time_s(activity: ParsedActivity) -> float:
    """Non-paused span of the activity, in seconds."""
    if is_short_activity(activity):
        return activity.duration_s or 0.0
    span = activity.duration_s or elapsed_time_s(activity)
    return max(0.0, span - (activity.paused_time_s or 0.0))


def timestamp_issues(activity: ParsedActivity) -> tuple[int, int]:
    """Count duplicate timestamps (``dt <= 0``) and gaps (``dt > 30 s``)."""
    timestamps = activity.records["timestamp"].to_list()
    duplicates = 0
    gaps = 0
    for index in range(len(timestamps) - 1):
        start = timestamps[index]
        end = timestamps[index + 1]
        if not isinstance(start, datetime) or not isinstance(end, datetime):
            continue
        dt_s = (end - start).total_seconds()
        if dt_s <= 0.0:
            duplicates += int(dt_s == 0.0)
        elif dt_s > MAX_INTERVAL_S:
            gaps += 1
    return duplicates, gaps


def build_intervals(activity: ParsedActivity) -> list[Interval]:
    """Build capped, pause-adjusted intervals with validated sensor samples."""
    records = activity.records
    if is_short_activity(activity):
        return []
    timestamps = records["timestamp"].to_list()
    pauses = merged_pauses(activity)
    intervals: list[Interval] = []
    elapsed_s = 0.0
    for index in range(records.height - 1):
        start = timestamps[index]
        end = timestamps[index + 1]
        if not isinstance(start, datetime) or not isinstance(end, datetime):
            continue
        dt_s = (end - start).total_seconds()
        if dt_s <= 0.0:
            continue
        counted_s = min(dt_s, MAX_INTERVAL_S)
        counted_end = start + timedelta(seconds=counted_s)
        active_s = max(0.0, counted_s - overlap_seconds(start, counted_end, pauses))
        if active_s <= 0.0:
            elapsed_s += dt_s
            continue

        speed, rejected_speed = accept(number(records["speed_mps"][index]), *SPEED_BOUNDS_MPS)
        heart_rate, rejected_heart_rate = accept(
            number(records["heart_rate"][index]), *HEART_RATE_BOUNDS_BPM
        )
        cadence_rpm, rejected_cadence = _cadence_sample(
            records["cadence"][index], records["fractional_cadence"][index]
        )
        distance = _distance_delta(
            records["distance_m"][index], records["distance_m"][index + 1], active_s, dt_s
        )

        intervals.append(
            Interval(
                elapsed_s=elapsed_s,
                active_s=active_s,
                speed_kmh=speed * 3.6 if speed is not None else None,
                heart_rate=heart_rate,
                cadence_rpm=cadence_rpm,
                distance_m=distance,
                rejected_values=int(rejected_speed)
                + int(rejected_heart_rate)
                + int(rejected_cadence),
            )
        )
        elapsed_s += dt_s
    return intervals


def _cadence_sample(cadence: object, fractional: object) -> tuple[float | None, bool]:
    cadence_rpm = number(cadence)
    fraction = number(fractional)
    if cadence_rpm is not None and fraction is not None:
        cadence_rpm += fraction / 128.0
    return accept(cadence_rpm, *CADENCE_BOUNDS_RPM)


def _distance_delta(previous: object, following: object, active_s: float, dt_s: float) -> float:
    before = number(previous)
    after = number(following)
    if before is None or after is None:
        return 0.0
    delta = after - before
    if delta <= 0.0:
        return 0.0
    return delta * min(1.0, active_s / dt_s)
