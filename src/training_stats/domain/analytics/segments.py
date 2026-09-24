"""Activity statistics derived from repaired raw FIT segments."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime

import polars as pl

from ..fit.types import (
    RECORD_COLUMNS,
    RECORD_SCHEMA,
    FitData,
    ParsedActivity,
    PauseInterval,
    TimerEvent,
)
from ..settings import AthleteConfig, QualityConfig
from .activities import AnalyzedActivity, compute_activity_metrics
from .intervals import interval_frame, is_paused, merge_pause_intervals

_MIN_SPEED_COVERAGE = 0.5
_MAX_MOVING_INTERVAL_S = 30.0
_FORGOTTEN_STOP_CHECK_DURATION_S = 7200.0


@dataclass(frozen=True, slots=True)
class SegmentAnalysis:
    """Analysis output for all candidates from one source file."""

    activities: tuple[AnalyzedActivity, ...]
    discarded_segment_count: int


@dataclass(frozen=True, slots=True)
class _Facts:
    records: pl.DataFrame
    start_time: datetime
    sport: str
    sub_sport: str | None
    duration_s: float
    moving_time_s: float
    paused_time_s: float
    distance_m: float | None
    elevation_gain_m: float | None
    avg_hr: float | None
    max_hr: float | None
    avg_speed_mps: float | None
    max_speed_mps: float | None
    pause_intervals: tuple[PauseInterval, ...]


def analyze_segments(
    segments: list[FitData],
    *,
    source_path: str,
    athlete: AthleteConfig,
    analysis_version: str,
    quality: QualityConfig,
) -> SegmentAnalysis:
    """Compute statistics for repaired segments and discard short activity legs."""
    candidates = [facts for data in segments if (facts := _derive_facts(data, quality)) is not None]
    retained = [
        facts for facts in candidates if facts.moving_time_s >= quality.minimum_segment_duration_s
    ]
    discarded = len(candidates) - len(retained)
    activities: list[AnalyzedActivity] = []

    for index, facts in enumerate(retained):
        flags = (
            ("forgotten_stop",)
            if _is_forgotten_stop(facts.duration_s, facts.moving_time_s, quality)
            else ()
        )
        activity = ParsedActivity(
            activity_id=_activity_id(facts.start_time, facts.sport, facts.duration_s, index),
            start_time=facts.start_time,
            sport=facts.sport,
            sub_sport=facts.sub_sport,
            duration_s=facts.duration_s,
            moving_time_s=facts.moving_time_s,
            paused_time_s=facts.paused_time_s,
            distance_m=facts.distance_m,
            elevation_gain_m=facts.elevation_gain_m,
            avg_hr=facts.avg_hr,
            max_hr=facts.max_hr,
            avg_speed_mps=facts.avg_speed_mps,
            max_speed_mps=facts.max_speed_mps,
            records=facts.records,
            source_path=source_path,
            segment_index=index,
            segment_count=len(retained),
            repair_flags=flags,
            pause_intervals=facts.pause_intervals,
        )
        activities.append(
            AnalyzedActivity(
                activity=activity,
                metrics=compute_activity_metrics(activity, athlete, analysis_version, quality),
            )
        )

    return SegmentAnalysis(activities=tuple(activities), discarded_segment_count=discarded)


def _derive_facts(data: FitData, quality: QualityConfig) -> _Facts | None:
    raw = data.frame
    records = (
        raw.filter(pl.col("message_type") == "record")
        .select(RECORD_COLUMNS)
        .cast(pl.Schema(RECORD_SCHEMA))
        .sort("timestamp", maintain_order=True)
    )
    if records.is_empty():
        return None

    first = records["timestamp"][0]
    last = records["timestamp"][-1]
    if not isinstance(first, datetime) or not isinstance(last, datetime):
        return None
    duration_s = max(0.0, (last - first).total_seconds())
    timer_events = _timer_events(raw)
    timer_pauses = _timer_pause_intervals(timer_events, first, last)
    if timer_pauses is None:
        moving_time_s, pauses = _speed_derived_movement(
            records, duration_s, quality.moving_speed_threshold_mps
        )
    else:
        pauses = timer_pauses
        paused_s = sum((pause.end - pause.start).total_seconds() for pause in pauses)
        moving_time_s = max(0.0, duration_s - paused_s)

    intervals = interval_frame(records, pauses, max_interval_s=_MAX_MOVING_INTERVAL_S)
    active_hr = intervals.filter(
        pl.col("heart_rate").is_not_null() & (pl.col("heart_rate") > 0) & (pl.col("active_s") > 0)
    )
    active_hr_time = _float(active_hr["active_s"].sum()) if not active_hr.is_empty() else 0.0
    avg_hr = (
        _float((active_hr["heart_rate"] * active_hr["active_s"]).sum()) / active_hr_time
        if active_hr_time > 0
        else None
    )
    active_records = [
        index
        for index, timestamp in enumerate(records["timestamp"].to_list())
        if isinstance(timestamp, datetime) and not is_paused(timestamp, pauses)
    ]
    record_hrs = [
        _float(records["heart_rate"][index])
        for index in active_records
        if records["heart_rate"][index] is not None and _float(records["heart_rate"][index]) > 0
    ]
    record_speeds = [
        _float(records["speed_mps"][index])
        for index in active_records
        if records["speed_mps"][index] is not None
    ]
    distance_m = _distance(records, intervals)
    elevation_gain_m = _elevation_gain(records, intervals)
    max_hr = max(record_hrs) if record_hrs else None
    max_speed = max(record_speeds) if record_speeds else None
    average_speed = (
        distance_m / moving_time_s
        if distance_m is not None and moving_time_s > 0
        else _weighted_speed(intervals)
    )
    sport, sub_sport = _sport_metadata(raw)

    return _Facts(
        records=records,
        start_time=first,
        sport=sport,
        sub_sport=sub_sport,
        duration_s=duration_s,
        moving_time_s=moving_time_s,
        paused_time_s=max(0.0, duration_s - moving_time_s),
        distance_m=distance_m,
        elevation_gain_m=elevation_gain_m,
        avg_hr=avg_hr,
        max_hr=max_hr,
        avg_speed_mps=average_speed,
        max_speed_mps=max_speed,
        pause_intervals=pauses,
    )


def _timer_events(data: pl.DataFrame) -> tuple[TimerEvent, ...]:
    events: list[TimerEvent] = []
    rows = data.filter((pl.col("message_type") == "event") & (pl.col("event") == "timer")).select(
        "timestamp", "event", "event_type"
    )
    for row in rows.iter_rows(named=True):
        timestamp = row["timestamp"]
        event = row["event"]
        event_type = row["event_type"]
        if (
            isinstance(timestamp, datetime)
            and isinstance(event, str)
            and isinstance(event_type, str)
        ):
            events.append(TimerEvent(timestamp, event, event_type))
    return tuple(sorted(events, key=lambda event: event.timestamp))


def _timer_pause_intervals(
    events: tuple[TimerEvent, ...], start: datetime, end: datetime
) -> tuple[PauseInterval, ...] | None:
    stop_types = {"stop", "stop_all", "stop_disable", "stop_disable_all"}
    start_types = {"start"}
    recognized = [event for event in events if event.event_type in stop_types | start_types]
    if not recognized:
        return None

    paused_at: datetime | None = None
    intervals: list[PauseInterval] = []
    for event in recognized:
        if event.event_type in stop_types and paused_at is None:
            paused_at = event.timestamp
        elif event.event_type in start_types and paused_at is not None:
            if event.timestamp > paused_at:
                intervals.append(PauseInterval(paused_at, event.timestamp))
            paused_at = None
    if paused_at is not None and end > paused_at:
        intervals.append(PauseInterval(paused_at, end))
    return merge_pause_intervals(tuple(intervals), start, end)


def _speed_derived_movement(
    records: pl.DataFrame, duration_s: float, threshold_mps: float
) -> tuple[float, tuple[PauseInterval, ...]]:
    if records.height < 2:
        return duration_s, ()
    timestamps = records["timestamp"].to_list()
    speeds = records["speed_mps"].to_list()
    valid_intervals = 0
    valid_speed_intervals = 0
    pauses: list[PauseInterval] = []
    for index in range(records.height - 1):
        start = timestamps[index]
        end = timestamps[index + 1]
        if not isinstance(start, datetime) or not isinstance(end, datetime) or end <= start:
            continue
        valid_intervals += 1
        speed = speeds[index]
        if isinstance(speed, bool) or not isinstance(speed, int | float):
            continue
        valid_speed_intervals += 1
        if speed < threshold_mps:
            pauses.append(PauseInterval(start, end))
    if valid_intervals == 0 or valid_speed_intervals / valid_intervals < _MIN_SPEED_COVERAGE:
        return duration_s, ()
    if not isinstance(timestamps[0], datetime) or not isinstance(timestamps[-1], datetime):
        return duration_s, ()
    merged_pauses = merge_pause_intervals(tuple(pauses), timestamps[0], timestamps[-1])
    paused_s = sum((pause.end - pause.start).total_seconds() for pause in merged_pauses)
    return max(0.0, duration_s - paused_s), merged_pauses


def _distance(records: pl.DataFrame, intervals: pl.DataFrame) -> float | None:
    values = records["distance_m"].to_list()
    active = intervals["active_duration_s"].to_list()
    durations = intervals["dt_s"].to_list()
    distance = 0.0
    observed = False
    for index, active_s in enumerate(active):
        previous = values[index]
        current = values[index + 1]
        dt_s = durations[index]
        if active_s <= 0 or dt_s <= 0 or previous is None or current is None:
            continue
        delta = _float(current) - _float(previous)
        if delta > 0:
            distance += delta * min(1.0, active_s / dt_s)
            observed = True
    return distance if observed else None


def _elevation_gain(records: pl.DataFrame, intervals: pl.DataFrame) -> float | None:
    values = records["altitude_m"].to_list()
    active = intervals["active_duration_s"].to_list()
    durations = intervals["dt_s"].to_list()
    gain = 0.0
    for index, active_s in enumerate(active):
        previous = values[index]
        current = values[index + 1]
        dt_s = durations[index]
        if active_s > 0 and dt_s > 0 and previous is not None and current is not None:
            gain += max(0.0, _float(current) - _float(previous)) * min(1.0, active_s / dt_s)
    return gain if gain > 0 else None


def _weighted_speed(intervals: pl.DataFrame) -> float | None:
    speeds = intervals.filter(pl.col("speed_mps").is_not_null() & (pl.col("active_s") > 0))
    if speeds.is_empty():
        return None
    total_s = _float(speeds["active_s"].sum())
    if total_s <= 0:
        return None
    weighted = _float((speeds["speed_mps"] * speeds["active_s"]).sum())
    return weighted / total_s


def _sport_metadata(data: pl.DataFrame) -> tuple[str, str | None]:
    sessions = data.filter(pl.col("message_type") == "session")
    sport = _first_string(sessions["sport"].to_list()) if not sessions.is_empty() else None
    sub_sport = _first_string(sessions["sub_sport"].to_list()) if not sessions.is_empty() else None
    return sport or "unknown", sub_sport


def _first_string(values: list[object]) -> str | None:
    return next((value for value in values if isinstance(value, str)), None)


def _is_forgotten_stop(duration_s: float, moving_time_s: float, quality: QualityConfig) -> bool:
    if duration_s > quality.forgotten_stop_max_duration_s:
        return True
    return (
        duration_s > _FORGOTTEN_STOP_CHECK_DURATION_S
        and moving_time_s / duration_s < quality.forgotten_stop_min_moving_ratio
    )


def _activity_id(start_time: datetime, sport: str, duration_s: float, segment_index: int) -> str:
    key = "|".join(
        [
            start_time.astimezone(UTC).isoformat(),
            sport,
            f"{duration_s:.0f}",
            str(segment_index),
        ]
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return float(value)
