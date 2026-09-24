"""Transient FIT data types.

Record data holds high-frequency time-series samples in memory only, as a
Polars frame with a fixed schema. It must never be persisted to the cache or
included in API responses.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

import polars as pl

# Fixed schema for transient per-record samples. Columns are read-only by
# convention: consumers must use select/with_columns/group_by, never mutate.
RECORD_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Datetime("us", "UTC"),
    "heart_rate": pl.Int32(),
    "distance_m": pl.Float64(),
    "altitude_m": pl.Float64(),
    "speed_mps": pl.Float64(),
}

RECORD_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "heart_rate",
    "distance_m",
    "altitude_m",
    "speed_mps",
)


def empty_records() -> pl.DataFrame:
    """Return an empty record frame with the canonical schema."""
    return pl.DataFrame(schema=RECORD_SCHEMA)


def empty_fit_data_frame() -> pl.DataFrame:
    """Return an empty frame for normalized raw FIT messages."""
    return pl.DataFrame(schema=FitData.SCHEMA)


@dataclass(frozen=True, slots=True)
class TimerEvent:
    """Normalized timer event decoded from a FIT event message."""

    timestamp: datetime
    event: str
    event_type: str


@dataclass(frozen=True, slots=True)
class PauseInterval:
    """Wall-clock interval during which the activity timer was stopped."""

    start: datetime
    end: datetime


@dataclass(frozen=True, slots=True)
class FitData:
    """Thin wrapper around the normalized raw FIT message dataframe."""

    SCHEMA: ClassVar[dict[str, pl.DataType]] = {
        "message_type": pl.String(),
        "timestamp": pl.Datetime("us", "UTC"),
        "heart_rate": pl.Int32(),
        "distance_m": pl.Float64(),
        "altitude_m": pl.Float64(),
        "speed_mps": pl.Float64(),
        "sport": pl.String(),
        "sub_sport": pl.String(),
        "event": pl.String(),
        "event_type": pl.String(),
    }

    frame: pl.DataFrame


@dataclass(frozen=True, slots=True)
class ParsedActivity:
    """Analyzed view of one FIT segment.

    ``records`` is transient high-frequency data used for analytics and then
    discarded; it is never cached or serialized. ``segment_index``/``segment_count``
    place the activity within its source file when recording breaks split a
    file into several activities.
    """

    activity_id: str
    start_time: datetime
    sport: str
    sub_sport: str | None
    duration_s: float | None
    moving_time_s: float | None
    distance_m: float | None
    elevation_gain_m: float | None
    avg_hr: float | None
    max_hr: float | None
    avg_speed_mps: float | None
    max_speed_mps: float | None
    records: pl.DataFrame
    source_path: str
    paused_time_s: float | None = None
    segment_index: int = 0
    segment_count: int = 1
    repair_flags: tuple[str, ...] = ()
    pause_intervals: tuple[PauseInterval, ...] = ()
