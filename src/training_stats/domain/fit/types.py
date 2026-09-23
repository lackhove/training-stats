"""Transient FIT data types.

Record data holds high-frequency time-series samples in memory only, as a
Polars frame with a fixed schema. It must never be persisted to the cache or
included in API responses.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

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


@dataclass(frozen=True, slots=True)
class ParsedActivity:
    """Normalized, session-level view of one FIT activity segment.

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
    calories: float | None
    records: pl.DataFrame
    source_path: str
    source_hash: str
    paused_time_s: float | None = None
    segment_index: int = 0
    segment_count: int = 1
    repair_flags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ParsedFile:
    """One parsed FIT file: retained segment activities plus repair diagnostics."""

    activities: tuple[ParsedActivity, ...]
    repair_flags: tuple[str, ...]
    discarded_segment_count: int
