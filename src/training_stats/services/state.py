"""Typed synchronization records shared by services and storage adapters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..domain.analytics.heart_rate import HrBucketAvg
from ..domain.frames import SourceStatus

__all__ = ["ActivityRecord", "SourceIndexEntry", "SourceStatus", "SyncState"]


@dataclass(frozen=True, slots=True)
class SourceIndexEntry:
    source_name: str
    source_path: str
    source_hash: str | None
    file_size: int | None
    modified_at: datetime | None
    activity_id: str | None
    segment_index: int
    segment_count: int
    file_repair_flags: tuple[str, ...]
    discarded_segment_count: int
    processed_at: datetime
    source_analysis_version: str
    status: SourceStatus
    error_message: str | None


@dataclass(frozen=True, slots=True)
class ActivityRecord:
    activity_id: str
    start_time: datetime
    sport: str
    sub_sport: str | None
    duration_s: float | None
    moving_time_s: float | None
    paused_time_s: float | None
    distance_m: float | None
    elevation_gain_m: float | None
    avg_hr: float | None
    max_hr: float | None
    avg_speed_mps: float | None
    max_speed_mps: float | None
    calories: float | None
    segment_index: int
    segment_count: int
    repair_flags: tuple[str, ...]
    source_name: str
    source_path: str
    hr_zone_1_s: float
    hr_zone_2_s: float
    hr_zone_3_s: float
    hr_zone_4_s: float
    hr_zone_5_s: float
    hr_coverage_ratio: float
    hr_histogram_s: dict[str, float]
    hr_10min_averages: tuple[HrBucketAvg, ...]
    hr_drift_bpm_per_hour: float | None
    load_value: float | None
    load_method: str | None
    load_confidence: str | None
    data_completeness: float
    ef_first_half: float | None
    ef_second_half: float | None
    aerobic_decoupling_pct: float | None
    decoupling_confidence: str
    analysis_version: str


@dataclass(frozen=True)
class SyncState:
    """Source outcomes plus accepted activity facts for an incremental sync."""

    index_rows: list[SourceIndexEntry]
    activities: dict[str, ActivityRecord]
