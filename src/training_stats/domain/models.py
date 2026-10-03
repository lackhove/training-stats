"""Domain input and cache-record models used by services and analytics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Final, Literal

from .metrics import Metric, TripPeriod


@dataclass(frozen=True, slots=True)
class SourceFile:
    """Metadata for a single FIT file provided by an activity source.

    ``path`` is a stable, source-specific identifier (relative path for local
    directories, remote path for WebDAV).
    """

    source_name: str
    path: str
    size: int | None = None
    modified_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ActivityRecord:
    """Persistable per-activity metadata and its derived metric collection."""

    activity_id: str
    start_time: datetime
    sport: str
    sub_sport: str | None
    segment_index: int
    segment_count: int
    repair_flags: tuple[str, ...]
    source_name: str
    source_path: str
    analysis_version: str
    metrics: tuple[Metric, ...]

    def __post_init__(self) -> None:
        if not self.metrics:
            msg = "an accepted ActivityRecord must contain at least one metric"
            raise ValueError(msg)
        if any(
            not isinstance(metric.period, TripPeriod)
            or metric.period.activity_id != self.activity_id
            for metric in self.metrics
        ):
            msg = "cached activity metrics must use their ActivityRecord trip period"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ActivitySummary:
    """Compact activity metadata for list responses."""

    activity_id: str
    start_time: datetime
    sport: str
    sub_sport: str | None

    @classmethod
    def from_record(cls, record: ActivityRecord) -> ActivitySummary:
        """Project a cached record into its compact list representation."""
        return cls(
            activity_id=record.activity_id,
            start_time=record.start_time,
            sport=record.sport,
            sub_sport=record.sub_sport,
        )


SourceStatus = Literal["ok", "failed", "duplicate"]
STATUS_OK: Final[SourceStatus] = "ok"
STATUS_FAILED: Final[SourceStatus] = "failed"
STATUS_DUPLICATE: Final[SourceStatus] = "duplicate"


@dataclass(frozen=True, slots=True)
class SourceIndexEntry:
    """Typed cache outcome for one source file/segment."""

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
class TrainingFormPoint:
    """Daily impulse-response training form point."""

    date: date
    ctl: float
    atl: float
    tsb: float
    daily_load: float


@dataclass(frozen=True, slots=True)
class TrainingFormResult:
    """Training form (fitness/fatigue/form) and tour-readiness summary."""

    current_ctl: float
    current_atl: float
    current_tsb: float
    form_classification: str
    is_warmup: bool
    history_days: int
    rolling_7d_distance_km: float
    rolling_28d_distance_km: float
    days_since_60k_ride: int | None
    trend_points: tuple[TrainingFormPoint, ...]
    load_confidence_ratio: float = 1.0
