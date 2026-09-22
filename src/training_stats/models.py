"""Shared domain models used across sources, cache, analytics and services.

Services return these models (plus analytics row types); serialization to
HTTP/CLI JSON happens only at the edges via :mod:`training_stats.wire`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .analytics.daily import DailySummaryRow
    from .analytics.heart_rate import HrBucketAvg
    from .analytics.weekly import WeeklySummaryRow

# Request caps shared by the API query constraints and the service validators.
MAX_LIST_LIMIT = 500
MAX_WEEKS = 104
MAX_DAYS = 3650


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
class ActivitySummary:
    """Compact, LLM-friendly activity summary returned by list tools."""

    activity_id: str
    start_time: datetime
    sport: str
    sub_sport: str | None
    duration_s: float | None
    distance_m: float | None
    elevation_gain_m: float | None
    avg_hr: float | None
    max_hr: float | None
    calories: float | None
    load_value: float | None
    moving_time_s: float | None = None
    local_start_time: str = ""


@dataclass(frozen=True, slots=True)
class ActivityDetails:
    """Full activity record including derived metrics and data quality."""

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
    analysis_version: str
    source_name: str
    source_path: str
    ef_first_half: float | None = None
    ef_second_half: float | None = None
    aerobic_decoupling_pct: float | None = None
    decoupling_confidence: str | None = None
    local_start_time: str = ""
    paused_time_s: float | None = None
    repair_flags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Decoupling:
    """Aerobic decoupling metrics for one activity."""

    ef_first_half: float | None
    ef_second_half: float | None
    aerobic_decoupling_pct: float | None
    confidence: str | None


@dataclass(frozen=True, slots=True)
class LatestActivity:
    """Most recent activity summary plus its decoupling metrics, if known."""

    summary: ActivitySummary
    decoupling: Decoupling | None = None


@dataclass(frozen=True, slots=True)
class DataQuality:
    """Aggregate data-quality indicators for a period."""

    activity_count: int
    mean_hr_coverage_ratio: float | None
    mean_data_completeness: float | None
    failed_file_count: int
    repaired_file_count: int = 0
    discarded_segment_count: int = 0
    duplicate_activity_count: int = 0


@dataclass(frozen=True, slots=True)
class FormSnapshot:
    """Current training form (fitness/fatigue) summary."""

    ctl: float
    atl: float
    tsb: float
    classification: str
    is_warmup: bool
    load_confidence_ratio: float


@dataclass(frozen=True, slots=True)
class TourReadiness:
    """Rolling endurance volume and long-ride recency for tour planning."""

    rolling_7d_km: float
    rolling_28d_km: float
    days_since_60k: int | None


@dataclass(frozen=True, slots=True)
class Overview:
    """Compact overview bundle for coaching conversations and scripts.

    Subsumes coach context, training form and metric trends: the daily
    series carries per-(date, sport) totals for every metric, and the form
    history carries the daily CTL/ATL/TSB curve.
    """

    athlete_profile: dict[str, object]
    recent_activities: tuple[ActivitySummary, ...]
    weekly_summaries: tuple[WeeklySummaryRow, ...]
    daily_series: tuple[DailySummaryRow, ...]
    form_history: tuple[TrainingFormPoint, ...]
    zone_distribution_s: dict[str, float]
    data_quality: DataQuality
    latest_activity: LatestActivity | None
    period_start: date
    period_end: date
    form: FormSnapshot
    tour_readiness: TourReadiness


@dataclass(frozen=True, slots=True)
class TrainingFormPoint:
    """Daily impulse-response training form point."""

    date: str
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


@dataclass(frozen=True, slots=True)
class SyncResult:
    """Outcome of one synchronization run."""

    files_discovered: int
    files_processed: int
    files_skipped: int
    files_failed: int
    cache_updated_at: datetime | None = field(default=None)
    source_errors: tuple[str, ...] = ()
    files_repaired: int = 0
    segments_discarded: int = 0
    duplicates_detected: int = 0


@dataclass(frozen=True, slots=True)
class ThrottledSync:
    """A skipped sync: the last successful one is still within the interval."""

    seconds_since_last_sync: float
