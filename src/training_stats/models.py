"""Shared domain models used across sources, cache, analytics and server."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

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

    def to_dict(self) -> dict[str, object]:
        return {
            "activity_id": self.activity_id,
            "start_time": self.start_time.isoformat(),
            "local_start_time": self.local_start_time,
            "sport": self.sport,
            "sub_sport": self.sub_sport,
            "duration_s": self.duration_s,
            "moving_time_s": self.moving_time_s,
            "distance_m": self.distance_m,
            "elevation_gain_m": self.elevation_gain_m,
            "avg_hr": self.avg_hr,
            "max_hr": self.max_hr,
            "calories": self.calories,
            "load_value": self.load_value,
        }


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
    hr_10min_averages: tuple[dict[str, object], ...]
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

    def to_dict(self) -> dict[str, object]:
        return {
            "activity_id": self.activity_id,
            "start_time": self.start_time.isoformat(),
            "sport": self.sport,
            "sub_sport": self.sub_sport,
            "duration_s": self.duration_s,
            "moving_time_s": self.moving_time_s,
            "distance_m": self.distance_m,
            "elevation_gain_m": self.elevation_gain_m,
            "avg_hr": self.avg_hr,
            "max_hr": self.max_hr,
            "avg_speed_mps": self.avg_speed_mps,
            "max_speed_mps": self.max_speed_mps,
            "calories": self.calories,
            "hr_zones_s": {
                "zone_1": self.hr_zone_1_s,
                "zone_2": self.hr_zone_2_s,
                "zone_3": self.hr_zone_3_s,
                "zone_4": self.hr_zone_4_s,
                "zone_5": self.hr_zone_5_s,
            },
            "hr_coverage_ratio": self.hr_coverage_ratio,
            "hr_histogram_s": self.hr_histogram_s,
            "hr_10min_averages": [dict(bucket) for bucket in self.hr_10min_averages],
            "hr_drift_bpm_per_hour": self.hr_drift_bpm_per_hour,
            "decoupling": {
                "ef_first_half": self.ef_first_half,
                "ef_second_half": self.ef_second_half,
                "aerobic_decoupling_pct": self.aerobic_decoupling_pct,
                "confidence": self.decoupling_confidence,
            }
            if self.decoupling_confidence is not None
            else None,
            "load": {
                "value": self.load_value,
                "method": self.load_method,
                "confidence": self.load_confidence,
            },
            "data_completeness": self.data_completeness,
            "analysis_version": self.analysis_version,
            "local_start_time": self.local_start_time,
            "paused_time_s": self.paused_time_s,
            "data_hygiene_flags": list(self.repair_flags),
            "source": {"name": self.source_name, "path": self.source_path},
        }


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

    def to_dict(self) -> dict[str, object]:
        return {
            "activity_count": self.activity_count,
            "mean_hr_coverage_ratio": self.mean_hr_coverage_ratio,
            "mean_data_completeness": self.mean_data_completeness,
            "failed_file_count": self.failed_file_count,
            "repaired_file_count": self.repaired_file_count,
            "discarded_segment_count": self.discarded_segment_count,
            "duplicate_activity_count": self.duplicate_activity_count,
        }


@dataclass(frozen=True, slots=True)
class Overview:
    """Compact overview bundle for coaching conversations and scripts.

    Subsumes coach context, training form and metric trends: the daily
    series carries per-(date, sport) totals for every metric, and the form
    history carries the daily CTL/ATL/TSB curve.
    """

    athlete_profile: dict[str, object]
    recent_activities: tuple[dict[str, object], ...]
    weekly_summaries: tuple[dict[str, object], ...]
    daily_series: tuple[dict[str, object], ...]
    form_history: tuple[dict[str, object], ...]
    zone_distribution_s: dict[str, float]
    data_quality: dict[str, object]
    latest_activity: dict[str, object] | None
    period_start: str
    period_end: str
    form: dict[str, object] = field(default_factory=dict)
    tour_readiness: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "athlete_profile": self.athlete_profile,
            "period": {"start": self.period_start, "end": self.period_end},
            "recent_activities": list(self.recent_activities),
            "weekly_summaries": list(self.weekly_summaries),
            "daily_series": list(self.daily_series),
            "zone_distribution_s": self.zone_distribution_s,
            "form": self.form,
            "form_history": list(self.form_history),
            "tour_readiness": self.tour_readiness,
            "latest_activity": self.latest_activity,
            "data_quality": self.data_quality,
        }


@dataclass(frozen=True, slots=True)
class TrainingFormPoint:
    """Daily impulse-response training form point."""

    date: str
    ctl: float
    atl: float
    tsb: float
    daily_load: float

    def to_dict(self) -> dict[str, object]:
        return {
            "date": self.date,
            "ctl": self.ctl,
            "atl": self.atl,
            "tsb": self.tsb,
            "daily_load": self.daily_load,
        }


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

    def to_dict(self) -> dict[str, object]:
        return {
            "current_ctl": self.current_ctl,
            "current_atl": self.current_atl,
            "current_tsb": self.current_tsb,
            "form_classification": self.form_classification,
            "is_warmup": self.is_warmup,
            "history_days": self.history_days,
            "rolling_7d_distance_km": self.rolling_7d_distance_km,
            "rolling_28d_distance_km": self.rolling_28d_distance_km,
            "days_since_60k_ride": self.days_since_60k_ride,
            "trend_points": [p.to_dict() for p in self.trend_points],
            "load_confidence_ratio": self.load_confidence_ratio,
        }


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

    def to_dict(self) -> dict[str, object]:
        return {
            "files_discovered": self.files_discovered,
            "files_processed": self.files_processed,
            "files_skipped": self.files_skipped,
            "files_failed": self.files_failed,
            "cache_updated_at": self.cache_updated_at.isoformat()
            if self.cache_updated_at is not None
            else None,
            "source_errors": list(self.source_errors),
            "files_repaired": self.files_repaired,
            "segments_discarded": self.segments_discarded,
            "duplicates_detected": self.duplicates_detected,
        }
