"""Typed dictionary contracts for service payloads returned by the API/CLI.

These mirror the domain ``to_dict()`` shapes exactly and are validated with
:class:`pydantic.TypeAdapter` at the API boundary, so the OpenAPI schema and
runtime validation stay in one place without hand-maintained Pydantic model
mirrors.
"""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class HrZonesDict(TypedDict):
    zone_1: float
    zone_2: float
    zone_3: float
    zone_4: float
    zone_5: float


class LoadDict(TypedDict):
    value: float | None
    method: str | None
    confidence: str | None


class DecouplingDict(TypedDict):
    ef_first_half: float | None
    ef_second_half: float | None
    aerobic_decoupling_pct: float | None
    confidence: str | None


class SourceDict(TypedDict):
    name: str
    path: str


class TenMinuteAverageDict(TypedDict):
    start_min: int
    avg_hr: float
    hr_time_s: float


class ComparisonDict(TypedDict):
    load_value: float | None
    duration_s: float | None
    distance_m: float | None


class TotalsDict(TypedDict):
    activity_count: int
    duration_s: float
    distance_m: float
    elevation_gain_m: float
    hr_zones_s: HrZonesDict
    load_value: float
    load_method: str
    hard_activity_count: int
    longest_activity_s: float
    longest_activity_id: str | None


class ActivitySummaryDict(TypedDict):
    activity_id: str
    start_time: str
    local_start_time: str
    sport: str
    sub_sport: str | None
    duration_s: float | None
    moving_time_s: float | None
    distance_m: float | None
    elevation_gain_m: float | None
    avg_hr: float | None
    max_hr: float | None
    calories: float | None
    load_value: float | None


class ActivityDetailsDict(TypedDict):
    activity_id: str
    start_time: str
    local_start_time: str
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
    hr_zones_s: HrZonesDict
    hr_coverage_ratio: float
    hr_histogram_s: dict[str, float]
    hr_10min_averages: list[TenMinuteAverageDict]
    hr_drift_bpm_per_hour: float | None
    decoupling: DecouplingDict | None
    load: LoadDict
    data_completeness: float
    analysis_version: str
    source: SourceDict
    data_hygiene_flags: list[str]


class WeeklyRowDict(TotalsDict):
    iso_year: int
    iso_week: int
    week_start: str
    sport: str
    comparison_vs_prev_week: ComparisonDict


class PeriodDict(TypedDict):
    start: str
    end: str


class LatestActivityDict(ActivitySummaryDict):
    decoupling: NotRequired[DecouplingDict | None]


class OverviewDict(TypedDict):
    athlete_profile: dict[str, Any]
    period: PeriodDict
    recent_activities: list[ActivitySummaryDict]
    weekly_summaries: list[WeeklyRowDict]
    daily_series: list[dict[str, Any]]
    zone_distribution_s: dict[str, float]
    form: dict[str, Any]
    form_history: list[dict[str, Any]]
    tour_readiness: dict[str, Any]
    latest_activity: LatestActivityDict | None
    data_quality: dict[str, Any]


class SyncFeedbackDict(TypedDict, total=False):
    throttled: bool
    seconds_since_last_sync: float
    files_discovered: int
    files_processed: int
    files_skipped: int
    files_failed: int
    files_repaired: int
    segments_discarded: int
    duplicates_detected: int
    cache_updated_at: str | None
    source_errors: list[str]
