"""Wire (transport) models shared by the REST API and CLI ``--json`` output.

Services return domain models; serialization happens only at the edges.
``api.py`` maps domain → wire models here (the anti-corruption layer for the
HTTP boundary); ``cli.py`` dumps the same wire models with
``model_dump(mode="json")`` so REST responses and CLI JSON stay identical.
"""

from __future__ import annotations

from pydantic import BaseModel

from . import models
from .analytics.daily import DailySummaryRow
from .analytics.heart_rate import HrBucketAvg
from .analytics.load import LOAD_METHOD
from .analytics.weekly import WeeklySummaryRow


class HrZones(BaseModel):
    zone_1: float
    zone_2: float
    zone_3: float
    zone_4: float
    zone_5: float


class LoadInfo(BaseModel):
    value: float | None
    method: str | None
    confidence: str | None


class Decoupling(BaseModel):
    ef_first_half: float | None
    ef_second_half: float | None
    aerobic_decoupling_pct: float | None
    confidence: str | None


class Source(BaseModel):
    name: str
    path: str


class TenMinuteAverage(BaseModel):
    start_min: int
    avg_hr: float
    hr_time_s: float


class Comparison(BaseModel):
    load_value: float | None
    duration_s: float | None
    distance_m: float | None


class Period(BaseModel):
    start: str
    end: str


class ActivitySummary(BaseModel):
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


class ActivityDetails(BaseModel):
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
    hr_zones_s: HrZones
    hr_coverage_ratio: float
    hr_histogram_s: dict[str, float]
    hr_10min_averages: list[TenMinuteAverage]
    hr_drift_bpm_per_hour: float | None
    decoupling: Decoupling | None
    load: LoadInfo
    data_completeness: float
    analysis_version: str
    source: Source
    data_hygiene_flags: list[str]


class WeeklyRow(BaseModel):
    activity_count: int
    duration_s: float
    distance_m: float
    elevation_gain_m: float
    hr_zones_s: HrZones
    load_value: float
    load_method: str
    hard_activity_count: int
    longest_activity_s: float
    longest_activity_id: str | None
    iso_year: int
    iso_week: int
    week_start: str
    sport: str
    comparison_vs_prev_week: Comparison


class DailySeriesPoint(BaseModel):
    date: str
    sport: str
    activity_count: int
    duration_s: float
    distance_m: float
    elevation_gain_m: float
    hr_zones_s: HrZones
    load_value: float
    load_method: str
    hard_activity_count: int
    longest_activity_s: float
    longest_activity_id: str | None


class FormSnapshot(BaseModel):
    ctl: float
    atl: float
    tsb: float
    classification: str
    is_warmup: bool
    load_confidence_ratio: float


class FormHistoryPoint(BaseModel):
    date: str
    ctl: float
    atl: float
    tsb: float
    daily_load: float


class TourReadiness(BaseModel):
    rolling_7d_km: float
    rolling_28d_km: float
    days_since_60k: int | None


class DataQuality(BaseModel):
    activity_count: int
    mean_hr_coverage_ratio: float | None
    mean_data_completeness: float | None
    failed_file_count: int
    repaired_file_count: int
    discarded_segment_count: int
    duplicate_activity_count: int


class LatestActivity(BaseModel):
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
    decoupling: Decoupling | None = None


class Overview(BaseModel):
    athlete_profile: dict[str, object]
    period: Period
    recent_activities: list[ActivitySummary]
    weekly_summaries: list[WeeklyRow]
    daily_series: list[DailySeriesPoint]
    zone_distribution_s: dict[str, float]
    form: FormSnapshot
    form_history: list[FormHistoryPoint]
    tour_readiness: TourReadiness
    latest_activity: LatestActivity | None
    data_quality: DataQuality


# -- domain → wire mappers ----------------------------------------------------


def activity_summary(domain: models.ActivitySummary) -> ActivitySummary:
    return ActivitySummary(
        activity_id=domain.activity_id,
        start_time=domain.start_time.isoformat(),
        local_start_time=domain.local_start_time,
        sport=domain.sport,
        sub_sport=domain.sub_sport,
        duration_s=domain.duration_s,
        moving_time_s=domain.moving_time_s,
        distance_m=domain.distance_m,
        elevation_gain_m=domain.elevation_gain_m,
        avg_hr=domain.avg_hr,
        max_hr=domain.max_hr,
        calories=domain.calories,
        load_value=domain.load_value,
    )


def activity_details(domain: models.ActivityDetails) -> ActivityDetails:
    decoupling: Decoupling | None = None
    if domain.decoupling_confidence is not None:
        decoupling = Decoupling(
            ef_first_half=domain.ef_first_half,
            ef_second_half=domain.ef_second_half,
            aerobic_decoupling_pct=domain.aerobic_decoupling_pct,
            confidence=domain.decoupling_confidence,
        )
    return ActivityDetails(
        activity_id=domain.activity_id,
        start_time=domain.start_time.isoformat(),
        local_start_time=domain.local_start_time,
        sport=domain.sport,
        sub_sport=domain.sub_sport,
        duration_s=domain.duration_s,
        moving_time_s=domain.moving_time_s,
        paused_time_s=domain.paused_time_s,
        distance_m=domain.distance_m,
        elevation_gain_m=domain.elevation_gain_m,
        avg_hr=domain.avg_hr,
        max_hr=domain.max_hr,
        avg_speed_mps=domain.avg_speed_mps,
        max_speed_mps=domain.max_speed_mps,
        calories=domain.calories,
        hr_zones_s=_hr_zones_from_details(domain),
        hr_coverage_ratio=domain.hr_coverage_ratio,
        hr_histogram_s=domain.hr_histogram_s,
        hr_10min_averages=[_ten_minute_average(bucket) for bucket in domain.hr_10min_averages],
        hr_drift_bpm_per_hour=domain.hr_drift_bpm_per_hour,
        decoupling=decoupling,
        load=LoadInfo(
            value=domain.load_value,
            method=domain.load_method,
            confidence=domain.load_confidence,
        ),
        data_completeness=domain.data_completeness,
        analysis_version=domain.analysis_version,
        source=Source(name=domain.source_name, path=domain.source_path),
        data_hygiene_flags=list(domain.repair_flags),
    )


def weekly_row(domain: WeeklySummaryRow) -> WeeklyRow:
    return WeeklyRow(
        activity_count=domain.activity_count,
        duration_s=domain.duration_s,
        distance_m=domain.distance_m,
        elevation_gain_m=domain.elevation_gain_m,
        hr_zones_s=_hr_zones_from_totals(domain),
        load_value=domain.load_value,
        load_method=LOAD_METHOD,
        hard_activity_count=domain.hard_activity_count,
        longest_activity_s=domain.longest_activity_s,
        longest_activity_id=domain.longest_activity_id,
        iso_year=domain.iso_year,
        iso_week=domain.iso_week,
        week_start=domain.week_start,
        sport=domain.sport,
        comparison_vs_prev_week=Comparison(
            load_value=domain.load_value_prev,
            duration_s=domain.duration_s_prev,
            distance_m=domain.distance_m_prev,
        ),
    )


def daily_point(domain: DailySummaryRow) -> DailySeriesPoint:
    return DailySeriesPoint(
        date=domain.date,
        sport=domain.sport,
        activity_count=domain.activity_count,
        duration_s=domain.duration_s,
        distance_m=domain.distance_m,
        elevation_gain_m=domain.elevation_gain_m,
        hr_zones_s=_hr_zones_from_totals(domain),
        load_value=domain.load_value,
        load_method=LOAD_METHOD,
        hard_activity_count=domain.hard_activity_count,
        longest_activity_s=domain.longest_activity_s,
        longest_activity_id=domain.longest_activity_id,
    )


def overview(domain: models.Overview) -> Overview:
    return Overview(
        athlete_profile=domain.athlete_profile,
        period=Period(
            start=domain.period_start.isoformat(),
            end=domain.period_end.isoformat(),
        ),
        recent_activities=[activity_summary(item) for item in domain.recent_activities],
        weekly_summaries=[weekly_row(row) for row in domain.weekly_summaries],
        daily_series=[daily_point(row) for row in domain.daily_series],
        zone_distribution_s=domain.zone_distribution_s,
        form=FormSnapshot(
            ctl=domain.form.ctl,
            atl=domain.form.atl,
            tsb=domain.form.tsb,
            classification=domain.form.classification,
            is_warmup=domain.form.is_warmup,
            load_confidence_ratio=domain.form.load_confidence_ratio,
        ),
        form_history=[
            FormHistoryPoint(
                date=point.date,
                ctl=point.ctl,
                atl=point.atl,
                tsb=point.tsb,
                daily_load=point.daily_load,
            )
            for point in domain.form_history
        ],
        tour_readiness=TourReadiness(
            rolling_7d_km=domain.tour_readiness.rolling_7d_km,
            rolling_28d_km=domain.tour_readiness.rolling_28d_km,
            days_since_60k=domain.tour_readiness.days_since_60k,
        ),
        latest_activity=_latest_activity(domain.latest_activity),
        data_quality=data_quality(domain.data_quality),
    )


def data_quality(domain: models.DataQuality) -> DataQuality:
    return DataQuality(
        activity_count=domain.activity_count,
        mean_hr_coverage_ratio=domain.mean_hr_coverage_ratio,
        mean_data_completeness=domain.mean_data_completeness,
        failed_file_count=domain.failed_file_count,
        repaired_file_count=domain.repaired_file_count,
        discarded_segment_count=domain.discarded_segment_count,
        duplicate_activity_count=domain.duplicate_activity_count,
    )


def _latest_activity(domain: models.LatestActivity | None) -> LatestActivity | None:
    if domain is None:
        return None
    summary = domain.summary
    decoupling: Decoupling | None = None
    if domain.decoupling is not None:
        decoupling = Decoupling(
            ef_first_half=domain.decoupling.ef_first_half,
            ef_second_half=domain.decoupling.ef_second_half,
            aerobic_decoupling_pct=domain.decoupling.aerobic_decoupling_pct,
            confidence=domain.decoupling.confidence,
        )
    return LatestActivity(
        activity_id=summary.activity_id,
        start_time=summary.start_time.isoformat(),
        local_start_time=summary.local_start_time,
        sport=summary.sport,
        sub_sport=summary.sub_sport,
        duration_s=summary.duration_s,
        moving_time_s=summary.moving_time_s,
        distance_m=summary.distance_m,
        elevation_gain_m=summary.elevation_gain_m,
        avg_hr=summary.avg_hr,
        max_hr=summary.max_hr,
        calories=summary.calories,
        load_value=summary.load_value,
        decoupling=decoupling,
    )


def _hr_zones_from_details(domain: models.ActivityDetails) -> HrZones:
    return HrZones(
        zone_1=domain.hr_zone_1_s,
        zone_2=domain.hr_zone_2_s,
        zone_3=domain.hr_zone_3_s,
        zone_4=domain.hr_zone_4_s,
        zone_5=domain.hr_zone_5_s,
    )


def _hr_zones_from_totals(domain: DailySummaryRow | WeeklySummaryRow) -> HrZones:
    return HrZones(
        zone_1=domain.hr_zone_1_s,
        zone_2=domain.hr_zone_2_s,
        zone_3=domain.hr_zone_3_s,
        zone_4=domain.hr_zone_4_s,
        zone_5=domain.hr_zone_5_s,
    )


def _ten_minute_average(bucket: HrBucketAvg) -> TenMinuteAverage:
    return TenMinuteAverage(
        start_min=bucket.start_min,
        avg_hr=bucket.avg_hr,
        hr_time_s=bucket.hr_time_s,
    )
