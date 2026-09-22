"""Compact overview assembly for coaching conversations and scripts."""

from __future__ import annotations

from datetime import datetime, timedelta, tzinfo
from typing import TYPE_CHECKING

from ..analytics.daily import DailySummaryRow
from ..config import AthleteConfig
from ..models import Decoupling, FormSnapshot, LatestActivity, Overview, TourReadiness
from .profile import athlete_profile

if TYPE_CHECKING:
    from ..cache.repository import TrainingRepository


def build_overview(
    repository: TrainingRepository,
    athlete: AthleteConfig,
    tz: tzinfo,
    days: int = 28,
    sport: str | None = None,
) -> Overview:
    """Compose a compact training overview for the last ``days`` days.

    The daily series carries per-(date, sport) totals for every metric, so
    any metric trend is a filter and projection over that series; the form
    history carries the daily CTL/ATL/TSB curve.
    """
    end_date = datetime.now(tz).date()
    start_date = end_date - timedelta(days=max(days, 1) - 1)

    recent = repository.list_activities(
        start_date=start_date, end_date=end_date, sport=sport, limit=5
    )
    weekly = repository.weekly_summary(weeks=4, sport=sport)
    daily_rows = repository.daily_summary(start_date=start_date, end_date=end_date, sport=sport)
    quality = repository.data_quality()
    latest = _latest_activity(repository, sport)
    form_result = repository.training_form(days=days, sport=sport)

    return Overview(
        athlete_profile=athlete_profile(athlete),
        recent_activities=tuple(recent),
        weekly_summaries=tuple(weekly),
        daily_series=tuple(daily_rows),
        form_history=tuple(form_result.trend_points),
        zone_distribution_s=_zone_distribution(daily_rows),
        data_quality=quality,
        latest_activity=latest,
        period_start=start_date,
        period_end=end_date,
        form=FormSnapshot(
            ctl=form_result.current_ctl,
            atl=form_result.current_atl,
            tsb=form_result.current_tsb,
            classification=form_result.form_classification,
            is_warmup=form_result.is_warmup,
            load_confidence_ratio=form_result.load_confidence_ratio,
        ),
        tour_readiness=TourReadiness(
            rolling_7d_km=form_result.rolling_7d_distance_km,
            rolling_28d_km=form_result.rolling_28d_distance_km,
            days_since_60k=form_result.days_since_60k_ride,
        ),
    )


def _zone_distribution(rows: list[DailySummaryRow]) -> dict[str, float]:
    zones = {f"zone_{i}": 0.0 for i in range(1, 6)}
    for row in rows:
        for key, value in row.zone_seconds().items():
            zones[key] += value
    return zones


def _latest_activity(repository: TrainingRepository, sport: str | None) -> LatestActivity | None:
    latest = repository.list_activities(sport=sport, limit=1)
    if not latest:
        return None
    summary = latest[0]
    details = repository.get_activity(summary.activity_id)
    decoupling: Decoupling | None = None
    if details is not None and details.decoupling_confidence is not None:
        decoupling = Decoupling(
            ef_first_half=details.ef_first_half,
            ef_second_half=details.ef_second_half,
            aerobic_decoupling_pct=details.aerobic_decoupling_pct,
            confidence=details.decoupling_confidence,
        )
    return LatestActivity(summary=summary, decoupling=decoupling)
