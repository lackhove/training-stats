"""Compact coach context assembly for LLM conversations."""

from __future__ import annotations

from datetime import date, datetime, timedelta, tzinfo
from typing import TYPE_CHECKING

from ..analytics.daily import DailySummaryRow
from ..config import AthleteConfig
from ..models import CoachContext
from .profile import athlete_profile

if TYPE_CHECKING:
    from ..cache.repository import TrainingRepository


def build_coach_context(
    repository: TrainingRepository,
    athlete: AthleteConfig,
    tz: tzinfo,
    days: int = 28,
    sport: str | None = None,
) -> CoachContext:
    """Compose a compact coaching context for the last ``days`` days."""
    end_date = datetime.now(tz).date()
    start_date = end_date - timedelta(days=max(days, 1) - 1)

    recent = repository.list_activities(
        start_date=start_date, end_date=end_date, sport=sport, limit=5
    )
    weekly = repository.weekly_summary(
        start_date=start_date, end_date=end_date, weeks=4, sport=sport
    )
    daily_rows = repository.daily_summary(start_date=start_date, end_date=end_date, sport=sport)
    zone_distribution = _zone_distribution(daily_rows)
    load_trend = repository.training_trends(metric="load", days=days, sport=sport)
    quality = repository.data_quality()
    latest = _latest_activity(repository, sport)
    form_result = repository.training_form(days=days, sport=sport)

    form_dict: dict[str, object] = {
        "ctl": form_result.current_ctl,
        "atl": form_result.current_atl,
        "tsb": form_result.current_tsb,
        "classification": form_result.form_classification,
        "is_warmup": form_result.is_warmup,
    }
    tour_readiness_dict: dict[str, object] = {
        "rolling_7d_km": form_result.rolling_7d_distance_km,
        "rolling_28d_km": form_result.rolling_28d_distance_km,
        "days_since_60k": form_result.days_since_60k_ride,
    }

    return CoachContext(
        athlete_profile=athlete_profile(athlete),
        recent_activities=tuple(item.to_dict() for item in recent),
        weekly_summaries=tuple(row.to_dict() for row in weekly),
        zone_distribution_s=zone_distribution,
        load_trend=load_trend.to_dict(),
        data_quality=quality.to_dict(),
        latest_activity=latest,
        period_start=start_date.isoformat(),
        period_end=end_date.isoformat(),
        form=form_dict,
        tour_readiness=tour_readiness_dict,
    )


def current_week_bounds(tz: tzinfo, today: date | None = None) -> tuple[date, date]:
    """Return Monday and Sunday of the current ISO week."""
    day = today if today is not None else datetime.now(tz).date()
    monday = day - timedelta(days=day.weekday())
    return monday, monday + timedelta(days=6)


def _zone_distribution(rows: list[DailySummaryRow]) -> dict[str, float]:
    zones = {f"zone_{i}": 0.0 for i in range(1, 6)}
    for row in rows:
        for key, value in row.zone_seconds().items():
            zones[key] += value
    return zones


def _latest_activity(repository: TrainingRepository, sport: str | None) -> dict[str, object] | None:
    latest = repository.list_activities(sport=sport, limit=1)
    if not latest:
        return None
    summary_dict = latest[0].to_dict()
    details = repository.get_activity(latest[0].activity_id)
    if details is not None:
        summary_dict["decoupling"] = details.to_dict().get("decoupling")
    return summary_dict
