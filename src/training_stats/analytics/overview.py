"""Overview assembly for coaching conversations and scripts.

Pure assembly: :func:`build_overview` composes pre-computed pieces — storage
reads happen in the repository, the analytics calls in the service layer.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from ..config import AthleteConfig
from ..models import (
    ActivityDetails,
    ActivitySummary,
    DataQuality,
    Decoupling,
    FormSnapshot,
    LatestActivity,
    Overview,
    TourReadiness,
    TrainingFormResult,
)
from .daily import DailySummaryRow
from .profile import athlete_profile
from .weekly import WeeklySummaryRow


def build_overview(
    *,
    athlete: AthleteConfig,
    period_start: date,
    period_end: date,
    sport: str | None,
    recent: Sequence[ActivitySummary],
    weekly: Sequence[WeeklySummaryRow],
    daily_rows: Sequence[DailySummaryRow],
    form_result: TrainingFormResult,
    quality: DataQuality,
    latest_summary: ActivitySummary | None,
    latest_details: ActivityDetails | None,
) -> Overview:
    """Compose the compact overview from pre-computed pieces.

    The daily series carries per-(date, sport) totals for every metric, so
    any metric trend is a filter and projection over that series; the form
    history carries the daily CTL/ATL/TSB curve.
    """
    return Overview(
        athlete_profile=athlete_profile(athlete),
        recent_activities=tuple(recent),
        weekly_summaries=tuple(weekly),
        daily_series=tuple(daily_rows),
        form_history=tuple(form_result.trend_points),
        zone_distribution_s=_zone_distribution(list(daily_rows)),
        data_quality=quality,
        latest_activity=latest_activity(latest_summary, latest_details),
        period_start=period_start,
        period_end=period_end,
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


def latest_activity(
    summary: ActivitySummary | None, details: ActivityDetails | None
) -> LatestActivity | None:
    """Wrap the most recent activity summary with its decoupling metrics."""
    if summary is None:
        return None
    decoupling: Decoupling | None = None
    if details is not None and details.decoupling_confidence is not None:
        decoupling = Decoupling(
            ef_first_half=details.ef_first_half,
            ef_second_half=details.ef_second_half,
            aerobic_decoupling_pct=details.aerobic_decoupling_pct,
            confidence=details.decoupling_confidence,
        )
    return LatestActivity(summary=summary, decoupling=decoupling)


def _zone_distribution(rows: list[DailySummaryRow]) -> dict[str, float]:
    zones = {f"zone_{i}": 0.0 for i in range(1, 6)}
    for row in rows:
        for key, value in row.zone_seconds().items():
            zones[key] += value
    return zones
