"""Training form impulse-response model (CTL, ATL, TSB) and tour readiness.

Implements the Banister/Coggan fitness-fatigue EWMA recursion over daily TRIMP
loads in local calendar days, plus rolling endurance volume and recency metrics.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

from ..models import TrainingFormPoint, TrainingFormResult

CTL_TIME_CONSTANT = 42
ATL_TIME_CONSTANT = 7

CTL_DECAY = 1.0 - math.exp(-1.0 / CTL_TIME_CONSTANT)
ATL_DECAY = 1.0 - math.exp(-1.0 / ATL_TIME_CONSTANT)

CLASSIFICATION_FRESH = "fresh"
CLASSIFICATION_OPTIMAL = "optimal"
CLASSIFICATION_HIGH_FATIGUE = "high_fatigue"
CLASSIFICATION_OVERTRAINING_RISK = "overtraining_risk"

MIN_LONG_RIDE_DISTANCE_M = 60_000.0


def classify_form(tsb: float) -> str:
    """Classify training stress balance (TSB) into form state.

    Bands:
    - TSB > +5.0: "fresh" (tapered, race-ready)
    - -10.0 <= TSB <= +5.0: "optimal" (neutral / productive maintenance)
    - -30.0 <= TSB < -10.0: "high_fatigue" (productive overload)
    - TSB < -30.0: "overtraining_risk" (excessive acute fatigue)
    """
    if tsb > 5.0:
        return CLASSIFICATION_FRESH
    if tsb >= -10.0:
        return CLASSIFICATION_OPTIMAL
    if tsb >= -30.0:
        return CLASSIFICATION_HIGH_FATIGUE
    return CLASSIFICATION_OVERTRAINING_RISK


def compute_form_curves(
    daily_loads: dict[date, float],
    first_date: date,
    end_date: date,
) -> list[TrainingFormPoint]:
    """Compute continuous daily CTL, ATL, and TSB from first_date to end_date.

    Calendar days without recorded activities contribute 0.0 TRIMP load.
    Initial conditions on the day before first_date are CTL=0.0, ATL=0.0.
    """
    if first_date > end_date:
        return []

    ctl = 0.0
    atl = 0.0
    points: list[TrainingFormPoint] = []
    current = first_date

    while current <= end_date:
        trimp = daily_loads.get(current, 0.0)
        ctl = ctl + (trimp - ctl) * CTL_DECAY
        atl = atl + (trimp - atl) * ATL_DECAY
        tsb = ctl - atl
        points.append(
            TrainingFormPoint(
                date=current.isoformat(),
                ctl=ctl,
                atl=atl,
                tsb=tsb,
                daily_load=trimp,
            )
        )
        current += timedelta(days=1)

    return points


def compute_rolling_distance(
    daily_distances: dict[date, float],
    today: date,
    window_days: int,
) -> float:
    """Sum distances in meters over the past window_days inclusive of today."""
    if window_days <= 0:
        return 0.0
    start = today - timedelta(days=window_days - 1)
    total_m = 0.0
    current = start
    while current <= today:
        total_m += daily_distances.get(current, 0.0)
        current += timedelta(days=1)
    return total_m


def compute_days_since_60k(
    activities: list[tuple[date, float]],
    today: date,
    min_distance_m: float = MIN_LONG_RIDE_DISTANCE_M,
) -> int | None:
    """Calendar days since the most recent activity with distance >= min_distance_m.

    Returns None if no qualifying activity exists in history on or before today.
    """
    qualifying_dates = [
        act_date for act_date, dist in activities if dist >= min_distance_m and act_date <= today
    ]
    if not qualifying_dates:
        return None
    latest_date = max(qualifying_dates)
    return (today - latest_date).days


def compute_training_form(
    daily_loads: dict[date, float],
    today: date,
    days: int = 42,
    rolling_7d_distance_km: float = 0.0,
    rolling_28d_distance_km: float = 0.0,
    days_since_60k_ride: int | None = None,
    load_confidence_ratio: float = 1.0,
) -> TrainingFormResult:
    """Assemble complete training form result with form curves and readiness."""
    if not daily_loads:
        return TrainingFormResult(
            current_ctl=0.0,
            current_atl=0.0,
            current_tsb=0.0,
            form_classification=CLASSIFICATION_OPTIMAL,
            is_warmup=True,
            history_days=0,
            rolling_7d_distance_km=rolling_7d_distance_km,
            rolling_28d_distance_km=rolling_28d_distance_km,
            days_since_60k_ride=days_since_60k_ride,
            trend_points=(),
            load_confidence_ratio=load_confidence_ratio,
        )

    first_date = min(min(daily_loads.keys()), today)
    points = compute_form_curves(daily_loads, first_date, today)

    current_point = points[-1]
    history_days = len(points)
    is_warmup = history_days < 42
    trend_slice = tuple(points[-days:]) if days > 0 else tuple(points)

    return TrainingFormResult(
        current_ctl=current_point.ctl,
        current_atl=current_point.atl,
        current_tsb=current_point.tsb,
        form_classification=classify_form(current_point.tsb),
        is_warmup=is_warmup,
        history_days=history_days,
        rolling_7d_distance_km=rolling_7d_distance_km,
        rolling_28d_distance_km=rolling_28d_distance_km,
        days_since_60k_ride=days_since_60k_ride,
        trend_points=trend_slice,
        load_confidence_ratio=load_confidence_ratio,
    )
