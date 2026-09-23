"""Analytics for activity metrics, aggregations, training form and overview.

Pure functions over activity data and Polars frames — no I/O, no locking.
The read-side entry points for summaries and training form consume lazy
activity frames so each consumer decides its own column projection; row
selection helpers operate on the (few) aggregated dataclass rows.
"""

from .activities import ActivityMetrics, AnalyzedActivity, compute_activity_metrics
from .daily import (
    HARD_LOAD_THRESHOLD,
    DailySummaryRow,
    aggregate_daily,
    select_daily,
    summarize_daily,
)
from .endurance import DecouplingResult, compute_aerobic_decoupling
from .fitness import (
    classify_form,
    compute_days_since_60k,
    compute_form_curves,
    compute_rolling_distance,
    compute_training_form,
    training_form_from,
)
from .heart_rate import MAX_RECORD_INTERVAL_S, ZoneBounds, ZoneResult, time_in_zones
from .load import LOAD_METHOD, LOAD_VERSION, hr_trimp_load, load_confidence
from .overview import build_overview, latest_activity
from .profile import athlete_profile, zone_descriptions
from .weekly import WeeklySummaryRow, aggregate_weekly, daily_to_weekly, select_weekly

__all__ = [
    "HARD_LOAD_THRESHOLD",
    "LOAD_METHOD",
    "LOAD_VERSION",
    "MAX_RECORD_INTERVAL_S",
    "ActivityMetrics",
    "AnalyzedActivity",
    "DailySummaryRow",
    "DecouplingResult",
    "WeeklySummaryRow",
    "ZoneBounds",
    "ZoneResult",
    "aggregate_daily",
    "aggregate_weekly",
    "athlete_profile",
    "build_overview",
    "classify_form",
    "compute_activity_metrics",
    "compute_aerobic_decoupling",
    "compute_days_since_60k",
    "compute_form_curves",
    "compute_rolling_distance",
    "compute_training_form",
    "daily_to_weekly",
    "hr_trimp_load",
    "latest_activity",
    "load_confidence",
    "select_daily",
    "select_weekly",
    "summarize_daily",
    "time_in_zones",
    "training_form_from",
    "zone_descriptions",
]
