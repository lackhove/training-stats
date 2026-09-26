"""Daily aggregations in the configured local timezone.

:func:`summarize_daily` groups cached activity records by local date and
sport, reading only the named metrics it aggregates: metadata supplies the
identity facts and every total comes from a cached metric. Selection helpers
operate on the (few) resulting dataclass rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, tzinfo

from ..metrics import Metric, ScalarValue, TableValue
from ..models import ActivityRecord

# An activity counts as "hard" when its estimated load reaches this value.
HARD_LOAD_THRESHOLD = 150.0

_ZONE_COUNT = 5


@dataclass(frozen=True, slots=True)
class PeriodTotalsRow:
    """Aggregated totals shared by daily and weekly rows."""

    activity_count: int
    duration_s: float
    distance_m: float
    elevation_gain_m: float
    hr_zone_1_s: float
    hr_zone_2_s: float
    hr_zone_3_s: float
    hr_zone_4_s: float
    hr_zone_5_s: float
    load_value: float
    hard_activity_count: int
    longest_activity_s: float
    longest_activity_id: str | None

    def zone_seconds(self) -> dict[str, float]:
        return {
            "zone_1": self.hr_zone_1_s,
            "zone_2": self.hr_zone_2_s,
            "zone_3": self.hr_zone_3_s,
            "zone_4": self.hr_zone_4_s,
            "zone_5": self.hr_zone_5_s,
        }


@dataclass(frozen=True, slots=True)
class DailySummaryRow(PeriodTotalsRow):
    """Totals for one local date and sport."""

    date: str = ""
    sport: str = ""


class _Group:
    """Accumulates one (local date, sport) bucket in sorted input order."""

    __slots__ = (
        "_has_duration",
        "activity_count",
        "distance_m",
        "duration_s",
        "elevation_gain_m",
        "hard_activity_count",
        "load_value",
        "longest_activity_id",
        "longest_activity_s",
        "zones",
    )

    def __init__(self) -> None:
        self.activity_count = 0
        self.duration_s = 0.0
        self.distance_m = 0.0
        self.elevation_gain_m = 0.0
        self.zones = [0.0] * _ZONE_COUNT
        self.load_value = 0.0
        self.hard_activity_count = 0
        self.longest_activity_s = 0.0
        self.longest_activity_id: str | None = None
        self._has_duration = False

    def add(self, activity_id: str, metrics: dict[str, Metric]) -> None:
        self.activity_count += 1
        duration = metric_number(metrics.get("activity.duration"))
        self.duration_s += duration if duration is not None else 0.0
        distance = metric_number(metrics.get("activity.distance"))
        self.distance_m += distance if distance is not None else 0.0
        elevation = metric_number(metrics.get("activity.elevation_gain"))
        self.elevation_gain_m += elevation if elevation is not None else 0.0
        zones = metric_table_values(metrics.get("heart_rate.zone_time"))
        for index in range(_ZONE_COUNT):
            self.zones[index] += zone_cell(zones, index)
        load = metric_number(metrics.get("training_load.hr_trimp"))
        if load is None:
            load = 0.0
        self.load_value += load
        if load >= HARD_LOAD_THRESHOLD:
            self.hard_activity_count += 1
        self._add_longest(activity_id, duration if duration is not None else 0.0)

    def _add_longest(self, activity_id: str, duration: float) -> None:
        if self._has_duration and duration <= self.longest_activity_s:
            return
        self._has_duration = True
        self.longest_activity_s = duration
        self.longest_activity_id = activity_id if duration > 0.0 else None


def summarize_daily(records: Sequence[ActivityRecord], tz: tzinfo) -> list[DailySummaryRow]:
    """Group activities by local date and sport.

    Records are visited in ``(start_time, activity_id)`` order so that the
    longest-activity pick resolves ties to the earliest activity. Returns rows
    sorted by date, then sport, with deterministic values.
    """
    groups: dict[tuple[date, str], _Group] = {}
    for record in sorted(records, key=lambda item: (item.start_time, item.activity_id)):
        metrics = {metric.name: metric for metric in record.metrics}
        key = (record.start_time.astimezone(tz).date(), record.sport)
        group = groups.get(key)
        if group is None:
            group = _Group()
            groups[key] = group
        group.add(record.activity_id, metrics)
    rows = [
        DailySummaryRow(
            activity_count=group.activity_count,
            duration_s=group.duration_s,
            distance_m=group.distance_m,
            elevation_gain_m=group.elevation_gain_m,
            hr_zone_1_s=group.zones[0],
            hr_zone_2_s=group.zones[1],
            hr_zone_3_s=group.zones[2],
            hr_zone_4_s=group.zones[3],
            hr_zone_5_s=group.zones[4],
            load_value=group.load_value,
            hard_activity_count=group.hard_activity_count,
            longest_activity_s=group.longest_activity_s,
            longest_activity_id=group.longest_activity_id,
            date=day.isoformat(),
            sport=sport,
        )
        for (day, sport), group in groups.items()
    ]
    return sorted(rows, key=lambda row: (row.date, row.sport))


def metric_number(metric: Metric | None) -> float | None:
    """Read a cached numeric scalar, rejecting non-numerics and booleans."""
    if metric is None or not isinstance(metric.value, ScalarValue):
        return None
    value = metric.value.value
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def metric_table_values(metric: Metric | None) -> tuple[object, ...]:
    """Read the raw cell values of a cached table metric."""
    if metric is None or not isinstance(metric.value, TableValue):
        return ()
    return metric.value.values


def zone_cell(values: tuple[object, ...], index: int) -> float:
    """Read one zone cell, defaulting missing or invalid cells to 0.0."""
    if index >= len(values):
        return 0.0
    value = values[index]
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return float(value)


def select_daily(
    rows: list[DailySummaryRow],
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    sport: str | None = None,
) -> list[DailySummaryRow]:
    """Filter daily rows by date range and sport; merge sports when unfiltered."""
    filtered = [
        row
        for row in rows
        if _date_in_range(date.fromisoformat(row.date), start_date, end_date)
        and (sport is None or row.sport.lower() == sport.lower())
    ]
    if sport is None:
        filtered = aggregate_daily(filtered)
    return sorted(filtered, key=lambda row: (row.date, row.sport))


def aggregate_daily(rows: list[DailySummaryRow]) -> list[DailySummaryRow]:
    """Aggregate per-(date, sport) rows into per-date totals."""
    by_date: dict[str, list[DailySummaryRow]] = {}
    for row in rows:
        by_date.setdefault(row.date, []).append(row)
    return [
        DailySummaryRow(
            date=day,
            sport="all",
            activity_count=sum(row.activity_count for row in group),
            duration_s=sum(row.duration_s for row in group),
            distance_m=sum(row.distance_m for row in group),
            elevation_gain_m=sum(row.elevation_gain_m for row in group),
            hr_zone_1_s=sum(row.hr_zone_1_s for row in group),
            hr_zone_2_s=sum(row.hr_zone_2_s for row in group),
            hr_zone_3_s=sum(row.hr_zone_3_s for row in group),
            hr_zone_4_s=sum(row.hr_zone_4_s for row in group),
            hr_zone_5_s=sum(row.hr_zone_5_s for row in group),
            load_value=sum(row.load_value for row in group),
            hard_activity_count=sum(row.hard_activity_count for row in group),
            longest_activity_s=max(row.longest_activity_s for row in group),
            longest_activity_id=max(
                group, key=lambda row: row.longest_activity_s
            ).longest_activity_id,
        )
        for day, group in sorted(by_date.items())
    ]


def _date_in_range(value: date, start_date: date | None, end_date: date | None) -> bool:
    if start_date is not None and value < start_date:
        return False
    return not (end_date is not None and value > end_date)
