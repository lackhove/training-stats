"""Daily aggregations in the configured local timezone.

:func:`summarize_daily` groups cached activity records by local date and
sport, reading only the named metrics it aggregates: metadata supplies the
identity facts and every total comes from a cached metric. Selection helpers
operate on the (few) resulting dataclass rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, tzinfo
from typing import TypedDict

from ..metrics import Metric, TableValue
from ..models import ActivityRecord
from .metric_helpers import metric_number

# An activity counts as "hard" when its estimated load reaches this value.
HARD_LOAD_THRESHOLD = 150.0

_ZONE_COUNT = 5


@dataclass(frozen=True, slots=True)
class PeriodMetricCounts:
    """Availability and numeric contribution counts for period totals."""

    duration_values: int = 0
    distance_available: int = 0
    distance_values: int = 0
    elevation_available: int = 0
    elevation_values: int = 0
    heart_rate_available: int = 0
    heart_rate_zone_cells: tuple[int, ...] = (0, 0, 0, 0, 0)
    load_values: int = 0

    def combine(self, other: PeriodMetricCounts) -> PeriodMetricCounts:
        return PeriodMetricCounts(
            duration_values=self.duration_values + other.duration_values,
            distance_available=self.distance_available + other.distance_available,
            distance_values=self.distance_values + other.distance_values,
            elevation_available=self.elevation_available + other.elevation_available,
            elevation_values=self.elevation_values + other.elevation_values,
            heart_rate_available=self.heart_rate_available + other.heart_rate_available,
            heart_rate_zone_cells=tuple(
                left + right
                for left, right in zip(
                    self.heart_rate_zone_cells, other.heart_rate_zone_cells, strict=True
                )
            ),
            load_values=self.load_values + other.load_values,
        )


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
    metric_counts: PeriodMetricCounts = field(default_factory=PeriodMetricCounts, kw_only=True)

    def zone_seconds(self) -> dict[str, float]:
        return {
            "zone_1": self.hr_zone_1_s,
            "zone_2": self.hr_zone_2_s,
            "zone_3": self.hr_zone_3_s,
            "zone_4": self.hr_zone_4_s,
            "zone_5": self.hr_zone_5_s,
        }


class _TotalsKwargs(TypedDict):
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
    metric_counts: PeriodMetricCounts


@dataclass(frozen=True, slots=True)
class DailySummaryRow(PeriodTotalsRow):
    """Totals for one local date and sport."""

    date: date
    sport: str


class _PeriodTotalsAccumulator:
    """Accumulate the common totals shared by daily and weekly rows."""

    __slots__ = (
        "_has_longest",
        "activity_count",
        "distance_m",
        "duration_s",
        "elevation_gain_m",
        "hard_activity_count",
        "load_value",
        "longest_activity_id",
        "longest_activity_s",
        "metric_counts",
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
        self._has_longest = False
        self.metric_counts = PeriodMetricCounts()

    def add_activity(
        self,
        activity_id: str,
        metrics: dict[str, Metric],
        present_quantities: tuple[str, ...],
    ) -> None:
        duration = metric_number(metrics.get("activity.duration"))
        distance = metric_number(metrics.get("activity.distance"))
        elevation = metric_number(metrics.get("activity.elevation_gain"))
        zone_metric = metrics.get("heart_rate.zone_time")
        has_zone_table = zone_metric is not None and isinstance(zone_metric.value, TableValue)
        raw_zones = metric_table_values(zone_metric) if has_zone_table else ()
        zones = tuple(zone_value(raw_zones, index) for index in range(_ZONE_COUNT))
        load = metric_number(metrics.get("training_load.hr_trimp"))
        metric_counts = PeriodMetricCounts(
            duration_values=int(duration is not None),
            distance_available=int("distance" in present_quantities),
            distance_values=int(distance is not None),
            elevation_available=int("elevation" in present_quantities),
            elevation_values=int(elevation is not None),
            heart_rate_available=int("heart_rate" in present_quantities),
            heart_rate_zone_cells=tuple(int(value is not None) for value in zones),
            load_values=int(load is not None),
        )
        self._add_totals(
            activity_count=1,
            duration_s=duration if duration is not None else 0.0,
            distance_m=distance if distance is not None else 0.0,
            elevation_gain_m=elevation if elevation is not None else 0.0,
            zones=zones,
            load_value=load if load is not None else 0.0,
            hard_activity_count=int(load is not None and load >= HARD_LOAD_THRESHOLD),
            longest_activity_s=duration if duration is not None else 0.0,
            longest_activity_id=(activity_id if duration is not None and duration > 0.0 else None),
            metric_counts=metric_counts,
            first_tie_wins=True,
        )

    def add_row(self, row: PeriodTotalsRow, *, first_tie_wins: bool = False) -> None:
        self._add_totals(
            activity_count=row.activity_count,
            duration_s=row.duration_s,
            distance_m=row.distance_m,
            elevation_gain_m=row.elevation_gain_m,
            zones=tuple(
                row.zone_seconds()[f"zone_{index + 1}"]
                if row.metric_counts.heart_rate_zone_cells[index]
                else None
                for index in range(_ZONE_COUNT)
            ),
            load_value=row.load_value,
            hard_activity_count=row.hard_activity_count,
            longest_activity_s=row.longest_activity_s,
            longest_activity_id=row.longest_activity_id,
            metric_counts=row.metric_counts,
            first_tie_wins=first_tie_wins,
        )

    def _add_totals(
        self,
        *,
        activity_count: int,
        duration_s: float,
        distance_m: float,
        elevation_gain_m: float,
        zones: tuple[float | None, ...],
        load_value: float,
        hard_activity_count: int,
        longest_activity_s: float,
        longest_activity_id: str | None,
        metric_counts: PeriodMetricCounts,
        first_tie_wins: bool,
    ) -> None:
        self.activity_count += activity_count
        self.duration_s += duration_s
        self.distance_m += distance_m
        self.elevation_gain_m += elevation_gain_m
        for index in range(_ZONE_COUNT):
            zone = zones[index]
            if zone is not None:
                self.zones[index] += zone
        self.load_value += load_value
        self.metric_counts = self.metric_counts.combine(metric_counts)
        self.hard_activity_count += hard_activity_count
        if longest_activity_s > self.longest_activity_s or (
            first_tie_wins and not self._has_longest
        ):
            self.longest_activity_s = longest_activity_s
            self.longest_activity_id = longest_activity_id
        self._has_longest = True

    def to_row(self) -> PeriodTotalsRow:
        return PeriodTotalsRow(
            activity_count=self.activity_count,
            duration_s=self.duration_s,
            distance_m=self.distance_m,
            elevation_gain_m=self.elevation_gain_m,
            hr_zone_1_s=self.zones[0],
            hr_zone_2_s=self.zones[1],
            hr_zone_3_s=self.zones[2],
            hr_zone_4_s=self.zones[3],
            hr_zone_5_s=self.zones[4],
            load_value=self.load_value,
            hard_activity_count=self.hard_activity_count,
            longest_activity_s=self.longest_activity_s,
            longest_activity_id=self.longest_activity_id,
            metric_counts=self.metric_counts,
        )


def summarize_daily(records: Sequence[ActivityRecord], tz: tzinfo) -> list[DailySummaryRow]:
    """Group activities by local date and sport.

    Records are visited in ``(start_time, activity_id)`` order so that the
    longest-activity pick resolves ties to the earliest activity. Returns rows
    sorted by date, then sport, with deterministic values.
    """
    groups: dict[tuple[date, str], _PeriodTotalsAccumulator] = {}
    for record in sorted(records, key=lambda item: (item.start_time, item.activity_id)):
        metrics = {metric.name: metric for metric in record.metrics}
        key = (record.start_time.astimezone(tz).date(), record.sport)
        group = groups.get(key)
        if group is None:
            group = _PeriodTotalsAccumulator()
            groups[key] = group
        group.add_activity(record.activity_id, metrics, record.present_quantities)
    rows = [
        DailySummaryRow(
            **_totals_kwargs(group.to_row()),
            date=day,
            sport=sport,
        )
        for (day, sport), group in groups.items()
    ]
    return sorted(rows, key=lambda row: (row.date, row.sport))


def metric_table_values(metric: Metric | None) -> tuple[object, ...]:
    """Read the raw cell values of a cached table metric."""
    if metric is None or not isinstance(metric.value, TableValue):
        return ()
    return metric.value.values


def zone_value(values: tuple[object, ...], index: int) -> float | None:
    """Read one numeric zone cell, returning ``None`` when it did not contribute."""
    if index >= len(values):
        return None
    value = values[index]
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
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
        if _date_in_range(row.date, start_date, end_date)
        and (sport is None or row.sport.lower() == sport.lower())
    ]
    if sport is None:
        filtered = aggregate_daily(filtered)
    return sorted(filtered, key=lambda row: (row.date, row.sport))


def aggregate_daily(rows: list[DailySummaryRow]) -> list[DailySummaryRow]:
    """Aggregate per-(date, sport) rows into per-date totals."""
    by_date: dict[date, list[DailySummaryRow]] = {}
    for row in rows:
        by_date.setdefault(row.date, []).append(row)
    result: list[DailySummaryRow] = []
    for day, group in sorted(by_date.items()):
        totals = _PeriodTotalsAccumulator()
        for row in group:
            totals.add_row(row, first_tie_wins=True)
        result.append(DailySummaryRow(**_totals_kwargs(totals.to_row()), date=day, sport="all"))
    return result


def _totals_kwargs(row: PeriodTotalsRow) -> _TotalsKwargs:
    return {
        "activity_count": row.activity_count,
        "duration_s": row.duration_s,
        "distance_m": row.distance_m,
        "elevation_gain_m": row.elevation_gain_m,
        "hr_zone_1_s": row.hr_zone_1_s,
        "hr_zone_2_s": row.hr_zone_2_s,
        "hr_zone_3_s": row.hr_zone_3_s,
        "hr_zone_4_s": row.hr_zone_4_s,
        "hr_zone_5_s": row.hr_zone_5_s,
        "load_value": row.load_value,
        "hard_activity_count": row.hard_activity_count,
        "longest_activity_s": row.longest_activity_s,
        "longest_activity_id": row.longest_activity_id,
        "metric_counts": row.metric_counts,
    }


def _date_in_range(value: date, start_date: date | None, end_date: date | None) -> bool:
    if start_date is not None and value < start_date:
        return False
    return not (end_date is not None and value > end_date)
