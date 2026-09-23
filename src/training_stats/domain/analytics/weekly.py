"""Weekly aggregations based on ISO weeks in the configured local timezone."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta

from .daily import DailySummaryRow, PeriodTotalsRow


@dataclass(frozen=True, slots=True)
class WeeklySummaryRow(PeriodTotalsRow):
    """Totals for one ISO week and sport, including previous-week comparison."""

    iso_year: int = 0
    iso_week: int = 0
    week_start: str = ""
    sport: str = ""
    load_value_prev: float | None = None
    duration_s_prev: float | None = None
    distance_m_prev: float | None = None


def iso_week_key(day: date) -> tuple[int, int]:
    """Return ``(iso_year, iso_week)`` for a date."""
    iso = day.isocalendar()
    return iso.year, iso.week


def week_start(day: date) -> date:
    """Return the Monday starting the ISO week of ``day``."""
    return day - timedelta(days=day.weekday())


def _previous_bucket_key(iso_year: int, iso_week: int) -> tuple[int, int]:
    if iso_week > 1:
        return iso_year, iso_week - 1
    previous_year = iso_year - 1
    for candidate_week in (53, 52):
        try:
            date.fromisocalendar(previous_year, candidate_week, 1)
        except ValueError:
            continue
        return previous_year, candidate_week
    return previous_year, 52


class _Aggregator:
    """Accumulates daily rows into weekly totals."""

    __slots__ = (
        "activity_count",
        "distance_m",
        "duration_s",
        "elevation_gain_m",
        "hard_activity_count",
        "load_value",
        "longest_activity_id",
        "longest_activity_s",
        "zone_1",
        "zone_2",
        "zone_3",
        "zone_4",
        "zone_5",
    )

    def __init__(self) -> None:
        self.activity_count = 0
        self.duration_s = 0.0
        self.distance_m = 0.0
        self.elevation_gain_m = 0.0
        self.zone_1 = 0.0
        self.zone_2 = 0.0
        self.zone_3 = 0.0
        self.zone_4 = 0.0
        self.zone_5 = 0.0
        self.load_value = 0.0
        self.hard_activity_count = 0
        self.longest_activity_s = 0.0
        self.longest_activity_id: str | None = None

    def add_daily(self, row: DailySummaryRow) -> None:
        self.activity_count += row.activity_count
        self.duration_s += row.duration_s
        self.distance_m += row.distance_m
        self.elevation_gain_m += row.elevation_gain_m
        self.zone_1 += row.hr_zone_1_s
        self.zone_2 += row.hr_zone_2_s
        self.zone_3 += row.hr_zone_3_s
        self.zone_4 += row.hr_zone_4_s
        self.zone_5 += row.hr_zone_5_s
        self.load_value += row.load_value
        self.hard_activity_count += row.hard_activity_count
        if row.longest_activity_s > self.longest_activity_s:
            self.longest_activity_s = row.longest_activity_s
            self.longest_activity_id = row.longest_activity_id


def daily_to_weekly(daily_rows: list[DailySummaryRow]) -> list[WeeklySummaryRow]:
    """Aggregate daily rows into weekly rows, sorted by ISO week.

    Weekly rows are derived from daily rows so that regeneration is
    deterministic and incremental syncs stay simple. Previous-week
    comparisons resolve against the full input set — pass unfiltered rows.
    """
    buckets: dict[tuple[int, int, str], _Aggregator] = {}
    week_starts: dict[tuple[int, int], date] = {}
    for row in daily_rows:
        day = date.fromisoformat(row.date)
        iso_year, iso_week = iso_week_key(day)
        week_starts[(iso_year, iso_week)] = week_start(day)
        key = (iso_year, iso_week, row.sport)
        aggregator = buckets.get(key)
        if aggregator is None:
            aggregator = _Aggregator()
            buckets[key] = aggregator
        aggregator.add_daily(row)

    rows: list[WeeklySummaryRow] = []
    for (iso_year, iso_week, sport), aggregator in sorted(buckets.items()):
        previous = buckets.get((*_previous_bucket_key(iso_year, iso_week), sport))
        rows.append(
            WeeklySummaryRow(
                iso_year=iso_year,
                iso_week=iso_week,
                week_start=week_starts[(iso_year, iso_week)].isoformat(),
                sport=sport,
                activity_count=aggregator.activity_count,
                duration_s=aggregator.duration_s,
                distance_m=aggregator.distance_m,
                elevation_gain_m=aggregator.elevation_gain_m,
                hr_zone_1_s=aggregator.zone_1,
                hr_zone_2_s=aggregator.zone_2,
                hr_zone_3_s=aggregator.zone_3,
                hr_zone_4_s=aggregator.zone_4,
                hr_zone_5_s=aggregator.zone_5,
                load_value=aggregator.load_value,
                hard_activity_count=aggregator.hard_activity_count,
                longest_activity_s=aggregator.longest_activity_s,
                longest_activity_id=aggregator.longest_activity_id,
                load_value_prev=previous.load_value if previous else None,
                duration_s_prev=previous.duration_s if previous else None,
                distance_m_prev=previous.distance_m if previous else None,
            )
        )
    return rows


def select_weekly(
    rows: list[WeeklySummaryRow],
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    weeks: int = 8,
    sport: str | None = None,
) -> list[WeeklySummaryRow]:
    """Filter weekly rows, merge sports when unfiltered, newest first.

    The ``weeks`` truncation only applies when no explicit date range is
    given (matching the API semantics).
    """
    filtered = [
        row
        for row in rows
        if (sport is None or row.sport.lower() == sport.lower())
        and (start_date is None or date.fromisoformat(row.week_start) >= start_date)
        and (end_date is None or date.fromisoformat(row.week_start) <= end_date)
    ]
    if sport is None:
        filtered = aggregate_weekly(filtered)
    filtered.sort(key=lambda row: (row.iso_year, row.iso_week), reverse=True)
    if start_date is None and end_date is None:
        filtered = filtered[:weeks]
    return filtered


def aggregate_weekly(rows: list[WeeklySummaryRow]) -> list[WeeklySummaryRow]:
    """Aggregate per-(week, sport) rows into per-week totals."""
    by_week: dict[tuple[int, int], list[WeeklySummaryRow]] = {}
    for row in rows:
        by_week.setdefault((row.iso_year, row.iso_week), []).append(row)
    return [
        WeeklySummaryRow(
            iso_year=year,
            iso_week=week,
            week_start=min(row.week_start for row in group),
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
            load_value_prev=_sum_optional(row.load_value_prev for row in group),
            duration_s_prev=_sum_optional(row.duration_s_prev for row in group),
            distance_m_prev=_sum_optional(row.distance_m_prev for row in group),
        )
        for (year, week), group in sorted(by_week.items())
    ]


def _sum_optional(values: Iterable[object]) -> float | None:
    present = [
        float(value)
        for value in values
        if isinstance(value, int | float) and not isinstance(value, bool)
    ]
    return sum(present) if present else None
