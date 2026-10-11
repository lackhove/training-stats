"""Weekly aggregations based on ISO weeks in the configured local timezone."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta

from .daily import (
    DailySummaryRow,
    PeriodTotalsRow,
    _PeriodTotalsAccumulator,
    _totals_kwargs,
)


@dataclass(frozen=True, slots=True)
class WeeklySummaryRow(PeriodTotalsRow):
    """Totals for one ISO week and sport, including previous-week comparison."""

    iso_year: int
    iso_week: int
    week_start: date
    sport: str
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


def daily_to_weekly(daily_rows: list[DailySummaryRow]) -> list[WeeklySummaryRow]:
    """Aggregate daily rows into weekly rows, sorted by ISO week.

    Weekly rows are derived from daily rows so that regeneration is
    deterministic and incremental syncs stay simple. Previous-week
    comparisons resolve against the full input set — pass unfiltered rows.
    """
    buckets: dict[tuple[int, int, str], _PeriodTotalsAccumulator] = {}
    week_starts: dict[tuple[int, int], date] = {}
    for row in daily_rows:
        iso_year, iso_week = iso_week_key(row.date)
        week_starts[(iso_year, iso_week)] = week_start(row.date)
        key = (iso_year, iso_week, row.sport)
        aggregator = buckets.get(key)
        if aggregator is None:
            aggregator = _PeriodTotalsAccumulator()
            buckets[key] = aggregator
        aggregator.add_row(row)

    rows: list[WeeklySummaryRow] = []
    for (iso_year, iso_week, sport), aggregator in sorted(buckets.items()):
        previous = buckets.get((*_previous_bucket_key(iso_year, iso_week), sport))
        rows.append(
            WeeklySummaryRow(
                **_totals_kwargs(aggregator.to_row()),
                iso_year=iso_year,
                iso_week=iso_week,
                week_start=week_starts[(iso_year, iso_week)],
                sport=sport,
                load_value_prev=(
                    previous.load_value
                    if previous is not None and previous.metric_counts.load_values > 0
                    else None
                ),
                duration_s_prev=previous.duration_s if previous else None,
                distance_m_prev=(
                    previous.distance_m
                    if previous is not None and previous.metric_counts.distance_values > 0
                    else None
                ),
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
        and (start_date is None or row.week_start >= start_date)
        and (end_date is None or row.week_start <= end_date)
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
    result: list[WeeklySummaryRow] = []
    for (year, week), group in sorted(by_week.items()):
        totals = _PeriodTotalsAccumulator()
        for row in group:
            totals.add_row(row, first_tie_wins=True)
        result.append(
            WeeklySummaryRow(
                **_totals_kwargs(totals.to_row()),
                iso_year=year,
                iso_week=week,
                week_start=min(row.week_start for row in group),
                sport="all",
                load_value_prev=_sum_optional(row.load_value_prev for row in group),
                duration_s_prev=_sum_optional(row.duration_s_prev for row in group),
                distance_m_prev=_sum_optional(row.distance_m_prev for row in group),
            )
        )
    return result


def _sum_optional(values: Iterable[object]) -> float | None:
    present = [
        float(value)
        for value in values
        if isinstance(value, int | float) and not isinstance(value, bool)
    ]
    return sum(present) if present else None
