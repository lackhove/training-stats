"""Weekly aggregations based on ISO weeks in the configured local timezone."""

from __future__ import annotations

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

    def to_dict(self) -> dict[str, object]:
        base = {
            "iso_year": self.iso_year,
            "iso_week": self.iso_week,
            "week_start": self.week_start,
            "sport": self.sport,
        }
        return {
            **base,
            **self.totals_dict(),
            "comparison_vs_prev_week": {
                "load_value": self.load_value_prev,
                "duration_s": self.duration_s_prev,
                "distance_m": self.distance_m_prev,
            },
        }


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
    """Aggregate stored daily rows into weekly rows, sorted by ISO week.

    Weekly rows are derived from daily rows so that cache regeneration is
    deterministic and incremental syncs stay simple.
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
