"""Daily aggregations in the configured local timezone.

Aggregation is pushed down to Polars: :func:`summarize_daily` consumes a
lazy activity frame, projects only the columns it needs, and groups by
local date and sport. Selection helpers operate on the (few) resulting
dataclass rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, tzinfo

import polars as pl

# An activity counts as "hard" when its estimated load reaches this value.
HARD_LOAD_THRESHOLD = 150.0


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


def summarize_daily(activities: pl.LazyFrame, tz: tzinfo) -> list[DailySummaryRow]:
    """Group activities by local date and sport.

    The frame is sorted by ``(start_time, activity_id)`` first so that the
    longest-activity pick resolves ties to the earliest activity. Returns
    rows sorted by date, then sport, with deterministic values.
    """
    frame = (
        activities.sort("start_time", "activity_id")
        .select(
            pl.col("start_time").dt.convert_time_zone(str(tz)).dt.date().alias("date"),
            "sport",
            "activity_id",
            pl.col("duration_s").fill_null(0.0),
            pl.col("distance_m").fill_null(0.0),
            pl.col("elevation_gain_m").fill_null(0.0),
            "hr_zone_1_s",
            "hr_zone_2_s",
            "hr_zone_3_s",
            "hr_zone_4_s",
            "hr_zone_5_s",
            pl.col("load_value").fill_null(0.0),
        )
        .group_by("date", "sport")
        .agg(
            pl.len().alias("activity_count"),
            pl.col("duration_s").sum(),
            pl.col("distance_m").sum(),
            pl.col("elevation_gain_m").sum(),
            pl.col("hr_zone_1_s").sum(),
            pl.col("hr_zone_2_s").sum(),
            pl.col("hr_zone_3_s").sum(),
            pl.col("hr_zone_4_s").sum(),
            pl.col("hr_zone_5_s").sum(),
            pl.col("load_value").sum(),
            (pl.col("load_value") >= HARD_LOAD_THRESHOLD).sum().alias("hard_activity_count"),
            pl.col("duration_s").max().alias("longest_activity_s"),
            pl.col("activity_id")
            .filter(
                (pl.col("duration_s") == pl.col("duration_s").max()) & (pl.col("duration_s") > 0.0)
            )
            .first()
            .alias("longest_activity_id"),
        )
        .sort("date", "sport")
        .collect()
    )
    return [_daily_row(row) for row in frame.to_dicts()]


def _daily_row(row: dict[str, object]) -> DailySummaryRow:
    day = row["date"]
    return DailySummaryRow(
        date=day.isoformat() if isinstance(day, date) else str(day),
        sport=str(row["sport"]),
        activity_count=_int(row["activity_count"]),
        duration_s=_float(row["duration_s"]),
        distance_m=_float(row["distance_m"]),
        elevation_gain_m=_float(row["elevation_gain_m"]),
        hr_zone_1_s=_float(row["hr_zone_1_s"]),
        hr_zone_2_s=_float(row["hr_zone_2_s"]),
        hr_zone_3_s=_float(row["hr_zone_3_s"]),
        hr_zone_4_s=_float(row["hr_zone_4_s"]),
        hr_zone_5_s=_float(row["hr_zone_5_s"]),
        load_value=_float(row["load_value"]),
        hard_activity_count=_int(row["hard_activity_count"]),
        longest_activity_s=_float(row["longest_activity_s"]),
        longest_activity_id=_optional_str(row["longest_activity_id"]),
    )


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


def _optional_str(value: object) -> str | None:
    return str(value) if isinstance(value, str) else None


def _int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return int(value)


def _float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return float(value)
