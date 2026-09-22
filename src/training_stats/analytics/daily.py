"""Daily aggregations in the configured local timezone."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, tzinfo

from .activities import AnalyzedActivity

# An activity counts as "hard" when its estimated load reaches this value.
HARD_LOAD_THRESHOLD = 150.0


def local_date(start_time: datetime, tz: tzinfo) -> date:
    """Return the local calendar date of a UTC activity start time."""
    return start_time.astimezone(tz).date()


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


class _Accumulator:
    """Mutable internal accumulator for grouping."""

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

    def add(self, item: AnalyzedActivity, duration: float) -> None:
        load = item.metrics.load_value or 0.0
        self.activity_count += 1
        self.duration_s += duration
        self.distance_m += item.activity.distance_m or 0.0
        self.elevation_gain_m += item.activity.elevation_gain_m or 0.0
        self.zone_1 += item.metrics.hr_zone_1_s
        self.zone_2 += item.metrics.hr_zone_2_s
        self.zone_3 += item.metrics.hr_zone_3_s
        self.zone_4 += item.metrics.hr_zone_4_s
        self.zone_5 += item.metrics.hr_zone_5_s
        self.load_value += load
        self.hard_activity_count += int(load >= HARD_LOAD_THRESHOLD)
        if duration > self.longest_activity_s:
            self.longest_activity_s = duration
            self.longest_activity_id = item.activity.activity_id

    def to_row(self, date_key: str, sport: str) -> DailySummaryRow:
        return DailySummaryRow(
            date=date_key,
            sport=sport,
            activity_count=self.activity_count,
            duration_s=self.duration_s,
            distance_m=self.distance_m,
            elevation_gain_m=self.elevation_gain_m,
            hr_zone_1_s=self.zone_1,
            hr_zone_2_s=self.zone_2,
            hr_zone_3_s=self.zone_3,
            hr_zone_4_s=self.zone_4,
            hr_zone_5_s=self.zone_5,
            load_value=self.load_value,
            hard_activity_count=self.hard_activity_count,
            longest_activity_s=self.longest_activity_s,
            longest_activity_id=self.longest_activity_id,
        )


def summarize_daily(activities: list[AnalyzedActivity], tz: tzinfo) -> list[DailySummaryRow]:
    """Group analyzed activities by local date and sport.

    Returns rows sorted by date, then sport, with deterministic values.
    """
    buckets: dict[tuple[date, str], _Accumulator] = {}
    for item in activities:
        day = local_date(item.activity.start_time, tz)
        key = (day, item.activity.sport)
        accumulator = buckets.get(key)
        if accumulator is None:
            accumulator = _Accumulator()
            buckets[key] = accumulator
        accumulator.add(item, item.activity.duration_s or 0.0)
    return [
        accumulator.to_row(day.isoformat(), sport)
        for (day, sport), accumulator in sorted(buckets.items())
    ]
