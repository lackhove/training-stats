"""Metric trend computation over daily summaries."""

from __future__ import annotations

from dataclasses import dataclass

from .daily import DailySummaryRow

ALLOWED_METRICS: tuple[str, ...] = ("duration", "distance", "elevation", "load", "activities")

_DIRECTION_THRESHOLD_PCT = 5.0


@dataclass(frozen=True, slots=True)
class TrendPoint:
    date: str
    value: float

    def to_dict(self) -> dict[str, object]:
        return {"date": self.date, "value": self.value}


@dataclass(frozen=True, slots=True)
class TrendResult:
    """Daily series of one metric plus summary statistics."""

    metric: str
    sport: str | None
    points: tuple[TrendPoint, ...]
    total: float
    mean: float
    change_pct: float | None
    direction: str

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "sport": self.sport,
            "points": [point.to_dict() for point in self.points],
            "total": self.total,
            "mean": self.mean,
            "change_pct": self.change_pct,
            "direction": self.direction,
        }


def validate_metric(metric: str) -> str:
    """Raise ``ValueError`` for unknown metric names."""
    normalized = metric.strip().lower()
    if normalized not in ALLOWED_METRICS:
        msg = f"unknown metric {metric!r}; allowed values: {', '.join(ALLOWED_METRICS)}"
        raise ValueError(msg)
    return normalized


def _row_value(row: DailySummaryRow, metric: str) -> float:
    if metric == "duration":
        return row.duration_s
    if metric == "distance":
        return row.distance_m
    if metric == "elevation":
        return row.elevation_gain_m
    if metric == "load":
        return row.load_value
    return float(row.activity_count)


def compute_trend(daily_rows: list[DailySummaryRow], metric: str, sport: str | None) -> TrendResult:
    """Build a daily trend for ``metric`` from daily rows.

    Rows are expected pre-filtered to the requested date window and sport.
    """
    validated = validate_metric(metric)
    points = tuple(
        TrendPoint(date=row.date, value=_row_value(row, validated)) for row in daily_rows
    )
    values = [point.value for point in points]
    total = sum(values)
    mean = total / len(values) if values else 0.0
    change_pct, direction = _direction(values)
    return TrendResult(
        metric=validated,
        sport=sport,
        points=points,
        total=total,
        mean=mean,
        change_pct=change_pct,
        direction=direction,
    )


def _direction(values: list[float]) -> tuple[float | None, str]:
    """Compare first-half vs second-half means to classify the direction."""
    if len(values) < 2:
        return None, "stable"
    half = len(values) // 2
    first_half = values[:half]
    second_half = values[half:]
    first_mean = sum(first_half) / len(first_half)
    second_mean = sum(second_half) / len(second_half)
    if first_mean <= 0:
        return None, "stable" if second_mean <= 0 else "increasing"
    change = (second_mean - first_mean) / first_mean * 100.0
    if change > _DIRECTION_THRESHOLD_PCT:
        return change, "increasing"
    if change < -_DIRECTION_THRESHOLD_PCT:
        return change, "decreasing"
    return change, "stable"
