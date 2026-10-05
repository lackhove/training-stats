"""Transport-neutral domain types for analytical metrics and reports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Literal

type MetricScalar = int | float | str | bool
type AxisPoint = str | int | float


class MetricKind(StrEnum):
    MEASURED = "measured"
    CALCULATED = "calculated"
    ESTIMATED = "estimated"
    INTERPRETED = "interpreted"


class Confidence(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class MetricMethod(StrEnum):
    HR_TRIMP = "hr_trimp"


class TimeBasis(StrEnum):
    ELAPSED_TIME = "elapsed_time"
    MOVING_TIME = "moving_time"
    ACTIVE_TIME = "active_time"
    PEDALING_TIME = "pedaling_time"
    VALID_HEART_RATE_TIME = "valid_heart_rate_time"


class MissingReason(StrEnum):
    NO_DATA = "no_data"
    INSUFFICIENT_DATA = "insufficient_data"
    NO_POWER_DATA = "no_power_data"
    INSUFFICIENT_DURATION = "insufficient_duration"
    INSUFFICIENT_COVERAGE = "insufficient_coverage"
    INSUFFICIENT_WINDOWS = "insufficient_windows"
    INVALID_EF = "invalid_ef"
    NO_CADENCE_DATA = "no_cadence_data"
    NO_PEDALING_INTERVALS = "no_pedaling_intervals"


@dataclass(frozen=True, slots=True)
class TripPeriod:
    """Metric scope for one accepted activity."""

    activity_id: str
    start: datetime
    end: datetime | None = None
    kind: Literal["trip"] = "trip"


@dataclass(frozen=True, slots=True)
class DateRangePeriod:
    """Inclusive local-calendar date range."""

    start: date
    end: date
    timezone: str
    kind: Literal["date_range"] = "date_range"


@dataclass(frozen=True, slots=True)
class SnapshotPeriod:
    """Metric scope at one point in time."""

    at: date | datetime
    kind: Literal["snapshot"] = "snapshot"


@dataclass(frozen=True, slots=True)
class AllTimePeriod:
    """Metric scope over all available history."""

    kind: Literal["all_time"] = "all_time"


type Period = TripPeriod | DateRangePeriod | SnapshotPeriod | AllTimePeriod


@dataclass(frozen=True, slots=True)
class AxisBin:
    """A labeled half-open numeric interval on an axis."""

    label: str
    lower: float
    upper: float | None


type AxisValue = AxisPoint | AxisBin


@dataclass(frozen=True, slots=True)
class Axis:
    """Labeled values for one table or matrix axis."""

    name: str
    values: tuple[AxisValue, ...]
    unit: str | None = None


@dataclass(frozen=True, slots=True)
class ScalarValue:
    value: MetricScalar
    shape: Literal["scalar"] = "scalar"


@dataclass(frozen=True, slots=True)
class TableValue:
    axis: Axis
    values: tuple[MetricScalar | None, ...]
    shape: Literal["table"] = "table"

    def __post_init__(self) -> None:
        if len(self.axis.values) != len(self.values):
            msg = "table axis and value lengths must match"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class MatrixValue:
    x_axis: Axis
    y_axis: Axis
    values: tuple[tuple[MetricScalar | None, ...], ...]
    shape: Literal["matrix"] = "matrix"

    def __post_init__(self) -> None:
        if len(self.values) != len(self.y_axis.values):
            msg = "matrix row count must match y-axis length"
            raise ValueError(msg)
        if any(len(row) != len(self.x_axis.values) for row in self.values):
            msg = "matrix column count must match x-axis length"
            raise ValueError(msg)


type MetricValue = ScalarValue | TableValue | MatrixValue


@dataclass(frozen=True, slots=True)
class Metric:
    """One named metric over a period, with an explicit shape and provenance."""

    name: str
    description: str
    period: Period
    value: MetricValue | None
    kind: MetricKind
    unit: str | None = None
    missing_reason: MissingReason | None = None
    method: MetricMethod | None = None
    algorithm_version: str | None = None
    confidence: Confidence | None = None
    formula: str | None = None
    coefficient_set: str | None = None
    time_basis: TimeBasis | None = None

    def __post_init__(self) -> None:
        if not self.name:
            msg = "metric name must not be empty"
            raise ValueError(msg)
        if not self.description.strip():
            msg = "metric description must not be empty"
            raise ValueError(msg)
        if (self.value is None) == (self.missing_reason is None):
            msg = "metric must have either a value or a missing reason"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class MetricReport:
    """Ordered metric collection returned by a public analytics summary."""

    metrics: tuple[Metric, ...]
