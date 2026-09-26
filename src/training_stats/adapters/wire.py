"""Generic Pydantic wire contracts for metric reports."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from ..domain.metrics import (
    Confidence,
    MetricKind,
    MetricMethod,
    MetricReport,
    MissingReason,
    TimeBasis,
)
from ..domain.models import ActivitySummary


class ActivitySummaryWire(BaseModel):
    model_config = ConfigDict(extra="forbid")

    activity_id: str
    start_time: datetime
    sport: str
    sub_sport: str | None


class TripPeriodWire(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["trip"]
    activity_id: str
    start: datetime
    end: datetime | None


class DateRangePeriodWire(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["date_range"]
    start: date
    end: date
    timezone: str


class SnapshotPeriodWire(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["snapshot"]
    at: date | datetime


class AllTimePeriodWire(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["all_time"]


PeriodWire = Annotated[
    TripPeriodWire | DateRangePeriodWire | SnapshotPeriodWire | AllTimePeriodWire,
    Field(discriminator="kind"),
]


class AxisBinWire(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    lower: float
    upper: float | None


AxisValueWire = str | int | float | AxisBinWire


class AxisWire(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    values: list[AxisValueWire]
    unit: str | None


class ScalarValueWire(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shape: Literal["scalar"]
    value: int | float | str | bool


class TableValueWire(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shape: Literal["table"]
    axis: AxisWire
    values: list[int | float | str | bool | None]


class MatrixValueWire(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shape: Literal["matrix"]
    x_axis: AxisWire
    y_axis: AxisWire
    values: list[list[int | float | str | bool | None]]


MetricValueWire = Annotated[
    ScalarValueWire | TableValueWire | MatrixValueWire,
    Field(discriminator="shape"),
]


class MetricWire(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    period: PeriodWire
    value: MetricValueWire | None
    kind: MetricKind
    unit: str | None
    missing_reason: MissingReason | None
    method: MetricMethod | None
    algorithm_version: str | None
    confidence: Confidence | None
    formula: str | None
    coefficient_set: str | None
    time_basis: TimeBasis | None


class MetricReportWire(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metrics: list[MetricWire]


METRIC_REPORT_ADAPTER = TypeAdapter(MetricReportWire)
_DOMAIN_REPORT_ADAPTER = TypeAdapter(MetricReport)
ACTIVITY_SUMMARIES_ADAPTER = TypeAdapter(list[ActivitySummaryWire])
_DOMAIN_ACTIVITY_SUMMARIES_ADAPTER = TypeAdapter(list[ActivitySummary])


def metric_report(report: MetricReport) -> MetricReportWire:
    """Validate and map a domain report into its generic transport model."""
    payload = _DOMAIN_REPORT_ADAPTER.dump_python(report, mode="json")
    return METRIC_REPORT_ADAPTER.validate_python(payload)


def activity_summaries(
    summaries: Sequence[ActivitySummary],
) -> list[ActivitySummaryWire]:
    """Validate and map activity summaries into their transport models."""
    payload = _DOMAIN_ACTIVITY_SUMMARIES_ADAPTER.dump_python(list(summaries), mode="json")
    return ACTIVITY_SUMMARIES_ADAPTER.validate_python(payload)
