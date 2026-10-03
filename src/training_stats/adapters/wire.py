"""Pydantic wire contracts for activity records and metric reports."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from ..domain.metrics import (
    Confidence,
    DateRangePeriod,
    MetricKind,
    MetricMethod,
    MetricReport,
    MissingReason,
    TimeBasis,
)
from ..domain.models import ActivityRecord, ActivitySummary


class ActivitySummaryWire(BaseModel):
    model_config = ConfigDict(extra="forbid")

    activity_id: str
    start_time: datetime
    sport: str
    sub_sport: str | None = None


class TripPeriodWire(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["trip"]
    activity_id: str
    start: datetime
    end: datetime | None = None


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
    upper: float | None = None


AxisValueWire = str | int | float | AxisBinWire


class AxisWire(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    values: list[AxisValueWire]
    unit: str | None = None


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


class _MetricTrailWire(BaseModel):
    """Value and provenance fields shared by both metric payloads."""

    model_config = ConfigDict(extra="forbid")

    value: MetricValueWire | None = None
    kind: MetricKind
    unit: str | None = None
    missing_reason: MissingReason | None = None
    method: MetricMethod | None = None
    algorithm_version: str | None = None
    confidence: Confidence | None = None
    formula: str | None = None
    coefficient_set: str | None = None
    time_basis: TimeBasis | None = None


class _MetricLeadWire(BaseModel):
    """Identity fields shared by both metric payloads."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str


class _MetricLeadPeriodWire(_MetricLeadWire):
    """Identity fields plus the period the metric was computed for."""

    period: PeriodWire


# Pydantic collects base fields in reverse MRO order, so the trailing base is
# listed first: that keeps the public key order (name, description, period,
# value, ...) identical to the flat declarations this structure replaced.
# Class docstrings would become JSON-schema descriptions, so these two public
# payload models stay undocumented on purpose.
class MetricWire(_MetricTrailWire, _MetricLeadPeriodWire):
    pass


class MetricWithoutPeriodWire(_MetricTrailWire, _MetricLeadWire):
    pass


class ActivityRecordWire(BaseModel):
    model_config = ConfigDict(extra="forbid")

    activity_id: str
    start_time: datetime
    sport: str
    sub_sport: str | None = None
    segment_index: int
    segment_count: int
    repair_flags: list[str]
    source_name: str
    source_path: str
    analysis_version: str
    metrics: list[MetricWithoutPeriodWire]


class WeeklyReportWire(BaseModel):
    model_config = ConfigDict(extra="forbid")

    period: DateRangePeriodWire
    metrics: list[MetricWithoutPeriodWire]


class MetricReportWire(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metrics: list[MetricWire]


METRIC_REPORT_ADAPTER = TypeAdapter(MetricReportWire)
ACTIVITY_SUMMARIES_ADAPTER = TypeAdapter(list[ActivitySummaryWire])
ACTIVITY_RECORD_ADAPTER = TypeAdapter(ActivityRecordWire)


def metric_report(report: MetricReport) -> MetricReportWire:
    """Map a domain report into its generic transport model."""
    return METRIC_REPORT_ADAPTER.validate_python(report, from_attributes=True)


def activity_summaries(
    summaries: Sequence[ActivitySummary],
) -> list[ActivitySummaryWire]:
    """Map activity summaries into their transport models."""
    return ACTIVITY_SUMMARIES_ADAPTER.validate_python(list(summaries), from_attributes=True)


def activity_record(record: ActivityRecord) -> ActivityRecordWire:
    """Map a complete domain record into its transport model.

    Metric periods are not read: activity metrics are scoped by the record's
    own trip, which the wire payload does not repeat per metric.
    """
    return ACTIVITY_RECORD_ADAPTER.validate_python(record, from_attributes=True)


def weekly_report(report: MetricReport) -> WeeklyReportWire:
    """Hoist the shared weekly date range out of its metric collection."""
    if not report.metrics:
        msg = "weekly reports must contain metrics"
        raise ValueError(msg)
    period = report.metrics[0].period
    if not isinstance(period, DateRangePeriod) or any(
        metric.period != period for metric in report.metrics
    ):
        msg = "weekly metrics must share one date-range period"
        raise ValueError(msg)
    return WeeklyReportWire(
        period=DateRangePeriodWire.model_validate(period, from_attributes=True),
        metrics=[
            MetricWithoutPeriodWire.model_validate(metric, from_attributes=True)
            for metric in report.metrics
        ],
    )
