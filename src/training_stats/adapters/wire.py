"""Pydantic wire contracts for activity records and metric reports."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from ..domain.metrics import (
    Confidence,
    Metric,
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


class MetricWire(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    period: PeriodWire
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


class ActivityMetricWire(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
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
    metrics: list[ActivityMetricWire]


class MetricReportWire(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metrics: list[MetricWire]


METRIC_REPORT_ADAPTER = TypeAdapter(MetricReportWire)
_DOMAIN_REPORT_ADAPTER = TypeAdapter(MetricReport)
_DOMAIN_METRIC_ADAPTER = TypeAdapter(Metric)
ACTIVITY_SUMMARIES_ADAPTER = TypeAdapter(list[ActivitySummaryWire])
_DOMAIN_ACTIVITY_SUMMARIES_ADAPTER = TypeAdapter(list[ActivitySummary])
ACTIVITY_RECORD_ADAPTER = TypeAdapter(ActivityRecordWire)
_DOMAIN_ACTIVITY_RECORD_ADAPTER = TypeAdapter(ActivityRecord)


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


def activity_record(record: ActivityRecord) -> ActivityRecordWire:
    """Validate and map a complete domain record into its transport model."""
    payload = _DOMAIN_ACTIVITY_RECORD_ADAPTER.dump_python(record, mode="json")
    payload["metrics"] = [
        _DOMAIN_METRIC_ADAPTER.dump_python(metric, mode="json", exclude={"period"})
        for metric in record.metrics
    ]
    return ACTIVITY_RECORD_ADAPTER.validate_python(payload)
