"""Tests for shared domain Metric types and wire serialization."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import TypeAdapter

from training_stats.adapters.wire import MetricReportWire, metric_report
from training_stats.domain.metrics import (
    AllTimePeriod,
    Axis,
    AxisBin,
    Confidence,
    DateRangePeriod,
    MatrixValue,
    Metric,
    MetricKind,
    MetricMethod,
    MetricReport,
    MissingReason,
    ScalarValue,
    SnapshotPeriod,
    TableValue,
    TripPeriod,
)
from training_stats.domain.models import ActivityRecord


def test_metric_requires_value_or_reason() -> None:
    period = AllTimePeriod()
    with pytest.raises(ValueError, match="either a value or a missing reason"):
        Metric("missing-both", period, None, MetricKind.MEASURED)
    with pytest.raises(ValueError, match="either a value or a missing reason"):
        Metric(
            "missing-neither",
            period,
            ScalarValue(0),
            MetricKind.MEASURED,
            missing_reason=MissingReason.NO_DATA,
        )


def test_activity_record_trip_period_must_match_id() -> None:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    metric = Metric(
        "distance",
        TripPeriod("different-ride", start),
        ScalarValue(1000.0),
        MetricKind.MEASURED,
    )
    with pytest.raises(ValueError, match="must use their ActivityRecord trip period"):
        ActivityRecord("ride-1", start, "cycling", None, 0, 1, (), "local", "x.fit", "1", (metric,))


def test_activity_record_rejects_period_aggregates() -> None:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    metric = Metric(
        "distance-by-day",
        DateRangePeriod(date(2026, 6, 1), date(2026, 6, 2), "UTC"),
        TableValue(Axis("date", ("2026-06-01",)), (1000.0,)),
        MetricKind.CALCULATED,
        "m",
    )
    with pytest.raises(ValueError, match="must use their ActivityRecord trip period"):
        ActivityRecord("ride-1", start, "cycling", None, 0, 1, (), "local", "x.fit", "1", (metric,))


def test_value_shapes_validate_axis_dimensions() -> None:
    with pytest.raises(ValueError, match="axis and value lengths"):
        TableValue(Axis("date", (date(2026, 1, 1),)), (1.0, 2.0))
    with pytest.raises(ValueError, match="row count"):
        MatrixValue(Axis("x", (1, 2)), Axis("y", (1, 2)), ((1.0, 2.0),))
    with pytest.raises(ValueError, match="column count"):
        MatrixValue(Axis("x", (1, 2)), Axis("y", (1,)), ((1.0,),))


def test_domain_report_json_round_trip_all_shapes_and_periods() -> None:
    trip = TripPeriod("ride-1", datetime(2026, 6, 1, tzinfo=UTC))
    window = DateRangePeriod(date(2026, 6, 1), date(2026, 6, 7), "Europe/Berlin")
    report = MetricReport(
        (
            Metric(
                "load",
                trip,
                ScalarValue(123.0),
                MetricKind.ESTIMATED,
                "TRIMP",
                method=MetricMethod.HR_TRIMP,
                algorithm_version="1",
                confidence=Confidence.HIGH,
            ),
            Metric(
                "daily-distance",
                window,
                TableValue(Axis("date", ("2026-06-01",)), (1000.0,)),
                MetricKind.CALCULATED,
                "m",
            ),
            Metric(
                "heatmap",
                trip,
                MatrixValue(
                    Axis("speed", (AxisBin("5-10", 5.0, 10.0),), "km/h"),
                    Axis("cadence", (AxisBin("80-90", 80.0, 90.0),), "rpm"),
                    ((42.0,),),
                ),
                MetricKind.CALCULATED,
                "s",
            ),
            Metric(
                "unavailable",
                SnapshotPeriod(date(2026, 6, 1)),
                None,
                MetricKind.MEASURED,
                missing_reason=MissingReason.NO_DATA,
            ),
            Metric("history", AllTimePeriod(), ScalarValue("all"), MetricKind.INTERPRETED),
        )
    )
    adapter = TypeAdapter(MetricReport)
    decoded = adapter.validate_json(adapter.dump_json(report))
    assert decoded == report
    wire = metric_report(report)
    assert wire.metrics[1].value.shape == "table"
    assert wire.metrics[2].value.shape == "matrix"


def test_wire_report_has_discriminated_shape_contract() -> None:
    report = MetricReport(
        (
            Metric(
                "metric.with.new.name",
                TripPeriod("ride-2", datetime(2026, 6, 1, tzinfo=UTC)),
                ScalarValue(2),
                MetricKind.MEASURED,
                unit="unknown-unit",
            ),
        )
    )
    wire = metric_report(report)
    assert isinstance(wire, MetricReportWire)
    dumped = wire.model_dump(mode="json")
    assert dumped["metrics"][0]["value"]["shape"] == "scalar"
    assert dumped["metrics"][0]["period"]["kind"] == "trip"
