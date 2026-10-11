"""Tests for shared domain Metric types and wire serialization."""

from __future__ import annotations

import json
from dataclasses import fields
from datetime import UTC, date, datetime
from typing import Any

import pytest
from pydantic import TypeAdapter

from training_stats.adapters.wire import (
    ActivityRecordWire,
    ActivitySummaryWire,
    MetricReportWire,
    MetricWire,
    MetricWithoutPeriodWire,
    activity_record,
    metric_report,
    weekly_report,
)
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
    TimeBasis,
    TripPeriod,
)
from training_stats.domain.models import ActivityRecord, ActivitySummary


def test_metric_requires_value_or_reason() -> None:
    period = AllTimePeriod()
    with pytest.raises(ValueError, match="either a value or a missing reason"):
        Metric("missing-both", "A missing test metric.", period, None, MetricKind.MEASURED)
    with pytest.raises(ValueError, match="either a value or a missing reason"):
        Metric(
            "missing-neither",
            "A metric with conflicting value and reason.",
            period,
            ScalarValue(0),
            MetricKind.MEASURED,
            missing_reason=MissingReason.NO_DATA,
        )
    with pytest.raises(ValueError, match="description must not be empty"):
        Metric("blank-description", "   ", period, ScalarValue(0), MetricKind.MEASURED)


def test_activity_record_trip_period_must_match_id() -> None:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    metric = Metric(
        "distance",
        "Recorded distance for one trip.",
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
        "Recorded distance for each day.",
        DateRangePeriod(date(2026, 6, 1), date(2026, 6, 2), "UTC"),
        TableValue(Axis("date", ("2026-06-01",)), (1000.0,)),
        MetricKind.CALCULATED,
        unit="m",
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
                "Banister heart-rate training-load estimate.",
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
                "Recorded distance on each day.",
                window,
                TableValue(Axis("date", ("2026-06-01",)), (1000.0,)),
                MetricKind.CALCULATED,
                unit="m",
            ),
            Metric(
                "heatmap",
                "Time spent in each speed and cadence bin.",
                trip,
                MatrixValue(
                    Axis("speed", (AxisBin("5-10", 5.0, 10.0),), "km/h"),
                    Axis("cadence", (AxisBin("80-90", 80.0, 90.0),), "rpm"),
                    ((42.0,),),
                ),
                MetricKind.CALCULATED,
                unit="s",
            ),
            Metric(
                "unavailable",
                "Unavailable measured input.",
                SnapshotPeriod(date(2026, 6, 1)),
                None,
                MetricKind.MEASURED,
                missing_reason=MissingReason.NO_DATA,
            ),
            Metric(
                "history",
                "A label for the full history scope.",
                AllTimePeriod(),
                ScalarValue("all"),
                MetricKind.INTERPRETED,
            ),
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
                "Human-readable description for an arbitrary metric.",
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
    assert (
        dumped["metrics"][0]["description"] == "Human-readable description for an arbitrary metric."
    )


def test_excluding_none_fields_preserves_null_array_cells() -> None:
    report = MetricReport(
        (
            Metric(
                "sparse.daily_distance",
                "Recorded distance when activity data is available.",
                DateRangePeriod(date(2026, 6, 1), date(2026, 6, 2), "UTC"),
                TableValue(Axis("date", ("2026-06-01", "2026-06-02")), (None, 1200.0)),
                MetricKind.CALCULATED,
            ),
        )
    )
    dumped = metric_report(report).model_dump(mode="json", exclude_none=True)
    metric = dumped["metrics"][0]
    assert "unit" not in metric
    assert metric["value"]["values"] == [None, 1200.0]

    weekly_dump = weekly_report(report).model_dump(mode="json", exclude_none=True)
    assert weekly_dump["period"]["kind"] == "date_range"
    weekly_metric = weekly_dump["metrics"][0]
    assert "period" not in weekly_metric
    assert weekly_metric["value"]["values"] == [None, 1200.0]


def _shapes_report() -> MetricReport:
    """Scalar/table/matrix values, a missing metric, and every period kind."""
    return MetricReport(
        (
            Metric(
                "training_load",
                "Banister heart-rate training-load estimate.",
                TripPeriod("ride-1", datetime(2026, 6, 1, 10, tzinfo=UTC)),
                ScalarValue(87.5),
                MetricKind.ESTIMATED,
                "TRIMP",
                method=MetricMethod.HR_TRIMP,
                algorithm_version="1",
                confidence=Confidence.HIGH,
                formula="sum of beat-to-beat stress",
                coefficient_set="banister",
                time_basis=TimeBasis.VALID_HEART_RATE_TIME,
            ),
            Metric(
                "heart_rate.zone_time",
                "Seconds spent in each configured heart-rate zone.",
                DateRangePeriod(date(2026, 6, 1), date(2026, 6, 7), "Europe/Berlin"),
                TableValue(Axis("zone", ("Z1", "Z2", "Z3")), (60.0, None, 30.0)),
                MetricKind.CALCULATED,
                "s",
            ),
            Metric(
                "heatmap.speed_cadence",
                "Time spent in each speed and cadence bin.",
                SnapshotPeriod(date(2026, 6, 8)),
                MatrixValue(
                    Axis(
                        "speed",
                        (AxisBin("5-10", 5.0, 10.0), AxisBin("10-15", 10.0, 15.0)),
                        "km/h",
                    ),
                    Axis("cadence", (AxisBin("80-90", 80.0, 90.0),), "rpm"),
                    ((42.0, None),),
                ),
                MetricKind.CALCULATED,
                "s",
            ),
            Metric(
                "cadence.mean",
                "Average pedaling rate over the full history.",
                AllTimePeriod(),
                None,
                MetricKind.MEASURED,
                missing_reason=MissingReason.NO_CADENCE_DATA,
            ),
        )
    )


def _activity_record_fixture() -> ActivityRecord:
    """A cached record whose trip metrics must not repeat their period."""
    start = datetime(2026, 6, 1, 10, tzinfo=UTC)
    return ActivityRecord(
        activity_id="ride-1",
        start_time=start,
        sport="cycling",
        sub_sport=None,
        segment_index=0,
        segment_count=2,
        repair_flags=("gap_repaired",),
        source_name="local",
        source_path="2026/ride-1.fit",
        analysis_version="1",
        metrics=(
            Metric(
                "distance",
                "Recorded distance for one trip.",
                TripPeriod("ride-1", start),
                ScalarValue(1000.0),
                MetricKind.MEASURED,
                "m",
            ),
            Metric(
                "heart_rate.zone_time",
                "Seconds spent in each configured heart-rate zone.",
                TripPeriod("ride-1", start),
                TableValue(Axis("zone", ("Z1", "Z2")), (60.0, None)),
                MetricKind.CALCULATED,
                "s",
            ),
            Metric(
                "cadence.mean",
                "Average pedaling rate for one trip.",
                TripPeriod("ride-1", start),
                None,
                MetricKind.MEASURED,
                missing_reason=MissingReason.NO_CADENCE_DATA,
            ),
        ),
    )


def _weekly_report_fixture() -> MetricReport:
    """Metrics sharing one date range so the period can be hoisted."""
    window = DateRangePeriod(date(2026, 6, 1), date(2026, 6, 7), "Europe/Berlin")
    return MetricReport(
        (
            Metric(
                "activity.distance_by_week",
                "Recorded distance in each ISO week.",
                window,
                ScalarValue(120000.0),
                MetricKind.CALCULATED,
                "m",
            ),
            Metric(
                "activity.count_by_week",
                "Accepted activities in each ISO week.",
                window,
                ScalarValue(3),
                MetricKind.CALCULATED,
            ),
            Metric(
                "heart_rate.zone_time",
                "Seconds spent in each configured heart-rate zone.",
                window,
                TableValue(Axis("zone", ("Z1", "Z2")), (60.0, None)),
                MetricKind.CALCULATED,
                "s",
            ),
        )
    )


# Captured from the dump-to-dict wire conversions they replace: direct attribute
# validation must keep producing these exact dictionaries, key order included.
# Omission under `exclude_none=True` is locked by the exclusion tests above and
# by the response-shape assertions in `test_api.py`.
EXPECTED_SHAPES_REPORT = {
    "metrics": [
        {
            "name": "training_load",
            "description": "Banister heart-rate training-load estimate.",
            "period": {
                "kind": "trip",
                "activity_id": "ride-1",
                "start": "2026-06-01T10:00:00Z",
                "end": None,
            },
            "value": {"shape": "scalar", "value": 87.5},
            "kind": "estimated",
            "unit": "TRIMP",
            "missing_reason": None,
            "method": "hr_trimp",
            "algorithm_version": "1",
            "confidence": "high",
            "formula": "sum of beat-to-beat stress",
            "coefficient_set": "banister",
            "time_basis": "valid_heart_rate_time",
        },
        {
            "name": "heart_rate.zone_time",
            "description": "Seconds spent in each configured heart-rate zone.",
            "period": {
                "kind": "date_range",
                "start": "2026-06-01",
                "end": "2026-06-07",
                "timezone": "Europe/Berlin",
            },
            "value": {
                "shape": "table",
                "axis": {"name": "zone", "values": ["Z1", "Z2", "Z3"], "unit": None},
                "values": [60.0, None, 30.0],
            },
            "kind": "calculated",
            "unit": "s",
            "missing_reason": None,
            "method": None,
            "algorithm_version": None,
            "confidence": None,
            "formula": None,
            "coefficient_set": None,
            "time_basis": None,
        },
        {
            "name": "heatmap.speed_cadence",
            "description": "Time spent in each speed and cadence bin.",
            "period": {"kind": "snapshot", "at": "2026-06-08"},
            "value": {
                "shape": "matrix",
                "x_axis": {
                    "name": "speed",
                    "values": [
                        {"label": "5-10", "lower": 5.0, "upper": 10.0},
                        {"label": "10-15", "lower": 10.0, "upper": 15.0},
                    ],
                    "unit": "km/h",
                },
                "y_axis": {
                    "name": "cadence",
                    "values": [{"label": "80-90", "lower": 80.0, "upper": 90.0}],
                    "unit": "rpm",
                },
                "values": [[42.0, None]],
            },
            "kind": "calculated",
            "unit": "s",
            "missing_reason": None,
            "method": None,
            "algorithm_version": None,
            "confidence": None,
            "formula": None,
            "coefficient_set": None,
            "time_basis": None,
        },
        {
            "name": "cadence.mean",
            "description": "Average pedaling rate over the full history.",
            "period": {"kind": "all_time"},
            "value": None,
            "kind": "measured",
            "unit": None,
            "missing_reason": "no_cadence_data",
            "method": None,
            "algorithm_version": None,
            "confidence": None,
            "formula": None,
            "coefficient_set": None,
            "time_basis": None,
        },
    ]
}
EXPECTED_ACTIVITY_RECORD = {
    "activity_id": "ride-1",
    "start_time": "2026-06-01T10:00:00Z",
    "sport": "cycling",
    "sub_sport": None,
    "segment_index": 0,
    "segment_count": 2,
    "repair_flags": ["gap_repaired"],
    "source_name": "local",
    "source_path": "2026/ride-1.fit",
    "analysis_version": "1",
    "metrics": [
        {
            "name": "distance",
            "description": "Recorded distance for one trip.",
            "value": {"shape": "scalar", "value": 1000.0},
            "kind": "measured",
            "unit": "m",
            "missing_reason": None,
            "method": None,
            "algorithm_version": None,
            "confidence": None,
            "formula": None,
            "coefficient_set": None,
            "time_basis": None,
        },
        {
            "name": "heart_rate.zone_time",
            "description": "Seconds spent in each configured heart-rate zone.",
            "value": {
                "shape": "table",
                "axis": {"name": "zone", "values": ["Z1", "Z2"], "unit": None},
                "values": [60.0, None],
            },
            "kind": "calculated",
            "unit": "s",
            "missing_reason": None,
            "method": None,
            "algorithm_version": None,
            "confidence": None,
            "formula": None,
            "coefficient_set": None,
            "time_basis": None,
        },
        {
            "name": "cadence.mean",
            "description": "Average pedaling rate for one trip.",
            "value": None,
            "kind": "measured",
            "unit": None,
            "missing_reason": "no_cadence_data",
            "method": None,
            "algorithm_version": None,
            "confidence": None,
            "formula": None,
            "coefficient_set": None,
            "time_basis": None,
        },
    ],
}
EXPECTED_WEEKLY_REPORT = {
    "period": {
        "kind": "date_range",
        "start": "2026-06-01",
        "end": "2026-06-07",
        "timezone": "Europe/Berlin",
    },
    "metrics": [
        {
            "name": "activity.distance_by_week",
            "description": "Recorded distance in each ISO week.",
            "value": {"shape": "scalar", "value": 120000.0},
            "kind": "calculated",
            "unit": "m",
            "missing_reason": None,
            "method": None,
            "algorithm_version": None,
            "confidence": None,
            "formula": None,
            "coefficient_set": None,
            "time_basis": None,
        },
        {
            "name": "activity.count_by_week",
            "description": "Accepted activities in each ISO week.",
            "value": {"shape": "scalar", "value": 3},
            "kind": "calculated",
            "unit": None,
            "missing_reason": None,
            "method": None,
            "algorithm_version": None,
            "confidence": None,
            "formula": None,
            "coefficient_set": None,
            "time_basis": None,
        },
        {
            "name": "heart_rate.zone_time",
            "description": "Seconds spent in each configured heart-rate zone.",
            "value": {
                "shape": "table",
                "axis": {"name": "zone", "values": ["Z1", "Z2"], "unit": None},
                "values": [60.0, None],
            },
            "kind": "calculated",
            "unit": "s",
            "missing_reason": None,
            "method": None,
            "algorithm_version": None,
            "confidence": None,
            "formula": None,
            "coefficient_set": None,
            "time_basis": None,
        },
    ],
}


def _assert_identical_json(payload: dict[str, Any], expected: dict[str, Any]) -> None:
    assert payload == expected
    # Dict equality ignores key order, so compare the serialized bytes too.
    assert json.dumps(payload) == json.dumps(expected)


def test_metric_report_wire_json_is_unchanged() -> None:
    _assert_identical_json(
        metric_report(_shapes_report()).model_dump(mode="json"), EXPECTED_SHAPES_REPORT
    )


def test_activity_record_wire_json_omits_metric_periods() -> None:
    payload = activity_record(_activity_record_fixture()).model_dump(mode="json")
    _assert_identical_json(payload, EXPECTED_ACTIVITY_RECORD)
    excluded = activity_record(_activity_record_fixture()).model_dump(
        mode="json", exclude_none=True
    )
    assert "sub_sport" not in excluded
    assert all("period" not in metric for metric in excluded["metrics"])


def test_weekly_report_wire_json_hoists_shared_period() -> None:
    _assert_identical_json(
        weekly_report(_weekly_report_fixture()).model_dump(mode="json"),
        EXPECTED_WEEKLY_REPORT,
    )
    excluded = weekly_report(_weekly_report_fixture()).model_dump(mode="json", exclude_none=True)
    assert excluded["period"]["kind"] == "date_range"
    assert all("period" not in metric for metric in excluded["metrics"])


@pytest.mark.parametrize(
    ("domain_type", "wire_type", "excluded"),
    [
        (Metric, MetricWire, frozenset()),
        (Metric, MetricWithoutPeriodWire, frozenset({"period"})),
        (MetricReport, MetricReportWire, frozenset()),
        (ActivityRecord, ActivityRecordWire, frozenset({"present_quantities"})),
        (ActivitySummary, ActivitySummaryWire, frozenset()),
    ],
    ids=["metric", "metric-without-period", "metric-report", "activity-record", "summary"],
)
def test_wire_fields_cover_every_domain_field(
    domain_type: type[Any],
    wire_type: type[Any],
    excluded: frozenset[str],
) -> None:
    """Attribute validation ignores undeclared domain data; keep the sets equal."""
    declared = {field.name for field in fields(domain_type)}
    assert set(wire_type.model_fields) == declared - excluded
