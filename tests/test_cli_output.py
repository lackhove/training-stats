"""Tests for generic shape-based Rich metric rendering."""

from __future__ import annotations

from datetime import UTC, date, datetime
from io import StringIO

from rich.console import Console

from training_stats.adapters.cli.output import format_duration, render_activity_list, render_report
from training_stats.domain.metrics import (
    Axis,
    AxisBin,
    Confidence,
    DateRangePeriod,
    MatrixValue,
    Metric,
    MetricKind,
    MetricReport,
    MissingReason,
    ScalarValue,
    TableValue,
    TripPeriod,
)
from training_stats.domain.models import ActivitySummary


def _render(report: MetricReport) -> str:
    output = StringIO()
    render_report(report, Console(file=output, width=180, force_terminal=False, color_system=None))
    return output.getvalue()


def test_activity_list_renders_metadata_without_metrics() -> None:
    output = StringIO()
    render_activity_list(
        (
            ActivitySummary(
                activity_id="ride-1",
                start_time=datetime(2026, 6, 1, 10, tzinfo=UTC),
                sport="cycling",
                sub_sport="road",
                segment_index=0,
                segment_count=1,
                repair_flags=(),
                source_name="local",
                source_path="ride.fit",
                analysis_version="1",
            ),
        ),
        Console(file=output, width=120, force_terminal=False, color_system=None),
    )
    rendered = output.getvalue()
    assert "ride-1" in rendered
    assert "cycling" in rendered
    assert "1/1" in rendered
    assert "training_load" not in rendered


def test_duration_formatting() -> None:
    assert format_duration(61) == "1:01"
    assert format_duration(3661) == "1:01:01"


def test_empty_report() -> None:
    assert "No metrics available." in _render(MetricReport(()))


def test_scalar_and_missing_metrics_are_generic() -> None:
    period = TripPeriod("ride-1", datetime(2026, 6, 1, 10, tzinfo=UTC))
    report = MetricReport(
        (
            Metric(
                "custom.unknown_measure",
                "The number of unknown widgets measured during a trip.",
                period,
                ScalarValue(12.5),
                MetricKind.ESTIMATED,
                unit="widgets",
                confidence=Confidence.MEDIUM,
            ),
            Metric(
                "custom.unavailable_measure",
                "Number of widgets when sensor data is available.",
                period,
                None,
                MetricKind.MEASURED,
                unit="widgets",
                missing_reason=MissingReason.NO_DATA,
            ),
        )
    )
    output = _render(report)
    assert "Custom Unknown Measure" in output
    assert "12.50 widgets" in output
    assert "The number of unknown widgets measured during a trip." in output
    assert "confidence medium" in output
    assert "No Data" in output


def test_table_uses_axis_and_unknown_unit_fallback() -> None:
    report = MetricReport(
        (
            Metric(
                "distance.over_days",
                "Recorded distance on each date in the requested range.",
                DateRangePeriod(date(2026, 6, 1), date(2026, 6, 2), "UTC"),
                TableValue(
                    Axis("date", (date(2026, 6, 1), date(2026, 6, 2))),
                    (1000.0, 2000.0),
                ),
                MetricKind.CALCULATED,
                unit="furlongs",
            ),
        )
    )
    output = _render(report)
    assert "Distance Over Days" in output
    assert "Recorded distance on each date" in output
    assert "2026-06-01" in output
    assert "1,000 furlongs" in output


def test_matrix_renders_axes_and_cell_measure() -> None:
    report = MetricReport(
        (
            Metric(
                "speed.cadence.distribution",
                "Duration in each binned speed and cadence combination.",
                TripPeriod("ride-2", datetime(2026, 6, 1, tzinfo=UTC)),
                MatrixValue(
                    x_axis=Axis("speed", (AxisBin("10\u201315", 10.0, 15.0),), "km/h"),
                    y_axis=Axis("cadence", (AxisBin("80\u201390", 80.0, 90.0),), "rpm"),
                    values=((42.0,),),
                ),
                MetricKind.CALCULATED,
                unit="s",
            ),
        )
    )
    output = _render(report)
    assert "Speed Cadence Distribution" in output
    assert "Duration in each binned speed and cadence combination." in output
    assert "10\u201315 km/h" in output
    assert "80\u201390 rpm" in output
    assert "0:42" in output
