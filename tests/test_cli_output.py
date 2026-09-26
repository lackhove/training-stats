"""Tests for generic shape-based Rich metric rendering."""

from __future__ import annotations

from datetime import UTC, date, datetime
from io import StringIO

from rich.console import Console

from training_stats.adapters.cli.output import format_duration, render_report
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


def _render(report: MetricReport) -> str:
    output = StringIO()
    render_report(report, Console(file=output, force_terminal=False, color_system=None))
    return output.getvalue()


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
                period,
                ScalarValue(12.5),
                MetricKind.ESTIMATED,
                unit="widgets",
                confidence=Confidence.MEDIUM,
            ),
            Metric(
                "custom.unavailable_measure",
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
    assert "confidence medium" in output
    assert "No Data" in output


def test_table_uses_axis_and_unknown_unit_fallback() -> None:
    report = MetricReport(
        (
            Metric(
                "distance.over_days",
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
    assert "2026-06-01" in output
    assert "1,000 furlongs" in output


def test_matrix_renders_axes_and_cell_measure() -> None:
    report = MetricReport(
        (
            Metric(
                "speed.cadence.distribution",
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
    assert "10\u201315 km/h" in output
    assert "80\u201390 rpm" in output
    assert "0:42" in output
