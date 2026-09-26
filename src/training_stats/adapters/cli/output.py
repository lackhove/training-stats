"""Shape-driven Rich rendering for generic metric reports."""

from __future__ import annotations

from datetime import date, datetime

from rich import box
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ...domain.metrics import (
    AllTimePeriod,
    AxisBin,
    DateRangePeriod,
    MatrixValue,
    Metric,
    MetricReport,
    MetricScalar,
    ScalarValue,
    SnapshotPeriod,
    TableValue,
    TripPeriod,
)
from ...domain.models import ActivitySummary

_EM_DASH = "\u2014"
_HEAT_COLORS = ("blue", "cyan", "green", "yellow", "red")


def render_activity_list(payload: object, console: Console) -> None:
    """Render compact activity metadata as a Rich table."""
    if not isinstance(payload, tuple):
        console.print(Text("Invalid activity list.", style="red"))
        return
    activities = tuple(activity for activity in payload if isinstance(activity, ActivitySummary))
    if len(activities) != len(payload):
        console.print(Text("Invalid activity list.", style="red"))
        return
    if not activities:
        console.print(Text("No activities found.", style="dim"))
        return

    table = Table(title="Activities", box=box.SIMPLE_HEAVY, pad_edge=False, header_style="bold")
    table.add_column("Activity ID", no_wrap=True)
    table.add_column("Started", no_wrap=True)
    table.add_column("Sport")
    table.add_column("Sub-sport")
    for activity in activities:
        table.add_row(
            activity.activity_id,
            activity.start_time.isoformat(sep=" ", timespec="minutes"),
            activity.sport,
            activity.sub_sport or _EM_DASH,
        )
    console.print(table)


def render_report(payload: object, console: Console) -> None:
    """Render any report using only metric value shape and its metadata."""
    if not isinstance(payload, MetricReport):
        console.print(Text("Invalid metric report.", style="red"))
        return
    if not payload.metrics:
        console.print(Text("No metrics available.", style="dim"))
        return

    for period, metrics in _group_periods(payload.metrics):
        period_label = format_period(period)
        scalar_metrics = [
            metric
            for metric in metrics
            if metric.value is None or isinstance(metric.value, ScalarValue)
        ]
        if scalar_metrics:
            console.print(_panel(f"Metrics · {period_label}", _scalar_table(scalar_metrics)))
        for metric in metrics:
            if isinstance(metric.value, TableValue):
                console.print(
                    _panel(
                        f"{format_name(metric.name)} · {period_label}",
                        Group(Text(metric.description, style="dim"), _table(metric)),
                    )
                )
            elif isinstance(metric.value, MatrixValue):
                console.print(
                    _panel(
                        f"{format_name(metric.name)} · {period_label}",
                        Group(Text(metric.description, style="dim"), _matrix(metric)),
                    )
                )


def format_name(name: str) -> str:
    """Humanize a metric identifier without a metric-specific label registry."""
    return " ".join(
        part.capitalize() for part in name.replace(".", "_").replace("-", "_").split("_")
    )


def format_period(period: object) -> str:
    """Format any declared period variant."""
    if isinstance(period, TripPeriod):
        start = period.start.strftime("%Y-%m-%d %H:%M")
        return f"trip {period.activity_id} · {start}"
    if isinstance(period, DateRangePeriod):
        return f"{period.start.isoformat()} → {period.end.isoformat()} ({period.timezone})"
    if isinstance(period, SnapshotPeriod):
        return f"as of {_format_date(period.at)}"
    if isinstance(period, AllTimePeriod):
        return "all time"
    return "period unknown"


def format_scalar(value: MetricScalar | None, unit: str | None) -> str:
    """Apply universal formatting based on value type and unit, with fallback."""
    if value is None:
        return _EM_DASH
    if isinstance(value, bool):
        rendered = "yes" if value else "no"
    elif isinstance(value, float):
        rendered = f"{value:,.2f}" if abs(value) < 1000 else f"{value:,.0f}"
    else:
        rendered = str(value)
    if unit == "s" and isinstance(value, int | float) and not isinstance(value, bool):
        rendered = format_duration(float(value))
    elif unit == "ratio" and isinstance(value, int | float) and not isinstance(value, bool):
        rendered = f"{float(value) * 100:.0f}%"
    if unit and unit not in {"s", "ratio"}:
        rendered = f"{rendered} {unit}"
    return rendered


def _group_periods(metrics: tuple[Metric, ...]) -> list[tuple[object, list[Metric]]]:
    groups: list[tuple[object, list[Metric]]] = []
    for metric in metrics:
        for period, group in groups:
            if period == metric.period:
                group.append(metric)
                break
        else:
            groups.append((metric.period, [metric]))
    return groups


def _scalar_table(metrics: list[Metric]) -> Table:
    table = Table(box=box.SIMPLE_HEAVY, pad_edge=False, header_style="bold")
    table.add_column("Metric / description", overflow="fold")
    table.add_column("Value", justify="right")
    table.add_column("Kind", no_wrap=True)
    table.add_column("Details", overflow="fold")
    for metric in metrics:
        if metric.value is None:
            reason = metric.missing_reason.value if metric.missing_reason is not None else "missing"
            value = Text(f"{_EM_DASH}  ({format_name(reason)})", style="dim")
        else:
            scalar = metric.value.value if isinstance(metric.value, ScalarValue) else None
            value = Text(format_scalar(scalar, metric.unit))
        label = Text(format_name(metric.name), style="bold")
        label.append("\n")
        label.append(metric.description, style="dim")
        table.add_row(label, value, metric.kind.value, _provenance(metric))
    return table


def _table(metric: Metric) -> Table:
    value = metric.value
    if not isinstance(value, TableValue):
        return Table()
    table = Table(box=box.SIMPLE, pad_edge=False, header_style="bold")
    table.add_column(format_name(value.axis.name), no_wrap=True)
    table.add_column(format_name(metric.name), justify="right")
    for axis_value, cell in zip(value.axis.values, value.values, strict=True):
        table.add_row(_format_axis(axis_value, value.axis.unit), format_scalar(cell, metric.unit))
    if metric.missing_reason is not None:
        table.caption = format_name(metric.missing_reason.value)
    return table


def _matrix(metric: Metric) -> Table:
    value = metric.value
    if not isinstance(value, MatrixValue):
        return Table()
    numeric_values = [
        float(cell)
        for row in value.values
        for cell in row
        if isinstance(cell, int | float) and not isinstance(cell, bool)
    ]
    low = min(numeric_values, default=0.0)
    high = max(numeric_values, default=0.0)
    table = Table(box=box.SIMPLE, pad_edge=False, header_style="bold")
    table.add_column(format_name(value.y_axis.name), no_wrap=True)
    for x_value in value.x_axis.values:
        table.add_column(_format_axis(x_value, value.x_axis.unit), justify="right")
    for y_index, y_value in enumerate(value.y_axis.values):
        cells: list[RenderableType] = [_format_axis(y_value, value.y_axis.unit)]
        for cell in value.values[y_index]:
            text = Text(format_scalar(cell, metric.unit))
            if isinstance(cell, int | float) and not isinstance(cell, bool):
                text.stylize(_heat_style(float(cell), low, high))
            cells.append(text)
        table.add_row(*cells)
    table.caption = f"Cell measure: {metric.unit or 'value'} · {_provenance(metric)}"
    return table


def _format_axis(value: object, unit: str | None) -> str:
    if isinstance(value, AxisBin):
        upper = "∞" if value.upper is None else f"{value.upper:g}"
        unit_text = f" {unit}" if unit else ""
        return f"{value.lower:g}\u2013{upper}{unit_text}"
    if isinstance(value, datetime):
        rendered = value.isoformat(sep=" ", timespec="minutes")
    elif isinstance(value, date):
        rendered = value.isoformat()
    else:
        rendered = str(value)
    return f"{rendered} {unit}" if unit else rendered


def _heat_style(value: float, low: float, high: float) -> str:
    if high <= low:
        return _HEAT_COLORS[2]
    index = round((value - low) / (high - low) * (len(_HEAT_COLORS) - 1))
    return _HEAT_COLORS[max(0, min(index, len(_HEAT_COLORS) - 1))]


def _provenance(metric: Metric) -> str:
    details = [metric.method.value] if metric.method is not None else []
    if metric.algorithm_version is not None:
        details.append(f"v{metric.algorithm_version}")
    if metric.confidence is not None:
        details.append(f"confidence {metric.confidence.value}")
    if metric.time_basis is not None:
        details.append(metric.time_basis.value)
    return ", ".join(details) if details else _EM_DASH


def _panel(title: str, renderable: RenderableType) -> Panel:
    return Panel(renderable, title=title, title_align="left", border_style="cyan", expand=False)


def format_duration(seconds: float) -> str:
    """Format elapsed seconds as H:MM:SS or M:SS."""
    total = max(round(seconds), 0)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _format_date(value: date | datetime) -> str:
    if isinstance(value, datetime):
        return value.isoformat(sep=" ", timespec="minutes")
    return value.isoformat()
