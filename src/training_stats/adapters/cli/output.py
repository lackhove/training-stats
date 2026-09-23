"""Rich rendering for the training-stats CLI.

This module owns all Rich rendering; the CLI passes it wire dictionaries
(``model_dump(mode="json")`` of the models in :mod:`training_stats.adapters.wire`)
and injects the target :class:`rich.console.Console`. Diagnostics (sync
feedback, errors) render on stderr, payloads on stdout.
"""

from __future__ import annotations

from datetime import datetime

from rich import box
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

_ZONE_COLORS = ("cyan", "green", "yellow", "dark_orange", "red")
_BAR_WIDTH = 16
_EM_DASH = "\u2014"
_BLOCKS = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"


# -- value coercion (payloads are dict[str, object]) -----------------------


def _opt_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _opt_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _str(value: object) -> str:
    return value if isinstance(value, str) else ""


def _mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _rows(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


# -- formatting helpers ----------------------------------------------------


def format_duration(seconds: float | None) -> str:
    """Format a duration as ``H:MM:SS`` or ``M:SS``."""
    if seconds is None:
        return _EM_DASH
    total = max(round(seconds), 0)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_distance(meters: float | None) -> str:
    """Format a distance in meters as meters or kilometers."""
    if meters is None:
        return _EM_DASH
    if meters < 1000:
        return f"{meters:.0f} m"
    return f"{meters / 1000:.2f} km"


def format_elevation(meters: float | None) -> str:
    """Format an elevation gain in meters."""
    return _EM_DASH if meters is None else f"{meters:.0f} m"


def format_speed_kmh(mps: float | None) -> str:
    """Format a speed in meters per second as km/h."""
    return _EM_DASH if mps is None else f"{mps * 3.6:.1f} km/h"


def format_number(value: float | None, decimals: int = 0, unit: str = "") -> str:
    """Format an optional number, falling back to an em dash."""
    if value is None:
        return _EM_DASH
    return f"{value:.{decimals}f}{unit}"


def format_percent(ratio: float | None) -> str:
    """Format a 0..1 ratio as a whole percentage."""
    return _EM_DASH if ratio is None else f"{ratio * 100:.0f}%"


def format_start(item: dict[str, object]) -> str:
    """Format the local start time of an activity summary or details payload."""
    local = _str(item.get("local_start_time"))
    if local:
        try:
            return datetime.fromisoformat(local).strftime("%Y-%m-%d %H:%M")
        except ValueError:
            return local
    return _str(item.get("start_time")) or _EM_DASH


def format_delta(
    current: float | None,
    previous: float | None,
    *,
    scale: float = 1.0,
    decimals: int = 0,
    unit: str = "",
) -> Text:
    """Format a signed change versus a previous value, colored green/red."""
    if current is None or previous is None:
        return Text(_EM_DASH, style="dim")
    difference = (current - previous) * scale
    if round(difference, decimals) == 0:
        return Text(f"+{0.0:.{decimals}f}{unit}", style="dim")
    style = "green" if difference > 0 else "red"
    return Text(f"{difference:+.{decimals}f}{unit}", style=style)


def _sport_label(item: dict[str, object]) -> str:
    sport = _str(item.get("sport")) or "unknown"
    sub_sport = _str(item.get("sub_sport"))
    return f"{sport} ({sub_sport})" if sub_sport else sport


def _panel(title: str, renderable: RenderableType, *, style: str = "cyan") -> Panel:
    return Panel(renderable, title=title, title_align="left", border_style=style, expand=False)


def _kv_grid(pairs: list[tuple[str, RenderableType]]) -> Table:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", no_wrap=True)
    grid.add_column()
    for label, value in pairs:
        grid.add_row(label, value)
    return grid


def _bar(seconds: float, total: float, color: str) -> ProgressBar:
    safe_total = total if total > 0 else 1.0
    return ProgressBar(
        total=safe_total,
        completed=max(min(seconds, safe_total), 0.0),
        width=_BAR_WIDTH,
        pulse=False,
        complete_style=color,
    )


def _zone_seconds(zones: dict[str, object]) -> list[float]:
    return [(_opt_float(zones.get(f"zone_{index}")) or 0.0) for index in range(1, 6)]


def _zone_table(zones: dict[str, object], total: float | None = None) -> Table:
    seconds = _zone_seconds(zones)
    zone_total = total if total is not None else sum(seconds)
    table = Table(box=box.SIMPLE, pad_edge=False, show_edge=False)
    table.add_column("Zone", style="bold", no_wrap=True)
    table.add_column("Time", justify="right")
    table.add_column("Share", justify="right")
    table.add_column("", min_width=_BAR_WIDTH + 2)
    for index, value in enumerate(seconds, start=1):
        share = value / zone_total if zone_total > 0 else 0.0
        table.add_row(
            f"Z{index}",
            format_duration(value),
            format_percent(share),
            _bar(value, zone_total, _ZONE_COLORS[index - 1]),
        )
    return table


# -- payload renderers -----------------------------------------------------


def render_list(payload: object, console: Console) -> None:
    """Render activity summaries as a table, newest first."""
    activities = _rows(payload)
    if not activities:
        console.print(Text("No activities found.", style="dim"))
        return
    table = Table(box=box.SIMPLE_HEAVY, pad_edge=False, header_style="bold")
    table.add_column("Start", no_wrap=True)
    table.add_column("Sport", no_wrap=True)
    table.add_column("Duration", justify="right")
    table.add_column("Distance", justify="right")
    table.add_column("Elev", justify="right")
    table.add_column("Avg HR", justify="right")
    table.add_column("Max HR", justify="right")
    table.add_column("Load", justify="right")
    table.add_column("ID", style="dim", overflow="fold")
    for item in activities:
        table.add_row(
            format_start(item),
            _sport_label(item),
            format_duration(_opt_float(item.get("duration_s"))),
            format_distance(_opt_float(item.get("distance_m"))),
            format_elevation(_opt_float(item.get("elevation_gain_m"))),
            format_number(_opt_float(item.get("avg_hr"))),
            format_number(_opt_float(item.get("max_hr"))),
            format_number(_opt_float(item.get("load_value"))),
            _str(item.get("activity_id")),
        )
    console.print(table)


def render_weekly(payload: object, console: Console) -> None:
    """Render weekly totals with previous-week comparisons as a table."""
    rows = _rows(payload)
    if not rows:
        console.print(Text("No weekly data found.", style="dim"))
        return
    table = Table(box=box.SIMPLE_HEAVY, pad_edge=False, header_style="bold")
    table.add_column("Week", no_wrap=True)
    table.add_column("Start", no_wrap=True)
    table.add_column("Sport", no_wrap=True)
    table.add_column("Activities", justify="right")
    table.add_column("Duration", justify="right")
    table.add_column("Distance", justify="right")
    table.add_column("Elev", justify="right")
    table.add_column("Load", justify="right")
    table.add_column("\u0394 Load", justify="right")
    table.add_column("\u0394 Dist", justify="right")
    for row in rows:
        comparison = _mapping(row.get("comparison_vs_prev_week"))
        iso_year = _opt_int(row.get("iso_year")) or 0
        iso_week = _opt_int(row.get("iso_week")) or 0
        table.add_row(
            f"{iso_year}-W{iso_week:02d}",
            _str(row.get("week_start")) or _EM_DASH,
            _str(row.get("sport")),
            format_number(_opt_float(row.get("activity_count"))),
            format_duration(_opt_float(row.get("duration_s"))),
            format_distance(_opt_float(row.get("distance_m"))),
            format_elevation(_opt_float(row.get("elevation_gain_m"))),
            format_number(_opt_float(row.get("load_value"))),
            format_delta(
                _opt_float(row.get("load_value")), _opt_float(comparison.get("load_value"))
            ),
            format_delta(
                _opt_float(row.get("distance_m")),
                _opt_float(comparison.get("distance_m")),
                scale=0.001,
                decimals=1,
                unit=" km",
            ),
        )
    console.print(table)


def _histogram_label(key: str) -> int:
    head = key.split("-", 1)[0]
    try:
        return int(head)
    except ValueError:
        return 0


def _histogram_table(histogram: dict[str, object]) -> Table:
    entries = sorted(
        ((key, _opt_float(value) or 0.0) for key, value in histogram.items()),
        key=lambda entry: _histogram_label(entry[0]),
    )
    visible = [(key, seconds) for key, seconds in entries if seconds > 0]
    peak = max((seconds for _, seconds in visible), default=0.0)
    table = Table(box=box.SIMPLE, pad_edge=False, show_edge=False)
    table.add_column("BPM", style="bold", no_wrap=True)
    table.add_column("Time", justify="right")
    table.add_column("", min_width=_BAR_WIDTH + 2)
    for key, seconds in visible:
        table.add_row(key, format_duration(seconds), _bar(seconds, peak, "cyan"))
    return table


def _bucket_table(buckets: list[dict[str, object]]) -> Table:
    table = Table(box=box.SIMPLE, pad_edge=False, show_edge=False)
    table.add_column("Start", style="bold", no_wrap=True)
    table.add_column("Avg HR", justify="right")
    table.add_column("HR time", justify="right")
    for bucket in buckets:
        start_min = _opt_int(bucket.get("start_min")) or 0
        table.add_row(
            f"{start_min} min",
            format_number(_opt_float(bucket.get("avg_hr"))),
            format_duration(_opt_float(bucket.get("hr_time_s"))),
        )
    return table


def render_show(payload: object, console: Console) -> None:
    """Render one activity with metrics, zones, load and data quality."""
    details = _mapping(payload)
    header = Text(_sport_label(details), style="bold")
    header.append(f"  {format_start(details)}")
    header.append(f"  {_str(details.get('activity_id'))}", style="dim")
    console.print(_panel("Activity", header))

    console.print(
        _panel(
            "Metrics",
            _kv_grid(
                [
                    ("Duration", format_duration(_opt_float(details.get("duration_s")))),
                    ("Moving time", format_duration(_opt_float(details.get("moving_time_s")))),
                    ("Paused time", format_duration(_opt_float(details.get("paused_time_s")))),
                    ("Distance", format_distance(_opt_float(details.get("distance_m")))),
                    ("Elevation", format_elevation(_opt_float(details.get("elevation_gain_m")))),
                    ("Avg HR", format_number(_opt_float(details.get("avg_hr")), unit=" bpm")),
                    ("Max HR", format_number(_opt_float(details.get("max_hr")), unit=" bpm")),
                    ("Avg speed", format_speed_kmh(_opt_float(details.get("avg_speed_mps")))),
                    ("Max speed", format_speed_kmh(_opt_float(details.get("max_speed_mps")))),
                    ("Calories", format_number(_opt_float(details.get("calories")), unit=" kcal")),
                ]
            ),
        )
    )

    zones = _mapping(details.get("hr_zones_s"))
    zone_total = sum(_zone_seconds(zones))
    console.print(_panel("Heart rate zones", _zone_table(zones, total=zone_total)))

    histogram = _mapping(details.get("hr_histogram_s"))
    if histogram:
        console.print(_panel("HR histogram", _histogram_table(histogram)))

    buckets = _rows(details.get("hr_10min_averages"))
    if buckets:
        console.print(_panel("10-minute averages", _bucket_table(buckets)))

    load = _mapping(details.get("load"))
    load_value = _opt_float(load.get("value"))
    load_text = Text(format_number(load_value, decimals=0), style="bold")
    if load.get("method") is not None:
        load_text.append(f"  ({_str(load.get('method'))})", style="dim")
    if load.get("confidence") is not None:
        load_text.append(f"  confidence: {_str(load.get('confidence'))}", style="dim")
    console.print(_panel("Load", load_text))

    decoupling = _mapping(details.get("decoupling"))
    if decoupling:
        console.print(
            _panel(
                "Decoupling",
                _kv_grid(
                    [
                        (
                            "Efficiency factor",
                            f"{format_number(_opt_float(decoupling.get('ef_first_half')), 3)}"
                            f" \u2192 "
                            f"{format_number(_opt_float(decoupling.get('ef_second_half')), 3)}",
                        ),
                        (
                            "Aerobic decoupling",
                            format_number(
                                _opt_float(decoupling.get("aerobic_decoupling_pct")),
                                1,
                                unit="%",
                            ),
                        ),
                        ("Confidence", _str(decoupling.get("confidence")) or _EM_DASH),
                    ]
                ),
            )
        )

    source = _mapping(details.get("source"))
    quality_pairs: list[tuple[str, RenderableType]] = [
        ("Completeness", format_percent(_opt_float(details.get("data_completeness")))),
        ("HR coverage", format_percent(_opt_float(details.get("hr_coverage_ratio")))),
        ("Analysis version", _str(details.get("analysis_version")) or _EM_DASH),
        ("Source", f"{_str(source.get('name'))}: {_str(source.get('path'))}"),
    ]
    flags = _strings(details.get("data_hygiene_flags"))
    if flags:
        quality_pairs.append(("Hygiene flags", Text(", ".join(flags), style="yellow")))
    console.print(_panel("Data quality", _kv_grid(quality_pairs)))


def _profile_grid(profile: dict[str, object]) -> Table:
    pairs: list[tuple[str, RenderableType]] = [
        ("Max HR", format_number(_opt_float(profile.get("max_heart_rate")), unit=" bpm")),
        ("Resting HR", format_number(_opt_float(profile.get("resting_heart_rate")), unit=" bpm")),
    ]
    if profile.get("primary_goal") is not None:
        pairs.append(("Primary goal", _str(profile.get("primary_goal"))))
    if profile.get("weight_kg") is not None:
        pairs.append(("Weight", format_number(_opt_float(profile.get("weight_kg")), 1, " kg")))
    if profile.get("ftp_watts") is not None:
        pairs.append(("FTP", format_number(_opt_float(profile.get("ftp_watts")), unit=" W")))
    zone_bounds = _rows(profile.get("heart_rate_zones"))
    if zone_bounds:
        zone_text = "  ".join(
            f"Z{_opt_int(bound.get('zone')) or 0}: "
            f"{_opt_int(bound.get('min_bpm')) or 0}\u2013{_opt_int(bound.get('max_bpm')) or 0}"
            for bound in zone_bounds
        )
        pairs.append(("HR zones", zone_text))
    return _kv_grid(pairs)


def _sparkline(values: list[float], width: int = 40) -> str:
    """Render a compact unicode sparkline for a numeric series."""
    data = values[-width:]
    if not data:
        return ""
    low = min(data)
    high = max(data)
    span = high - low
    if span <= 0:
        return _BLOCKS[3] * len(data)
    highest = len(_BLOCKS) - 1
    return "".join(_BLOCKS[min(round((value - low) / span * highest), highest)] for value in data)


def _form_grid(form: dict[str, object], history: list[dict[str, object]]) -> RenderableType:
    pairs: list[tuple[str, RenderableType]] = [
        ("CTL (fitness)", format_number(_opt_float(form.get("ctl")), 1)),
        ("ATL (fatigue)", format_number(_opt_float(form.get("atl")), 1)),
        ("TSB (form)", format_number(_opt_float(form.get("tsb")), 1)),
        ("Classification", _str(form.get("classification")) or _EM_DASH),
        ("Load confidence", format_percent(_opt_float(form.get("load_confidence_ratio")))),
    ]
    if form.get("is_warmup"):
        pairs.append(("Note", Text("warm-up period", style="yellow")))
    grid = _kv_grid(pairs)
    tsb_series = [
        value for value in (_opt_float(point.get("tsb")) for point in history) if value is not None
    ]
    if len(tsb_series) > 1:
        sparkline = Table.grid(padding=(0, 2))
        sparkline.add_column(style="dim", no_wrap=True)
        sparkline.add_column()
        sparkline.add_row("TSB trend", Text(_sparkline(tsb_series), style="cyan"))
        return Group(grid, sparkline)
    return grid


def _latest_activity_grid(latest: dict[str, object]) -> Table:
    decoupling = _mapping(latest.get("decoupling"))
    pairs: list[tuple[str, RenderableType]] = [
        ("Sport", _sport_label(latest)),
        ("Start", format_start(latest)),
        ("Duration", format_duration(_opt_float(latest.get("duration_s")))),
        ("Distance", format_distance(_opt_float(latest.get("distance_m")))),
        ("Load", format_number(_opt_float(latest.get("load_value")))),
    ]
    if decoupling:
        pairs.append(
            (
                "Decoupling",
                format_number(_opt_float(decoupling.get("aerobic_decoupling_pct")), 1, unit="%"),
            )
        )
    pairs.append(("Activity ID", _str(latest.get("activity_id")) or _EM_DASH))
    return _kv_grid(pairs)


def render_overview(payload: object, console: Console) -> None:
    """Render the coaching overview as a sequence of panels."""
    overview = _mapping(payload)
    period = _mapping(overview.get("period"))
    period_text = f"{_str(period.get('start'))} \u2192 {_str(period.get('end'))}"
    console.print(_panel("Training overview", Text(period_text, style="bold")))

    profile = _mapping(overview.get("athlete_profile"))
    if profile:
        console.print(_panel("Athlete", _profile_grid(profile)))

    form = _mapping(overview.get("form"))
    if form:
        console.print(_panel("Form", _form_grid(form, _rows(overview.get("form_history")))))

    readiness = _mapping(overview.get("tour_readiness"))
    if readiness:
        console.print(
            _panel(
                "Tour readiness",
                _kv_grid(
                    [
                        (
                            "Rolling 7d",
                            f"{format_number(_opt_float(readiness.get('rolling_7d_km')), 1)} km",
                        ),
                        (
                            "Rolling 28d",
                            f"{format_number(_opt_float(readiness.get('rolling_28d_km')), 1)} km",
                        ),
                        (
                            "Days since 60 km ride",
                            format_number(_opt_float(readiness.get("days_since_60k"))),
                        ),
                    ]
                ),
            )
        )

    console.print(Text("Weekly summaries", style="bold cyan"))
    render_weekly(overview.get("weekly_summaries"), console)

    console.print(Text("Recent activities", style="bold cyan"))
    render_list(overview.get("recent_activities"), console)

    zones = _mapping(overview.get("zone_distribution_s"))
    if zones:
        console.print(
            _panel("Zone distribution", _zone_table(zones, total=sum(_zone_seconds(zones))))
        )

    quality = _mapping(overview.get("data_quality"))
    if quality:
        console.print(
            _panel(
                "Data quality",
                _kv_grid(
                    [
                        ("Activities", format_number(_opt_float(quality.get("activity_count")))),
                        (
                            "Mean HR coverage",
                            format_percent(_opt_float(quality.get("mean_hr_coverage_ratio"))),
                        ),
                        (
                            "Mean completeness",
                            format_percent(_opt_float(quality.get("mean_data_completeness"))),
                        ),
                        (
                            "Failed files",
                            format_number(_opt_float(quality.get("failed_file_count"))),
                        ),
                        (
                            "Repaired files",
                            format_number(_opt_float(quality.get("repaired_file_count"))),
                        ),
                        (
                            "Discarded segments",
                            format_number(_opt_float(quality.get("discarded_segment_count"))),
                        ),
                        (
                            "Duplicate activities",
                            format_number(_opt_float(quality.get("duplicate_activity_count"))),
                        ),
                    ]
                ),
            )
        )

    latest = _mapping(overview.get("latest_activity"))
    if latest:
        console.print(_panel("Latest activity", _latest_activity_grid(latest)))
