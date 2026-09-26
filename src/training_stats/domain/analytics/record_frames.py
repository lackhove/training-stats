"""Transient Polars projections from cached activity metric records."""

from __future__ import annotations

import polars as pl

from ..frames import ACTIVITY_COLUMNS, ACTIVITY_FRAME_SCHEMA
from ..metrics import AxisBin, Metric, ScalarValue, TableValue
from ..models import ActivityRecord


def activity_records_frame(
    records: tuple[ActivityRecord, ...] | list[ActivityRecord],
) -> pl.LazyFrame:
    """Project stable activity facts from metric-bearing records for analytics."""
    rows = [_activity_row(record) for record in records]
    if not rows:
        return pl.DataFrame(schema=ACTIVITY_FRAME_SCHEMA).lazy()
    return pl.DataFrame(rows, schema=ACTIVITY_FRAME_SCHEMA).select(ACTIVITY_COLUMNS).lazy()


def _activity_row(record: ActivityRecord) -> dict[str, object]:
    metrics = {metric.name: metric for metric in record.metrics}
    zones = _table_values(metrics.get("heart_rate.zone_time"))
    histogram = _table_rows(metrics.get("heart_rate.histogram"))
    bucket_avg = _table(metrics.get("heart_rate.average_by_10_minutes"))
    bucket_time = _table(metrics.get("heart_rate.observed_time_by_10_minutes"))
    bucket_rows = (
        [
            {"start_min": start, "avg_hr": average, "hr_time_s": seconds}
            for start, average, seconds in zip(
                bucket_avg[0], bucket_avg[1], bucket_time[1], strict=True
            )
        ]
        if bucket_avg is not None and bucket_time is not None
        else []
    )
    decoupling = metrics.get("endurance.aerobic_decoupling")
    return {
        "activity_id": record.activity_id,
        "start_time": record.start_time,
        "sport": record.sport,
        "sub_sport": record.sub_sport,
        "duration_s": _number(metrics.get("activity.duration")),
        "moving_time_s": _number(metrics.get("activity.moving_time")),
        "paused_time_s": _number(metrics.get("activity.paused_time")),
        "distance_m": _number(metrics.get("activity.distance")),
        "elevation_gain_m": _number(metrics.get("activity.elevation_gain")),
        "avg_hr": _number(metrics.get("activity.average_heart_rate")),
        "max_hr": _number(metrics.get("activity.maximum_heart_rate")),
        "avg_speed_mps": _number(metrics.get("activity.average_speed")),
        "max_speed_mps": _number(metrics.get("activity.maximum_speed")),
        "segment_index": record.segment_index,
        "segment_count": record.segment_count,
        "repair_flags": list(record.repair_flags),
        "source_name": record.source_name,
        "source_path": record.source_path,
        "hr_zone_1_s": _zone(zones, 0),
        "hr_zone_2_s": _zone(zones, 1),
        "hr_zone_3_s": _zone(zones, 2),
        "hr_zone_4_s": _zone(zones, 3),
        "hr_zone_5_s": _zone(zones, 4),
        "hr_coverage_ratio": _number(metrics.get("heart_rate.coverage")) or 0.0,
        "hr_histogram": histogram,
        "hr_10min_averages": bucket_rows,
        "hr_drift_bpm_per_hour": _number(metrics.get("heart_rate.drift")),
        "load_value": _number(metrics.get("training_load.hr_trimp")),
        "load_method": _metric_string(metrics.get("training_load.hr_trimp"), "method"),
        "load_confidence": _metric_string(metrics.get("training_load.hr_trimp"), "confidence"),
        "data_completeness": _number(metrics.get("data.completeness")) or 0.0,
        "ef_first_half": _number(metrics.get("endurance.efficiency_first_half")),
        "ef_second_half": _number(metrics.get("endurance.efficiency_second_half")),
        "aerobic_decoupling_pct": _number(decoupling),
        "decoupling_confidence": (
            _metric_string(decoupling, "confidence")
            or _metric_string(decoupling, "missing_reason")
            or "insufficient_duration"
        ),
        "analysis_version": record.analysis_version,
    }


def _table(metric: Metric | None) -> tuple[list[object], list[object]] | None:
    if metric is None or not isinstance(metric.value, TableValue):
        return None
    return (
        [_axis_value(value) for value in metric.value.axis.values],
        list(metric.value.values),
    )


def _table_rows(metric: Metric | None) -> list[dict[str, object]]:
    data = _table(metric)
    if data is None:
        return []
    axes, values = data
    return [
        {"bin": str(axis), "seconds": value}
        for axis, value in zip(axes, values, strict=True)
        if value is not None
    ]


def _table_values(metric: Metric | None) -> tuple[object, ...]:
    if metric is None or not isinstance(metric.value, TableValue):
        return ()
    return metric.value.values


def _axis_value(value: object) -> object:
    return value.label if isinstance(value, AxisBin) else value


def _number(metric: Metric | None) -> float | None:
    if metric is None or not isinstance(metric.value, ScalarValue):
        return None
    value = metric.value.value
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _metric_string(metric: Metric | None, field: str) -> str | None:
    if metric is None:
        return None
    value: object = getattr(metric, field)
    if value is None:
        return None
    return str(value)


def _zone(values: tuple[object, ...], index: int) -> float:
    if index >= len(values):
        return 0.0
    value = values[index]
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return float(value)
