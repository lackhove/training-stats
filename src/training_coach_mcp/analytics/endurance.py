"""Aerobic decoupling (efficiency factor drift) analytics.

Measures cardiac drift across two equal moving-time halves of an activity:
EF = speed / heart_rate, and decoupling_pct = (EF_1 - EF_2) / EF_1 * 100.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import polars as pl

from .heart_rate import MAX_RECORD_INTERVAL_S

CONFIDENCE_HIGH = "high"
CONFIDENCE_INSUFFICIENT_DURATION = "insufficient_duration"
CONFIDENCE_INSUFFICIENT_COVERAGE = "insufficient_coverage"
CONFIDENCE_INVALID_EF = "invalid_ef"


@dataclass(frozen=True, slots=True)
class DecouplingResult:
    """Aerobic decoupling metrics for one activity."""

    ef_first_half: float | None
    ef_second_half: float | None
    aerobic_decoupling_pct: float | None
    confidence: str

    def to_dict(self) -> dict[str, object]:
        return {
            "ef_first_half": self.ef_first_half,
            "ef_second_half": self.ef_second_half,
            "aerobic_decoupling_pct": self.aerobic_decoupling_pct,
            "confidence": self.confidence,
        }


def _empty_decoupling(confidence: str) -> DecouplingResult:
    return DecouplingResult(
        ef_first_half=None,
        ef_second_half=None,
        aerobic_decoupling_pct=None,
        confidence=confidence,
    )


def compute_aerobic_decoupling(
    records: pl.DataFrame,
    moving_speed_threshold_mps: float = 1.0,
    min_duration_s: float = 1200.0,
    min_coverage: float = 0.80,
    max_interval_s: float = MAX_RECORD_INTERVAL_S,
    session_moving_time_s: float | None = None,
) -> DecouplingResult:
    """Compute efficiency factor drift across two equal moving-time halves.

    Intervals crossing segment breaks or capped gaps (> max_interval_s) are
    excluded. Moving intervals require speed >= moving_speed_threshold_mps.
    Returns clean None values with descriptive confidence on quality failure.
    """
    if records.height < 2 or "timestamp" not in records.columns:
        return _empty_decoupling(CONFIDENCE_INSUFFICIENT_DURATION)

    first = records["timestamp"].min()
    last = records["timestamp"].max()
    if not isinstance(first, datetime) or not isinstance(last, datetime):
        return _empty_decoupling(CONFIDENCE_INSUFFICIENT_DURATION)

    # Calculate raw dt to next record
    dt_expr = (
        (pl.col("timestamp").shift(-1) - pl.col("timestamp")).dt.total_microseconds() / 1_000_000.0
    ).alias("dt")

    frame = records.select(
        pl.col("timestamp"),
        pl.col("heart_rate")
        if "heart_rate" in records.columns
        else pl.lit(None).alias("heart_rate"),
        pl.col("speed_mps") if "speed_mps" in records.columns else pl.lit(None).alias("speed_mps"),
        dt_expr,
    )

    # Exclude segment breaks if a segment column is present
    if "segment" in records.columns:
        frame = frame.with_columns(
            (records["segment"].shift(-1) == records["segment"]).alias("same_segment")
        ).filter(pl.col("same_segment").fill_null(False))
    elif "segment_index" in records.columns:
        frame = frame.with_columns(
            (records["segment_index"].shift(-1) == records["segment_index"]).alias("same_segment")
        ).filter(pl.col("same_segment").fill_null(False))

    # Exclude intervals crossing capped gaps or out-of-order timestamps
    valid_intervals = frame.filter(
        pl.col("dt").is_not_null() & (pl.col("dt") > 0.0) & (pl.col("dt") <= max_interval_s)
    )
    if valid_intervals.is_empty():
        return _empty_decoupling(CONFIDENCE_INSUFFICIENT_DURATION)

    # Moving intervals based on speed
    moving_intervals = valid_intervals.filter(
        pl.col("speed_mps").is_not_null() & (pl.col("speed_mps") >= moving_speed_threshold_mps)
    )
    record_moving_duration = _sum_col(moving_intervals, "dt")

    total_moving_time = (
        session_moving_time_s
        if session_moving_time_s is not None and session_moving_time_s > record_moving_duration
        else record_moving_duration
    )

    # Intervals with BOTH moving speed and valid HR
    both_valid = moving_intervals.filter(
        pl.col("heart_rate").is_not_null() & (pl.col("heart_rate") > 0)
    )
    t_moving = _sum_col(both_valid, "dt")

    # Gate 1: Minimum moving duration
    if t_moving < min_duration_s:
        return _empty_decoupling(CONFIDENCE_INSUFFICIENT_DURATION)

    # Gate 2: Sensor coverage across moving time
    coverage = t_moving / total_moving_time if total_moving_time > 0.0 else 0.0
    if coverage < min_coverage:
        return _empty_decoupling(CONFIDENCE_INSUFFICIENT_COVERAGE)

    # Moving-Time Split: two equal halves by valid moving time
    half_time = t_moving / 2.0

    half1_speed_time = 0.0
    half1_hr_time = 0.0
    half1_time = 0.0

    half2_speed_time = 0.0
    half2_hr_time = 0.0
    half2_time = 0.0

    speeds: list[float] = [float(v) for v in both_valid["speed_mps"].to_list()]
    hrs: list[float] = [float(v) for v in both_valid["heart_rate"].to_list()]
    dts: list[float] = [float(v) for v in both_valid["dt"].to_list()]

    acc = 0.0
    for speed, hr, dt in zip(speeds, hrs, dts, strict=True):
        start_t = acc
        end_t = acc + dt
        acc = end_t

        if end_t <= half_time:
            half1_speed_time += speed * dt
            half1_hr_time += hr * dt
            half1_time += dt
        elif start_t >= half_time:
            half2_speed_time += speed * dt
            half2_hr_time += hr * dt
            half2_time += dt
        else:
            dt1 = half_time - start_t
            dt2 = end_t - half_time

            half1_speed_time += speed * dt1
            half1_hr_time += hr * dt1
            half1_time += dt1

            half2_speed_time += speed * dt2
            half2_hr_time += hr * dt2
            half2_time += dt2

    if half1_time <= 0.0 or half2_time <= 0.0:
        return _empty_decoupling(CONFIDENCE_INVALID_EF)

    mean_speed_1 = half1_speed_time / half1_time
    mean_hr_1 = half1_hr_time / half1_time
    mean_speed_2 = half2_speed_time / half2_time
    mean_hr_2 = half2_hr_time / half2_time

    if mean_hr_1 <= 0.0 or mean_speed_1 <= 0.0 or mean_hr_2 <= 0.0 or mean_speed_2 <= 0.0:
        return _empty_decoupling(CONFIDENCE_INVALID_EF)

    ef_1 = mean_speed_1 / mean_hr_1
    ef_2 = mean_speed_2 / mean_hr_2

    if ef_1 <= 0.0 or ef_2 <= 0.0:
        return _empty_decoupling(CONFIDENCE_INVALID_EF)

    decoupling_pct = ((ef_1 - ef_2) / ef_1) * 100.0

    return DecouplingResult(
        ef_first_half=ef_1,
        ef_second_half=ef_2,
        aerobic_decoupling_pct=decoupling_pct,
        confidence=CONFIDENCE_HIGH,
    )


def _sum_col(frame: pl.DataFrame, col: str) -> float:
    if frame.is_empty():
        return 0.0
    val = frame[col].sum()
    if isinstance(val, int | float):
        return float(val)
    return 0.0
