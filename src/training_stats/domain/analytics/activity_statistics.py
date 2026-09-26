"""Detailed, coarse activity analytics computed from transient FIT records."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..fit.types import ParsedActivity
from ..metrics import (
    Axis,
    AxisBin,
    MatrixValue,
    Metric,
    MetricKind,
    MetricScalar,
    MissingReason,
    TimeBasis,
    TripPeriod,
)
from ..settings import AthleteConfig, QualityConfig
from .intervals import merge_pause_intervals, overlap_seconds
from .metric_helpers import scalar_metric, table_metric

_MAX_INTERVAL_S = 30.0
_WINDOW_S = 300.0
_CADENCE_BANDS: tuple[tuple[str, float, float | None], ...] = (
    ("rpm_1_59", 1.0, 60.0),
    ("rpm_60_69", 60.0, 70.0),
    ("rpm_70_79", 70.0, 80.0),
    ("rpm_80_89", 80.0, 90.0),
    ("rpm_90_99", 90.0, 100.0),
    ("rpm_100_plus", 100.0, None),
)


@dataclass(frozen=True, slots=True)
class _Interval:
    timestamp: datetime
    elapsed_s: float
    moving_elapsed_s: float
    active_s: float
    moving_s: float
    speed_kmh: float | None
    heart_rate: float | None
    cadence_rpm: float | None
    distance_m: float


def _compute_activity_statistics_data(
    activity: ParsedActivity,
    athlete: AthleteConfig,
    quality: QualityConfig,
) -> dict[str, object]:
    """Compute bounded distributions, five-minute windows and quality facts."""
    records = activity.records
    raw_points = records.height
    if raw_points < 2:
        return _empty_statistics(activity, raw_points)

    timestamps = records["timestamp"].to_list()
    first = timestamps[0]
    last = timestamps[-1]
    if not isinstance(first, datetime) or not isinstance(last, datetime):
        return _empty_statistics(activity, raw_points)

    pauses = merge_pause_intervals(activity.pause_intervals, first, last)
    intervals: list[_Interval] = []
    duplicates = 0
    gaps = 0
    invalid_speed = 0
    valid_speed_samples = 0
    speed_samples_during_movement = 0
    invalid_hr = 0
    invalid_cadence = 0
    outliers = 0
    elapsed_s = 0.0
    moving_elapsed_s = 0.0
    speed_valid_s = 0.0
    hr_valid_s = 0.0
    cadence_valid_s = 0.0
    pedaling_s = 0.0
    no_pedaling_s = 0.0
    speed_values: list[tuple[float, float]] = []
    hr_values: list[tuple[float, float]] = []
    cadence_values: list[tuple[float, float]] = []
    pedaling_values: list[tuple[float, float]] = []
    zones_s = [0.0] * 5
    zone_bounds = athlete.heart_rate_zones.as_bounds()
    thresholds = quality.speed_thresholds_kmh
    speed_threshold_time = {f"{threshold:g}": 0.0 for threshold in thresholds}
    cadence_band_time = {name: 0.0 for name, _low, _high in _CADENCE_BANDS}

    for index in range(raw_points - 1):
        start = timestamps[index]
        end = timestamps[index + 1]
        if not isinstance(start, datetime) or not isinstance(end, datetime):
            continue
        dt_s = (end - start).total_seconds()
        if dt_s <= 0:
            duplicates += int(dt_s == 0)
            continue
        if dt_s > _MAX_INTERVAL_S:
            gaps += 1
        counted_s = min(dt_s, _MAX_INTERVAL_S)
        counted_end = start + timedelta(seconds=counted_s)
        active_s = max(0.0, counted_s - overlap_seconds(start, counted_end, pauses))
        if active_s <= 0:
            elapsed_s += dt_s
            continue

        raw_speed = records["speed_mps"][index]
        speed = _number(raw_speed)
        if speed is not None and (speed < 0.0 or speed > 40.0):
            outliers += 1
            speed = None
        if speed is None:
            invalid_speed += 1
        else:
            valid_speed_samples += 1
        speed_kmh = speed * 3.6 if speed is not None else None
        moving_s = (
            active_s if speed is not None and speed >= quality.moving_speed_threshold_mps else 0.0
        )

        hr_raw = records["heart_rate"][index]
        hr_number = _number(hr_raw)
        if hr_number is not None and not 25.0 <= hr_number <= 240.0:
            outliers += 1
            hr_number = None
        if hr_number is None:
            invalid_hr += 1

        cadence_raw = records["cadence"][index]
        fractional_raw = records["fractional_cadence"][index]
        cadence_number = _number(cadence_raw)
        fractional = _number(fractional_raw)
        cadence = (
            cadence_number + fractional / 128.0
            if cadence_number is not None and fractional is not None
            else cadence_number
        )
        if cadence is not None and (cadence < 0.0 or cadence > 220.0):
            outliers += 1
            cadence = None
        if cadence is None:
            invalid_cadence += 1

        distance = 0.0
        previous_distance = _number(records["distance_m"][index])
        current_distance = _number(records["distance_m"][index + 1])
        if previous_distance is not None and current_distance is not None:
            delta = current_distance - previous_distance
            if delta > 0.0:
                distance = delta * min(1.0, active_s / dt_s)

        interval = _Interval(
            timestamp=start,
            elapsed_s=elapsed_s,
            moving_elapsed_s=moving_elapsed_s,
            active_s=active_s,
            moving_s=moving_s,
            speed_kmh=speed_kmh,
            heart_rate=hr_number,
            cadence_rpm=cadence,
            distance_m=distance,
        )
        intervals.append(interval)

        if speed_kmh is not None and moving_s > 0.0:
            speed_samples_during_movement += 1
            speed_values.append((speed_kmh, moving_s))
            speed_valid_s += moving_s
            for threshold in thresholds:
                if speed_kmh >= threshold:
                    speed_threshold_time[f"{threshold:g}"] += moving_s
        if hr_number is not None:
            hr_values.append((hr_number, active_s))
            hr_valid_s += active_s
            for zone_index, (_zone_number, low, high) in enumerate(zone_bounds):
                if low <= hr_number <= high:
                    zones_s[zone_index] += active_s
                    break
        if cadence is not None:
            cadence_values.append((cadence, active_s))
            cadence_valid_s += active_s
            if cadence <= 0.0:
                no_pedaling_s += active_s
            else:
                pedaling_s += active_s
                pedaling_values.append((cadence, active_s))
                for band_name, cadence_low, cadence_high in _CADENCE_BANDS:
                    if cadence >= cadence_low and (cadence_high is None or cadence < cadence_high):
                        cadence_band_time[band_name] += active_s

        elapsed_s += dt_s
        moving_elapsed_s += moving_s

    moving_time_s = activity.moving_time_s or moving_elapsed_s
    active_time_s = max(0.0, (activity.duration_s or elapsed_s) - (activity.paused_time_s or 0.0))
    hr_coverage = min(hr_valid_s / active_time_s, 1.0) if active_time_s > 0.0 else 0.0
    speed_coverage = min(speed_valid_s / moving_time_s, 1.0) if moving_time_s > 0.0 else 0.0
    cadence_coverage = min(cadence_valid_s / moving_time_s, 1.0) if moving_time_s > 0.0 else 0.0
    cadence_present = bool(cadence_values)
    cadence_available = cadence_present and cadence_coverage >= quality.cadence_min_coverage
    cadence_reason = (
        None
        if cadence_available
        else "no_cadence_data"
        if not cadence_present
        else "insufficient_coverage"
    )

    time_windows = _time_windows(
        intervals, activity.duration_s or elapsed_s, _WINDOW_S, zone_bounds=zone_bounds
    )
    moving_windows = (
        _time_windows(
            intervals, moving_time_s, _WINDOW_S, moving_basis=True, zone_bounds=zone_bounds
        )
        if abs((activity.duration_s or elapsed_s) - moving_time_s) > 1.0
        else []
    )
    if not cadence_available:
        unavailable_reason = cadence_reason or "insufficient_coverage"
        _mask_window_cadence(time_windows, unavailable_reason)
        _mask_window_cadence(moving_windows, unavailable_reason)
    efficiency = _efficiency(
        intervals,
        athlete,
        min_duration_s=quality.decoupling_min_duration_s,
        min_coverage=quality.decoupling_min_coverage,
        moving_time_s=moving_time_s,
    )
    relationships = _relationships(time_windows)
    heatmaps = _heatmaps(intervals)
    if not cadence_available:
        heatmaps["heart_rate_cadence"] = {"valid_observation_count": 0, "bins": []}
        heatmaps["speed_cadence"] = {"valid_observation_count": 0, "bins": []}
    total_zone_s = sum(zones_s)
    zone_percent = [value / hr_valid_s * 100.0 if hr_valid_s > 0.0 else 0.0 for value in zones_s]

    cadence_pedaling_reason = None
    if cadence_available and not pedaling_values:
        cadence_pedaling_reason = "no_pedaling_intervals"
    quality_data = {
        "raw_record_points": raw_points,
        "valid_speed_intervals": len(speed_values),
        "valid_speed_samples": valid_speed_samples,
        "speed_samples_during_movement": speed_samples_during_movement,
        "valid_heart_rate_intervals": len(hr_values),
        "valid_cadence_intervals": len(cadence_values),
        "invalid_speed_intervals": invalid_speed,
        "invalid_heart_rate_intervals": invalid_hr,
        "invalid_cadence_intervals": invalid_cadence,
        "duplicate_timestamps": duplicates,
        "gap_intervals_over_30s": gaps,
        "rejected_outlier_values": outliers,
        "pause_intervals": len(pauses),
        "speed_coverage_ratio": speed_coverage,
        "heart_rate_coverage_ratio": hr_coverage,
        "cadence_coverage_ratio": cadence_coverage,
        "filters": [
            "known pauses excluded",
            "interval weights capped at 30 seconds",
            f"moving speed >= {quality.moving_speed_threshold_mps:g} m/s",
            "heart rate range 25-240 bpm",
            "cadence range 0-220 rpm",
            f"cadence statistics require coverage >= {quality.cadence_min_coverage:g}",
        ],
    }
    return {
        "speed_statistics": {
            **_statistics(speed_values, moving_time_s, "moving_time"),
            "speed_above_threshold_s": [
                {"threshold_kmh": float(threshold), "seconds": seconds}
                for threshold, seconds in speed_threshold_time.items()
            ],
            "valid_moving_time_ratio": speed_coverage,
            "valid_measurement_ratio": (
                valid_speed_samples / (valid_speed_samples + invalid_speed)
                if valid_speed_samples + invalid_speed > 0
                else 0.0
            ),
            "measurements_during_movement_ratio": (
                speed_samples_during_movement / valid_speed_samples
                if valid_speed_samples > 0
                else 0.0
            ),
        },
        "heart_rate_statistics": {
            **_statistics(hr_values, active_time_s, "active_time"),
            "coverage_ratio": hr_coverage,
            "histogram_s": [
                {"bin": label, "seconds": seconds}
                for label, seconds in _hr_histogram(hr_values).items()
            ],
        },
        "cadence_statistics": {
            "available": cadence_available,
            "reason": cadence_reason,
            "overall": _statistics(
                cadence_values if cadence_available else [],
                moving_time_s,
                "moving_time",
                empty_reason=cadence_reason or "insufficient_data",
            ),
            "pedaling": {
                **_statistics(
                    pedaling_values if cadence_available else [],
                    moving_time_s,
                    "pedaling_time",
                    empty_reason=cadence_pedaling_reason or cadence_reason or "insufficient_data",
                ),
            },
            "pedaling_time_s": pedaling_s if cadence_available else None,
            "no_pedaling_time_s": no_pedaling_s if cadence_available else None,
            "pedaling_time_ratio": (
                pedaling_s / moving_time_s if cadence_available and moving_time_s > 0.0 else None
            ),
            "coverage_ratio": cadence_coverage,
            "bands_s": (
                [{"band": band, "seconds": seconds} for band, seconds in cadence_band_time.items()]
                if cadence_available
                else []
            ),
        },
        "zone_distribution": {
            "seconds": {f"zone_{index + 1}": value for index, value in enumerate(zones_s)},
            "percent": {f"zone_{index + 1}": value for index, value in enumerate(zone_percent)},
            "valid_heart_rate_time_s": hr_valid_s,
            "unclassified_time_s": max(0.0, hr_valid_s - total_zone_s),
            "percent_denominator": "valid_heart_rate_time",
        },
        "time_windows": {
            "time_basis": "elapsed_time",
            "window_s": _WINDOW_S,
            "items": time_windows,
            "movement_time_variant": moving_windows if moving_windows else None,
        },
        "efficiency": efficiency,
        "fatigue_response": {
            "heart_rate_drift_bpm_per_hour": _heart_rate_drift(time_windows),
            **efficiency,
        },
        "relationships": relationships,
        "heatmap_data": heatmaps,
        "data_quality": quality_data,
    }


def _empty_statistics(activity: ParsedActivity, raw_points: int) -> dict[str, object]:
    return {
        "speed_statistics": _statistics([], activity.moving_time_s or 0.0, "moving_time"),
        "heart_rate_statistics": _statistics([], activity.duration_s or 0.0, "active_time"),
        "cadence_statistics": {
            "available": False,
            "reason": "no_cadence_data",
            "overall": _statistics([], activity.moving_time_s or 0.0, "moving_time"),
            "pedaling": _statistics([], activity.moving_time_s or 0.0, "pedaling_time"),
            "pedaling_time_s": 0.0,
            "no_pedaling_time_s": 0.0,
            "pedaling_time_ratio": 0.0,
            "coverage_ratio": 0.0,
            "bands_s": [{"band": name, "seconds": 0.0} for name, _low, _high in _CADENCE_BANDS],
        },
        "zone_distribution": {
            "seconds": {f"zone_{index}": 0.0 for index in range(1, 6)},
            "percent": {f"zone_{index}": 0.0 for index in range(1, 6)},
            "valid_heart_rate_time_s": 0.0,
            "unclassified_time_s": 0.0,
            "percent_denominator": "valid_heart_rate_time",
        },
        "time_windows": {"time_basis": "elapsed_time", "window_s": _WINDOW_S, "items": []},
        "efficiency": {"reason": "insufficient_data"},
        "fatigue_response": {
            "reason": "insufficient_data",
            "heart_rate_drift_bpm_per_hour": None,
        },
        "relationships": {},
        "heatmap_data": {},
        "data_quality": {
            "raw_record_points": raw_points,
            "valid_speed_intervals": 0,
            "valid_speed_samples": 0,
            "speed_samples_during_movement": 0,
            "valid_heart_rate_intervals": 0,
            "valid_cadence_intervals": 0,
            "invalid_speed_intervals": 0,
            "invalid_heart_rate_intervals": 0,
            "invalid_cadence_intervals": 0,
            "duplicate_timestamps": 0,
            "gap_intervals_over_30s": 0,
            "rejected_outlier_values": 0,
            "pause_intervals": len(activity.pause_intervals),
            "filters": [],
        },
    }


def _statistics(
    values: list[tuple[float, float]],
    denominator_s: float,
    time_basis: str,
    *,
    empty_reason: str = "insufficient_data",
) -> dict[str, object]:
    total_weight = sum(weight for _value, weight in values)
    if not values or total_weight <= 0.0:
        return {
            "mean": None,
            "median": None,
            "standard_deviation": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "minimum": None,
            "maximum": None,
            "valid_sample_count": 0,
            "valid_time_s": 0.0,
            "coverage_ratio": 0.0,
            "time_basis": time_basis,
            "reason": empty_reason,
        }
    mean = sum(value * weight for value, weight in values) / total_weight
    variance = sum(weight * (value - mean) ** 2 for value, weight in values) / total_weight
    return {
        "mean": mean,
        "median": _weighted_quantile(values, 0.5),
        "standard_deviation": math.sqrt(max(0.0, variance)),
        "p10": _weighted_quantile(values, 0.1),
        "p25": _weighted_quantile(values, 0.25),
        "p75": _weighted_quantile(values, 0.75),
        "p90": _weighted_quantile(values, 0.9),
        "minimum": min(value for value, _weight in values),
        "maximum": max(value for value, _weight in values),
        "valid_sample_count": len(values),
        "valid_time_s": total_weight,
        "coverage_ratio": min(total_weight / denominator_s, 1.0) if denominator_s > 0 else 0.0,
        "time_basis": time_basis,
        "reason": None,
    }


def _weighted_quantile(values: list[tuple[float, float]], quantile: float) -> float:
    ordered = sorted(values)
    total_weight = sum(weight for _value, weight in ordered)
    target = total_weight * quantile
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= target:
            return value
    return ordered[-1][0]


def _time_windows(
    intervals: list[_Interval],
    span_s: float,
    window_s: float,
    *,
    moving_basis: bool = False,
    zone_bounds: list[tuple[int, int, int]],
) -> list[dict[str, object]]:
    groups: dict[int, list[_Interval]] = defaultdict(list)
    for interval in intervals:
        axis = interval.moving_elapsed_s if moving_basis else interval.elapsed_s
        if moving_basis and interval.moving_s <= 0.0:
            continue
        groups[int(axis // window_s)].append(interval)
    windows: list[dict[str, object]] = []
    window_count = math.ceil(span_s / window_s) if span_s > 0.0 else 0
    for index in range(window_count):
        start_s = index * window_s
        end_s = min(span_s, start_s + window_s)
        group = groups.get(index, [])
        speed = [(item.speed_kmh, item.moving_s) for item in group if item.speed_kmh is not None]
        heart = [(item.heart_rate, item.active_s) for item in group if item.heart_rate is not None]
        cadence = [
            (item.cadence_rpm, item.active_s)
            for item in group
            if item.cadence_rpm is not None and item.cadence_rpm > 0.0
        ]
        zones = {f"zone_{zone}": 0.0 for zone in range(1, 6)}
        for value, weight in heart:
            for zone, low, high in zone_bounds:
                if low <= value <= high:
                    zones[f"zone_{zone}"] += weight
                    break
        moving = sum(item.moving_s for item in group)
        active = sum(item.active_s for item in group)
        hr_valid_s = sum(weight for _value, weight in heart)
        distance = sum(item.distance_m for item in group)
        windows.append(
            {
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": end_s - start_s,
                "moving_time_s": moving,
                "distance_m": distance,
                "moving_time_ratio": moving / (end_s - start_s) if end_s > start_s else 0.0,
                "average_speed_kmh": _weighted_mean(speed),
                "median_speed_kmh": _weighted_quantile(speed, 0.5) if speed else None,
                "speed_p25_kmh": _weighted_quantile(speed, 0.25) if speed else None,
                "speed_p75_kmh": _weighted_quantile(speed, 0.75) if speed else None,
                "average_heart_rate_bpm": _weighted_mean(heart),
                "median_heart_rate_bpm": _weighted_quantile(heart, 0.5) if heart else None,
                "heart_rate_p25_bpm": _weighted_quantile(heart, 0.25) if heart else None,
                "heart_rate_p75_bpm": _weighted_quantile(heart, 0.75) if heart else None,
                "heart_rate_valid_time_s": hr_valid_s,
                "heart_rate_zone_seconds": zones,
                "heart_rate_zone_percent": {
                    key: value / hr_valid_s * 100.0 if hr_valid_s > 0.0 else 0.0
                    for key, value in zones.items()
                },
                "heart_rate_zone_percent_denominator": "valid_heart_rate_time",
                "average_pedaling_cadence_rpm": _weighted_mean(cadence),
                "median_pedaling_cadence_rpm": (
                    _weighted_quantile(cadence, 0.5) if cadence else None
                ),
                "cadence_coverage_ratio": (
                    sum(weight for _value, weight in cadence) / moving if moving > 0.0 else 0.0
                ),
                "data_validity": {
                    "speed_intervals": len(speed),
                    "heart_rate_intervals": len(heart),
                    "cadence_intervals": len(cadence),
                    "active_time_s": active,
                },
            }
        )
    return windows


def _mask_window_cadence(windows: list[dict[str, object]], reason: str) -> None:
    for window in windows:
        window["average_pedaling_cadence_rpm"] = None
        window["median_pedaling_cadence_rpm"] = None
        window["cadence_reason"] = reason


def _efficiency(
    intervals: list[_Interval],
    athlete: AthleteConfig,
    *,
    min_duration_s: float,
    min_coverage: float,
    moving_time_s: float,
) -> dict[str, object]:
    pairs: list[tuple[float, float, float]] = []
    for item in intervals:
        if item.speed_kmh is None or item.heart_rate is None or item.moving_s <= 0.0:
            continue
        pairs.append((item.speed_kmh, item.heart_rate, item.moving_s))
    valid_s = sum(duration for _speed, _hr, duration in pairs)
    coverage = valid_s / moving_time_s if moving_time_s > 0.0 else 0.0
    if valid_s < min_duration_s:
        return {
            "speed_per_heart_rate": None,
            "speed_per_heart_rate_reserve": None,
            "first_half_average_speed_kmh": None,
            "second_half_average_speed_kmh": None,
            "first_half_average_heart_rate_bpm": None,
            "second_half_average_heart_rate_bpm": None,
            "ef_first_half": None,
            "ef_second_half": None,
            "aerobic_decoupling_pct": None,
            "valid_time_s": valid_s,
            "coverage_ratio": coverage,
            "reason": "insufficient_duration",
            "formula": "speed_kmh / heart_rate_bpm; halves split by valid moving time",
        }
    if coverage < min_coverage:
        return {
            "speed_per_heart_rate": None,
            "speed_per_heart_rate_reserve": None,
            "first_half_average_speed_kmh": None,
            "second_half_average_speed_kmh": None,
            "first_half_average_heart_rate_bpm": None,
            "second_half_average_heart_rate_bpm": None,
            "ef_first_half": None,
            "ef_second_half": None,
            "aerobic_decoupling_pct": None,
            "valid_time_s": valid_s,
            "coverage_ratio": coverage,
            "reason": "insufficient_coverage",
            "formula": "speed_kmh / heart_rate_bpm; halves split by valid moving time",
        }
    mean_speed = sum(speed * duration for speed, _hr, duration in pairs) / valid_s
    mean_hr = sum(hr * duration for _speed, hr, duration in pairs) / valid_s
    hr_reserve = athlete.max_heart_rate - athlete.resting_heart_rate
    reserve_ratio = (mean_hr - athlete.resting_heart_rate) / hr_reserve if hr_reserve > 0 else 0.0
    first_half_speed, first_half_hr, first_half, second_half_speed, second_half_hr, second_half = (
        _half_efficiency(pairs, valid_s)
    )
    decoupling = (
        (first_half - second_half) / first_half * 100.0
        if first_half is not None and second_half is not None and first_half > 0.0
        else None
    )
    return {
        "speed_per_heart_rate": mean_speed / mean_hr if mean_hr > 0.0 else None,
        "speed_per_heart_rate_reserve": (
            mean_speed / reserve_ratio if reserve_ratio > 0.0 else None
        ),
        "first_half_average_speed_kmh": first_half_speed,
        "second_half_average_speed_kmh": second_half_speed,
        "first_half_average_heart_rate_bpm": first_half_hr,
        "second_half_average_heart_rate_bpm": second_half_hr,
        "ef_first_half": first_half,
        "ef_second_half": second_half,
        "aerobic_decoupling_pct": decoupling,
        "valid_time_s": valid_s,
        "coverage_ratio": coverage,
        "reason": None,
        "formula": "EF = time-weighted speed_kmh / heart_rate_bpm; decoupling = (EF1-EF2)/EF1*100",
        "time_basis": "valid moving time",
        "heart_rate_reserve_formula": "(mean HR - resting HR) / (maximum HR - resting HR)",
    }


def _half_efficiency(
    pairs: list[tuple[float, float, float]], total_s: float
) -> tuple[
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
]:
    half = total_s / 2.0
    values = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    cursor = 0.0
    for speed, hr, duration in pairs:
        end = cursor + duration
        first_s = max(0.0, min(end, half) - cursor)
        second_s = max(0.0, end - max(cursor, half))
        if first_s > 0.0:
            values[0][0] += speed * first_s
            values[0][1] += hr * first_s
            values[0][2] += first_s
        if second_s > 0.0:
            values[1][0] += speed * second_s
            values[1][1] += hr * second_s
            values[1][2] += second_s
        cursor = end
    summaries = [
        (
            speed / duration,
            hr / duration,
            speed / hr,
        )
        if duration > 0.0 and hr > 0.0
        else (None, None, None)
        for speed, hr, duration in values
    ]
    return (*summaries[0], *summaries[1])


def _relationships(windows: list[dict[str, object]]) -> dict[str, object]:
    pairs: dict[str, list[tuple[float, float]]] = {
        "speed_heart_rate": [],
        "speed_cadence": [],
        "heart_rate_cadence": [],
    }
    for window in windows:
        speed = _number(window.get("average_speed_kmh"))
        hr = _number(window.get("average_heart_rate_bpm"))
        cadence = _number(window.get("average_pedaling_cadence_rpm"))
        if speed is not None and hr is not None:
            pairs["speed_heart_rate"].append((speed, hr))
        if speed is not None and cadence is not None:
            pairs["speed_cadence"].append((speed, cadence))
        if hr is not None and cadence is not None:
            pairs["heart_rate_cadence"].append((hr, cadence))
    relationships: dict[str, object] = {}
    for name, values in pairs.items():
        if name != "speed_heart_rate" and not values:
            continue
        spearman = _spearman(values) if len(values) >= 3 else None
        pearson = _pearson(values) if len(values) >= 3 else None
        reason = (
            "insufficient_windows"
            if len(values) < 3
            else "zero_variance"
            if spearman is None and pearson is None
            else None
        )
        relationships[name] = {
            "spearman": spearman,
            "pearson": pearson,
            "window_count": len(values),
            "filters": ["valid five-minute windows", "moving intervals", "pauses excluded"],
            "reason": reason,
            "causality_note": "Correlation does not establish causation.",
        }
    return relationships


def _heatmaps(intervals: list[_Interval]) -> dict[str, object]:
    cells: dict[str, dict[tuple[int, int], list[float]]] = {
        "heart_rate_speed": defaultdict(lambda: [0.0, 0.0]),
        "heart_rate_cadence": defaultdict(lambda: [0.0, 0.0]),
        "speed_cadence": defaultdict(lambda: [0.0, 0.0]),
    }
    for item in intervals:
        if item.heart_rate is not None and item.speed_kmh is not None and item.moving_s > 0.0:
            key = (int(item.heart_rate // 10) * 10, int(item.speed_kmh // 5) * 5)
            cells["heart_rate_speed"][key][0] += item.moving_s
            cells["heart_rate_speed"][key][1] += 1.0
        if item.heart_rate is not None and item.cadence_rpm is not None and item.cadence_rpm > 0:
            key = (int(item.heart_rate // 10) * 10, int(item.cadence_rpm // 10) * 10)
            cells["heart_rate_cadence"][key][0] += item.active_s
            cells["heart_rate_cadence"][key][1] += 1.0
        if item.speed_kmh is not None and item.cadence_rpm is not None and item.cadence_rpm > 0:
            key = (int(item.speed_kmh // 5) * 5, int(item.cadence_rpm // 10) * 10)
            cells["speed_cadence"][key][0] += item.moving_s
            cells["speed_cadence"][key][1] += 1.0
    return {
        "time_basis": "moving_time; HR-cadence uses active time",
        "filters": ["pauses excluded", "invalid values excluded", "duration-weighted"],
        "heart_rate_speed": _cells(cells["heart_rate_speed"], 10, 5),
        "heart_rate_cadence": _cells(cells["heart_rate_cadence"], 10, 10),
        "speed_cadence": _cells(cells["speed_cadence"], 5, 10),
    }


def _cells(
    values: dict[tuple[int, int], list[float]], x_width: int, y_width: int
) -> dict[str, object]:
    return {
        "valid_observation_count": int(sum(value[1] for value in values.values())),
        "bins": [
            {
                "x_min": x,
                "x_max": x + x_width,
                "y_min": y,
                "y_max": y + y_width,
                "duration_s": value[0],
                "observation_count": int(value[1]),
            }
            for (x, y), value in sorted(values.items())
        ],
    }


def _hr_histogram(values: list[tuple[float, float]]) -> dict[str, float]:
    histogram: dict[str, float] = defaultdict(float)
    for value, duration in values:
        low = int(value // 5) * 5
        histogram[f"{low}-{low + 4}"] += duration
    return dict(sorted(histogram.items()))


def _heart_rate_drift(windows: list[dict[str, object]]) -> float | None:
    points: list[tuple[float, float]] = []
    for window in windows:
        start = _number(window.get("start_s"))
        duration = _number(window.get("duration_s"))
        heart_rate = _number(window.get("average_heart_rate_bpm"))
        if start is not None and duration is not None and heart_rate is not None:
            points.append((start + duration / 2.0, heart_rate))
    if len(points) < 2:
        return None
    mean_x = sum(x for x, _y in points) / len(points)
    mean_y = sum(y for _x, y in points) / len(points)
    variance = sum((x - mean_x) ** 2 for x, _y in points)
    if variance <= 0.0:
        return None
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in points)
    return covariance / variance * 3600.0


def _weighted_mean(values: list[tuple[float, float]]) -> float | None:
    total = sum(weight for _value, weight in values)
    return sum(value * weight for value, weight in values) / total if total > 0.0 else None


def _spearman(values: list[tuple[float, float]]) -> float | None:
    return _pearson(
        list(zip(_ranks([x for x, _y in values]), _ranks([y for _x, y in values]), strict=True))
    )


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _pearson(values: list[tuple[float, float]]) -> float | None:
    if len(values) < 2:
        return None
    mean_x = sum(x for x, _y in values) / len(values)
    mean_y = sum(y for _x, y in values) / len(values)
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in values)
    variance_x = sum((x - mean_x) ** 2 for x, _y in values)
    variance_y = sum((y - mean_y) ** 2 for _x, y in values)
    denominator = math.sqrt(variance_x * variance_y)
    return covariance / denominator if denominator > 0.0 else None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def compute_activity_statistics(
    activity: ParsedActivity,
    athlete: AthleteConfig,
    quality: QualityConfig,
) -> tuple[Metric, ...]:
    """Return detailed distribution, window, relationship, and heatmap metrics."""
    raw = _compute_activity_statistics_data(activity, athlete, quality)
    end = (
        activity.start_time + timedelta(seconds=activity.duration_s)
        if activity.duration_s is not None
        else None
    )
    period = TripPeriod(activity_id=activity.activity_id, start=activity.start_time, end=end)
    return _statistics_metrics(raw, period)


def _statistics_metrics(data: dict[str, object], period: TripPeriod) -> tuple[Metric, ...]:
    metrics: list[Metric] = []

    def walk(value: object, path: tuple[str, ...], basis: TimeBasis | None = None) -> None:
        if isinstance(value, dict):
            current_basis = _basis(value.get("time_basis")) or basis
            bins = value.get("bins")
            if isinstance(bins, list) and bins and all(isinstance(item, dict) for item in bins):
                _append_heatmap(metrics, path, bins, period, current_basis)
                for key, child in value.items():
                    if key not in {"bins", "time_basis", "filters"}:
                        walk(child, (*path, key), current_basis)
                return
            scalar_map = {
                key: child
                for key, child in value.items()
                if key not in {"reason", "time_basis", "filters", "formula", "repair_flags"}
                and isinstance(child, (str, int, float, bool))
            }
            if (
                scalar_map
                and len(scalar_map)
                == len(
                    [
                        key
                        for key in value
                        if key not in {"reason", "time_basis", "filters", "formula", "repair_flags"}
                    ]
                )
                and _is_axis_mapping(path)
            ):
                _append_table(
                    metrics,
                    path,
                    tuple(scalar_map),
                    tuple(scalar_map.values()),
                    period,
                    current_basis,
                )
                return
            local_reason = _reason(value.get("reason"))
            for key, child in value.items():
                if key in {
                    "reason",
                    "time_basis",
                    "filters",
                    "formula",
                    "repair_flags",
                    "causality_note",
                }:
                    continue
                if child is None:
                    _append_scalar(metrics, (*path, key), None, period, current_basis, local_reason)
                else:
                    walk(child, (*path, key), current_basis)
            return
        if isinstance(value, list):
            if not value:
                return
            if all(isinstance(item, dict) for item in value):
                _append_rows(metrics, path, value, period, basis)
                return
            if all(isinstance(item, (str, int, float, bool)) for item in value):
                axis = Axis(name="index", values=tuple(range(len(value))))
                _append_table(metrics, path, axis.values, tuple(value), period, basis)
            return
        if value is None or isinstance(value, (str, int, float, bool)):
            _append_scalar(metrics, path, value, period, basis, MissingReason.NO_DATA)

    walk(data, ("activity", "statistics"))
    return tuple(metrics)


def _append_rows(
    metrics: list[Metric],
    path: tuple[str, ...],
    rows: list[object],
    period: TripPeriod,
    basis: TimeBasis | None,
) -> None:
    entries = [row for row in rows if isinstance(row, dict)]
    if not entries:
        return
    axis_key = next(
        (
            key
            for key in ("bin", "band", "threshold_kmh", "start_min", "start_s", "date")
            if key in entries[0]
        ),
        None,
    )
    if axis_key is None:
        for index, row in enumerate(entries):
            for key, value in row.items():
                if isinstance(value, (dict, list)):
                    _statistics_metrics_for_nested(
                        metrics, (*path, str(index), key), value, period, basis
                    )
        return
    axis_values = tuple(_axis_value(row[axis_key]) for row in entries)
    axis = Axis(name=axis_key, values=axis_values, unit=_unit((axis_key,)))
    excluded = {axis_key}
    for key in ("x_min", "x_max", "y_min", "y_max"):
        excluded.add(key)
    all_keys = set().union(*(row.keys() for row in entries)) - excluded
    for key in sorted(all_keys):
        values = tuple(row.get(key) for row in entries)
        if all(value is None or isinstance(value, (str, int, float, bool)) for value in values):
            typed_values = tuple(
                value if isinstance(value, (str, int, float, bool)) else None for value in values
            )
            metrics.append(
                table_metric(
                    ".".join((*path, key)),
                    period,
                    axis,
                    typed_values,
                    unit=_unit((*path, key)),
                    kind=MetricKind.CALCULATED,
                    time_basis=basis,
                )
            )
        else:
            for index, row in enumerate(entries):
                if isinstance(row.get(key), dict):
                    walk_data = row[key]
                    _statistics_metrics_for_nested(
                        metrics, (*path, key, str(index)), walk_data, period, basis
                    )


def _statistics_metrics_for_nested(
    metrics: list[Metric],
    path: tuple[str, ...],
    value: object,
    period: TripPeriod,
    basis: TimeBasis | None,
) -> None:
    if isinstance(value, dict):
        if _is_axis_mapping(path) and all(
            isinstance(item, (str, int, float, bool)) for item in value.values()
        ):
            _append_table(metrics, path, tuple(value), tuple(value.values()), period, basis)
            return
        for key, child in value.items():
            if key not in {"reason", "filters", "time_basis"}:
                _statistics_metrics_for_nested(metrics, (*path, str(key)), child, period, basis)
    elif isinstance(value, list):
        if value and all(isinstance(item, dict) for item in value):
            _append_rows(metrics, path, value, period, basis)


def _append_heatmap(
    metrics: list[Metric],
    path: tuple[str, ...],
    bins: list[dict[str, object]],
    period: TripPeriod,
    basis: TimeBasis | None,
) -> None:
    x_bounds = sorted(
        {
            (low, high)
            for row in bins
            if (low := _number(row.get("x_min"))) is not None
            and (high := _number(row.get("x_max"))) is not None
        }
    )
    y_bounds = sorted(
        {
            (low, high)
            for row in bins
            if (low := _number(row.get("y_min"))) is not None
            and (high := _number(row.get("y_max"))) is not None
        }
    )
    if not x_bounds or not y_bounds:
        return
    x_bins = tuple(AxisBin(f"{low:g}\u2013{high:g}", low, high) for low, high in x_bounds)
    y_bins = tuple(AxisBin(f"{low:g}\u2013{high:g}", low, high) for low, high in y_bounds)
    x_index = {bounds: index for index, bounds in enumerate(x_bounds)}
    y_index = {bounds: index for index, bounds in enumerate(y_bounds)}
    for measure in ("duration_s", "observation_count"):
        cells: list[list[MetricScalar | None]] = [[0 for _ in x_bins] for _ in y_bins]
        for row in bins:
            x_low, x_high = _number(row.get("x_min")), _number(row.get("x_max"))
            y_low, y_high = _number(row.get("y_min")), _number(row.get("y_max"))
            value = row.get(measure)
            if (
                x_low is not None
                and x_high is not None
                and y_low is not None
                and y_high is not None
                and isinstance(value, int | float)
            ):
                x = (x_low, x_high)
                y = (y_low, y_high)
                if x in x_index and y in y_index:
                    cells[y_index[y]][x_index[x]] = value
        metrics.append(
            Metric(
                name=".".join((*path, measure)),
                period=period,
                value=MatrixValue(
                    x_axis=Axis(name="x", values=x_bins),
                    y_axis=Axis(name="y", values=y_bins),
                    values=tuple(tuple(row) for row in cells),
                ),
                kind=MetricKind.CALCULATED,
                unit="s" if measure == "duration_s" else "observations",
                time_basis=basis,
            )
        )


def _append_table(
    metrics: list[Metric],
    path: tuple[str, ...],
    axis_values: tuple[object, ...],
    values: tuple[object, ...],
    period: TripPeriod,
    basis: TimeBasis | None,
) -> None:
    name = ".".join(path)
    typed_values: tuple[MetricScalar | None, ...] = tuple(
        value if isinstance(value, (str, int, float, bool)) else None for value in values
    )
    metrics.append(
        table_metric(
            name,
            period,
            Axis(
                name=path[-1] if path else "index",
                values=tuple(_axis_value(v) for v in axis_values),
                unit=_unit(path),
            ),
            typed_values,
            unit=_unit(path),
            kind=MetricKind.CALCULATED,
            time_basis=basis,
        )
    )


def _append_scalar(
    metrics: list[Metric],
    path: tuple[str, ...],
    value: object,
    period: TripPeriod,
    basis: TimeBasis | None,
    missing_reason: MissingReason,
) -> None:
    if value is None or isinstance(value, (str, int, float, bool)):
        metrics.append(
            scalar_metric(
                ".".join(path),
                period,
                value,
                unit=_unit(path),
                kind=(
                    MetricKind.INTERPRETED
                    if isinstance(value, (str, bool))
                    else MetricKind.CALCULATED
                ),
                missing_reason=missing_reason,
                time_basis=basis,
            )
        )


def _axis_value(value: object) -> str | int | float:
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return value
    return str(value)


def _unit(path: tuple[str, ...]) -> str | None:
    name = ".".join(path).casefold()
    if name.endswith("_s") or "seconds" in name or "time" in name:
        return "s"
    if "heart_rate" in name or name.endswith("_bpm") or "bpm" in name:
        return "bpm"
    if "cadence" in name or "rpm" in name:
        return "rpm"
    if "speed" in name or "kmh" in name:
        return "km/h"
    if "coverage" in name or "ratio" in name:
        return "ratio"
    if "percent" in name or "_pct" in name:
        return "%"
    if "count" in name or "observations" in name or "intervals" in name:
        return "count"
    return None


def _is_axis_mapping(path: tuple[str, ...]) -> bool:
    return bool(path) and path[-1] in {"seconds", "percent", "histogram_s", "bands_s"}


def _basis(value: object) -> TimeBasis | None:
    if not isinstance(value, str):
        return None
    for basis in TimeBasis:
        if value == basis.value:
            return basis
    return None


def _reason(value: object) -> MissingReason:
    if isinstance(value, str):
        try:
            return MissingReason(value)
        except ValueError:
            return MissingReason.INSUFFICIENT_DATA
    return MissingReason.NO_DATA
