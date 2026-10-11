"""Omit activity analysis whose required record quantities were not measured."""

from __future__ import annotations

from collections.abc import Collection, Sequence

from ..metrics import Metric

_HEART_RATE = frozenset({"heart_rate"})
_CADENCE = frozenset({"cadence"})
_POWER = frozenset({"power"})
_SPEED = frozenset({"speed"})
_DISTANCE = frozenset({"distance"})
_ELEVATION = frozenset({"elevation"})
_SPEED_AND_HEART_RATE = frozenset({"speed", "heart_rate"})

_PAIR_REQUIREMENTS: dict[str, frozenset[str]] = {
    "speed_heart_rate": _SPEED_AND_HEART_RATE,
    "heart_rate_speed": _SPEED_AND_HEART_RATE,
    "speed_cadence": frozenset({"speed", "cadence"}),
    "heart_rate_cadence": frozenset({"heart_rate", "cadence"}),
    "power_heart_rate": frozenset({"power", "heart_rate"}),
    "power_speed": frozenset({"power", "speed"}),
    "power_cadence": frozenset({"power", "cadence"}),
}

_DISTRIBUTION_DIAGNOSTICS = {"valid_sample_count", "valid_time_s", "coverage_ratio"}
_SPEED_DIAGNOSTICS = _DISTRIBUTION_DIAGNOSTICS | {
    "valid_moving_time_ratio",
    "valid_measurement_ratio",
    "measurements_during_movement_ratio",
}
_SENSOR_WINDOWS = {
    "average_heart_rate_bpm": _HEART_RATE,
    "heart_rate_p25_bpm": _HEART_RATE,
    "heart_rate_p75_bpm": _HEART_RATE,
    "median_heart_rate_bpm": _HEART_RATE,
    "average_pedaling_cadence_rpm": _CADENCE,
    "median_pedaling_cadence_rpm": _CADENCE,
    "average_power_w": _POWER,
    "average_speed_kmh": _SPEED,
    "median_speed_kmh": _SPEED,
    "speed_p25_kmh": _SPEED,
    "speed_p75_kmh": _SPEED,
    "distance_m": _DISTANCE,
}


def omit_unmeasured_metrics(
    metrics: Sequence[Metric], present_quantities: Collection[str]
) -> tuple[Metric, ...]:
    """Keep diagnostics and metrics whose required record quantities are present."""
    present = frozenset(present_quantities)
    return tuple(
        metric
        for metric in metrics
        if (required := _required_quantities(metric.name)) is None
        or any(requirement <= present for requirement in required)
    )


def _required_quantities(name: str) -> tuple[frozenset[str], ...] | None:
    if name == "activity.distance":
        return (_DISTANCE,)
    if name == "activity.elevation_gain":
        return (_ELEVATION,)
    if name in {"activity.average_heart_rate", "activity.maximum_heart_rate"}:
        return (_HEART_RATE,)
    if name == "activity.maximum_speed":
        return (_SPEED,)
    if name == "activity.average_speed":
        return (_SPEED, _DISTANCE)
    if name.startswith("heart_rate."):
        return None if name == "heart_rate.coverage" else (_HEART_RATE,)
    if name == "training_load.hr_trimp":
        return (_HEART_RATE,)
    if name.startswith("activity.statistics.data_quality."):
        return None
    if name.startswith("activity.statistics.speed_statistics."):
        return _family_requirement(
            name,
            "activity.statistics.speed_statistics.",
            _SPEED,
            diagnostics=_SPEED_DIAGNOSTICS,
        )
    if name.startswith("activity.statistics.heart_rate_statistics."):
        return _family_requirement(name, "activity.statistics.heart_rate_statistics.", _HEART_RATE)
    if name.startswith("activity.statistics.cadence_statistics."):
        return _cadence_requirement(name)
    if name.startswith("activity.statistics.power_statistics."):
        return _power_requirement(name)
    if name.startswith("activity.statistics.zone_distribution."):
        return None if name.endswith(".valid_heart_rate_time_s") else (_HEART_RATE,)
    if name.startswith("activity.statistics.time_windows."):
        field = name.removeprefix("activity.statistics.time_windows.")
        if field in {"window_s", "movement_time_variant"}:
            return None
        if field.startswith(("items.", "movement_time_variant.")):
            field = field.split(".", 1)[1]
        if field in {
            "cadence_coverage_ratio",
            "cadence_reason",
            "heart_rate_valid_time_s",
            "heart_rate_zone_percent_denominator",
            "power_coverage_ratio",
            "power_reason",
        }:
            return None
        requirement = _SENSOR_WINDOWS.get(field)
        return (requirement,) if requirement is not None else None
    if name.startswith("activity.statistics.efficiency."):
        return (
            None
            if name.endswith((".valid_time_s", ".coverage_ratio"))
            else (_SPEED_AND_HEART_RATE,)
        )
    if name.startswith("activity.statistics.fatigue_response."):
        return (_HEART_RATE,)
    if name.startswith("activity.statistics.relationships."):
        family = name.removeprefix("activity.statistics.relationships.").split(".", 1)[0]
        requirement = _PAIR_REQUIREMENTS.get(family)
        return (requirement,) if requirement is not None else None
    if name.startswith("activity.statistics.heatmap_data."):
        family = name.removeprefix("activity.statistics.heatmap_data.").split(".", 1)[0]
        requirement = _PAIR_REQUIREMENTS.get(family)
        return (requirement,) if requirement is not None else None
    return None


def _family_requirement(
    name: str,
    prefix: str,
    requirement: frozenset[str],
    *,
    diagnostics: set[str] = _DISTRIBUTION_DIAGNOSTICS,
) -> tuple[frozenset[str], ...] | None:
    field = name.removeprefix(prefix).rsplit(".", 1)[-1]
    return None if field in diagnostics else (requirement,)


def _cadence_requirement(name: str) -> tuple[frozenset[str], ...] | None:
    field = name.removeprefix("activity.statistics.cadence_statistics.")
    if field == "available" or field == "coverage_ratio":
        return None
    if field.startswith(("overall.", "pedaling.")):
        return _family_requirement(name, "activity.statistics.cadence_statistics.", _CADENCE)
    return (_CADENCE,)


def _power_requirement(name: str) -> tuple[frozenset[str], ...] | None:
    field = name.removeprefix("activity.statistics.power_statistics.")
    if field in {"available", "coverage_ratio"}:
        return None
    if field.startswith("overall."):
        return _family_requirement(name, "activity.statistics.power_statistics.", _POWER)
    return (_POWER,)
