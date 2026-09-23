"""Estimated training load based on heart-rate zones (Banister-style TRIMP).

Method ``hr_trimp``: for each zone, a Banister weighting factor is derived
from the zone's midpoint heart-rate ratio
``HRr = (HR_mid - HRrest) / (HRmax - HRrest)``::

    weight_k = 0.64 * exp(1.92 * HRr_k)          (male coefficients)
    load      = sum_k (minutes_in_zone_k * weight_k)

The result is an *estimate*: it depends on configured HR zones and the
fraction of the activity with valid HR data (see ``load_confidence``).
"""

from __future__ import annotations

import math

from .heart_rate import ZoneBounds, ZoneResult

LOAD_METHOD = "hr_trimp"
LOAD_VERSION = "1"
# Banister male coefficients; documented as part of the method definition.
_TRIMP_COEFFICIENT = 0.64
_TRIMP_EXPONENT = 1.92

_CONFIDENCE_HIGH = 0.9
_CONFIDENCE_MEDIUM = 0.5


def zone_trimp_weight(zone: ZoneBounds, max_hr: int, resting_hr: int) -> float:
    """Banister weighting factor for a zone's midpoint heart rate."""
    midpoint = (zone.min_bpm + zone.max_bpm) / 2.0
    hr_reserve = max_hr - resting_hr
    if hr_reserve <= 0:
        return 0.0
    ratio = (midpoint - resting_hr) / hr_reserve
    ratio = max(0.0, min(ratio, 1.2))
    return _TRIMP_COEFFICIENT * math.exp(_TRIMP_EXPONENT * ratio)


def hr_trimp_load(
    zones_result: ZoneResult, zones: list[ZoneBounds], max_hr: int, resting_hr: int
) -> float | None:
    """Estimated TRIMP-like load from time-in-zone, or ``None`` without HR data."""
    if zones_result.valid_hr_time_s <= 0 or not zones:
        return None
    load = 0.0
    for seconds, zone in zip(zones_result.zone_seconds, zones, strict=True):
        weight = zone_trimp_weight(zone, max_hr, resting_hr)
        load += (seconds / 60.0) * weight
    return load


def load_confidence(hr_coverage_ratio: float) -> str:
    """Map HR data coverage to a confidence label."""
    if hr_coverage_ratio >= _CONFIDENCE_HIGH:
        return "high"
    if hr_coverage_ratio >= _CONFIDENCE_MEDIUM:
        return "medium"
    return "low"
