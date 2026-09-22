"""Shared coercion helpers for values read back from Polars/JSON payloads.

Polars ``to_dicts()`` yields ``dict[str, object]``; these helpers convert
those untyped values safely without lying about types.
"""

from __future__ import annotations

import json
from typing import Any

from .analytics.heart_rate import HrBucketAvg


def optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


def optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return int(value)


def int_value(value: object) -> int:
    result = optional_int(value)
    return result if result is not None else 0


def optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def required_float(value: object) -> float:
    return optional_float(value) or 0.0


def flags_to_json(flags: tuple[str, ...]) -> str | None:
    return json.dumps(list(flags)) if flags else None


def flags_from_json(value: object) -> tuple[str, ...]:
    if not isinstance(value, str) or not value:
        return ()
    try:
        raw = json.loads(value)
    except json.JSONDecodeError:
        return ()
    if not isinstance(raw, list):
        return ()
    return tuple(str(flag) for flag in raw)


def histogram_to_json(histogram: dict[str, float]) -> str:
    return json.dumps({bin_key: round(seconds, 3) for bin_key, seconds in histogram.items()})


def histogram_from_json(value: object) -> dict[str, float]:
    if not isinstance(value, str) or not value:
        return {}
    try:
        raw = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(key): required_float(seconds) for key, seconds in raw.items()}


def buckets_to_json(buckets: tuple[HrBucketAvg, ...]) -> str:
    return json.dumps([bucket.to_dict() for bucket in buckets])


def buckets_from_json(value: object) -> tuple[HrBucketAvg, ...]:
    if not isinstance(value, str) or not value:
        return ()
    try:
        raw = json.loads(value)
    except json.JSONDecodeError:
        return ()
    if not isinstance(raw, list):
        return ()
    buckets: list[HrBucketAvg] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        buckets.append(
            HrBucketAvg(
                start_min=int_value(entry.get("start_min")),
                avg_hr=required_float(entry.get("avg_hr")),
                hr_time_s=required_float(entry.get("hr_time_s")),
            )
        )
    return tuple(buckets)


def buckets_as_dicts(value: object) -> tuple[dict[str, Any], ...]:
    """Parse stored 10-minute buckets as plain dicts (activity details payload)."""
    return tuple(bucket.to_dict() for bucket in buckets_from_json(value))
