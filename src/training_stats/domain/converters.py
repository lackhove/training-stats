"""Safe coercions for values materialized from domain Polars frames."""

from __future__ import annotations


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
