"""Field normalization helpers for FIT values."""

from __future__ import annotations


def coerce_float(value: object) -> float | None:
    """Convert a FIT field value to ``float``, mapping invalid values to ``None``."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        result = float(value)
        return None if result != result else result  # filter NaN
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def coerce_int(value: object) -> int | None:
    """Convert a FIT field value to ``int``, mapping invalid values to ``None``."""
    as_float = coerce_float(value)
    if as_float is None:
        return None
    return round(as_float)


def coerce_str(value: object) -> str | None:
    """Convert a FIT field value to a cleaned lowercase string or ``None``."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower()
    return cleaned or None
