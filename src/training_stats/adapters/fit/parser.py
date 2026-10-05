"""Concrete FIT decoder adapter backed by ``fitdecode``.

Rows carry only the fields their message provides; the declared
``FitData.SCHEMA`` fills absent columns with nulls and fixes dtypes, so
session, record and event rows may omit fields freely.
"""

from __future__ import annotations

import io
from datetime import UTC, datetime

import fitdecode
import polars as pl

from ...domain.errors import FitParseError
from ...domain.fit.types import FitData
from ...domain.models import SourceFile


class FitDecodeParser:
    """Decode FIT messages into normalized transient inputs."""

    def parse(self, content: bytes, source_file: SourceFile) -> FitData:
        try:
            reader = fitdecode.FitReader(io.BytesIO(content))
            rows: list[dict[str, object]] = []
            for frame in reader:
                if not isinstance(frame, fitdecode.FitDataMessage):
                    continue
                if frame.name == "session":
                    rows.append(_parse_session_metadata(frame))
                elif frame.name == "record":
                    row = _parse_record_row(frame)
                    if row is not None:
                        rows.append(row)
                elif frame.name == "event":
                    event = _parse_timer_event(frame)
                    if event is not None:
                        rows.append(event)
        except fitdecode.exceptions.FitError as exc:
            msg = f"failed to parse FIT file {source_file.path!r}: {type(exc).__name__}"
            raise FitParseError(msg) from exc
        except (OSError, ValueError) as exc:
            msg = f"failed to read FIT file {source_file.path!r}: {type(exc).__name__}"
            raise FitParseError(msg) from exc

        if not rows:
            msg = f"no session or record messages found in FIT file {source_file.path!r}"
            raise FitParseError(msg)
        rows = _coalesce_record_rows(rows)
        return FitData(frame=pl.DataFrame(rows, schema=FitData.SCHEMA))


def _parse_session_metadata(frame: fitdecode.FitDataMessage) -> dict[str, object]:
    row: dict[str, object] = {"message_type": "session"}
    for name in ("sport", "sub_sport"):
        value = coerce_str(frame.get_value(name, fallback=None))
        if value is not None:
            row[name] = value
    return row


def _parse_record_row(frame: fitdecode.FitDataMessage) -> dict[str, object] | None:
    timestamp = _as_datetime(frame.get_value("timestamp", fallback=None))
    if timestamp is None:
        return None
    row: dict[str, object] = {
        "message_type": "record",
        "timestamp": _ensure_utc(timestamp),
    }
    values = {
        "heart_rate": coerce_int(frame.get_value("heart_rate", fallback=None)),
        "cadence": coerce_int(frame.get_value("cadence", fallback=None)),
        "fractional_cadence": coerce_int(frame.get_value("fractional_cadence", fallback=None)),
        "power_w": coerce_int(frame.get_value("power", fallback=None)),
        "distance_m": coerce_float(frame.get_value("distance", fallback=None)),
        "altitude_m": _altitude(frame),
        "speed_mps": _speed(frame, "enhanced_speed", "speed"),
    }
    row.update({name: value for name, value in values.items() if value is not None})
    return row


def _coalesce_record_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Merge sensor fields emitted in separate record messages at one timestamp."""
    normalized: list[dict[str, object]] = []
    record_indices: dict[datetime, int] = {}
    for row in rows:
        timestamp = row.get("timestamp")
        if row.get("message_type") != "record" or not isinstance(timestamp, datetime):
            normalized.append(row)
            continue
        index = record_indices.get(timestamp)
        if index is None:
            record_indices[timestamp] = len(normalized)
            normalized.append(row.copy())
            continue
        merged = normalized[index]
        merged.update(
            {
                name: value
                for name, value in row.items()
                if name not in {"message_type", "timestamp"} and value is not None
            }
        )
    return normalized


def _parse_timer_event(frame: fitdecode.FitDataMessage) -> dict[str, object] | None:
    timestamp = _as_datetime(frame.get_value("timestamp", fallback=None))
    event = coerce_str(frame.get_value("event", fallback=None))
    event_type = coerce_str(frame.get_value("event_type", fallback=None))
    if timestamp is None or event is None:
        return None
    row: dict[str, object] = {
        "message_type": "event",
        "timestamp": _ensure_utc(timestamp),
        "event": event,
    }
    if event_type is not None:
        row["event_type"] = event_type
    return row


def _ensure_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _as_datetime(value: object) -> datetime | None:
    return value if isinstance(value, datetime) else None


def _speed(frame: fitdecode.FitDataMessage, enhanced: str, plain: str) -> float | None:
    value = coerce_float(frame.get_value(enhanced, fallback=None))
    if value is None:
        value = coerce_float(frame.get_value(plain, fallback=None))
    return value


def _altitude(frame: fitdecode.FitDataMessage) -> float | None:
    value = coerce_float(frame.get_value("enhanced_altitude", fallback=None))
    if value is None:
        value = coerce_float(frame.get_value("altitude", fallback=None))
    return value


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
