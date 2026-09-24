"""Concrete FIT decoder adapter backed by ``fitdecode``."""

from __future__ import annotations

import io
from datetime import UTC, datetime

import fitdecode
import polars as pl

from ...domain.errors import FitParseError
from ...domain.fit.types import FitData
from ...domain.models import SourceFile
from .normalize import coerce_float, coerce_int, coerce_str


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
        return FitData(frame=pl.DataFrame(rows, schema=FitData.SCHEMA))


def _parse_session_metadata(frame: fitdecode.FitDataMessage) -> dict[str, object]:
    return {
        "message_type": "session",
        "timestamp": None,
        "heart_rate": None,
        "distance_m": None,
        "altitude_m": None,
        "speed_mps": None,
        "sport": coerce_str(frame.get_value("sport", fallback=None)),
        "sub_sport": coerce_str(frame.get_value("sub_sport", fallback=None)),
        "event": None,
        "event_type": None,
    }


def _parse_record_row(frame: fitdecode.FitDataMessage) -> dict[str, object] | None:
    timestamp = _as_datetime(frame.get_value("timestamp", fallback=None))
    if timestamp is None:
        return None
    return {
        "message_type": "record",
        "timestamp": _ensure_utc(timestamp),
        "heart_rate": coerce_int(frame.get_value("heart_rate", fallback=None)),
        "distance_m": coerce_float(frame.get_value("distance", fallback=None)),
        "altitude_m": _altitude(frame),
        "speed_mps": _speed(frame, "enhanced_speed", "speed"),
        "sport": None,
        "sub_sport": None,
        "event": None,
        "event_type": None,
    }


def _parse_timer_event(frame: fitdecode.FitDataMessage) -> dict[str, object] | None:
    timestamp = _as_datetime(frame.get_value("timestamp", fallback=None))
    event = coerce_str(frame.get_value("event", fallback=None))
    event_type = coerce_str(frame.get_value("event_type", fallback=None))
    if timestamp is None or event is None:
        return None
    return {
        "message_type": "event",
        "timestamp": _ensure_utc(timestamp),
        "heart_rate": None,
        "distance_m": None,
        "altitude_m": None,
        "speed_mps": None,
        "sport": None,
        "sub_sport": None,
        "event": event,
        "event_type": event_type,
    }


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
