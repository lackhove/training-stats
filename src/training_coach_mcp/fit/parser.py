"""FIT parsing boundary.

The rest of the system depends only on the :class:`FitParser` protocol and
the :mod:`training_coach_mcp.fit.types` dataclasses, never on fitdecode.
"""

from __future__ import annotations

import hashlib
import io
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import fitdecode
import polars as pl

from ..config import QualityConfig
from ..errors import FitParseError
from ..models import SourceFile
from .normalize import coerce_float, coerce_int, coerce_str
from .repair import moving_time_from_records, repair_timeline
from .types import RECORD_SCHEMA, ParsedActivity, ParsedFile, empty_records

logger = logging.getLogger(__name__)


class FitParser(Protocol):
    def parse(self, content: bytes, source_file: SourceFile) -> ParsedFile:
        """Parse raw FIT bytes into a :class:`ParsedFile`."""
        ...


@dataclass(frozen=True, slots=True)
class _SessionValues:
    start_time: datetime | None = None
    sport: str | None = None
    sub_sport: str | None = None
    elapsed_s: float | None = None
    timer_s: float | None = None
    moving_s: float | None = None
    distance_m: float | None = None
    ascent_m: float | None = None
    avg_hr: float | None = None
    max_hr: float | None = None
    avg_speed: float | None = None
    max_speed: float | None = None
    calories: float | None = None


class FitDecodeParser:
    """Default parser implementation based on the ``fitdecode`` library.

    Session headers are never trusted blindly: start time and duration are
    re-derived from the repaired record timeline, and recording breaks are
    split into independent segment activities.
    """

    def __init__(self, quality: QualityConfig | None = None) -> None:
        self.quality = quality or QualityConfig()

    def parse(self, content: bytes, source_file: SourceFile) -> ParsedFile:
        source_hash = hashlib.sha256(content).hexdigest()
        try:
            reader = fitdecode.FitReader(io.BytesIO(content))
            sessions: list[_SessionValues] = []
            record_rows: list[dict[str, object]] = []
            for frame in reader:
                if not isinstance(frame, fitdecode.FitDataMessage):
                    continue
                if frame.name == "session":
                    sessions.append(self._parse_session(frame))
                elif frame.name == "record":
                    row = self._parse_record_row(frame)
                    if row is not None:
                        record_rows.append(row)
        except fitdecode.exceptions.FitError as exc:
            msg = f"failed to parse FIT file {source_file.path!r}: {type(exc).__name__}"
            raise FitParseError(msg) from exc
        except (OSError, ValueError) as exc:
            msg = f"failed to read FIT file {source_file.path!r}: {type(exc).__name__}"
            raise FitParseError(msg) from exc

        records_frame = _frame_from_rows(record_rows)
        if not sessions and records_frame.is_empty():
            msg = f"no session or record messages found in FIT file {source_file.path!r}"
            raise FitParseError(msg)

        repair = repair_timeline(
            records_frame,
            gap_threshold_s=self.quality.record_gap_threshold_s,
            clock_jump_min_gap_s=self.quality.clock_jump_min_gap_s,
            minimum_segment_duration_s=self.quality.minimum_segment_duration_s,
            clock_shift_max_pre_moving_s=self.quality.clock_shift_max_pre_moving_s,
            moving_speed_threshold_mps=self.quality.moving_speed_threshold_mps,
        )
        total_span_s = sum(
            (
                span
                for span in (self._span_s(segment.records) for segment in repair.segments)
                if span is not None
            ),
            0.0,
        )
        segment_count = len(repair.segments) if repair.segments else 1
        if repair.segments:
            activities = tuple(
                self._segment_activity(
                    sessions,
                    segment.records,
                    segment.segment_index,
                    segment_count,
                    segment.flags,
                    total_span_s,
                    source_file,
                    source_hash,
                )
                for segment in repair.segments
            )
        else:
            activities = (
                self._segment_activity(sessions, None, 0, 1, (), 0.0, source_file, source_hash),
            )
        first_record: datetime | None = None
        if not records_frame.is_empty() and "clock_corrected" in repair.flags:
            first = records_frame["timestamp"].min()
            if isinstance(first, datetime):
                first_record = first
        logger.info(
            "parsed %s: sport=%s segments=%d discarded=%d flags=%s start=%s",
            source_file.path,
            activities[0].sport,
            segment_count,
            repair.discarded_segment_count,
            repair.flags,
            activities[0].start_time.isoformat(),
        )
        if first_record is not None:
            logger.info(
                "repaired timeline for %s: first record %s -> start %s",
                source_file.path,
                first_record.isoformat(),
                activities[0].start_time.isoformat(),
            )
        return ParsedFile(
            activities=activities,
            repair_flags=repair.flags,
            discarded_segment_count=repair.discarded_segment_count,
        )

    def _parse_session(self, frame: fitdecode.FitDataMessage) -> _SessionValues:
        return _SessionValues(
            start_time=_as_datetime(frame.get_value("start_time", fallback=None)),
            sport=coerce_str(frame.get_value("sport", fallback=None)),
            sub_sport=coerce_str(frame.get_value("sub_sport", fallback=None)),
            elapsed_s=coerce_float(frame.get_value("total_elapsed_time", fallback=None)),
            timer_s=coerce_float(frame.get_value("total_timer_time", fallback=None)),
            moving_s=coerce_float(frame.get_value("total_moving_time", fallback=None)),
            distance_m=coerce_float(frame.get_value("total_distance", fallback=None)),
            ascent_m=coerce_float(frame.get_value("total_ascent", fallback=None)),
            avg_hr=coerce_float(frame.get_value("avg_heart_rate", fallback=None)),
            max_hr=coerce_float(frame.get_value("max_heart_rate", fallback=None)),
            avg_speed=_speed(frame, "enhanced_avg_speed", "avg_speed"),
            max_speed=_speed(frame, "enhanced_max_speed", "max_speed"),
            calories=coerce_float(frame.get_value("total_calories", fallback=None)),
        )

    def _parse_record_row(self, frame: fitdecode.FitDataMessage) -> dict[str, object] | None:
        timestamp = _as_datetime(frame.get_value("timestamp", fallback=None))
        if timestamp is None:
            return None
        return {
            "timestamp": _ensure_utc(timestamp),
            "heart_rate": coerce_int(frame.get_value("heart_rate", fallback=None)),
            "distance_m": coerce_float(frame.get_value("distance", fallback=None)),
            "altitude_m": _altitude(frame),
            "speed_mps": _speed(frame, "enhanced_speed", "speed"),
        }

    def _segment_activity(
        self,
        sessions: list[_SessionValues],
        segment_records: pl.DataFrame | None,
        segment_index: int,
        segment_count: int,
        segment_flags: tuple[str, ...],
        total_span_s: float,
        source_file: SourceFile,
        source_hash: str,
    ) -> ParsedActivity:
        first_session = sessions[0] if sessions else None
        records = segment_records if segment_records is not None else empty_records()
        multi_segment = segment_count > 1
        span_s = self._span_s(records)

        start_time: datetime | None = None
        if not records.is_empty():
            first = records["timestamp"][0]
            if isinstance(first, datetime):
                start_time = first
        if start_time is None and first_session is not None:
            start_time = first_session.start_time
        if start_time is None:
            msg = f"FIT file {source_file.path!r} has no start time"
            raise FitParseError(msg)

        duration_s = self._duration_s(sessions, span_s)
        moving_time_s = self._moving_time_s(sessions, records, span_s, total_span_s, duration_s)
        paused_time_s = (
            max(0.0, duration_s - moving_time_s)
            if duration_s is not None and moving_time_s is not None
            else None
        )
        distance_m = self._distance_m(sessions, records, span_s, total_span_s)
        elevation_gain_m = self._elevation_gain_m(sessions, records, span_s, total_span_s)
        avg_hr, max_hr = self._heart_rate(sessions, records, multi_segment)
        avg_speed, max_speed = self._speeds(sessions, records, multi_segment, span_s, distance_m)
        calories = self._calories(sessions, span_s, total_span_s, multi_segment)

        repair_flags = segment_flags
        if self._is_forgotten_stop(duration_s, moving_time_s):
            repair_flags = (*segment_flags, "forgotten_stop")

        sport = first_session.sport if first_session else None
        return ParsedActivity(
            activity_id=_activity_id(start_time, sport, duration_s, segment_index),
            start_time=start_time,
            sport=sport or "unknown",
            sub_sport=first_session.sub_sport if first_session else None,
            duration_s=duration_s,
            moving_time_s=moving_time_s,
            distance_m=distance_m,
            elevation_gain_m=elevation_gain_m,
            avg_hr=avg_hr,
            max_hr=max_hr,
            avg_speed_mps=avg_speed,
            max_speed_mps=max_speed,
            calories=calories,
            records=records,
            source_path=source_file.path,
            source_hash=source_hash,
            paused_time_s=paused_time_s,
            segment_index=segment_index,
            segment_count=segment_count,
            repair_flags=repair_flags,
        )

    @staticmethod
    def _span_s(records: pl.DataFrame) -> float | None:
        if records.is_empty():
            return None
        first = records["timestamp"][0]
        last = records["timestamp"][-1]
        if not isinstance(first, datetime) or not isinstance(last, datetime):
            return None
        return (last - first).total_seconds()

    def _duration_s(self, sessions: list[_SessionValues], span_s: float | None) -> float | None:
        elapsed_s = _sum_opt(s.elapsed_s for s in sessions)
        timer_s = _sum_opt(s.timer_s for s in sessions)
        if span_s is None:
            return elapsed_s if elapsed_s is not None else timer_s
        if elapsed_s is not None and abs(elapsed_s - span_s) <= self.quality.record_gap_threshold_s:
            return elapsed_s
        return span_s

    def _moving_time_s(
        self,
        sessions: list[_SessionValues],
        records: pl.DataFrame,
        span_s: float | None,
        total_span_s: float,
        duration_s: float | None,
    ) -> float | None:
        moving = moving_time_from_records(
            records, threshold_mps=self.quality.moving_speed_threshold_mps
        )
        if moving is not None:
            return moving
        session_moving = _sum_opt(s.moving_s for s in sessions)
        if session_moving is None:
            return duration_s
        share = _span_share(span_s, total_span_s)
        if share is None:
            return session_moving
        return session_moving * share

    def _distance_m(
        self,
        sessions: list[_SessionValues],
        records: pl.DataFrame,
        span_s: float | None,
        total_span_s: float,
    ) -> float | None:
        session_distance = _sum_opt(s.distance_m for s in sessions)
        distances = records["distance_m"].drop_nulls()
        if distances.len() >= 2:
            first = _as_float(distances[0])
            last = _as_float(distances[-1])
            if first is not None and last is not None:
                delta = last - first
                if delta >= 0:
                    return delta
        share = _span_share(span_s, total_span_s)
        if share is None:
            return session_distance
        if session_distance is None:
            return None
        return session_distance * share

    def _elevation_gain_m(
        self,
        sessions: list[_SessionValues],
        records: pl.DataFrame,
        span_s: float | None,
        total_span_s: float,
    ) -> float | None:
        session_ascent = _sum_opt(s.ascent_m for s in sessions)
        share = _span_share(span_s, total_span_s)
        if share is None:
            return (
                session_ascent
                if session_ascent is not None
                else _elevation_gain_from_records(records)
            )
        if session_ascent is not None:
            return session_ascent * share
        return _elevation_gain_from_records(records)

    def _heart_rate(
        self, sessions: list[_SessionValues], records: pl.DataFrame, multi_segment: bool
    ) -> tuple[float | None, float | None]:
        avg_hr = _weighted_avg_hr(sessions)
        max_hr = _max_opt(s.max_hr for s in sessions)
        record_hrs = records["heart_rate"].drop_nulls()
        if multi_segment:
            if record_hrs.len() == 0:
                return None, None
            return _as_float(record_hrs.mean()), _as_float(record_hrs.max())
        if avg_hr is None and max_hr is None and record_hrs.len() > 0:
            return _as_float(record_hrs.mean()), _as_float(record_hrs.max())
        return avg_hr, max_hr

    def _speeds(
        self,
        sessions: list[_SessionValues],
        records: pl.DataFrame,
        multi_segment: bool,
        span_s: float | None,
        distance_m: float | None,
    ) -> tuple[float | None, float | None]:
        avg_speed = _weighted_speed(sessions)
        max_speed = _max_opt(s.max_speed for s in sessions)
        record_speeds = records["speed_mps"].drop_nulls()
        record_max = _as_float(record_speeds.max()) if record_speeds.len() > 0 else None
        if multi_segment:
            avg = (
                distance_m / span_s
                if distance_m is not None and span_s is not None and span_s > 0
                else None
            )
            return avg, record_max
        if max_speed is None:
            max_speed = record_max
        return avg_speed, max_speed

    def _calories(
        self,
        sessions: list[_SessionValues],
        span_s: float | None,
        total_span_s: float,
        multi_segment: bool,
    ) -> float | None:
        total = _sum_opt(s.calories for s in sessions)
        if not multi_segment:
            return total
        share = _span_share(span_s, total_span_s)
        if total is None or share is None:
            return total
        return total * share

    @staticmethod
    def _is_forgotten_stop(duration_s: float | None, moving_time_s: float | None) -> bool:
        if duration_s is None:
            return False
        if duration_s > _FORGOTTEN_STOP_MAX_DURATION_S:
            return True
        if duration_s > _FORGOTTEN_STOP_CHECK_DURATION_S and moving_time_s is not None:
            return moving_time_s / duration_s < _FORGOTTEN_STOP_MIN_MOVING_RATIO
        return False


_FORGOTTEN_STOP_MAX_DURATION_S = 86400.0
_FORGOTTEN_STOP_CHECK_DURATION_S = 7200.0
_FORGOTTEN_STOP_MIN_MOVING_RATIO = 0.20


def _frame_from_rows(rows: list[dict[str, object]]) -> pl.DataFrame:
    if not rows:
        return empty_records()
    return pl.DataFrame(rows, schema=RECORD_SCHEMA)


def _span_share(span_s: float | None, total_span_s: float) -> float | None:
    """Proportional share of one segment within the file; 1.0 for single segments."""
    if span_s is None or total_span_s <= 0:
        return None
    share = span_s / total_span_s
    return share if share < 1 else 1.0


def _as_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _ensure_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _activity_id(
    start_time: datetime, sport: str | None, duration_s: float | None, segment_index: int = 0
) -> str:
    """Deterministic activity ID stable across sources and re-downloads."""
    key = "|".join(
        [
            start_time.astimezone(UTC).isoformat(),
            sport or "unknown",
            f"{duration_s:.0f}" if duration_s is not None else "-",
            str(segment_index),
        ]
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


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


def _sum_opt(values: Iterable[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) if present else None


def _max_opt(values: Iterable[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return max(present) if present else None


def _weighted_avg_hr(sessions: list[_SessionValues]) -> float | None:
    weighted: list[tuple[float, float]] = []
    for session in sessions:
        if session.avg_hr is None:
            continue
        weight = session.elapsed_s if session.elapsed_s is not None else session.timer_s
        weighted.append((session.avg_hr, weight or 0.0))
    if not weighted:
        return None
    total_weight = sum(weight for _, weight in weighted)
    if total_weight <= 0:
        return sum(hr for hr, _ in weighted) / len(weighted)
    return sum(hr * weight for hr, weight in weighted) / total_weight


def _weighted_speed(sessions: list[_SessionValues]) -> float | None:
    weighted: list[tuple[float, float]] = []
    for session in sessions:
        if session.avg_speed is None:
            continue
        weight = session.elapsed_s if session.elapsed_s is not None else session.timer_s
        weighted.append((session.avg_speed, weight or 0.0))
    if not weighted:
        return None
    total_weight = sum(weight for _, weight in weighted)
    if total_weight <= 0:
        return sum(speed for speed, _ in weighted) / len(weighted)
    return sum(speed * weight for speed, weight in weighted) / total_weight


def _elevation_gain_from_records(records: pl.DataFrame) -> float | None:
    if records.is_empty():
        return None
    altitudes = records["altitude_m"].drop_nulls().to_list()
    if len(altitudes) < 2:
        return None
    gain = 0.0
    previous = altitudes[0]
    for altitude in altitudes[1:]:
        if not isinstance(altitude, int | float):
            continue
        delta = float(altitude) - float(previous)
        if delta > 0:
            gain += delta
        previous = altitude
    return gain if gain > 0 else None
