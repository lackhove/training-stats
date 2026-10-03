"""Tests for the FIT parser using synthetic fixtures and synthetic inputs."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

import fitdecode
import polars as pl
import pytest

from tests.conftest import FIXTURES, ZONE_BOUNDS, make_athlete
from training_stats.adapters.fit.parser import FitDecodeParser, _parse_record_row
from training_stats.domain.analytics.heart_rate import build_zone_bounds, time_in_zones
from training_stats.domain.analytics.segments import analyze_segments
from training_stats.domain.errors import FitParseError
from training_stats.domain.fit.repair import repair_timeline
from training_stats.domain.fit.types import (
    RECORD_SCHEMA,
    FitData,
    ParsedActivity,
    empty_records,
)
from training_stats.domain.models import SourceFile
from training_stats.domain.settings import QualityConfig

PARSER = FitDecodeParser()


def _source_file(path: str = "test.fit") -> SourceFile:
    return SourceFile(source_name="test", path=path)


def _parse(name: str) -> ParsedActivity:
    source = _source_file(name)
    decoded = PARSER.parse((FIXTURES / name).read_bytes(), source)
    quality = QualityConfig()
    segments = repair_timeline(
        decoded,
        gap_threshold_s=quality.activity_gap_threshold_s,
        distance_gap_threshold_m=quality.activity_distance_gap_threshold_m,
    )
    analyzed = analyze_segments(
        segments,
        source_path=source.path,
        athlete=make_athlete(),
        analysis_version="test",
        quality=quality,
    )
    assert analyzed.discarded_segment_count == 0
    assert len(analyzed.activities) == 1
    return analyzed.activities[0].activity


class _SyntheticRecordMessage:
    def __init__(self, values: dict[str, object]) -> None:
        self.values = values

    def get_value(self, name: str, fallback: object | None = None) -> object | None:
        return self.values.get(name, fallback)


class TestFixtures:
    def test_cycling_fixture(self) -> None:
        activity = _parse("cycling.fit")
        assert activity.sport == "cycling"
        assert activity.start_time == datetime(2024, 1, 15, 9, 0, 0, tzinfo=UTC)
        assert activity.duration_s == pytest.approx(3439.0)
        assert activity.distance_m == pytest.approx(16002.0)
        assert activity.avg_hr == pytest.approx(123.4258)
        assert activity.max_hr == pytest.approx(154.0)
        assert activity.avg_speed_mps == pytest.approx(4.77814)
        assert activity.max_speed_mps == pytest.approx(7.333)
        assert activity.records.height > 3000
        assert activity.records["heart_rate"].head(10).null_count() == 0
        assert activity.repair_flags == ()
        assert activity.segment_count == 1

    def test_second_fixture(self) -> None:
        activity = _parse("activity_2.fit")
        assert activity.sport == "cycling"
        assert activity.duration_s == pytest.approx(3498.0)
        assert activity.max_hr == pytest.approx(142.0)
        assert activity.records.height > 800

    def test_moving_time_derived_from_records(self) -> None:
        """Moving time is derived from record speeds, not session aggregates."""
        activity = _parse("activity_2.fit")
        assert activity.moving_time_s == pytest.approx(3383.0)
        assert activity.paused_time_s is not None
        assert activity.paused_time_s > 0
        assert activity.sub_sport == "generic"
        # Record samples provide the heart-rate data for analysis.
        assert activity.avg_hr is not None

    def test_record_frame_schema(self) -> None:
        activity = _parse("cycling.fit")
        assert activity.records.schema == pl.Schema(RECORD_SCHEMA)
        assert activity.records["timestamp"].dtype == pl.Datetime("us", "UTC")


class TestStableIds:
    def test_stable_activity_id(self) -> None:
        first = _parse("cycling.fit")
        second = _parse("cycling.fit")
        assert first.activity_id == second.activity_id
        assert len(first.activity_id) == 16

    def test_different_files_have_different_ids(self) -> None:
        first = _parse("cycling.fit")
        second = _parse("activity_2.fit")
        assert first.activity_id != second.activity_id


class TestRawFitData:
    def test_parser_returns_only_raw_records_events_and_sport_metadata(self) -> None:
        decoded = PARSER.parse((FIXTURES / "cycling.fit").read_bytes(), _source_file())
        assert isinstance(decoded, FitData)
        assert decoded.frame.schema == pl.Schema(FitData.SCHEMA)
        assert set(decoded.frame["message_type"].unique().to_list()) <= {
            "record",
            "event",
            "session",
        }
        assert {
            "total_elapsed_time",
            "total_timer_time",
            "total_moving_time",
            "total_distance",
            "total_ascent",
            "avg_heart_rate",
            "max_heart_rate",
            "avg_speed",
            "max_speed",
        }.isdisjoint(decoded.frame.columns)
        session_rows = decoded.frame.filter(pl.col("message_type") == "session")
        assert session_rows["sport"][0] == "cycling"
        assert session_rows["sub_sport"][0] == "generic"
        # Session rows carry no record fields; the schema fills them with nulls.
        assert session_rows["heart_rate"][0] is None
        assert session_rows["distance_m"][0] is None


class TestCadenceParsing:
    def test_record_cadence_fields_are_normalized(self) -> None:
        timestamp = datetime(2026, 9, 24, 10, 27, 59, tzinfo=UTC)
        message = _SyntheticRecordMessage(
            {
                "timestamp": timestamp,
                "cadence": 87,
                "fractional_cadence": 64,
            }
        )

        row = _parse_record_row(cast(fitdecode.FitDataMessage, message))

        assert row is not None
        assert row["cadence"] == 87
        assert row["fractional_cadence"] == 64

    def test_missing_record_cadence_fields_are_null(self) -> None:
        message = _SyntheticRecordMessage(
            {"timestamp": datetime(2026, 9, 24, 10, 27, 59, tzinfo=UTC)}
        )

        row = _parse_record_row(cast(fitdecode.FitDataMessage, message))

        assert row is not None
        assert "cadence" not in row
        assert "fractional_cadence" not in row
        # Rows carry no null padding: the declared schema fills the absent
        # columns (and every column a record row never provides) with nulls.
        frame = pl.DataFrame([row], schema=FitData.SCHEMA)
        assert frame.schema == pl.Schema(FitData.SCHEMA)
        assert frame["cadence"][0] is None
        assert frame["fractional_cadence"][0] is None
        assert frame["sport"][0] is None
        assert frame["event"][0] is None


class TestMalformedInput:
    def test_garbage_bytes(self) -> None:
        with pytest.raises(FitParseError, match="failed to parse"):
            PARSER.parse(b"this is not a fit file", _source_file())

    def test_empty_content(self) -> None:
        with pytest.raises(FitParseError):
            PARSER.parse(b"", _source_file())

    def test_truncated_fixture(self) -> None:
        content = (FIXTURES / "cycling.fit").read_bytes()
        with pytest.raises(FitParseError):
            PARSER.parse(content[: len(content) // 2], _source_file())


class TestRecordFrames:
    def test_empty_records_schema(self) -> None:
        empty = empty_records()
        assert empty.height == 0
        assert empty.schema == pl.Schema(RECORD_SCHEMA)

    def test_golden_zones_cycling_fixture(self) -> None:
        """Lock vectorized zone results to the reference-loop values."""
        activity = _parse("cycling.fit")
        result = time_in_zones(
            activity.records,
            build_zone_bounds(ZONE_BOUNDS),
            pause_intervals=activity.pause_intervals,
        )
        assert result.zone_seconds == (61.0, 2299.0, 972.0, 17.0, 0.0)
        assert result.valid_hr_time_s == 3349.0
        assert result.hr_coverage_ratio == pytest.approx(1.0)

    def test_golden_zones_second_fixture(self) -> None:
        activity = _parse("activity_2.fit")
        result = time_in_zones(
            activity.records,
            build_zone_bounds(ZONE_BOUNDS),
            pause_intervals=activity.pause_intervals,
        )
        assert result.zone_seconds == (177.0, 1741.0, 1465.0, 0.0, 0.0)
        assert result.valid_hr_time_s == 3383.0
        assert result.hr_coverage_ratio == pytest.approx(1.0)

    def test_time_in_zones_does_not_mutate_input(self) -> None:
        activity = _parse("cycling.fit")
        before = activity.records.clone()
        time_in_zones(
            activity.records,
            build_zone_bounds(ZONE_BOUNDS),
            pause_intervals=activity.pause_intervals,
        )
        assert activity.records.equals(before)
