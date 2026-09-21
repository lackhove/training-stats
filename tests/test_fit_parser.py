"""Tests for the FIT parser using synthetic fixtures and synthetic inputs."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from tests.conftest import FIXTURES, ZONE_BOUNDS
from training_stats.analytics.heart_rate import build_zone_bounds, time_in_zones
from training_stats.errors import FitParseError
from training_stats.fit.parser import FitDecodeParser
from training_stats.fit.types import RECORD_SCHEMA, ParsedActivity, empty_records
from training_stats.models import SourceFile

PARSER = FitDecodeParser()


def _source_file(path: str = "test.fit") -> SourceFile:
    return SourceFile(source_name="test", path=path)


def _parse(name: str) -> ParsedActivity:
    parsed = PARSER.parse((FIXTURES / name).read_bytes(), _source_file(name))
    assert parsed.discarded_segment_count == 0
    assert len(parsed.activities) == 1
    return parsed.activities[0]


class TestFixtures:
    def test_cycling_fixture(self) -> None:
        activity = _parse("cycling.fit")
        assert activity.sport == "cycling"
        assert activity.start_time == datetime(2024, 1, 15, 9, 0, 0, tzinfo=UTC)
        assert activity.duration_s == pytest.approx(3441.0)
        assert activity.distance_m == pytest.approx(16036.0)
        assert activity.avg_hr == pytest.approx(123.0)
        assert activity.max_hr == pytest.approx(154.0)
        assert activity.calories == pytest.approx(505.0)
        assert activity.avg_speed_mps == pytest.approx(4.661)
        assert activity.max_speed_mps == pytest.approx(7.333)
        assert activity.records.height > 3000
        assert activity.records["heart_rate"].head(10).null_count() == 0
        assert activity.repair_flags == ()
        assert activity.segment_count == 1

    def test_second_fixture(self) -> None:
        activity = _parse("activity_2.fit")
        assert activity.sport == "cycling"
        assert activity.duration_s == pytest.approx(3498.447)
        assert activity.max_hr == pytest.approx(142.0)
        assert activity.records.height > 800

    def test_moving_time_derived_from_records(self) -> None:
        """The second fixture lacks session moving time; records provide speed."""
        activity = _parse("activity_2.fit")
        assert activity.moving_time_s == pytest.approx(3371.0)
        assert activity.paused_time_s is not None
        assert activity.paused_time_s > 0
        assert activity.sub_sport == "generic"
        # Session has total_moving_time absent but records provide HR.
        assert activity.avg_hr is not None

    def test_record_frame_schema(self) -> None:
        activity = _parse("cycling.fit")
        assert activity.records.schema == pl.Schema(RECORD_SCHEMA)
        assert activity.records["timestamp"].dtype == pl.Datetime("us", "UTC")


class TestStableIds:
    def test_stable_activity_id(self) -> None:
        content = (FIXTURES / "cycling.fit").read_bytes()
        first = PARSER.parse(content, _source_file("cycling.fit"))
        second = PARSER.parse(content, _source_file("cycling.fit"))
        assert first.activities[0].activity_id == second.activities[0].activity_id
        assert len(first.activities[0].activity_id) == 16
        assert first.activities[0].source_hash

    def test_different_files_have_different_ids(self) -> None:
        first = PARSER.parse((FIXTURES / "cycling.fit").read_bytes(), _source_file("a.fit"))
        second = PARSER.parse((FIXTURES / "activity_2.fit").read_bytes(), _source_file("b.fit"))
        assert first.activities[0].activity_id != second.activities[0].activity_id


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
        result = time_in_zones(activity.records, build_zone_bounds(ZONE_BOUNDS))
        assert result.zone_seconds == (71.0, 2366.0, 985.0, 17.0, 0.0)
        assert result.valid_hr_time_s == 3439.0
        assert result.hr_coverage_ratio == pytest.approx(1.0)

    def test_golden_zones_second_fixture(self) -> None:
        activity = _parse("activity_2.fit")
        result = time_in_zones(activity.records, build_zone_bounds(ZONE_BOUNDS))
        assert result.zone_seconds == (190.0, 1812.0, 1496.0, 0.0, 0.0)
        assert result.valid_hr_time_s == 3498.0
        assert result.hr_coverage_ratio == pytest.approx(1.0)

    def test_time_in_zones_does_not_mutate_input(self) -> None:
        activity = _parse("cycling.fit")
        before = activity.records.clone()
        time_in_zones(activity.records, build_zone_bounds(ZONE_BOUNDS))
        assert activity.records.equals(before)
