"""Shared pytest fixtures."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from training_stats.config import (
    AppConfig,
    ApplicationConfig,
    AthleteConfig,
    CacheConfig,
    HeartRateZonesConfig,
    LocalSourceConfig,
    QualityConfig,
    ServerConfig,
)
from training_stats.errors import FitParseError, SourceError
from training_stats.fit.parser import FitParser
from training_stats.fit.types import RECORD_SCHEMA, ParsedActivity, ParsedFile
from training_stats.ingest import FileProcessor
from training_stats.models import SourceFile
from training_stats.sources.base import ActivitySource

FIXTURES = Path(__file__).parent / "fixtures"

ZONE_BOUNDS = [
    (1, 90, 108),
    (2, 109, 126),
    (3, 127, 144),
    (4, 145, 162),
    (5, 163, 180),
]


def make_athlete(**overrides: object) -> AthleteConfig:
    values: dict[str, object] = {
        "max_heart_rate": 180,
        "resting_heart_rate": 55,
        "heart_rate_zones": HeartRateZonesConfig(
            z1_min=ZONE_BOUNDS[0][1],
            z1_max=ZONE_BOUNDS[0][2],
            z2_min=ZONE_BOUNDS[1][1],
            z2_max=ZONE_BOUNDS[1][2],
            z3_min=ZONE_BOUNDS[2][1],
            z3_max=ZONE_BOUNDS[2][2],
            z4_min=ZONE_BOUNDS[3][1],
            z4_max=ZONE_BOUNDS[3][2],
            z5_min=ZONE_BOUNDS[4][1],
            z5_max=ZONE_BOUNDS[4][2],
        ),
    }
    values.update(overrides)
    return AthleteConfig(**values)  # type: ignore[arg-type]


def make_app_config(
    cache_dir: Path, source_dir: Path, athlete: AthleteConfig | None = None
) -> AppConfig:
    return AppConfig(
        application=ApplicationConfig(timezone="Europe/Berlin"),
        cache=CacheConfig(directory=cache_dir),
        athlete=athlete or make_athlete(),
        sources=[LocalSourceConfig(name="local", path=source_dir)],
        server=ServerConfig(),
    )


def make_processor(
    *,
    athlete: AthleteConfig,
    analysis_version: str,
    parser: FitParser | None = None,
    quality: QualityConfig | None = None,
) -> FileProcessor:
    """FileProcessor wired with the deterministic StubParser for cache tests."""
    return FileProcessor(
        parser=parser or StubParser(),
        athlete=athlete,
        quality=quality or QualityConfig(),
        analysis_version=analysis_version,
    )


def make_record_frame(
    start: datetime, heart_rates: Iterable[int | None], interval_s: int = 60
) -> pl.DataFrame:
    """Create a record frame with samples spaced ``interval_s`` apart."""
    rows: list[dict[str, object]] = []
    timestamp = start
    for heart_rate in heart_rates:
        rows.append({"timestamp": timestamp, "heart_rate": heart_rate})
        timestamp = timestamp + timedelta(seconds=interval_s)
    return pl.DataFrame(rows, schema=RECORD_SCHEMA)


class StubSource(ActivitySource):
    """In-memory source for tests; content is an opaque byte string."""

    def __init__(self, name: str = "stub", files: dict[str, bytes] | None = None) -> None:
        super().__init__(name)
        self.files: dict[str, bytes] = dict(files or {})

    def list_files(self) -> list[SourceFile]:
        return [
            SourceFile(
                source_name=self.name,
                path=path,
                size=len(content),
                modified_at=_stub_modified_at(content),
            )
            for path, content in sorted(self.files.items())
        ]

    def read_file(self, source_file: SourceFile) -> bytes:
        if source_file.path not in self.files:
            msg = f"file not found: {source_file.path}"
            raise SourceError(msg)
        return self.files[source_file.path]


def _stub_modified_at(content: bytes) -> datetime:
    """Deterministic mtime derived from the content, like a real filesystem.

    An in-place content change therefore bumps the mtime, which is what the
    sync's metadata short-circuit keys on.
    """
    digest = hashlib.sha256(content).digest()
    return datetime(2026, 1, 1, tzinfo=UTC) + timedelta(
        seconds=int.from_bytes(digest[:4], "big") % 100_000_000
    )


class StubParser:
    """Deterministic parser for cache tests.

    Content grammar: ``ISO_START|SPORT|DURATION_S|DISTANCE_M[,hr,hr,...]``
    The optional heart-rate list creates one record per minute. Multiple
    lines in the content produce one segment activity per line.
    """

    def parse(self, content: bytes, source_file: SourceFile) -> ParsedFile:
        try:
            lines = [line for line in content.decode("utf-8").splitlines() if line.strip()]
        except UnicodeDecodeError as exc:
            msg = f"invalid stub content in {source_file.path}"
            raise FitParseError(msg) from exc
        if not lines:
            msg = f"invalid stub content in {source_file.path}"
            raise FitParseError(msg)
        segment_count = len(lines)
        activities = tuple(
            self._parse_line(line, index, segment_count, source_file)
            for index, line in enumerate(lines)
        )
        return ParsedFile(activities=activities, repair_flags=(), discarded_segment_count=0)

    def _parse_line(
        self, text: str, segment_index: int, segment_count: int, source_file: SourceFile
    ) -> ParsedActivity:
        try:
            head, _, hr_part = text.partition(",")
            start_raw, sport, duration_raw, distance_raw = head.split("|")
            start = datetime.fromisoformat(start_raw)
            duration = float(duration_raw)
        except ValueError as exc:
            msg = f"invalid stub content in {source_file.path}"
            raise FitParseError(msg) from exc
        records: pl.DataFrame = pl.DataFrame(schema=RECORD_SCHEMA)
        if hr_part:
            heart_rates = [int(value) for value in hr_part.split(";")]
            records = make_record_frame(start, heart_rates)
        return ParsedActivity(
            activity_id=self._activity_id(start, sport),
            start_time=start,
            sport=sport,
            sub_sport=None,
            duration_s=duration,
            moving_time_s=duration,
            distance_m=float(distance_raw),
            elevation_gain_m=None,
            avg_hr=None,
            max_hr=None,
            avg_speed_mps=None,
            max_speed_mps=None,
            calories=None,
            records=records,
            source_path=source_file.path,
            source_hash="",
            segment_index=segment_index,
            segment_count=segment_count,
        )

    @staticmethod
    def _activity_id(start: datetime, sport: str) -> str:
        import hashlib

        key = f"{start.isoformat()}|{sport}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def stub_activity_content(
    start: datetime,
    sport: str,
    duration_s: float,
    distance_m: float = 10000.0,
    heart_rates: list[int] | None = None,
) -> bytes:
    head = f"{start.isoformat()}|{sport}|{duration_s}|{distance_m}"
    if heart_rates is None:
        return head.encode("utf-8")
    return (head + "," + ";".join(str(hr) for hr in heart_rates)).encode("utf-8")


@pytest.fixture
def athlete() -> AthleteConfig:
    return make_athlete()


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    return make_app_config(tmp_path / "cache", tmp_path / "activities")


@pytest.fixture
def fit_file_bytes() -> Callable[[str], bytes]:
    def _load(name: str) -> bytes:
        return (FIXTURES / name).read_bytes()

    return _load
