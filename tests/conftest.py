"""Shared pytest fixtures."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from math import ceil
from pathlib import Path

import polars as pl
import pytest

from training_stats.adapters.bootstrap import create_service
from training_stats.adapters.cache.repository import JsonTrainingRepository
from training_stats.adapters.config import (
    AppConfig,
    ApplicationConfig,
    CacheConfig,
    LocalSourceConfig,
    ServerConfig,
)
from training_stats.domain.errors import FitParseError
from training_stats.domain.fit.types import RECORD_SCHEMA, FitData
from training_stats.domain.models import SourceFile
from training_stats.domain.settings import AthleteConfig, HeartRateZonesConfig, QualityConfig
from training_stats.services.errors import SourceError
from training_stats.services.ingest import FileProcessor
from training_stats.services.ports import ActivitySource, FitParser
from training_stats.services.results import SyncResult
from training_stats.services.sync import SyncService
from training_stats.services.training import TrainingStatsService

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
    parser: FitParser | None = None,
    quality: QualityConfig | None = None,
) -> FileProcessor:
    """FileProcessor wired with the deterministic StubParser for cache tests."""
    return FileProcessor(
        parser=parser or StubParser(),
        athlete=athlete,
        quality=quality or QualityConfig(),
    )


def make_record_frame(
    start: datetime, heart_rates: Iterable[int | None], interval_s: int = 60
) -> pl.DataFrame:
    """Create a record frame with samples spaced ``interval_s`` apart."""
    rows: list[dict[str, object]] = []
    timestamp = start
    for heart_rate in heart_rates:
        rows.append(
            {
                "timestamp": timestamp,
                "heart_rate": heart_rate,
                "cadence": None,
                "fractional_cadence": None,
                "power_w": None,
            }
        )
        timestamp = timestamp + timedelta(seconds=interval_s)
    return pl.DataFrame(rows, schema=RECORD_SCHEMA)


class StubSource(ActivitySource):
    """In-memory source for tests; content is an opaque byte string."""

    def __init__(self, name: str = "stub", files: dict[str, bytes] | None = None) -> None:
        self.name = name
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

    def parse(self, content: bytes, source_file: SourceFile) -> FitData:
        try:
            lines = [line for line in content.decode("utf-8").splitlines() if line.strip()]
        except UnicodeDecodeError as exc:
            msg = f"invalid stub content in {source_file.path}"
            raise FitParseError(msg) from exc
        if not lines:
            msg = f"invalid stub content in {source_file.path}"
            raise FitParseError(msg)
        rows: list[dict[str, object]] = []
        for line in lines:
            rows.extend(self._parse_line(line, source_file))
        return FitData(frame=pl.DataFrame(rows, schema=FitData.SCHEMA))

    @staticmethod
    def _parse_line(text: str, source_file: SourceFile) -> list[dict[str, object]]:
        try:
            head, _, hr_part = text.partition(",")
            start_raw, sport, duration_raw, distance_raw = head.split("|")
            start = datetime.fromisoformat(start_raw)
            duration = float(duration_raw)
            distance = float(distance_raw)
            heart_rates = [int(value) for value in hr_part.split(";")] if hr_part else []
        except ValueError as exc:
            msg = f"invalid stub content in {source_file.path}"
            raise FitParseError(msg) from exc

        intervals = max(1, ceil(duration / 30), ceil(distance / 500))
        intervals = max(intervals, len(heart_rates) - 1)
        rows: list[dict[str, object]] = [
            {
                "message_type": "session",
                "timestamp": None,
                "heart_rate": None,
                "cadence": None,
                "fractional_cadence": None,
                "power_w": None,
                "distance_m": None,
                "altitude_m": None,
                "speed_mps": None,
                "sport": sport,
                "sub_sport": None,
                "event": None,
                "event_type": None,
            }
        ]
        for index in range(intervals + 1):
            hr_index = min(index * len(heart_rates) // (intervals + 1), len(heart_rates) - 1)
            heart_rate = heart_rates[hr_index] if heart_rates else None
            rows.append(
                {
                    "message_type": "record",
                    "timestamp": start + timedelta(seconds=duration * index / intervals),
                    "heart_rate": heart_rate,
                    "cadence": None,
                    "fractional_cadence": None,
                    "power_w": None,
                    "distance_m": distance * index / intervals,
                    "altitude_m": None,
                    "speed_mps": 5.0,
                    "sport": None,
                    "sub_sport": None,
                    "event": None,
                    "event_type": None,
                }
            )
        return rows


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


T0 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)


def two_files() -> dict[str, bytes]:
    return {
        "easy.fit": stub_activity_content(T0, "cycling", 3600.0, 30000.0, heart_rates=[120] * 60),
        "intervals.fit": stub_activity_content(
            T0 + timedelta(days=2), "running", 1800.0, 8000.0, heart_rates=[150] * 30
        ),
    }


def make_sync_service(
    tmp_path: Path,
    sources: list[StubSource],
    *,
    version: str = "1",
    quality: QualityConfig | None = None,
) -> SyncService:
    """SyncService wired with stub sources and an isolated cache directory."""
    cache_dir = tmp_path / "cache"
    config = make_app_config(cache_dir, tmp_path / "src")
    return SyncService(
        sources=sources,
        processor=make_processor(athlete=config.athlete, quality=quality),
        repository=JsonTrainingRepository(cache_dir),
        analysis_version=version,
    )


def build_repository(tmp_path: Path, files: dict[str, bytes]) -> JsonTrainingRepository:
    """Repository holding the activities synced from ``files``."""
    service = make_sync_service(tmp_path, [StubSource(files=files)])
    service.sync()
    return JsonTrainingRepository(tmp_path / "cache")


def make_server(
    tmp_path: Path,
    files: dict[str, bytes],
    *,
    interval_s: float = 0.0,
    config: AppConfig | None = None,
) -> TrainingStatsService:
    """Server backed by the deterministic stub source/parser.

    ``config`` lets callers keep a pre-configured application config (for
    example one carrying auth settings). The sync throttle is disabled unless
    ``interval_s`` says otherwise.
    """
    cfg = config if config is not None else make_app_config(tmp_path / "cache", tmp_path / "src")
    cfg.cache.sync_min_interval_seconds = interval_s
    server = create_service(cfg)
    # Swap in the deterministic stub source/parser for cache independence.
    server.sync_service.sources = [StubSource(files=files)]
    server.sync_service.processor = make_processor(athlete=cfg.athlete, quality=cfg.quality)
    return server


def spy_sync_calls(server: TrainingStatsService) -> Callable[[], int]:
    """Count how often the underlying sync actually runs."""
    calls: list[bool] = []
    original = server.sync_service.sync

    def counting_sync() -> SyncResult:
        calls.append(True)
        return original()

    server.sync_service.sync = counting_sync
    return lambda: len(calls)


def assert_no_none_fields(value: object) -> None:
    if isinstance(value, dict):
        assert all(item is not None for item in value.values())
        for item in value.values():
            assert_no_none_fields(item)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict | list):
                assert_no_none_fields(item)


def json_metrics(report: object) -> list[dict[str, object]]:
    assert isinstance(report, dict)
    metrics = report.get("metrics")
    assert isinstance(metrics, list)
    return metrics


def json_metric(report: object, name: str) -> dict[str, object]:
    return next(metric for metric in json_metrics(report) if metric["name"] == name)


def json_activity_ids(activities: object) -> set[str]:
    assert isinstance(activities, list)
    return {str(activity["activity_id"]) for activity in activities}


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
