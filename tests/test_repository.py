"""Tests for JSON cache records and source-outcome round trips."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tests.conftest import StubSource, make_app_config, make_processor, stub_activity_content
from training_stats.adapters.cache import jsonio
from training_stats.adapters.cache.repository import JsonTrainingRepository
from training_stats.adapters.cache.schemas import CACHE_FILE, CACHE_FORMAT_VERSION, METADATA_FILE
from training_stats.services.sync import SyncService

T0 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)


def spy_on_reads(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Record every JSON file the cache layer opens."""
    seen: list[Path] = []
    original = jsonio.read_json

    def spy(path: Path) -> dict[str, object] | None:
        seen.append(path)
        return original(path)

    monkeypatch.setattr(jsonio, "read_json", spy)
    return seen


def _cache_with_snapshot(tmp_path: Path) -> Path:
    """A cache directory holding a valid, current-format (but empty) snapshot."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / CACHE_FILE).write_text(
        json.dumps(
            {
                "format_version": CACHE_FORMAT_VERSION,
                "activities": [],
                "source_index": [],
            }
        )
    )
    return cache_dir


def build_repository(tmp_path: Path, files: dict[str, bytes]) -> JsonTrainingRepository:
    source = StubSource(files=files)
    cache_dir = tmp_path / "cache"
    config = make_app_config(cache_dir, tmp_path / "src")
    repository = JsonTrainingRepository(cache_dir=cache_dir)
    service = SyncService(
        sources=[source],
        processor=make_processor(athlete=config.athlete, analysis_version="1"),
        repository=repository,
        analysis_version="1",
    )
    service.sync()
    return repository


def two_files() -> dict[str, bytes]:
    return {
        "easy.fit": stub_activity_content(T0, "cycling", 3600.0, 30000.0, heart_rates=[120] * 60),
        "intervals.fit": stub_activity_content(
            T0 + timedelta(days=2), "running", 1800.0, 8000.0, heart_rates=[150] * 30
        ),
    }


class TestJsonCache:
    def test_empty_cache_and_empty_sync(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, {})
        assert repository.load_sync_state().activities == {}
        assert repository.load_sync_state().index_rows == []
        assert repository.last_sync_stamp() is not None

    def test_cache_stores_activity_metric_records(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, two_files())
        path = tmp_path / "cache" / CACHE_FILE
        payload = json.loads(path.read_text())
        assert payload["format_version"] == CACHE_FORMAT_VERSION
        assert len(payload["activities"]) == 2
        assert len(payload["source_index"]) == 2
        assert all(item["metrics"] for item in payload["activities"])
        assert all(
            metric["description"].strip()
            for activity in payload["activities"]
            for metric in activity["metrics"]
        )
        assert '"records":' not in path.read_text()

        state = repository.load_sync_state()
        assert {record.sport for record in state.activities.values()} == {"cycling", "running"}
        names = {metric.name for record in state.activities.values() for metric in record.metrics}
        assert "training_load.hr_trimp" in names
        assert all(record.start_time.tzinfo is not None for record in state.activities.values())

    def test_source_only_failure_has_no_activity_record(self, tmp_path: Path) -> None:
        repository = build_repository(tmp_path, {"broken.fit": b"garbage"})
        state = repository.load_sync_state()
        assert state.activities == {}
        assert len(state.index_rows) == 1
        assert state.index_rows[0].status == "failed"
        assert state.index_rows[0].error_message is not None

    def test_duplicate_outcome_keeps_one_activity(self, tmp_path: Path) -> None:
        content = stub_activity_content(T0, "cycling", 1800.0, 20000.0)
        repository = build_repository(tmp_path, {"one.fit": content, "two.fit": content})
        state = repository.load_sync_state()
        assert len(state.activities) == 1
        assert sorted(row.status for row in state.index_rows) == ["duplicate", "ok"]

    def test_multiple_segments_have_distinct_activity_records(self, tmp_path: Path) -> None:
        first = stub_activity_content(T0, "cycling", 1800.0, 20000.0)
        second = stub_activity_content(T0 + timedelta(hours=2), "cycling", 1200.0, 8000.0)
        repository = build_repository(tmp_path, {"ride.fit": first + b"\n" + second})
        state = repository.load_sync_state()
        assert len(state.activities) == 2
        assert sorted(record.segment_index for record in state.activities.values()) == [0, 1]
        assert {record.segment_count for record in state.activities.values()} == {2}

    def test_format_version_mismatch_requests_full_rebuild(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / CACHE_FILE).write_text('{"format_version": 0}')
        repository = JsonTrainingRepository(cache_dir=cache_dir)
        assert repository.load_sync_state().activities == {}
        assert repository.load_sync_state().index_rows == []

    def test_metadata_file_keeps_stamp_and_format_version(self, tmp_path: Path) -> None:
        build_repository(tmp_path, two_files())
        metadata = json.loads((tmp_path / "cache" / METADATA_FILE).read_text())
        assert set(metadata) == {"last_sync", "format_version"}
        assert metadata["format_version"] == CACHE_FORMAT_VERSION
        assert datetime.fromisoformat(metadata["last_sync"]).tzinfo is not None


class TestLastSyncStamp:
    def test_reads_metadata_without_opening_snapshot(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repository = JsonTrainingRepository(cache_dir=tmp_path / "cache")
        repository.save_sync_stamp(last_sync=T0)
        seen = spy_on_reads(monkeypatch)
        assert repository.last_sync_stamp() == T0
        assert [path.name for path in seen] == [METADATA_FILE]

    def test_legacy_metadata_without_version_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache_dir = _cache_with_snapshot(tmp_path)
        (cache_dir / METADATA_FILE).write_text(json.dumps({"last_sync": T0.isoformat()}))
        repository = JsonTrainingRepository(cache_dir=cache_dir)
        seen = spy_on_reads(monkeypatch)
        assert repository.last_sync_stamp() is None
        assert [path.name for path in seen] == [METADATA_FILE]

    def test_format_version_mismatch_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache_dir = _cache_with_snapshot(tmp_path)
        (cache_dir / METADATA_FILE).write_text(
            json.dumps(
                {
                    "last_sync": T0.isoformat(),
                    "format_version": CACHE_FORMAT_VERSION + 1,
                }
            )
        )
        repository = JsonTrainingRepository(cache_dir=cache_dir)
        seen = spy_on_reads(monkeypatch)
        assert repository.last_sync_stamp() is None
        assert [path.name for path in seen] == [METADATA_FILE]

    def test_absent_metadata_returns_none(self, tmp_path: Path) -> None:
        assert JsonTrainingRepository(cache_dir=tmp_path / "cache").last_sync_stamp() is None

    def test_corrupt_metadata_returns_none(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)
        (cache_dir / METADATA_FILE).write_text("not json at all")
        assert JsonTrainingRepository(cache_dir=cache_dir).last_sync_stamp() is None

    def test_naive_timestamp_is_read_as_utc(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        repository = JsonTrainingRepository(cache_dir=cache_dir)
        repository.save_sync_stamp(last_sync=T0.replace(tzinfo=None))
        assert repository.last_sync_stamp() == T0
