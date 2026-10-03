"""Tests for JSON cache records and source-outcome round trips."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from tests.conftest import (
    T0,
    StubSource,
    build_repository,
    make_sync_service,
    stub_activity_content,
    two_files,
)
from training_stats.adapters.cache import jsonio
from training_stats.adapters.cache.repository import (
    SNAPSHOT_FINGERPRINT_KEY,
    JsonTrainingRepository,
)
from training_stats.adapters.cache.schemas import CACHE_FILE, CACHE_FORMAT_VERSION, METADATA_FILE
from training_stats.services.state import SyncState


def spy_on_reads(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Record every JSON file the cache layer opens."""
    seen: list[Path] = []
    original = jsonio.read_json

    def spy(path: Path) -> dict[str, object] | None:
        seen.append(path)
        return original(path)

    monkeypatch.setattr(jsonio, "read_json", spy)
    return seen


def count_snapshot_writes(monkeypatch: pytest.MonkeyPatch) -> list[SyncState]:
    """Record every full-snapshot write performed by the repository."""
    writes: list[SyncState] = []
    original = JsonTrainingRepository.save_sync_snapshot

    def counting(self: JsonTrainingRepository, state: SyncState, *, last_sync: datetime) -> None:
        writes.append(state)
        original(self, state, last_sync=last_sync)

    monkeypatch.setattr(JsonTrainingRepository, "save_sync_snapshot", counting)
    return writes


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


def _write_snapshot_and_stamp(cache_dir: Path, *, last_sync: datetime) -> JsonTrainingRepository:
    """Repository whose metadata fingerprint describes an existing snapshot."""
    repository = JsonTrainingRepository(cache_dir=cache_dir)
    repository.save_sync_snapshot(SyncState(index_rows=[], activities={}), last_sync=last_sync)
    return repository


def _fingerprint(cache_dir: Path) -> dict[str, int]:
    info = (cache_dir / CACHE_FILE).stat()
    return {"size": info.st_size, "mtime_ns": info.st_mtime_ns, "inode": info.st_ino}


class TestJsonCache:
    def test_empty_cache_and_empty_sync(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        writes = count_snapshot_writes(monkeypatch)
        cache_dir = tmp_path / "cache"
        service = make_sync_service(tmp_path, [StubSource(files={})])
        assert service.sync().cache_updated_at is not None
        assert len(writes) == 1  # the empty sync wrote the snapshot exactly once

        snapshot = cache_dir / CACHE_FILE
        assert json.loads(snapshot.read_text()) == {
            "format_version": CACHE_FORMAT_VERSION,
            "activities": [],
            "source_index": [],
        }
        assert JsonTrainingRepository(cache_dir=cache_dir).load_sync_state().activities == {}
        # A fresh process throttles against the snapshot the empty sync wrote.
        assert JsonTrainingRepository(cache_dir=cache_dir).last_sync_stamp() is not None

        written = snapshot.read_bytes()
        modified = snapshot.stat().st_mtime_ns
        result = service.sync()
        assert result.cache_updated_at is not None
        assert len(writes) == 1  # unchanged: metadata-only, no snapshot rewrite
        assert snapshot.read_bytes() == written
        assert snapshot.stat().st_mtime_ns == modified
        metadata = json.loads((cache_dir / METADATA_FILE).read_text())
        assert metadata["last_sync"] == result.cache_updated_at.isoformat()
        assert metadata[SNAPSHOT_FINGERPRINT_KEY] == _fingerprint(cache_dir)

        snapshot.unlink()
        assert JsonTrainingRepository(cache_dir=cache_dir).last_sync_stamp() is None
        assert service.sync().cache_updated_at is not None
        assert len(writes) == 2  # the stamp only came back with a fresh snapshot
        assert JsonTrainingRepository(cache_dir=cache_dir).last_sync_stamp() is not None

        snapshot.write_text("not json")
        assert JsonTrainingRepository(cache_dir=cache_dir).last_sync_stamp() is None

    def test_noop_sync_reuses_snapshot_and_writes_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        files = {"a.fit": stub_activity_content(T0, "cycling", 1800.0, 20000.0)}
        service = make_sync_service(tmp_path, [StubSource(files=files)])
        assert service.sync().files_processed == 1

        snapshot = tmp_path / "cache" / CACHE_FILE
        written = snapshot.read_bytes()
        modified = snapshot.stat().st_mtime_ns
        writes = count_snapshot_writes(monkeypatch)
        seen = spy_on_reads(monkeypatch)

        assert service.sync().files_skipped == 1
        # The repository memo covers the just-written snapshot; only the small
        # metadata read is needed to validate the stamp fingerprint.
        assert [path.name for path in seen] == [METADATA_FILE]
        assert writes == []
        assert snapshot.read_bytes() == written
        assert snapshot.stat().st_mtime_ns == modified

        seen.clear()
        assert service.repository.last_sync_stamp() is not None
        assert [path.name for path in seen] == [METADATA_FILE]

    def test_snapshot_state_is_memoized_until_atomic_replacement(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache_dir = _cache_with_snapshot(tmp_path)
        snapshot = cache_dir / CACHE_FILE
        repository = JsonTrainingRepository(cache_dir)
        seen = spy_on_reads(monkeypatch)

        first = repository.load_sync_state()
        second = repository.load_sync_state()

        assert [path.name for path in seen] == [CACHE_FILE]
        assert first.index_rows is not second.index_rows
        assert first.activities is not second.activities

        jsonio.write_json_atomic({"format_version": 0}, snapshot)
        repaired = repository.load_sync_state()
        assert [path.name for path in seen] == [CACHE_FILE, CACHE_FILE]
        assert repaired.index_rows == []
        assert repaired.activities == {}

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
        cache_dir = tmp_path / "cache"
        metadata = json.loads((cache_dir / METADATA_FILE).read_text())
        assert set(metadata) == {"last_sync", "format_version", SNAPSHOT_FINGERPRINT_KEY}
        assert metadata["format_version"] == CACHE_FORMAT_VERSION
        assert datetime.fromisoformat(metadata["last_sync"]).tzinfo is not None
        assert metadata[SNAPSHOT_FINGERPRINT_KEY] == _fingerprint(cache_dir)

    def test_metadata_only_stamp_records_snapshot_fingerprint(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        repository = JsonTrainingRepository(cache_dir=cache_dir)
        repository.save_sync_snapshot(SyncState(index_rows=[], activities={}), last_sync=T0)
        snapshot = cache_dir / CACHE_FILE
        written = snapshot.read_bytes()

        repository.save_sync_stamp(
            SyncState(index_rows=[], activities={}), last_sync=T0 + timedelta(minutes=1)
        )
        assert snapshot.read_bytes() == written  # no snapshot serialization or replacement
        metadata = json.loads((cache_dir / METADATA_FILE).read_text())
        assert metadata[SNAPSHOT_FINGERPRINT_KEY] == _fingerprint(cache_dir)
        assert metadata["last_sync"] == (T0 + timedelta(minutes=1)).isoformat()

    @pytest.mark.parametrize("damage", ["missing", "malformed", "obsolete"])
    def test_stamp_repairs_damaged_snapshot_before_stamping(
        self, tmp_path: Path, damage: str
    ) -> None:
        cache_dir = tmp_path / "cache"
        repository = JsonTrainingRepository(cache_dir=cache_dir)
        repository.save_sync_snapshot(SyncState(index_rows=[], activities={}), last_sync=T0)
        snapshot = cache_dir / CACHE_FILE
        if damage == "missing":
            snapshot.unlink()
        elif damage == "malformed":
            snapshot.write_text("not json")
        else:
            snapshot.write_text('{"format_version": 0}')

        repository.save_sync_stamp(SyncState(index_rows=[], activities={}), last_sync=T0)
        assert json.loads(snapshot.read_text()) == {
            "format_version": CACHE_FORMAT_VERSION,
            "activities": [],
            "source_index": [],
        }
        assert repository.last_sync_stamp() == T0

    @pytest.mark.parametrize("metadata_case", ["absent", "legacy"])
    def test_stamp_writes_snapshot_without_a_matching_stamp(
        self, tmp_path: Path, metadata_case: str
    ) -> None:
        cache_dir = _cache_with_snapshot(tmp_path)
        if metadata_case == "legacy":
            (cache_dir / METADATA_FILE).write_text(json.dumps({"last_sync": T0.isoformat()}))
        snapshot = cache_dir / CACHE_FILE
        written = snapshot.read_bytes()
        repository = JsonTrainingRepository(cache_dir=cache_dir)

        repository.save_sync_stamp(SyncState(index_rows=[], activities={}), last_sync=T0)
        assert snapshot.read_bytes() != written  # stale metadata cannot vouch for a snapshot
        assert json.loads(snapshot.read_text())["format_version"] == CACHE_FORMAT_VERSION
        assert repository.last_sync_stamp() == T0


class TestLastSyncStamp:
    def test_reads_metadata_without_opening_snapshot(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repository = _write_snapshot_and_stamp(tmp_path / "cache", last_sync=T0)
        seen = spy_on_reads(monkeypatch)
        assert repository.last_sync_stamp() == T0
        assert [path.name for path in seen] == [METADATA_FILE]

    def test_removed_snapshot_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache_dir = tmp_path / "cache"
        repository = _write_snapshot_and_stamp(cache_dir, last_sync=T0)
        (cache_dir / CACHE_FILE).unlink()
        seen = spy_on_reads(monkeypatch)
        assert repository.last_sync_stamp() is None
        assert [path.name for path in seen] == [METADATA_FILE]

    @pytest.mark.parametrize(
        "alteration",
        ["truncate", "rewrite_same_size", "replace"],
        ids=["truncated", "rewritten-in-place", "replaced-file"],
    )
    def test_altered_snapshot_returns_none_without_reading_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alteration: str
    ) -> None:
        cache_dir = tmp_path / "cache"
        repository = _write_snapshot_and_stamp(cache_dir, last_sync=T0)
        snapshot = cache_dir / CACHE_FILE
        original = snapshot.read_bytes()
        if alteration == "truncate":
            snapshot.write_bytes(original[: len(original) // 2])
        elif alteration == "rewrite_same_size":
            info = snapshot.stat()
            snapshot.write_bytes(b" " * len(original))
            # Keep the size identical and bump mtime explicitly, so detection
            # relies on the fingerprint rather than filesystem granularity.
            os.utime(snapshot, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
        else:
            snapshot.unlink()
            snapshot.write_bytes(b'{"format_version": %d}' % CACHE_FORMAT_VERSION)

        seen = spy_on_reads(monkeypatch)
        assert repository.last_sync_stamp() is None
        assert [path.name for path in seen] == [METADATA_FILE]

    def test_legacy_metadata_without_fingerprint_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache_dir = _cache_with_snapshot(tmp_path)
        (cache_dir / METADATA_FILE).write_text(
            json.dumps(
                {
                    "last_sync": T0.isoformat(),
                    "format_version": CACHE_FORMAT_VERSION,
                }
            )
        )
        repository = JsonTrainingRepository(cache_dir=cache_dir)
        seen = spy_on_reads(monkeypatch)
        assert repository.last_sync_stamp() is None
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
        repository = _write_snapshot_and_stamp(
            tmp_path / "cache", last_sync=T0.replace(tzinfo=None)
        )
        assert repository.last_sync_stamp() == T0
