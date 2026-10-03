"""Tests for atomic JSON cache persistence, locking, and incremental sync."""

from __future__ import annotations

import fcntl
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from tests.conftest import T0, StubSource, make_sync_service, stub_activity_content
from training_stats.adapters.cache import jsonio, schemas
from training_stats.adapters.cache.locking import acquire_app_lock
from training_stats.adapters.cache.repository import JsonTrainingRepository
from training_stats.adapters.errors import CacheError
from training_stats.domain.models import SourceFile
from training_stats.services.errors import SourceError
from training_stats.services.state import SyncState
from training_stats.services.sync import SyncService


def one_file(duration_s: float = 1800.0) -> dict[str, bytes]:
    return {"a.fit": stub_activity_content(T0, "cycling", duration_s, 20000.0)}


def make_sync(
    tmp_path: Path,
    files: dict[str, bytes] | None = None,
    *,
    version: str = "1",
    source: StubSource | None = None,
) -> tuple[SyncService, StubSource, Path]:
    stub = source or StubSource(files=files)
    service = make_sync_service(tmp_path, [stub], version=version)
    return service, stub, tmp_path / "cache"


class CountingSource(StubSource):
    def __init__(self, files: dict[str, bytes]) -> None:
        super().__init__(files=files)
        self.read_count = 0

    def read_file(self, source_file: SourceFile) -> bytes:
        self.read_count += 1
        return super().read_file(source_file)


class OfflineSource(StubSource):
    """A configured source whose directory listing is unavailable."""

    def list_files(self) -> list[SourceFile]:
        raise SourceError("offline")


def two_stub_sources() -> tuple[StubSource, StubSource]:
    cloud = StubSource(
        name="cloud",
        files={"ride.fit": stub_activity_content(T0, "cycling", 1800.0, 20000.0)},
    )
    local = StubSource(
        name="local",
        files={"run.fit": stub_activity_content(T0 + timedelta(days=1), "running", 900.0, 10000.0)},
    )
    return cloud, local


def count_snapshot_writes(monkeypatch: pytest.MonkeyPatch) -> list[SyncState]:
    """Record every full-snapshot write performed by the repository."""
    calls: list[SyncState] = []
    original = JsonTrainingRepository.save_sync_snapshot

    def counting(self: JsonTrainingRepository, state: SyncState, *, last_sync: datetime) -> None:
        calls.append(state)
        original(self, state, last_sync=last_sync)

    monkeypatch.setattr(JsonTrainingRepository, "save_sync_snapshot", counting)
    return calls


def metadata_stamp(cache_dir: Path) -> str:
    return str(json.loads((cache_dir / schemas.METADATA_FILE).read_text())["last_sync"])


class TestAtomicJson:
    def test_round_trip_and_no_tmp_files(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "cache.json"
        jsonio.write_json_atomic({"value": [1, 2, 3]}, path)
        assert jsonio.read_json(path) == {"value": [1, 2, 3]}
        assert [item.name for item in path.parent.iterdir()] == [path.name]

    def test_missing_and_corrupt_json(self, tmp_path: Path) -> None:
        assert jsonio.read_json(tmp_path / "missing.json") is None
        broken = tmp_path / "broken.json"
        broken.write_text("not json")
        assert jsonio.read_json(broken) is None

    def test_write_fsyncs_once_before_replace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Durability: one fsync of the temporary file, before os.replace."""
        syncs: list[int] = []
        real_fsync = os.fsync

        def counting_fsync(descriptor: int) -> None:
            syncs.append(descriptor)
            real_fsync(descriptor)

        monkeypatch.setattr(os, "fsync", counting_fsync)
        path = tmp_path / "cache.json"
        jsonio.write_json_atomic({"value": [1, 2, 3]}, path)
        assert len(syncs) == 1
        assert jsonio.read_json(path) == {"value": [1, 2, 3]}


class TestAppLock:
    def test_acquire_is_idempotent(self, tmp_path: Path) -> None:
        acquire_app_lock(tmp_path, 1.0)
        acquire_app_lock(tmp_path, 1.0)

    def test_lock_conflict_raises(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        handle = os.open(cache_dir / schemas.LOCK_FILE, os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
            with pytest.raises(CacheError, match="another instance"):
                acquire_app_lock(cache_dir, 0.1)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
            os.close(handle)


class TestIncrementalSync:
    def test_initial_and_idempotent_sync(self, tmp_path: Path) -> None:
        service, _source, cache_dir = make_sync(tmp_path, one_file())
        first = service.sync()
        second = service.sync()
        assert first.files_processed == 1
        assert second.files_skipped == 1
        assert second.files_processed == 0
        assert (cache_dir / schemas.CACHE_FILE).exists()
        assert (cache_dir / schemas.METADATA_FILE).exists()
        payload = json.loads((cache_dir / schemas.CACHE_FILE).read_text())
        assert payload["format_version"] == schemas.CACHE_FORMAT_VERSION
        assert len(payload["activities"]) == 1

    def test_unchanged_source_is_not_read_again(self, tmp_path: Path) -> None:
        source = CountingSource(one_file())
        service, _source, _cache_dir = make_sync(tmp_path, source=source)
        service.sync()
        service.sync()
        assert source.read_count == 1

    def test_noop_sync_updates_only_stamp_metadata(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service, _source, cache_dir = make_sync(tmp_path, one_file())
        assert service.sync().files_processed == 1
        snapshot_path = cache_dir / schemas.CACHE_FILE
        snapshot_bytes = snapshot_path.read_bytes()
        snapshot_mtime = snapshot_path.stat().st_mtime_ns
        writes = count_snapshot_writes(monkeypatch)

        result = service.sync()

        assert result.files_skipped == 1
        assert result.files_processed == 0
        assert result.cache_updated_at is not None
        assert writes == []
        assert snapshot_path.read_bytes() == snapshot_bytes
        assert snapshot_path.stat().st_mtime_ns == snapshot_mtime
        assert metadata_stamp(cache_dir) == result.cache_updated_at.isoformat()

    def test_content_change_reprocesses_activity(self, tmp_path: Path) -> None:
        service, source, cache_dir = make_sync(tmp_path, one_file())
        service.sync()
        source.files["a.fit"] = stub_activity_content(T0, "cycling", 3600.0, 22000.0)
        result = service.sync()
        assert result.files_processed == 1
        state = JsonTrainingRepository(cache_dir).load_sync_state()
        activity = next(iter(state.activities.values()))
        assert activity.analysis_version == "1"
        distance = next(metric for metric in activity.metrics if metric.name == "activity.distance")
        assert distance.value is not None

    def test_analysis_version_reprocesses_every_file(self, tmp_path: Path) -> None:
        service, source, cache_dir = make_sync(tmp_path, one_file(), version="1")
        service.sync()
        new_service, _source, _cache = make_sync(tmp_path, source=source, version="2")
        result = new_service.sync()
        assert result.files_processed == 1
        state = JsonTrainingRepository(cache_dir).load_sync_state()
        assert {entry.source_analysis_version for entry in state.index_rows} == {"2"}

    def test_changed_file_removes_old_activity_and_adds_new(self, tmp_path: Path) -> None:
        service, source, cache_dir = make_sync(tmp_path, one_file())
        service.sync()
        source.files["a.fit"] = stub_activity_content(T0.replace(day=2), "cycling", 1800.0, 20000.0)
        service.sync()
        state = JsonTrainingRepository(cache_dir).load_sync_state()
        assert len(state.activities) == 1
        assert next(iter(state.activities.values())).start_time.day == 2

    def test_removed_file_is_purged(self, tmp_path: Path) -> None:
        service, source, cache_dir = make_sync(tmp_path, one_file())
        service.sync()
        source.files.clear()
        service.sync()
        state = JsonTrainingRepository(cache_dir).load_sync_state()
        assert state.activities == {}
        assert state.index_rows == []

    def test_failed_file_is_retried_without_hash(self, tmp_path: Path) -> None:
        class FailingSource(CountingSource):
            fail = True

            def read_file(self, source_file: SourceFile) -> bytes:
                self.read_count += 1
                if self.fail:
                    raise SourceError("temporary failure")
                return StubSource.read_file(self, source_file)

        source = FailingSource(one_file())
        service, _source, cache_dir = make_sync(tmp_path, source=source)
        assert service.sync().files_failed == 1
        source.fail = False
        assert service.sync().files_processed == 1
        assert source.read_count == 2
        assert len(JsonTrainingRepository(cache_dir).load_sync_state().activities) == 1

    def test_failed_parse_is_kept_as_source_outcome(self, tmp_path: Path) -> None:
        service, source, cache_dir = make_sync(tmp_path, {})
        source.files["broken.fit"] = b"garbage"
        assert service.sync().files_failed == 1
        state = JsonTrainingRepository(cache_dir).load_sync_state()
        assert state.activities == {}
        assert len(state.index_rows) == 1
        assert state.index_rows[0].error_message is not None

    def test_full_resync_after_cache_format_change(self, tmp_path: Path) -> None:
        service, source, cache_dir = make_sync(tmp_path, one_file())
        service.sync()
        path = cache_dir / schemas.CACHE_FILE
        payload = json.loads(path.read_text())
        payload["format_version"] = 0
        path.write_text(json.dumps(payload))
        new_service, _source, _cache_dir = make_sync(tmp_path, source=source)
        assert new_service.sync().files_processed == 1

    def test_failed_source_listing_does_not_overwrite_cache(self, tmp_path: Path) -> None:
        service, source, cache_dir = make_sync(tmp_path, one_file())
        service.sync()
        cache_before = (cache_dir / schemas.CACHE_FILE).read_bytes()
        metadata_before = (cache_dir / schemas.METADATA_FILE).read_bytes()

        offline = make_sync_service(tmp_path, [OfflineSource(name=source.name)])
        result = offline.sync()
        assert result.cache_updated_at is None
        assert (cache_dir / schemas.CACHE_FILE).read_bytes() == cache_before
        assert (cache_dir / schemas.METADATA_FILE).read_bytes() == metadata_before

    def test_raw_time_series_is_not_serialized(self, tmp_path: Path) -> None:
        service, _source, cache_dir = make_sync(tmp_path, one_file())
        service.sync()
        content = (cache_dir / schemas.CACHE_FILE).read_text()
        assert '"timestamp"' not in content
        assert '"records":' not in content


class TestMixedSourceListingFailure:
    def test_failed_source_keeps_rows_while_listed_source_changes(self, tmp_path: Path) -> None:
        cloud, local = two_stub_sources()
        first = make_sync_service(tmp_path, [cloud, local]).sync()
        assert first.files_processed == 2
        cache_dir = tmp_path / "cache"
        prior = JsonTrainingRepository(cache_dir).load_sync_state()
        preserved = next(
            record for record in prior.activities.values() if record.sport == "running"
        )
        stale_id = next(
            record.activity_id for record in prior.activities.values() if record.sport == "cycling"
        )

        cloud.files = {
            "century.fit": stub_activity_content(
                T0 + timedelta(days=2), "cycling", 7200.0, 200000.0
            )
        }
        service = make_sync_service(tmp_path, [cloud, OfflineSource(name="local")])
        result = service.sync()

        assert result.cache_updated_at is not None
        assert result.source_errors == ("local: offline",)
        assert result.files_discovered == 1
        assert result.files_processed == 1
        state = JsonTrainingRepository(cache_dir).load_sync_state()
        assert state.activities[preserved.activity_id] == preserved
        assert stale_id not in state.activities
        local_rows = [row for row in state.index_rows if row.source_name == "local"]
        cloud_rows = [row for row in state.index_rows if row.source_name == "cloud"]
        assert [(row.source_path, row.status) for row in local_rows] == [("run.fit", "ok")]
        assert [row.source_path for row in cloud_rows] == ["century.fit"]
        assert len(state.activities) == 2

    def test_failed_source_with_unchanged_peer_only_updates_stamp(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cloud, local = two_stub_sources()
        assert make_sync_service(tmp_path, [cloud, local]).sync().files_processed == 2
        cache_dir = tmp_path / "cache"
        snapshot_path = cache_dir / schemas.CACHE_FILE
        snapshot_bytes = snapshot_path.read_bytes()
        snapshot_mtime = snapshot_path.stat().st_mtime_ns
        prior = JsonTrainingRepository(cache_dir).load_sync_state()
        writes = count_snapshot_writes(monkeypatch)

        service = make_sync_service(tmp_path, [cloud, OfflineSource(name="local")])
        result = service.sync()

        assert result.cache_updated_at is not None
        assert result.files_skipped == 1
        assert result.source_errors == ("local: offline",)
        assert writes == []
        assert snapshot_path.read_bytes() == snapshot_bytes
        assert snapshot_path.stat().st_mtime_ns == snapshot_mtime
        assert metadata_stamp(cache_dir) == result.cache_updated_at.isoformat()
        state = JsonTrainingRepository(cache_dir).load_sync_state()
        assert state.activities == prior.activities
        assert state.index_rows == prior.index_rows
