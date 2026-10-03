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
from training_stats.adapters.cache.repository import (
    SNAPSHOT_FINGERPRINT_KEY,
    JsonTrainingRepository,
)
from training_stats.adapters.errors import CacheError
from training_stats.domain.models import (
    STATUS_DUPLICATE,
    STATUS_FAILED,
    STATUS_OK,
    ActivityRecord,
    SourceFile,
)
from training_stats.services.errors import SourceError
from training_stats.services.state import SyncState
from training_stats.services.sync import SyncService


def one_file(duration_s: float = 1800.0) -> dict[str, bytes]:
    return {"a.fit": stub_activity_content(T0, "cycling", duration_s, 20000.0)}


def duplicate_pair() -> dict[str, bytes]:
    """Two files holding the same activity: ``a.fit`` accepted, ``b.fit`` duplicate."""
    shared = stub_activity_content(T0, "cycling", 1800.0, 20000.0)
    return {"a.fit": shared, "b.fit": shared}


def _distance_of(record: ActivityRecord) -> object:
    """Distance value of a record, comparable across rebuilt copies."""
    return next(metric.value for metric in record.metrics if metric.name == "activity.distance")


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


class FlakyReadSource(CountingSource):
    """Counting source whose reads fail on demand while listing stays intact."""

    def __init__(self, files: dict[str, bytes]) -> None:
        super().__init__(files)
        self.read_error: str | None = None

    def read_file(self, source_file: SourceFile) -> bytes:
        if self.read_error is not None:
            self.read_count += 1
            raise SourceError(self.read_error)
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
        metadata = json.loads((cache_dir / schemas.METADATA_FILE).read_text())
        info = snapshot_path.stat()
        assert metadata[SNAPSHOT_FINGERPRINT_KEY] == {
            "size": info.st_size,
            "mtime_ns": info.st_mtime_ns,
            "inode": info.st_ino,
        }
        assert JsonTrainingRepository(cache_dir).last_sync_stamp() == result.cache_updated_at

    def test_metadata_only_stamp_is_voided_by_snapshot_removal(self, tmp_path: Path) -> None:
        service, _source, cache_dir = make_sync(tmp_path, one_file())
        assert service.sync().files_processed == 1
        assert service.sync().files_skipped == 1  # metadata-only stamp
        assert JsonTrainingRepository(cache_dir).last_sync_stamp() is not None

        (cache_dir / schemas.CACHE_FILE).unlink()
        assert JsonTrainingRepository(cache_dir).last_sync_stamp() is None

    def test_corrupt_snapshot_is_repaired_before_the_stamp(self, tmp_path: Path) -> None:
        service, _source, cache_dir = make_sync(tmp_path, {})
        assert service.sync().cache_updated_at is not None
        snapshot = cache_dir / schemas.CACHE_FILE
        assert snapshot.exists()
        snapshot.write_text("not json")

        result = service.sync()
        assert result.cache_updated_at is not None
        payload = json.loads(snapshot.read_text())
        assert payload["format_version"] == schemas.CACHE_FORMAT_VERSION
        assert payload["activities"] == []
        assert payload["source_index"] == []
        assert JsonTrainingRepository(cache_dir).last_sync_stamp() is not None

        written = snapshot.read_bytes()
        assert service.sync().cache_updated_at is not None
        assert snapshot.read_bytes() == written  # repaired snapshot is now the fast path

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

    def test_changed_duplicate_file_keeps_the_canonical_activity(self, tmp_path: Path) -> None:
        """A duplicate-reference row never owns the record another file accepted."""
        shared = stub_activity_content(T0, "cycling", 1800.0, 20000.0)
        unique = stub_activity_content(T0 + timedelta(days=3), "running", 900.0, 10000.0)
        service, source, cache_dir = make_sync(tmp_path, {"a.fit": shared, "b.fit": shared})
        first = service.sync()
        assert first.files_processed == 2
        assert first.duplicates_detected == 1
        repository = JsonTrainingRepository(cache_dir)
        canonical = next(iter(repository.load_sync_state().activities))
        assert [row.status for row in repository.load_sync_state().index_rows] == [
            STATUS_OK,
            STATUS_DUPLICATE,
        ]

        source.files["b.fit"] = unique
        second = service.sync()

        assert second.files_processed == 1
        state = repository.load_sync_state()
        assert len(state.activities) == 2
        assert canonical in state.activities
        accepted = [row for row in state.index_rows if row.status == STATUS_OK]
        updated_id = next(iter(set(state.activities) - {canonical}))
        assert {(row.source_path, row.activity_id) for row in accepted} == {
            ("a.fit", canonical),
            ("b.fit", updated_id),
        }

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

    def test_invalid_cached_record_triggers_full_reprocessing(self, tmp_path: Path) -> None:
        service, _source, cache_dir = make_sync(tmp_path, one_file())
        assert service.sync().files_processed == 1
        cache_path = cache_dir / schemas.CACHE_FILE
        payload = json.loads(cache_path.read_text())
        payload["activities"] = [{}]
        cache_path.write_text(json.dumps(payload))

        result = service.sync()

        assert result.files_processed == 1
        state = JsonTrainingRepository(cache_dir).load_sync_state()
        assert len(state.activities) == 1
        assert next(iter(state.activities.values())).activity_id

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


class TestTransientReadFailure:
    def test_read_failure_keeps_last_known_good_record(self, tmp_path: Path) -> None:
        source = FlakyReadSource(one_file())
        service, _source, cache_dir = make_sync(tmp_path, source=source)
        assert service.sync().files_processed == 1
        repository = JsonTrainingRepository(cache_dir)
        original_id, original = next(iter(repository.load_sync_state().activities.items()))

        source.files["a.fit"] = stub_activity_content(T0.replace(day=2), "cycling", 1800.0, 20000.0)
        source.read_error = "transient read failure"
        failed = service.sync()

        assert failed.files_failed == 1
        assert failed.files_processed == 0
        state = repository.load_sync_state()
        assert state.activities == {original_id: original}
        failure_rows = [row for row in state.index_rows if row.status == STATUS_FAILED]
        assert len(failure_rows) == 1
        assert failure_rows[0].source_hash is None
        assert failure_rows[0].activity_id is None
        assert failure_rows[0].error_message == "transient read failure"
        assert any(
            row.status == STATUS_OK and row.activity_id == original_id for row in state.index_rows
        )

        source.read_error = None
        retried = service.sync()

        assert retried.files_processed == 1
        state = repository.load_sync_state()
        assert len(state.activities) == 1
        updated = next(iter(state.activities.values()))
        assert updated.activity_id != original_id
        assert updated.start_time.day == 2
        assert not any(row.status == STATUS_FAILED for row in state.index_rows)
        assert {row.activity_id for row in state.index_rows if row.status == STATUS_OK} == {
            updated.activity_id
        }

    def test_repeated_identical_read_failure_reuses_the_failure_row(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = FlakyReadSource(one_file())
        service, _source, cache_dir = make_sync(tmp_path, source=source)
        assert service.sync().files_processed == 1
        source.files["a.fit"] = stub_activity_content(T0.replace(day=2), "cycling", 1800.0, 20000.0)
        source.read_error = "permission denied"
        assert service.sync().files_failed == 1

        snapshot_path = cache_dir / schemas.CACHE_FILE
        snapshot_bytes = snapshot_path.read_bytes()
        snapshot_mtime = snapshot_path.stat().st_mtime_ns
        writes = count_snapshot_writes(monkeypatch)

        for _ in range(3):
            assert service.sync().files_failed == 1

        assert writes == []
        assert snapshot_path.read_bytes() == snapshot_bytes
        assert snapshot_path.stat().st_mtime_ns == snapshot_mtime
        state = JsonTrainingRepository(cache_dir).load_sync_state()
        failed_rows = [row for row in state.index_rows if row.status == STATUS_FAILED]
        assert len(failed_rows) == 1
        assert len(state.activities) == 1


class TestDuplicatePromotion:
    def test_deleted_canonical_promotes_the_unchanged_duplicate(self, tmp_path: Path) -> None:
        service, source, cache_dir = make_sync(tmp_path, duplicate_pair())
        first = service.sync()
        assert first.files_processed == 2
        assert first.duplicates_detected == 1
        repository = JsonTrainingRepository(cache_dir)
        state = repository.load_sync_state()
        canonical_id, canonical = next(iter(state.activities.items()))

        del source.files["a.fit"]
        second = service.sync()

        assert second.files_discovered == 1
        assert second.files_processed == 1
        assert second.files_skipped == 0
        assert second.files_failed == 0
        state = repository.load_sync_state()
        assert set(state.activities) == {canonical_id}
        promoted = state.activities[canonical_id]
        assert promoted.sport == canonical.sport
        assert promoted.source_path == "b.fit"
        assert _distance_of(promoted) == _distance_of(canonical)
        assert [(row.source_path, row.status, row.activity_id) for row in state.index_rows] == [
            ("b.fit", STATUS_OK, canonical_id)
        ]
        assert state.index_rows[0].source_hash is not None

    def test_changed_canonical_promotes_the_stale_duplicate(self, tmp_path: Path) -> None:
        service, source, cache_dir = make_sync(tmp_path, duplicate_pair())
        service.sync()
        repository = JsonTrainingRepository(cache_dir)
        old_id = next(iter(repository.load_sync_state().activities))

        source.files["a.fit"] = stub_activity_content(
            T0 + timedelta(days=2), "running", 900.0, 10000.0
        )
        second = service.sync()

        assert second.files_processed == 2
        assert second.files_skipped == 0
        assert second.duplicates_detected == 0
        state = repository.load_sync_state()
        new_id = next(iter(set(state.activities) - {old_id}))
        assert state.activities[old_id].sport == "cycling"
        assert state.activities[new_id].sport == "running"
        assert {(row.source_path, row.activity_id) for row in state.index_rows} == {
            ("a.fit", new_id),
            ("b.fit", old_id),
        }
        assert all(row.status == STATUS_OK for row in state.index_rows)

    def test_unchanged_duplicate_is_not_reread_while_its_owner_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = CountingSource(duplicate_pair())
        service, _source, cache_dir = make_sync(tmp_path, source=source)
        assert service.sync().files_processed == 2
        assert source.read_count == 2
        snapshot_path = cache_dir / schemas.CACHE_FILE
        snapshot_bytes = snapshot_path.read_bytes()
        writes = count_snapshot_writes(monkeypatch)

        second = service.sync()

        assert second.files_skipped == 2
        assert second.files_processed == 0
        assert source.read_count == 2
        assert writes == []
        assert snapshot_path.read_bytes() == snapshot_bytes

    def test_failed_source_duplicate_is_not_promoted(self, tmp_path: Path) -> None:
        shared = stub_activity_content(T0, "cycling", 1800.0, 20000.0)
        local = StubSource(name="local", files={"a.fit": shared})
        cloud = StubSource(name="cloud", files={"b.fit": shared})
        assert make_sync_service(tmp_path, [local, cloud]).sync().duplicates_detected == 1
        repository = JsonTrainingRepository(tmp_path / "cache")
        canonical_id, canonical = next(iter(repository.load_sync_state().activities.items()))

        local.files = {}
        offline = make_sync_service(tmp_path, [OfflineSource(name="cloud"), local]).sync()
        assert offline.source_errors == ("cloud: offline",)
        assert offline.files_processed == 0
        assert offline.files_failed == 0
        state = repository.load_sync_state()
        assert state.activities == {}
        assert [(row.source_name, row.source_path, row.status) for row in state.index_rows] == [
            ("cloud", "b.fit", STATUS_DUPLICATE)
        ]

        cloud.files = {"b.fit": shared}
        healed = make_sync_service(tmp_path, [cloud, local]).sync()

        assert healed.files_processed == 1
        state = repository.load_sync_state()
        assert set(state.activities) == {canonical_id}
        assert state.activities[canonical_id].sport == canonical.sport
        assert [(row.source_path, row.status) for row in state.index_rows] == [("b.fit", STATUS_OK)]

    def test_changed_duplicate_promotes_when_its_canonical_was_deleted(
        self, tmp_path: Path
    ) -> None:
        service, source, cache_dir = make_sync(tmp_path, duplicate_pair())
        service.sync()
        repository = JsonTrainingRepository(cache_dir)
        canonical_id = next(iter(repository.load_sync_state().activities))

        del source.files["a.fit"]
        # Same activity id (start/sport/duration unchanged) with new bytes.
        source.files["b.fit"] = stub_activity_content(T0, "cycling", 1800.0, 25000.0)
        second = service.sync()

        assert second.files_processed == 1
        assert second.files_skipped == 0
        assert second.files_failed == 0
        assert second.duplicates_detected == 0
        state = repository.load_sync_state()
        assert set(state.activities) == {canonical_id}
        assert [(row.source_path, row.status) for row in state.index_rows] == [("b.fit", STATUS_OK)]

    def test_promotion_reprocesses_every_segment_of_the_surviving_file(
        self, tmp_path: Path
    ) -> None:
        shared = stub_activity_content(T0, "cycling", 1800.0, 20000.0)
        extra = stub_activity_content(T0 + timedelta(days=4), "running", 600.0, 5000.0)
        service, source, cache_dir = make_sync(
            tmp_path, {"a.fit": shared, "b.fit": shared + b"\n" + extra}
        )
        first = service.sync()
        assert first.files_processed == 2
        assert first.duplicates_detected == 1
        repository = JsonTrainingRepository(cache_dir)
        original_ids = set(repository.load_sync_state().activities)
        assert len(original_ids) == 2

        del source.files["a.fit"]
        second = service.sync()

        assert second.files_processed == 1
        state = repository.load_sync_state()
        assert set(state.activities) == original_ids
        assert [row.source_path for row in state.index_rows] == ["b.fit", "b.fit"]
        assert [row.status for row in state.index_rows] == [STATUS_OK, STATUS_OK]

    def test_only_one_of_several_copies_is_promoted(self, tmp_path: Path) -> None:
        shared = stub_activity_content(T0, "cycling", 1800.0, 20000.0)
        service, source, cache_dir = make_sync(
            tmp_path, {"a.fit": shared, "b.fit": shared, "c.fit": shared}
        )
        first = service.sync()
        assert first.files_processed == 3
        assert first.duplicates_detected == 2
        repository = JsonTrainingRepository(cache_dir)
        canonical_id = next(iter(repository.load_sync_state().activities))

        del source.files["a.fit"]
        second = service.sync()

        assert second.files_processed == 1
        assert second.files_skipped == 1
        assert second.duplicates_detected == 0
        state = repository.load_sync_state()
        assert set(state.activities) == {canonical_id}
        assert [(row.source_path, row.status) for row in state.index_rows] == [
            ("b.fit", STATUS_OK),
            ("c.fit", STATUS_DUPLICATE),
        ]

    def test_promotion_read_failure_counts_as_failed_and_stays_retryable(
        self, tmp_path: Path
    ) -> None:
        source = FlakyReadSource(duplicate_pair())
        service, _source, cache_dir = make_sync(tmp_path, source=source)
        assert service.sync().files_processed == 2
        repository = JsonTrainingRepository(cache_dir)
        canonical_id = next(iter(repository.load_sync_state().activities))

        del source.files["a.fit"]
        source.read_error = "vanished mid-fetch"
        failed = service.sync()

        assert failed.files_processed == 0
        assert failed.files_skipped == 0
        assert failed.files_failed == 1
        state = repository.load_sync_state()
        assert state.activities == {}
        assert [row.status for row in state.index_rows] == [STATUS_DUPLICATE, STATUS_FAILED]
        assert state.index_rows[1].source_hash is None

        source.read_error = None
        healed = service.sync()

        assert healed.files_processed == 1
        assert set(repository.load_sync_state().activities) == {canonical_id}


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
