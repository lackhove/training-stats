"""Tests for the Parquet cache: atomic writes, locking and sync behavior."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from tests.conftest import (
    StubParser,
    StubSource,
    make_app_config,
    make_processor,
    stub_activity_content,
)
from training_stats.cache import parquet, schemas
from training_stats.cache.locking import sync_lock
from training_stats.cache.sync import SyncService
from training_stats.errors import CacheError, SourceError
from training_stats.fit.types import ParsedFile
from training_stats.models import SourceFile

T0 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)


def make_sync(
    tmp_path: Path,
    files: dict[str, bytes] | None = None,
    analysis_version: str = "1",
    source: StubSource | None = None,
) -> tuple[SyncService, StubSource, Path]:
    source = source or StubSource(files=files)
    cache_dir = tmp_path / "cache"
    config = make_app_config(cache_dir, tmp_path / "src")
    service = SyncService(
        sources=[source],
        processor=make_processor(athlete=config.athlete, analysis_version=analysis_version),
        cache_dir=cache_dir,
        timezone=config.application.zoneinfo,
        analysis_version=analysis_version,
    )
    return service, source, cache_dir


class _CountingSource(StubSource):
    """Stub source that counts read_file calls (downloads)."""

    def __init__(self, files: dict[str, bytes]) -> None:
        super().__init__(files=files)
        self.read_count = 0

    def read_file(self, source_file: SourceFile) -> bytes:
        self.read_count += 1
        return super().read_file(source_file)


class _MetadatalessSource(_CountingSource):
    """Stub source that reports no size/mtime metadata."""

    def list_files(self) -> list[SourceFile]:
        return [
            SourceFile(source_name=self.name, path=path, size=None, modified_at=None)
            for path in sorted(self.files)
        ]


def one_file(**kwargs: object) -> dict[str, bytes]:
    values: dict[str, object] = {
        "start": T0,
        "sport": "cycling",
        "duration_s": 1800.0,
        "distance_m": 20000.0,
    }
    values.update(kwargs)
    return {"a.fit": stub_activity_content(**values)}  # type: ignore[arg-type]


class TestParquetHelpers:
    def test_atomic_write_roundtrip(self, tmp_path: Path) -> None:
        frame = pl.DataFrame({"x": [1, 2, 3]})
        target = tmp_path / "sub" / "data.parquet"
        parquet.write_parquet_atomic(frame, target)
        assert parquet.read_parquet(target).equals(frame)

    def test_write_leaves_no_tmp_files(self, tmp_path: Path) -> None:
        frame = pl.DataFrame({"x": [1]})
        parquet.write_parquet_atomic(frame, tmp_path / "data.parquet")
        assert [p.name for p in tmp_path.iterdir()] == ["data.parquet"]

    def test_json_atomic(self, tmp_path: Path) -> None:
        target = tmp_path / "meta.json"
        parquet.write_json_atomic({"a": 1}, target)
        assert json.loads(target.read_text()) == {"a": 1}

    def test_read_missing_returns_empty(self, tmp_path: Path) -> None:
        assert parquet.read_parquet(tmp_path / "nope.parquet").is_empty()

    def test_read_corrupt_raises_cache_error(self, tmp_path: Path) -> None:
        target = tmp_path / "bad.parquet"
        target.write_bytes(b"not parquet")
        with pytest.raises(CacheError, match="cannot read"):
            parquet.read_parquet(target)


class TestSyncLock:
    def test_lock_is_reentrant_across_processes(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "sync.lock"
        with sync_lock(lock_path, 1.0):
            pass  # released properly
        with sync_lock(lock_path, 1.0):
            pass

    def test_lock_conflict_raises(self, tmp_path: Path) -> None:
        import fcntl
        import os

        lock_path = tmp_path / "sync.lock"
        handle = os.open(lock_path, os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
            with pytest.raises(CacheError, match="sync lock"), sync_lock(lock_path, 0.2):
                pass
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
            os.close(handle)


class TestIncrementalSync:
    def test_initial_sync(self, tmp_path: Path) -> None:
        service, _source, cache_dir = make_sync(tmp_path, one_file())
        result = service.sync()
        assert result.files_discovered == 1
        assert result.files_processed == 1
        assert result.files_skipped == 0
        assert result.files_failed == 0
        assert (cache_dir / schemas.ACTIVITIES_FILE).exists()
        assert (cache_dir / schemas.DAILY_FILE).exists()
        assert (cache_dir / schemas.WEEKLY_FILE).exists()
        metadata = json.loads((cache_dir / schemas.METADATA_FILE).read_text())
        assert metadata["activity_count"] == 1

    def test_repeated_sync_is_idempotent(self, tmp_path: Path) -> None:
        service, _source, cache_dir = make_sync(tmp_path, one_file())
        first = service.sync()
        second = service.sync()
        assert second.files_processed == 0
        assert second.files_skipped == 1
        frame = parquet.read_parquet(cache_dir / schemas.ACTIVITIES_FILE)
        assert frame.height == 1
        assert first.cache_updated_at is not None

    def test_unchanged_files_not_downloaded(self, tmp_path: Path) -> None:
        source = _CountingSource(one_file())
        service, _source, _cache_dir = make_sync(tmp_path, source=source)
        service.sync()
        assert source.read_count == 1
        result = service.sync()
        assert source.read_count == 1  # metadata matched: no second read
        assert result.files_skipped == 1
        assert result.files_processed == 0

    def test_changed_content_triggers_read(self, tmp_path: Path) -> None:
        source = _CountingSource(one_file(duration_s=1800.0))
        service, _source, _cache_dir = make_sync(tmp_path, source=source)
        service.sync()
        # Same size on purpose: only the content-derived mtime differs.
        source.files["a.fit"] = stub_activity_content(T0, "cycling", 3600.0, 20000.0)
        result = service.sync()
        assert source.read_count == 2
        assert result.files_processed == 1

    def test_missing_metadata_always_reads(self, tmp_path: Path) -> None:
        source = _MetadatalessSource(one_file())
        service, _source, _cache_dir = make_sync(tmp_path, source=source)
        service.sync()
        result = service.sync()
        # No metadata short-circuit: every sync re-reads and reprocesses.
        assert source.read_count == 2
        assert result.files_skipped == 0
        assert result.files_processed == 1

    def test_failed_read_retries_with_matching_metadata(self, tmp_path: Path) -> None:
        class _FailingReadSource(_CountingSource):
            def __init__(self, files: dict[str, bytes]) -> None:
                super().__init__(files)
                self.fail_reads = True

            def read_file(self, source_file: SourceFile) -> bytes:
                self.read_count += 1
                if self.fail_reads:
                    msg = "simulated network failure"
                    raise SourceError(msg)
                return StubSource.read_file(self, source_file)

        source = _FailingReadSource(one_file())
        service, _source, _cache_dir = make_sync(tmp_path, source=source)
        result = service.sync()
        assert result.files_failed == 1
        # A failed read leaves no stored hash: the next sync must retry even
        # though size and mtime are identical.
        source.fail_reads = False
        result = service.sync()
        assert source.read_count == 2
        assert result.files_processed == 1

    def test_changed_file_reprocessed(self, tmp_path: Path) -> None:
        service, source, cache_dir = make_sync(tmp_path, one_file(duration_s=1800.0))
        service.sync()
        source.files["a.fit"] = stub_activity_content(T0, "cycling", 3600.0, 20000.0)
        result = service.sync()
        assert result.files_processed == 1
        frame = parquet.read_parquet(cache_dir / schemas.ACTIVITIES_FILE)
        assert frame.height == 1
        assert frame["duration_s"].to_list() == [3600.0]

    def test_analysis_version_invalidates(self, tmp_path: Path) -> None:
        service, _source, cache_dir = make_sync(tmp_path, one_file(), analysis_version="1")
        service.sync()
        source = service.sources[0]
        assert isinstance(source, StubSource)
        new_service, _unused, _dir = make_sync(tmp_path, source.files, analysis_version="2")
        result = new_service.sync()
        assert result.files_processed == 1
        metrics = parquet.read_parquet(cache_dir / schemas.ACTIVITY_METRICS_FILE)
        assert metrics["analysis_version"].to_list() == ["2:1"]

    def test_histogram_and_buckets_round_trip(self, tmp_path: Path) -> None:
        content = stub_activity_content(T0, "cycling", 1800.0, 20000.0, heart_rates=[130] * 30)
        service, _source, cache_dir = make_sync(tmp_path, {"a.fit": content}, analysis_version="2")
        service.sync()
        metrics = parquet.read_parquet(cache_dir / schemas.ACTIVITY_METRICS_FILE)
        row = metrics.to_dicts()[0]
        assert row["hr_histogram_json"] is not None
        histogram = json.loads(str(row["hr_histogram_json"]))
        assert histogram == {"130-134": 870.0}  # 29 intervals x 30 s cap
        buckets = json.loads(str(row["hr_10min_json"]))
        # 29 capped 30 s intervals split across three 10-min buckets.
        assert buckets == [
            {"start_min": 0, "avg_hr": 130.0, "hr_time_s": 300.0},
            {"start_min": 10, "avg_hr": 130.0, "hr_time_s": 300.0},
            {"start_min": 20, "avg_hr": 130.0, "hr_time_s": 270.0},
        ]
        assert row["hr_drift_bpm_per_hour"] is not None
        assert abs(float(row["hr_drift_bpm_per_hour"])) < 1e-9  # constant HR

    def test_removed_files_dropped_from_index(self, tmp_path: Path) -> None:
        service, source, _cache_dir = make_sync(tmp_path, one_file())
        service.sync()
        source.files.clear()
        service.sync()
        index = parquet.read_parquet(service.cache_dir / schemas.SOURCE_INDEX_FILE)
        assert index.is_empty()

    def test_sync_logs_summary_and_duplicates(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        content = stub_activity_content(T0, "cycling", 1800.0, 20000.0)
        service, _source, _cache_dir = make_sync(tmp_path, {"one.fit": content, "two.fit": content})
        with caplog.at_level(logging.INFO):
            service.sync()
        messages = [record.getMessage() for record in caplog.records]
        assert any(
            "sync complete: 2 discovered, 2 processed, 0 skipped, 0 failed, "
            "1 duplicates, 0 repaired files, 0 discarded segments" in message
            for message in messages
        )
        assert any("duplicate stub/two.fit: activity" in message for message in messages)


class TestCacheProtection:
    """A sync without any successful listing must never blank the cache."""

    def test_all_sources_failing_leaves_cache_untouched(self, tmp_path: Path) -> None:
        service, source, cache_dir = make_sync(tmp_path, one_file())
        service.sync()
        before = parquet.read_parquet(cache_dir / schemas.ACTIVITIES_FILE)

        class FailingSource(StubSource):
            def list_files(self) -> list[SourceFile]:
                msg = "remote down"
                raise SourceError(msg)

        broken = SyncService(
            sources=[FailingSource(name="stub", files=source.files)],
            processor=make_processor(
                athlete=make_app_config(cache_dir, tmp_path / "src").athlete,
                analysis_version="1",
            ),
            cache_dir=cache_dir,
            timezone=make_app_config(cache_dir, tmp_path / "src").application.zoneinfo,
            analysis_version="1",
        )
        result = broken.sync()
        assert result.files_failed == 0
        assert result.cache_updated_at is None
        assert "stub: remote down" in result.source_errors
        after = parquet.read_parquet(cache_dir / schemas.ACTIVITIES_FILE)
        assert after.equals(before)

    def test_no_sources_leaves_cache_untouched(self, tmp_path: Path) -> None:
        service, _source, cache_dir = make_sync(tmp_path, one_file())
        service.sync()
        before = parquet.read_parquet(cache_dir / schemas.ACTIVITIES_FILE)
        config = make_app_config(cache_dir, tmp_path / "src")
        empty = SyncService(
            sources=[],
            processor=make_processor(athlete=config.athlete, analysis_version="1"),
            cache_dir=cache_dir,
            timezone=config.application.zoneinfo,
            analysis_version="1",
        )
        result = empty.sync()
        assert result.files_discovered == 0
        assert result.cache_updated_at is None
        assert parquet.read_parquet(cache_dir / schemas.ACTIVITIES_FILE).equals(before)

    def test_emptied_source_purges_cache(self, tmp_path: Path) -> None:
        """A successful listing with zero files legitimately purges the cache."""
        service, source, cache_dir = make_sync(tmp_path, one_file())
        service.sync()
        source.files.clear()
        result = service.sync()
        assert result.files_discovered == 0
        assert result.cache_updated_at is not None
        assert parquet.read_parquet(cache_dir / schemas.ACTIVITIES_FILE).is_empty()


class TestFailureHandling:
    def test_failed_file_recorded(self, tmp_path: Path) -> None:
        service, source, cache_dir = make_sync(tmp_path, {})
        source.files["broken.fit"] = b"garbage"
        result = service.sync()
        assert result.files_failed == 1
        index = parquet.read_parquet(cache_dir / schemas.SOURCE_INDEX_FILE)
        row = index.to_dicts()[0]
        assert row["status"] == "failed"
        assert row["error_message"] is not None
        metadata = json.loads((cache_dir / schemas.METADATA_FILE).read_text())
        assert metadata["failed_file_count"] == 1

    def test_failed_file_skipped_when_unchanged(self, tmp_path: Path) -> None:
        service, source, _cache_dir = make_sync(tmp_path, {})
        source.files["broken.fit"] = b"garbage"
        service.sync()
        result = service.sync()
        assert result.files_failed == 0
        assert result.files_skipped == 1

    def test_failed_file_retried_when_changed(self, tmp_path: Path) -> None:
        service, source, _cache_dir = make_sync(tmp_path, {})
        source.files["broken.fit"] = b"garbage"
        service.sync()
        source.files["broken.fit"] = stub_activity_content(T0, "cycling", 1800.0, 20000.0)
        result = service.sync()
        assert result.files_processed == 1

    def test_duplicate_activity_detected(self, tmp_path: Path) -> None:
        content = stub_activity_content(T0, "cycling", 1800.0, 20000.0)
        service, _source, cache_dir = make_sync(tmp_path, {"one.fit": content, "two.fit": content})
        result = service.sync()
        assert result.files_processed == 2
        activities = parquet.read_parquet(cache_dir / schemas.ACTIVITIES_FILE)
        assert activities.height == 1
        index = parquet.read_parquet(cache_dir / schemas.SOURCE_INDEX_FILE)
        statuses = sorted(row["status"] for row in index.to_dicts())
        assert statuses == ["duplicate", "ok"]

    def test_source_error_does_not_abort(self, tmp_path: Path) -> None:
        source_a = StubSource(name="a", files=one_file())

        class FailingSource(StubSource):
            def list_files(self) -> list[SourceFile]:
                msg = "remote down"
                raise SourceError(msg)

        source_b = FailingSource(name="b", files={})
        cache_dir = tmp_path / "cache"
        config = make_app_config(cache_dir, tmp_path / "src")
        service = SyncService(
            sources=[source_b, source_a],
            processor=make_processor(athlete=config.athlete, analysis_version="1"),
            cache_dir=cache_dir,
            timezone=config.application.zoneinfo,
            analysis_version="1",
        )
        result = service.sync()
        assert result.files_processed == 1
        assert result.source_errors == ("b: remote down",)


class TestSegmentSync:
    """Multi-segment files and the ok-status aggregate filter."""

    def _two_segments(self) -> bytes:
        first = stub_activity_content(T0, "cycling", 1800.0, 20000.0)
        second = stub_activity_content(T0 + timedelta(hours=2), "cycling", 1200.0, 8000.0)
        return first + b"\n" + second

    def test_multi_segment_file_creates_two_activities(self, tmp_path: Path) -> None:
        service, _source, cache_dir = make_sync(tmp_path, {"ride.fit": self._two_segments()})
        result = service.sync()
        assert result.files_processed == 1
        activities = parquet.read_parquet(cache_dir / schemas.ACTIVITIES_FILE)
        assert activities.height == 2
        assert activities["segment_count"].to_list() == [2, 2]
        assert sorted(activities["segment_index"].to_list()) == [0, 1]
        index = parquet.read_parquet(cache_dir / schemas.SOURCE_INDEX_FILE)
        rows = index.to_dicts()
        assert len(rows) == 2
        assert sorted(str(row["status"]) for row in rows) == ["ok", "ok"]
        daily = parquet.read_parquet(cache_dir / schemas.DAILY_FILE)
        assert daily["activity_count"].to_list() == [2]

    def test_multi_segment_file_skipped_when_unchanged(self, tmp_path: Path) -> None:
        service, _source, cache_dir = make_sync(tmp_path, {"ride.fit": self._two_segments()})
        service.sync()
        result = service.sync()
        assert result.files_skipped == 1
        assert result.files_processed == 0
        activities = parquet.read_parquet(cache_dir / schemas.ACTIVITIES_FILE)
        assert activities.height == 2

    def test_multi_segment_purge_on_change(self, tmp_path: Path) -> None:
        service, source, cache_dir = make_sync(tmp_path, {"ride.fit": self._two_segments()})
        service.sync()
        source.files["ride.fit"] = stub_activity_content(T0, "cycling", 1800.0, 20000.0)
        result = service.sync()
        assert result.files_processed == 1
        activities = parquet.read_parquet(cache_dir / schemas.ACTIVITIES_FILE)
        assert activities.height == 1
        index = parquet.read_parquet(cache_dir / schemas.SOURCE_INDEX_FILE)
        assert index.height == 1
        daily = parquet.read_parquet(cache_dir / schemas.DAILY_FILE)
        assert daily["activity_count"].to_list() == [1]

    def test_duplicate_rows_excluded_from_aggregates(self, tmp_path: Path) -> None:
        """Regression: duplicate-status index rows must not double-count daily/weekly."""
        content = stub_activity_content(T0, "cycling", 1800.0, 20000.0)
        service, _source, cache_dir = make_sync(tmp_path, {"one.fit": content, "two.fit": content})
        service.sync()
        daily = parquet.read_parquet(cache_dir / schemas.DAILY_FILE)
        assert daily["activity_count"].to_list() == [1]
        weekly = parquet.read_parquet(cache_dir / schemas.WEEKLY_FILE)
        assert weekly["activity_count"].to_list() == [1]

    def test_repair_counters_in_sync_result(self, tmp_path: Path) -> None:
        from dataclasses import replace

        class FlagParser(StubParser):
            def parse(self, content: bytes, source_file: SourceFile) -> ParsedFile:
                parsed = super().parse(content, source_file)
                flagged = tuple(
                    replace(activity, repair_flags=("clock_corrected",))
                    for activity in parsed.activities
                )
                return replace(parsed, activities=flagged, repair_flags=("clock_corrected",))

        source = StubSource(files={"ride.fit": self._two_segments()})
        cache_dir = tmp_path / "cache"
        config = make_app_config(cache_dir, tmp_path / "src")
        service = SyncService(
            sources=[source],
            processor=make_processor(
                athlete=config.athlete, analysis_version="1", parser=FlagParser()
            ),
            cache_dir=cache_dir,
            timezone=config.application.zoneinfo,
            analysis_version="1",
        )
        result = service.sync()
        assert result.files_repaired == 1
        assert result.duplicates_detected == 0
        index = parquet.read_parquet(cache_dir / schemas.SOURCE_INDEX_FILE)
        assert all(
            json.loads(str(row["repair_flags_json"])) == ["clock_corrected"]
            for row in index.to_dicts()
        )
