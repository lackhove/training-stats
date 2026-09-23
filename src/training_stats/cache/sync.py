"""Incremental synchronization from activity sources into the Parquet cache.

Change detection is metadata-only: files whose stored size and modification
time match the previous sync are skipped without being read at all; downloaded
candidates are reprocessed unconditionally. FIT files are only ever added,
deleted or rewritten, and a rewrite changes size or modification time.
Parsing and metric computation are delegated to the injected
:class:`~training_stats.ingest.FileProcessor`; all cache I/O goes through the
injected :class:`SyncCacheRepository`.

Locking is an entry-point concern: the process holds the application-wide
cache lock (see :mod:`training_stats.cache.locking`) for its whole lifetime,
so neither sync nor repository code takes locks of its own.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from ..errors import SourceError
from ..models import SourceFile, SyncResult
from ..sources.base import ActivitySource
from . import schemas
from .repository import SyncState

if TYPE_CHECKING:
    # Runtime import would cycle: ingest -> cache.schemas -> this module.
    from ..ingest import FileProcessor

logger = logging.getLogger(__name__)


class SyncCacheRepository(Protocol):
    """Storage interface :class:`SyncService` needs from the cache repository.

    Narrow, write-oriented counterpart of ``TrainingRepository``; implemented
    structurally by ``ParquetTrainingRepository`` so the storage engine stays
    swappable behind the repository.
    """

    def load_sync_state(self) -> SyncState: ...

    def save_sync_snapshot(self, state: SyncState, *, last_sync: datetime) -> None: ...


class SyncService:
    """Synchronizes FIT files from all sources into the Parquet cache."""

    def __init__(
        self,
        *,
        sources: list[ActivitySource],
        processor: FileProcessor,
        repository: SyncCacheRepository,
        analysis_version: str,
    ) -> None:
        self.sources = sources
        self.processor = processor
        self.repository = repository
        self.analysis_version = analysis_version

    def sync(self) -> SyncResult:
        """Process new and changed files; skip unchanged ones.

        A failing source (e.g. WebDAV unreachable) does not abort the whole
        sync: the error is reported in ``source_errors`` and other sources
        continue.
        """
        return self._run_sync()

    def _run_sync(self) -> SyncResult:
        started_at = datetime.now(UTC)
        state = self.repository.load_sync_state()

        discovered = 0
        processed = 0
        skipped = 0
        failed = 0
        duplicates_detected = 0
        files_repaired = 0
        segments_discarded = 0
        sources_listed = 0
        source_errors: list[str] = []
        kept_index: list[dict[str, object]] = []

        for source in self.sources:
            try:
                files = source.list_files()
            except SourceError as exc:
                logger.error("source %s failed to list files: %s", source.name, exc)
                source_errors.append(f"{source.name}: {exc}")
                continue
            sources_listed += 1
            discovered += len(files)
            for source_file in files:
                previous_rows = self._previous_rows(state.index_rows, source.name, source_file.path)
                if previous_rows and self._metadata_unchanged(previous_rows, source_file):
                    kept_index.extend(previous_rows)
                    skipped += 1
                    continue
                try:
                    content = self._read_content(source_file)
                except SourceError as exc:
                    logger.error("cannot read %s/%s: %s", source.name, source_file.path, exc)
                    failed += 1
                    kept_index.append(self._failure_row(source_file, started_at, str(exc), None))
                    continue
                file_hash = hashlib.sha256(content).hexdigest()
                self._purge_previous(previous_rows, state.activities)
                outcome = self.processor.process(
                    source_file, content, file_hash, set(state.activities), started_at
                )
                kept_index.extend(outcome.index_rows)
                for activity_row in outcome.ok_activity_rows:
                    state.activities[str(activity_row["activity_id"])] = activity_row
                if outcome.ok_activity_rows or outcome.duplicates:
                    processed += 1
                else:
                    failed += 1
                duplicates_detected += outcome.duplicates
                files_repaired += 1 if outcome.repaired else 0
                segments_discarded += outcome.discarded_segments

        cache_updated_at: datetime | None = None
        if sources_listed == 0:
            # No source produced a file listing (none enabled or all failed).
            # Rewriting the stores here would blank the cache on a transient
            # outage or misconfiguration, so the existing cache is preserved.
            logger.warning(
                "no source produced a file listing; cache left untouched (%d source errors)",
                len(source_errors),
            )
        else:
            self.repository.save_sync_snapshot(
                SyncState(index_rows=kept_index, activities=state.activities),
                last_sync=started_at,
            )
            logger.info(
                "sync complete: %d discovered, %d processed, %d skipped, %d failed, "
                "%d duplicates, %d repaired files, %d discarded segments",
                discovered,
                processed,
                skipped,
                failed,
                duplicates_detected,
                files_repaired,
                segments_discarded,
            )
            cache_updated_at = started_at
        return SyncResult(
            files_discovered=discovered,
            files_processed=processed,
            files_skipped=skipped,
            files_failed=failed,
            cache_updated_at=cache_updated_at,
            source_errors=tuple(source_errors),
            files_repaired=files_repaired,
            segments_discarded=segments_discarded,
            duplicates_detected=duplicates_detected,
        )

    def _failure_row(
        self, source_file: SourceFile, processed_at: datetime, message: str, file_hash: str | None
    ) -> dict[str, object]:
        return {
            "source_name": source_file.source_name,
            "source_path": source_file.path,
            "source_hash": file_hash,
            "file_size": source_file.size,
            "modified_at": source_file.modified_at,
            "activity_id": None,
            "segment_index": 0,
            "segment_count": 0,
            schemas.FILE_REPAIR_FLAGS: None,
            "discarded_segment_count": 0,
            schemas.PROCESSED_AT: processed_at,
            schemas.SOURCE_ANALYSIS_VERSION: self.analysis_version,
            "status": schemas.STATUS_FAILED,
            "error_message": message,
        }

    def _read_content(self, source_file: SourceFile) -> bytes:
        for source in self.sources:
            if source.name == source_file.source_name:
                return source.read_file(source_file)
        msg = f"source {source_file.source_name!r} is not configured"
        raise SourceError(msg)

    def _previous_rows(
        self, index_rows: list[dict[str, object]], source_name: str, path: str
    ) -> list[dict[str, object]]:
        return [
            row
            for row in index_rows
            if row.get("source_name") == source_name and row.get("source_path") == path
        ]

    def _metadata_unchanged(
        self, previous_rows: list[dict[str, object]], source_file: SourceFile
    ) -> bool:
        """True when size and modification time match all previous rows.

        This avoids reading files at all. Rows without a stored hash (failed
        reads) or missing metadata never short-circuit, so they are re-read
        and reprocessed.
        """
        if source_file.size is None or source_file.modified_at is None:
            return False
        return all(
            row.get("source_hash") is not None
            and row.get(schemas.SOURCE_ANALYSIS_VERSION) == self.analysis_version
            and self._same_size(row.get("file_size"), source_file.size)
            and self._same_time(row.get("modified_at"), source_file.modified_at)
            for row in previous_rows
        )

    @staticmethod
    def _same_size(stored: object, size: int) -> bool:
        return isinstance(stored, int) and not isinstance(stored, bool) and stored == size

    @staticmethod
    def _same_time(stored: object, modified_at: datetime) -> bool:
        if not isinstance(stored, datetime) or stored.tzinfo is None:
            return False
        return stored.astimezone(UTC) == modified_at.astimezone(UTC)

    @staticmethod
    def _purge_previous(
        previous_rows: list[dict[str, object]],
        activities: dict[str, dict[str, object]],
    ) -> None:
        for row in previous_rows:
            activity_id = row.get("activity_id")
            if activity_id:
                activities.pop(str(activity_id), None)
