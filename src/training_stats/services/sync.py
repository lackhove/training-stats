"""Incremental synchronization application service."""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ..domain.models import (
    STATUS_DUPLICATE,
    STATUS_FAILED,
    STATUS_OK,
    ActivityRecord,
    SourceFile,
    SourceIndexEntry,
)
from .errors import SourceError
from .ports import ActivitySource, SyncRepository
from .results import SyncResult
from .state import SyncState

if TYPE_CHECKING:
    from .ingest import FileProcessor

logger = logging.getLogger(__name__)


class SyncService:
    """Synchronize files from configured activity sources."""

    def __init__(
        self,
        *,
        sources: list[ActivitySource],
        processor: FileProcessor,
        repository: SyncRepository,
        analysis_version: str,
    ) -> None:
        self.sources = sources
        self.processor = processor
        self.repository = repository
        self.analysis_version = analysis_version

    def sync(self) -> SyncResult:
        """Process new and changed files; isolate failures by source and file."""
        started_at = datetime.now(UTC)
        state = self.repository.load_sync_state()
        activities = dict(state.activities)
        activity_ids = set(activities)

        discovered = 0
        processed = 0
        skipped = 0
        failed = 0
        duplicates_detected = 0
        files_repaired = 0
        segments_discarded = 0
        sources_listed = 0
        source_errors: list[str] = []
        failed_sources: set[str] = set()
        kept_index: list[SourceIndexEntry] = []

        for source in self.sources:
            try:
                files = source.list_files()
            except SourceError as exc:
                logger.error("source %s failed to list files: %s", source.name, exc)
                source_errors.append(f"{source.name}: {exc}")
                failed_sources.add(source.name)
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
                    content = source.read_file(source_file)
                except SourceError as exc:
                    logger.error("cannot read %s/%s: %s", source.name, source_file.path, exc)
                    failed += 1
                    kept_index.append(self._failure_entry(source_file, started_at, str(exc)))
                    continue
                file_hash = hashlib.sha256(content).hexdigest()
                self._purge_previous(previous_rows, activities, activity_ids)
                outcome = self.processor.process(
                    source_file, content, file_hash, activity_ids, started_at
                )
                kept_index.extend(outcome.index_rows)
                for activity in outcome.activities:
                    activities[activity.activity_id] = activity
                    activity_ids.add(activity.activity_id)
                if any(row.status in {STATUS_OK, STATUS_DUPLICATE} for row in outcome.index_rows):
                    processed += 1
                else:
                    failed += 1
                duplicates_detected += outcome.duplicates
                files_repaired += int(outcome.repaired)
                segments_discarded += outcome.discarded_segments

        cache_updated_at: datetime | None = None
        if sources_listed == 0:
            logger.warning(
                "no source produced a file listing; cache left untouched (%d source errors)",
                len(source_errors),
            )
        else:
            # A source that failed to list is not evidence of deletion: carry its rows over.
            kept_index.extend(row for row in state.index_rows if row.source_name in failed_sources)
            accepted_ids = {
                row.activity_id
                for row in kept_index
                if row.status == STATUS_OK and row.activity_id is not None
            }
            persisted_activities = {
                activity_id: activity
                for activity_id, activity in activities.items()
                if activity_id in accepted_ids
            }
            snapshot = SyncState(index_rows=kept_index, activities=persisted_activities)
            if self._state_unchanged(state, snapshot):
                self.repository.save_sync_stamp(last_sync=started_at)
            else:
                self.repository.save_sync_snapshot(snapshot, last_sync=started_at)
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

    def _failure_entry(
        self, source_file: SourceFile, processed_at: datetime, message: str
    ) -> SourceIndexEntry:
        return SourceIndexEntry(
            source_name=source_file.source_name,
            source_path=source_file.path,
            source_hash=None,
            file_size=source_file.size,
            modified_at=source_file.modified_at,
            activity_id=None,
            segment_index=0,
            segment_count=0,
            file_repair_flags=(),
            discarded_segment_count=0,
            processed_at=processed_at,
            source_analysis_version=self.analysis_version,
            status=STATUS_FAILED,
            error_message=message,
        )

    @staticmethod
    def _previous_rows(
        index_rows: list[SourceIndexEntry], source_name: str, path: str
    ) -> list[SourceIndexEntry]:
        return [
            row for row in index_rows if row.source_name == source_name and row.source_path == path
        ]

    @staticmethod
    def _state_unchanged(previous: SyncState, current: SyncState) -> bool:
        """True when activities match and index rows differ at most in order."""
        if previous.activities != current.activities:
            return False
        return Counter(previous.index_rows) == Counter(current.index_rows)

    def _metadata_unchanged(
        self, previous_rows: list[SourceIndexEntry], source_file: SourceFile
    ) -> bool:
        """True when all stored rows match size, mtime, hash, and analysis version."""
        if source_file.size is None or source_file.modified_at is None:
            return False
        return all(
            row.source_hash is not None
            and row.source_analysis_version == self.analysis_version
            and row.file_size == source_file.size
            and self._same_time(row.modified_at, source_file.modified_at)
            for row in previous_rows
        )

    @staticmethod
    def _same_time(stored: datetime | None, modified_at: datetime) -> bool:
        if stored is None or stored.tzinfo is None:
            return False
        return stored.astimezone(UTC) == modified_at.astimezone(UTC)

    @staticmethod
    def _purge_previous(
        previous_rows: list[SourceIndexEntry],
        activities: dict[str, ActivityRecord],
        activity_ids: set[str],
    ) -> None:
        for row in previous_rows:
            if row.activity_id is not None:
                activities.pop(row.activity_id, None)
                activity_ids.discard(row.activity_id)
