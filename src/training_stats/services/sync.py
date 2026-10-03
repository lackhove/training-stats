"""Incremental synchronization application service."""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import chain
from typing import TYPE_CHECKING, Literal

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
    from .ingest import FileProcessor, Outcome

logger = logging.getLogger(__name__)

FileKey = tuple[str, str]


@dataclass(frozen=True, slots=True)
class _FileOutcome:
    """Counter contribution of one listed file, refreshed by promotion."""

    kind: Literal["skipped", "processed", "failed"]
    duplicates: int = 0
    repaired: int = 0
    discarded_segments: int = 0


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
        sources_listed = 0
        source_errors: list[str] = []
        failed_sources: set[str] = set()
        kept_index: list[SourceIndexEntry] = []
        outcomes: dict[FileKey, _FileOutcome] = {}
        listed: dict[FileKey, tuple[ActivitySource, SourceFile]] = {}

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
                key = (source.name, source_file.path)
                listed[key] = (source, source_file)
                previous_rows = self._previous_rows(state.index_rows, source.name, source_file.path)
                if previous_rows and self._metadata_unchanged(previous_rows, source_file):
                    kept_index.extend(previous_rows)
                    outcomes[key] = _FileOutcome(kind="skipped")
                    continue
                try:
                    content = source.read_file(source_file)
                except SourceError as exc:
                    logger.error("cannot read %s/%s: %s", source.name, source_file.path, exc)
                    kept_index.extend(
                        self._read_failure_rows(previous_rows, source_file, started_at, str(exc))
                    )
                    outcomes[key] = _FileOutcome(kind="failed")
                    continue
                file_hash = hashlib.sha256(content).hexdigest()
                self._purge_previous(
                    previous_rows,
                    activities,
                    activity_ids,
                    self._ids_owned_by_others(
                        chain(state.index_rows, kept_index),
                        source_name=source.name,
                        path=source_file.path,
                    ),
                )
                outcome = self.processor.process(
                    source_file, content, file_hash, activity_ids, started_at
                )
                kept_index.extend(outcome.index_rows)
                for activity in outcome.activities:
                    activities[activity.activity_id] = activity
                    activity_ids.add(activity.activity_id)
                outcomes[key] = _outcome_of(outcome)

        if sources_listed > 0:
            # A source that failed to list is not evidence of deletion: carry its rows over.
            kept_index.extend(row for row in state.index_rows if row.source_name in failed_sources)
            self._promote_orphan_duplicates(
                kept_index=kept_index,
                activities=activities,
                activity_ids=activity_ids,
                listed=listed,
                outcomes=outcomes,
                processed_at=started_at,
            )

        processed = sum(1 for entry in outcomes.values() if entry.kind == "processed")
        skipped = sum(1 for entry in outcomes.values() if entry.kind == "skipped")
        failed = sum(1 for entry in outcomes.values() if entry.kind == "failed")
        duplicates_detected = sum(entry.duplicates for entry in outcomes.values())
        files_repaired = sum(entry.repaired for entry in outcomes.values())
        segments_discarded = sum(entry.discarded_segments for entry in outcomes.values())

        cache_updated_at: datetime | None = None
        if sources_listed == 0:
            logger.warning(
                "no source produced a file listing; cache left untouched (%d source errors)",
                len(source_errors),
            )
        else:
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
                self.repository.save_sync_stamp(snapshot, last_sync=started_at)
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

    def _read_failure_rows(
        self,
        previous_rows: list[SourceIndexEntry],
        source_file: SourceFile,
        processed_at: datetime,
        message: str,
    ) -> list[SourceIndexEntry]:
        """Record an unreadable file without discarding its last known good rows.

        Accepted and duplicate rows are retained so a transient read failure
        never drops the record another sync produced. The failure outcome is a
        source-only row (``source_hash=None``); an identical earlier failure is
        reused so a persistent error does not rewrite the snapshot.
        """
        retained = [row for row in previous_rows if row.status in {STATUS_OK, STATUS_DUPLICATE}]
        candidate = self._failure_entry(source_file, processed_at, message)
        for row in previous_rows:
            if row.status == STATUS_FAILED and self._same_failure(row, candidate):
                return [*retained, row]
        return [*retained, candidate]

    @staticmethod
    def _same_failure(stored: SourceIndexEntry, candidate: SourceIndexEntry) -> bool:
        """True when ``stored`` already reports this attempt's failure."""
        return (
            stored.source_name == candidate.source_name
            and stored.source_path == candidate.source_path
            and stored.source_hash is None
            and stored.file_size == candidate.file_size
            and stored.modified_at == candidate.modified_at
            and stored.source_analysis_version == candidate.source_analysis_version
            and stored.error_message == candidate.error_message
        )

    @staticmethod
    def _ids_owned_by_others(
        rows: Iterable[SourceIndexEntry], *, source_name: str, path: str
    ) -> set[str]:
        """Activity ids another file still claims through an accepted row."""
        return {
            row.activity_id
            for row in rows
            if row.status == STATUS_OK
            and row.activity_id is not None
            and not (row.source_name == source_name and row.source_path == path)
        }

    def _promote_orphan_duplicates(
        self,
        *,
        kept_index: list[SourceIndexEntry],
        activities: dict[str, ActivityRecord],
        activity_ids: set[str],
        listed: dict[FileKey, tuple[ActivitySource, SourceFile]],
        outcomes: dict[FileKey, _FileOutcome],
        processed_at: datetime,
    ) -> None:
        """Re-home an orphaned duplicate copy when its accepted owner is gone.

        Runs after the listing pass. Duplicate rows whose ``activity_id`` no
        accepted ``ok`` row claims any more belong to a file whose canonical
        copy was deleted or rewritten; when that file is still present in a
        source that listed, it is re-read once so one surviving copy is
        promoted back to an accepted record and further copies stay duplicate.
        A source that failed to list never promotes: its rows stay carried over.
        """
        accepted = {
            row.activity_id
            for row in kept_index
            if row.status == STATUS_OK and row.activity_id is not None
        }
        orphaned: list[str] = []
        candidates: dict[FileKey, None] = {}
        for row in kept_index:
            if row.status != STATUS_DUPLICATE or row.activity_id is None:
                continue
            if row.activity_id in accepted:
                continue
            orphaned.append(row.activity_id)
            key = (row.source_name, row.source_path)
            if key in listed:
                candidates[key] = None
        if not orphaned:
            return
        # The unowned ids must leave the id set, or the promoted copy would be
        # marked duplicate of an activity that no longer exists.
        for activity_id in dict.fromkeys(orphaned):
            activities.pop(activity_id, None)
            activity_ids.discard(activity_id)

        for key in candidates:
            rows = [row for row in kept_index if (row.source_name, row.source_path) == key]
            if not any(
                row.status == STATUS_DUPLICATE
                and row.activity_id is not None
                and row.activity_id not in accepted
                for row in rows
            ):
                continue  # an earlier promotion in this pass restored its owner
            source, source_file = listed[key]
            try:
                content = source.read_file(source_file)
            except SourceError as exc:
                logger.error("cannot read %s/%s: %s", source.name, source_file.path, exc)
                new_rows = self._read_failure_rows(rows, source_file, processed_at, str(exc))
                outcomes[key] = _FileOutcome(kind="failed")
            else:
                file_hash = hashlib.sha256(content).hexdigest()
                self._purge_previous(
                    rows,
                    activities,
                    activity_ids,
                    self._ids_owned_by_others(
                        kept_index, source_name=source.name, path=source_file.path
                    ),
                )
                outcome = self.processor.process(
                    source_file, content, file_hash, activity_ids, processed_at
                )
                new_rows = outcome.index_rows
                for activity in outcome.activities:
                    activities[activity.activity_id] = activity
                    activity_ids.add(activity.activity_id)
                outcomes[key] = _outcome_of(outcome)
                accepted.update(
                    row.activity_id
                    for row in outcome.index_rows
                    if row.status == STATUS_OK and row.activity_id is not None
                )
            self._replace_rows(kept_index, key, new_rows)
            logger.info(
                "promoted %s/%s: its accepted owner vanished, the surviving copy was reprocessed",
                source.name,
                source_file.path,
            )

    @staticmethod
    def _replace_rows(
        kept_index: list[SourceIndexEntry], key: FileKey, new_rows: list[SourceIndexEntry]
    ) -> None:
        """Swap one file's rows in place so source/path row order stays stable."""
        positions = [
            index
            for index, row in enumerate(kept_index)
            if (row.source_name, row.source_path) == key
        ]
        if not positions:
            kept_index.extend(new_rows)
            return
        insert_at = positions[0]
        for index in reversed(positions):
            kept_index.pop(index)
        kept_index[insert_at:insert_at] = new_rows

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
        owned_by_others: set[str],
    ) -> None:
        """Drop the records this file's accepted rows own and no one else claims.

        Duplicate rows never own a record: purging their id would delete the
        canonical activity another file still serves.
        """
        for row in previous_rows:
            if row.status != STATUS_OK or row.activity_id is None:
                continue
            if row.activity_id in owned_by_others:
                continue
            activities.pop(row.activity_id, None)
            activity_ids.discard(row.activity_id)


def _outcome_of(outcome: Outcome) -> _FileOutcome:
    """Map one processing result to its counter contribution."""
    return _FileOutcome(
        kind="processed"
        if any(row.status in {STATUS_OK, STATUS_DUPLICATE} for row in outcome.index_rows)
        else "failed",
        duplicates=outcome.duplicates,
        repaired=int(outcome.repaired),
        discarded_segments=outcome.discarded_segments,
    )
