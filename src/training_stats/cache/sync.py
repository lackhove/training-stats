"""Incremental synchronization from activity sources into the Parquet cache.

Change detection is two-tier: files whose stored size and modification time
match the previous sync are skipped without being read at all; downloaded
candidates are then SHA-256 hashed, and only files whose hash or analysis
version changed are reprocessed. Parsing and metric computation are delegated
to the injected :class:`~training_stats.ingest.FileProcessor`.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime, tzinfo
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from ..analytics.activities import ActivityMetrics, AnalyzedActivity
from ..analytics.daily import DailySummaryRow, summarize_daily
from ..analytics.weekly import WeeklySummaryRow, daily_to_weekly
from ..converters import (
    buckets_from_json,
    flags_from_json,
    histogram_from_json,
    optional_float,
    optional_int,
    optional_str,
    required_float,
)
from ..errors import SourceError
from ..fit.types import ParsedActivity, empty_records
from ..models import SourceFile, SyncResult
from ..sources.base import ActivitySource
from . import parquet, schemas
from .locking import sync_lock

if TYPE_CHECKING:
    # Runtime import would cycle: ingest -> cache.schemas -> this module.
    from ..ingest import FileProcessor

logger = logging.getLogger(__name__)


class SyncService:
    """Synchronizes FIT files from all sources into the Parquet cache."""

    def __init__(
        self,
        *,
        sources: list[ActivitySource],
        processor: FileProcessor,
        cache_dir: Path,
        timezone: tzinfo,
        analysis_version: str,
        lock_timeout_seconds: float = 30.0,
    ) -> None:
        self.sources = sources
        self.processor = processor
        self.cache_dir = cache_dir
        self.tz = timezone
        self.analysis_version = analysis_version
        self.lock_timeout_seconds = lock_timeout_seconds

    def sync(self, force: bool = False, source_name: str | None = None) -> SyncResult:
        """Process new and changed files; skip unchanged ones.

        A failing source (e.g. WebDAV unreachable) does not abort the whole
        sync: the error is reported in ``source_errors`` and other sources
        continue.
        """
        self._validate_source_filter(source_name)
        lock_path = self.cache_dir / schemas.LOCK_FILE
        with sync_lock(lock_path, self.lock_timeout_seconds):
            return self._run_sync(force=force, source_name=source_name, reset=False)

    def rebuild_cache(self) -> SyncResult:
        """Ignore existing derived data and reprocess all current files."""
        lock_path = self.cache_dir / schemas.LOCK_FILE
        with sync_lock(lock_path, self.lock_timeout_seconds):
            return self._run_sync(force=True, source_name=None, reset=True)

    def _validate_source_filter(self, source_name: str | None) -> None:
        if source_name is None:
            return
        known = {source.name for source in self.sources}
        if source_name not in known:
            msg = f"unknown source {source_name!r}; configured sources: {sorted(known)}"
            raise SourceError(msg)

    def _run_sync(self, *, force: bool, source_name: str | None, reset: bool) -> SyncResult:
        started_at = datetime.now(UTC)
        index_rows = self._load_index(reset)
        activities, metrics = self._load_store(reset)

        selected_sources = [s for s in self.sources if source_name is None or s.name == source_name]

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

        for source in selected_sources:
            try:
                files = source.list_files()
            except SourceError as exc:
                logger.error("source %s failed to list files: %s", source.name, exc)
                source_errors.append(f"{source.name}: {exc}")
                continue
            sources_listed += 1
            discovered += len(files)
            for source_file in files:
                previous_rows = self._previous_rows(index_rows, source.name, source_file.path)
                if (
                    not force
                    and previous_rows
                    and self._metadata_unchanged(previous_rows, source_file)
                ):
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
                if not force and previous_rows and self._unchanged(previous_rows, file_hash):
                    kept_index.extend(previous_rows)
                    skipped += 1
                    continue
                self._purge_previous(previous_rows, activities, metrics)
                outcome = self.processor.process(
                    source_file, content, file_hash, set(activities), started_at
                )
                kept_index.extend(outcome.index_rows)
                for activity_row in outcome.ok_activity_rows:
                    activities[str(activity_row["activity_id"])] = activity_row
                for metrics_row in outcome.ok_metrics_rows:
                    metrics[str(metrics_row["activity_id"])] = metrics_row
                if outcome.ok_activity_rows or outcome.duplicates:
                    processed += 1
                else:
                    failed += 1
                duplicates_detected += outcome.duplicates
                files_repaired += 1 if outcome.repaired else 0
                segments_discarded += outcome.discarded_segments

        if sources_listed == 0:
            # No source produced a file listing (none enabled or all failed).
            # Rewriting the stores here would blank the cache on a transient
            # outage or misconfiguration, so the existing cache is preserved.
            logger.warning(
                "no source produced a file listing; cache left untouched (%d source errors)",
                len(source_errors),
            )
            return SyncResult(
                files_discovered=discovered,
                files_processed=processed,
                files_skipped=skipped,
                files_failed=failed,
                source_errors=tuple(source_errors),
                files_repaired=files_repaired,
                segments_discarded=segments_discarded,
                duplicates_detected=duplicates_detected,
            )

        self._write_cache(activities, metrics, kept_index, started_at)
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
        return SyncResult(
            files_discovered=discovered,
            files_processed=processed,
            files_skipped=skipped,
            files_failed=failed,
            cache_updated_at=started_at,
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
            "repair_flags_json": None,
            "discarded_segment_count": 0,
            schemas.PROCESSED_AT: processed_at,
            schemas.ANALYSIS_VERSION: self.analysis_version,
            "status": schemas.STATUS_FAILED,
            "error_message": message,
        }

    def _read_content(self, source_file: SourceFile) -> bytes:
        for source in self.sources:
            if source.name == source_file.source_name:
                return source.read_file(source_file)
        msg = f"source {source_file.source_name!r} is not configured"
        raise SourceError(msg)

    def _load_index(self, reset: bool) -> list[dict[str, object]]:
        if reset:
            return []
        frame = parquet.read_parquet(self.cache_dir / schemas.SOURCE_INDEX_FILE)
        if frame.is_empty():
            return []
        return frame.to_dicts()

    def _load_store(
        self, reset: bool
    ) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
        if reset:
            return {}, {}
        activities_frame = parquet.read_parquet(self.cache_dir / schemas.ACTIVITIES_FILE)
        metrics_frame = parquet.read_parquet(self.cache_dir / schemas.ACTIVITY_METRICS_FILE)
        activities = (
            {str(row["activity_id"]): row for row in activities_frame.to_dicts()}
            if not activities_frame.is_empty()
            else {}
        )
        metrics = (
            {str(row["activity_id"]): row for row in metrics_frame.to_dicts()}
            if not metrics_frame.is_empty()
            else {}
        )
        return activities, metrics

    def _previous_rows(
        self, index_rows: list[dict[str, object]], source_name: str, path: str
    ) -> list[dict[str, object]]:
        return [
            row
            for row in index_rows
            if row.get("source_name") == source_name and row.get("source_path") == path
        ]

    def _unchanged(self, previous_rows: list[dict[str, object]], file_hash: str) -> bool:
        return all(
            row.get("source_hash") == file_hash
            and row.get(schemas.ANALYSIS_VERSION) == self.analysis_version
            for row in previous_rows
        )

    def _metadata_unchanged(
        self, previous_rows: list[dict[str, object]], source_file: SourceFile
    ) -> bool:
        """True when size and modification time match all previous rows.

        This avoids reading files at all. Rows without a stored hash (failed
        reads) or missing metadata never short-circuit, so they are re-read
        and the SHA-256 hash decides.
        """
        if source_file.size is None or source_file.modified_at is None:
            return False
        return all(
            row.get("source_hash") is not None
            and row.get(schemas.ANALYSIS_VERSION) == self.analysis_version
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
        metrics: dict[str, dict[str, object]],
    ) -> None:
        for row in previous_rows:
            activity_id = row.get("activity_id")
            if activity_id:
                activities.pop(str(activity_id), None)
                metrics.pop(str(activity_id), None)

    def _write_cache(
        self,
        activities: dict[str, dict[str, object]],
        metrics: dict[str, dict[str, object]],
        index_rows: list[dict[str, object]],
        started_at: datetime,
    ) -> None:
        # Prune store rows without an active "ok" index row (removed or
        # re-specified files, historical orphans) so activities, aggregates
        # and the index stay consistent.
        ok_ids = {
            str(row["activity_id"])
            for row in index_rows
            if row.get("status") == schemas.STATUS_OK and row.get("activity_id") is not None
        }
        activities = {key: row for key, row in activities.items() if key in ok_ids}
        metrics = {key: row for key, row in metrics.items() if key in ok_ids}
        analyzed = [
            self._rebuild_analyzed(
                activities[str(row["activity_id"])], metrics[str(row["activity_id"])]
            )
            for row in index_rows
            if row.get("status") == schemas.STATUS_OK
            and row.get("activity_id") is not None
            and str(row["activity_id"]) in activities
            and str(row["activity_id"]) in metrics
        ]
        daily_rows = summarize_daily(analyzed, self.tz)
        weekly_rows = daily_to_weekly(daily_rows)

        parquet.write_parquet_atomic(
            _frame(activities, schemas.ACTIVITY_SCHEMA), self.cache_dir / schemas.ACTIVITIES_FILE
        )
        parquet.write_parquet_atomic(
            _frame(metrics, schemas.ACTIVITY_METRICS_SCHEMA),
            self.cache_dir / schemas.ACTIVITY_METRICS_FILE,
        )
        parquet.write_parquet_atomic(
            pl.DataFrame(_daily_records(daily_rows), schema=schemas.DAILY_SCHEMA),
            self.cache_dir / schemas.DAILY_FILE,
        )
        parquet.write_parquet_atomic(
            pl.DataFrame(_weekly_records(weekly_rows), schema=schemas.WEEKLY_SCHEMA),
            self.cache_dir / schemas.WEEKLY_FILE,
        )
        parquet.write_parquet_atomic(
            _index_frame(index_rows), self.cache_dir / schemas.SOURCE_INDEX_FILE
        )
        parquet.write_json_atomic(
            {
                "last_sync": started_at.isoformat(),
                "analysis_version": self.analysis_version,
                "activity_count": len(activities),
                "failed_file_count": sum(
                    1 for row in index_rows if row.get("status") == schemas.STATUS_FAILED
                ),
            },
            self.cache_dir / schemas.METADATA_FILE,
        )

    def _rebuild_analyzed(
        self, activity_row: dict[str, object], metrics_row: dict[str, object]
    ) -> AnalyzedActivity:
        start_time = activity_row["start_time"]
        if not isinstance(start_time, datetime):
            msg = "activity start_time has unexpected type"
            raise SourceError(msg)
        parsed = ParsedActivity(
            activity_id=str(activity_row["activity_id"]),
            start_time=start_time,
            sport=str(activity_row["sport"]),
            sub_sport=optional_str(activity_row.get("sub_sport")),
            duration_s=optional_float(activity_row.get("duration_s")),
            moving_time_s=optional_float(activity_row.get("moving_time_s")),
            distance_m=optional_float(activity_row.get("distance_m")),
            elevation_gain_m=optional_float(activity_row.get("elevation_gain_m")),
            avg_hr=optional_float(activity_row.get("avg_hr")),
            max_hr=optional_float(activity_row.get("max_hr")),
            avg_speed_mps=optional_float(activity_row.get("avg_speed_mps")),
            max_speed_mps=optional_float(activity_row.get("max_speed_mps")),
            calories=optional_float(activity_row.get("calories")),
            records=empty_records(),
            source_path=str(activity_row["source_path"]),
            source_hash="",
            paused_time_s=optional_float(activity_row.get("paused_time_s")),
            segment_index=optional_int(activity_row.get("segment_index")) or 0,
            segment_count=optional_int(activity_row.get("segment_count")) or 1,
            repair_flags=flags_from_json(activity_row.get("repair_flags_json")),
        )
        metric_values = ActivityMetrics(
            activity_id=str(metrics_row["activity_id"]),
            hr_zone_1_s=required_float(metrics_row.get("hr_zone_1_s")),
            hr_zone_2_s=required_float(metrics_row.get("hr_zone_2_s")),
            hr_zone_3_s=required_float(metrics_row.get("hr_zone_3_s")),
            hr_zone_4_s=required_float(metrics_row.get("hr_zone_4_s")),
            hr_zone_5_s=required_float(metrics_row.get("hr_zone_5_s")),
            hr_coverage_ratio=required_float(metrics_row.get("hr_coverage_ratio")),
            hr_histogram_s=histogram_from_json(metrics_row.get("hr_histogram_json")),
            hr_10min_averages=buckets_from_json(metrics_row.get("hr_10min_json")),
            hr_drift_bpm_per_hour=optional_float(metrics_row.get("hr_drift_bpm_per_hour")),
            load_value=optional_float(metrics_row.get("load_value")),
            load_method=optional_str(metrics_row.get("load_method")),
            load_confidence=optional_str(metrics_row.get("load_confidence")),
            data_completeness=required_float(metrics_row.get("data_completeness")),
            analysis_version=str(metrics_row[schemas.ANALYSIS_VERSION]),
            ef_first_half=optional_float(metrics_row.get("ef_first_half")),
            ef_second_half=optional_float(metrics_row.get("ef_second_half")),
            aerobic_decoupling_pct=optional_float(metrics_row.get("aerobic_decoupling_pct")),
            decoupling_confidence=optional_str(metrics_row.get("decoupling_confidence"))
            or "insufficient_duration",
        )
        return AnalyzedActivity(activity=parsed, metrics=metric_values)


def _frame(rows: dict[str, dict[str, object]], schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(list(rows.values()), schema=schema)


def _index_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=schemas.SOURCE_INDEX_SCHEMA)
    return pl.DataFrame(rows, schema=schemas.SOURCE_INDEX_SCHEMA)


def _daily_records(rows: list[DailySummaryRow]) -> list[dict[str, object]]:
    return [
        {
            "date": row.date,
            "sport": row.sport,
            "activity_count": row.activity_count,
            "duration_s": row.duration_s,
            "distance_m": row.distance_m,
            "elevation_gain_m": row.elevation_gain_m,
            "hr_zone_1_s": row.hr_zone_1_s,
            "hr_zone_2_s": row.hr_zone_2_s,
            "hr_zone_3_s": row.hr_zone_3_s,
            "hr_zone_4_s": row.hr_zone_4_s,
            "hr_zone_5_s": row.hr_zone_5_s,
            "load_value": row.load_value,
            "hard_activity_count": row.hard_activity_count,
            "longest_activity_s": row.longest_activity_s,
            "longest_activity_id": row.longest_activity_id,
        }
        for row in rows
    ]


def _weekly_records(rows: list[WeeklySummaryRow]) -> list[dict[str, object]]:
    return [
        {
            "iso_year": row.iso_year,
            "iso_week": row.iso_week,
            "week_start": row.week_start,
            "sport": row.sport,
            "activity_count": row.activity_count,
            "duration_s": row.duration_s,
            "distance_m": row.distance_m,
            "elevation_gain_m": row.elevation_gain_m,
            "hr_zone_1_s": row.hr_zone_1_s,
            "hr_zone_2_s": row.hr_zone_2_s,
            "hr_zone_3_s": row.hr_zone_3_s,
            "hr_zone_4_s": row.hr_zone_4_s,
            "hr_zone_5_s": row.hr_zone_5_s,
            "load_value": row.load_value,
            "hard_activity_count": row.hard_activity_count,
            "longest_activity_s": row.longest_activity_s,
            "longest_activity_id": row.longest_activity_id,
            "load_value_prev": row.load_value_prev,
            "duration_s_prev": row.duration_s_prev,
            "distance_m_prev": row.distance_m_prev,
        }
        for row in rows
    ]
