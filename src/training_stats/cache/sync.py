"""Incremental synchronization from activity sources into the Parquet cache.

Change detection is two-tier: files whose stored size and modification time
match the previous sync are skipped without being read at all; downloaded
candidates are then SHA-256 hashed, and only files whose hash or analysis
version changed are reprocessed.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, tzinfo
from pathlib import Path

import polars as pl

from ..analytics.activities import ActivityMetrics, AnalyzedActivity, compute_activity_metrics
from ..analytics.daily import DailySummaryRow, summarize_daily
from ..analytics.heart_rate import HrBucketAvg
from ..analytics.weekly import WeeklySummaryRow, daily_to_weekly
from ..config import AthleteConfig, QualityConfig
from ..errors import FitParseError, SourceError
from ..fit.parser import FitParser
from ..fit.types import ParsedActivity, empty_records
from ..models import SourceFile, SyncResult
from ..sources.base import ActivitySource
from . import parquet, schemas
from .locking import sync_lock

logger = logging.getLogger(__name__)

_STATUS_OK = "ok"
_STATUS_FAILED = "failed"
_STATUS_DUPLICATE = "duplicate"


@dataclass(frozen=True, slots=True)
class _Outcome:
    """Result of processing one file during a sync."""

    index_rows: list[dict[str, object]]
    ok_activity_rows: list[dict[str, object]]
    ok_metrics_rows: list[dict[str, object]]
    duplicates: int
    repaired: bool
    discarded_segments: int


def _empty_outcome(index_rows: list[dict[str, object]]) -> _Outcome:
    return _Outcome(
        index_rows=index_rows,
        ok_activity_rows=[],
        ok_metrics_rows=[],
        duplicates=0,
        repaired=False,
        discarded_segments=0,
    )


class SyncService:
    """Synchronizes FIT files from all sources into the Parquet cache."""

    def __init__(
        self,
        *,
        sources: list[ActivitySource],
        parser: FitParser,
        cache_dir: Path,
        athlete: AthleteConfig,
        timezone: tzinfo,
        analysis_version: str,
        quality: QualityConfig | None = None,
        lock_timeout_seconds: float = 30.0,
    ) -> None:
        self.sources = sources
        self.parser = parser
        self.cache_dir = cache_dir
        self.athlete = athlete
        self.tz = timezone
        self.analysis_version = analysis_version
        self.quality = quality or QualityConfig()
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
                outcome = self._process_file(
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

    def _process_file(
        self,
        source_file: SourceFile,
        content: bytes,
        file_hash: str,
        existing_ids: set[str],
        processed_at: datetime,
    ) -> _Outcome:
        base_row: dict[str, object] = {
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
            "status": _STATUS_FAILED,
            "error_message": None,
        }
        try:
            parsed_file = self.parser.parse(content, source_file)
        except FitParseError as exc:
            logger.error("cannot parse %s/%s: %s", source_file.source_name, source_file.path, exc)
            base_row["error_message"] = str(exc)
            return _empty_outcome([base_row])

        file_flags_json = _flags_to_json(parsed_file.repair_flags)
        index_rows: list[dict[str, object]] = []
        ok_activity_rows: list[dict[str, object]] = []
        ok_metrics_rows: list[dict[str, object]] = []
        duplicates = 0
        for activity in parsed_file.activities:
            analyzed = AnalyzedActivity(
                activity=activity,
                metrics=compute_activity_metrics(
                    activity, self.athlete, self.analysis_version, self.quality
                ),
            )
            row = dict(base_row)
            row["activity_id"] = activity.activity_id
            row["segment_index"] = activity.segment_index
            row["segment_count"] = activity.segment_count
            row["repair_flags_json"] = file_flags_json
            row["discarded_segment_count"] = parsed_file.discarded_segment_count
            if activity.activity_id in existing_ids:
                row["status"] = _STATUS_DUPLICATE
                row["error_message"] = "duplicate activity id"
                duplicates += 1
                logger.info(
                    "duplicate %s/%s: activity %s already cached",
                    source_file.source_name,
                    source_file.path,
                    activity.activity_id,
                )
            else:
                row["status"] = _STATUS_OK
                existing_ids.add(activity.activity_id)
                ok_activity_rows.append(self._activity_row(analyzed, source_file.source_name))
                ok_metrics_rows.append(self._metrics_row(analyzed))
            index_rows.append(row)
        return _Outcome(
            index_rows=index_rows,
            ok_activity_rows=ok_activity_rows,
            ok_metrics_rows=ok_metrics_rows,
            duplicates=duplicates,
            repaired=bool(parsed_file.repair_flags),
            discarded_segments=parsed_file.discarded_segment_count,
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
            "status": _STATUS_FAILED,
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
            if row.get("status") == _STATUS_OK and row.get("activity_id") is not None
        }
        activities = {key: row for key, row in activities.items() if key in ok_ids}
        metrics = {key: row for key, row in metrics.items() if key in ok_ids}
        analyzed = [
            self._rebuild_analyzed(
                activities[str(row["activity_id"])], metrics[str(row["activity_id"])]
            )
            for row in index_rows
            if row.get("status") == _STATUS_OK
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
                    1 for row in index_rows if row.get("status") == _STATUS_FAILED
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
            sub_sport=_optional_str(activity_row.get("sub_sport")),
            duration_s=_optional_float(activity_row.get("duration_s")),
            moving_time_s=_optional_float(activity_row.get("moving_time_s")),
            distance_m=_optional_float(activity_row.get("distance_m")),
            elevation_gain_m=_optional_float(activity_row.get("elevation_gain_m")),
            avg_hr=_optional_float(activity_row.get("avg_hr")),
            max_hr=_optional_float(activity_row.get("max_hr")),
            avg_speed_mps=_optional_float(activity_row.get("avg_speed_mps")),
            max_speed_mps=_optional_float(activity_row.get("max_speed_mps")),
            calories=_optional_float(activity_row.get("calories")),
            records=empty_records(),
            source_path=str(activity_row["source_path"]),
            source_hash="",
            paused_time_s=_optional_float(activity_row.get("paused_time_s")),
            segment_index=_optional_int(activity_row.get("segment_index")) or 0,
            segment_count=_optional_int(activity_row.get("segment_count")) or 1,
            repair_flags=_flags_from_json(activity_row.get("repair_flags_json")),
        )
        metric_values = ActivityMetrics(
            activity_id=str(metrics_row["activity_id"]),
            hr_zone_1_s=_required_float(metrics_row.get("hr_zone_1_s")),
            hr_zone_2_s=_required_float(metrics_row.get("hr_zone_2_s")),
            hr_zone_3_s=_required_float(metrics_row.get("hr_zone_3_s")),
            hr_zone_4_s=_required_float(metrics_row.get("hr_zone_4_s")),
            hr_zone_5_s=_required_float(metrics_row.get("hr_zone_5_s")),
            hr_coverage_ratio=_required_float(metrics_row.get("hr_coverage_ratio")),
            hr_histogram_s=_histogram_from_json(metrics_row.get("hr_histogram_json")),
            hr_10min_averages=_buckets_from_json(metrics_row.get("hr_10min_json")),
            hr_drift_bpm_per_hour=_optional_float(metrics_row.get("hr_drift_bpm_per_hour")),
            load_value=_optional_float(metrics_row.get("load_value")),
            load_method=_optional_str(metrics_row.get("load_method")),
            load_confidence=_optional_str(metrics_row.get("load_confidence")),
            data_completeness=_required_float(metrics_row.get("data_completeness")),
            analysis_version=str(metrics_row[schemas.ANALYSIS_VERSION]),
            ef_first_half=_optional_float(metrics_row.get("ef_first_half")),
            ef_second_half=_optional_float(metrics_row.get("ef_second_half")),
            aerobic_decoupling_pct=_optional_float(metrics_row.get("aerobic_decoupling_pct")),
            decoupling_confidence=_optional_str(metrics_row.get("decoupling_confidence"))
            or "insufficient_duration",
        )
        return AnalyzedActivity(activity=parsed, metrics=metric_values)

    @staticmethod
    def _activity_row(item: AnalyzedActivity, source_name: str) -> dict[str, object]:
        activity = item.activity
        return {
            "activity_id": activity.activity_id,
            "start_time": _ensure_utc(activity.start_time),
            "sport": activity.sport,
            "sub_sport": activity.sub_sport,
            "duration_s": activity.duration_s,
            "moving_time_s": activity.moving_time_s,
            "paused_time_s": activity.paused_time_s,
            "distance_m": activity.distance_m,
            "elevation_gain_m": activity.elevation_gain_m,
            "avg_hr": activity.avg_hr,
            "max_hr": activity.max_hr,
            "avg_speed_mps": activity.avg_speed_mps,
            "max_speed_mps": activity.max_speed_mps,
            "calories": activity.calories,
            "segment_index": activity.segment_index,
            "segment_count": activity.segment_count,
            "repair_flags_json": _flags_to_json(activity.repair_flags),
            "source_name": source_name,
            "source_path": activity.source_path,
        }

    @staticmethod
    def _metrics_row(item: AnalyzedActivity) -> dict[str, object]:
        metric_values = item.metrics
        return {
            "activity_id": metric_values.activity_id,
            "hr_zone_1_s": metric_values.hr_zone_1_s,
            "hr_zone_2_s": metric_values.hr_zone_2_s,
            "hr_zone_3_s": metric_values.hr_zone_3_s,
            "hr_zone_4_s": metric_values.hr_zone_4_s,
            "hr_zone_5_s": metric_values.hr_zone_5_s,
            "hr_coverage_ratio": metric_values.hr_coverage_ratio,
            "hr_histogram_json": _histogram_to_json(metric_values.hr_histogram_s),
            "hr_10min_json": _buckets_to_json(metric_values.hr_10min_averages),
            "hr_drift_bpm_per_hour": metric_values.hr_drift_bpm_per_hour,
            "load_value": metric_values.load_value,
            "load_method": metric_values.load_method,
            "load_confidence": metric_values.load_confidence,
            "data_completeness": metric_values.data_completeness,
            "ef_first_half": metric_values.ef_first_half,
            "ef_second_half": metric_values.ef_second_half,
            "aerobic_decoupling_pct": metric_values.aerobic_decoupling_pct,
            "decoupling_confidence": metric_values.decoupling_confidence,
            schemas.ANALYSIS_VERSION: metric_values.analysis_version,
        }


def _frame(rows: dict[str, dict[str, object]], schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(list(rows.values()), schema=schema)


def _histogram_to_json(histogram: dict[str, float]) -> str:
    return json.dumps({bin_key: round(seconds, 3) for bin_key, seconds in histogram.items()})


def _buckets_to_json(buckets: tuple[HrBucketAvg, ...]) -> str:
    return json.dumps([bucket.to_dict() for bucket in buckets])


def _histogram_from_json(value: object) -> dict[str, float]:
    if not isinstance(value, str) or not value:
        return {}
    try:
        raw = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(key): _required_float(seconds) for key, seconds in raw.items()}


def _buckets_from_json(value: object) -> tuple[HrBucketAvg, ...]:
    if not isinstance(value, str) or not value:
        return ()
    try:
        raw = json.loads(value)
    except json.JSONDecodeError:
        return ()
    if not isinstance(raw, list):
        return ()
    buckets: list[HrBucketAvg] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        buckets.append(
            HrBucketAvg(
                start_min=_int_value(entry.get("start_min")),
                avg_hr=_required_float(entry.get("avg_hr")),
                hr_time_s=_required_float(entry.get("hr_time_s")),
            )
        )
    return tuple(buckets)


def _int_value(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return int(value)


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


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return int(value)


def _flags_to_json(flags: tuple[str, ...]) -> str | None:
    return json.dumps(list(flags)) if flags else None


def _flags_from_json(value: object) -> tuple[str, ...]:
    if not isinstance(value, str) or not value:
        return ()
    try:
        raw = json.loads(value)
    except json.JSONDecodeError:
        return ()
    if not isinstance(raw, list):
        return ()
    return tuple(str(flag) for flag in raw)


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _required_float(value: object) -> float:
    return _optional_float(value) or 0.0


def _ensure_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
