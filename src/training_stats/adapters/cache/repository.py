"""Parquet adapter implementing the service read and sync-state ports."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from ...domain import frames
from ...domain.analytics.heart_rate import HrBucketAvg
from ...services.ports import SyncRepository, TrainingRepository
from ...services.state import ActivityRecord, SourceIndexEntry, SyncState
from ..errors import CacheError
from . import parquet, schemas


class ParquetTrainingRepository(TrainingRepository, SyncRepository):
    """Parquet-backed repository over the unified physical cache file."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir

    def cache_scan(self) -> pl.LazyFrame:
        """Lazy scan of the physical cache table, including source-only rows."""
        return self._scan(schemas.CACHE_FILE, schemas.CACHE_SCHEMA)

    def activities_scan(self) -> pl.LazyFrame:
        """Return the canonical domain activity frame without collecting it."""
        return (
            self.cache_scan()
            .filter(pl.col("start_time").is_not_null())
            .select(frames.ACTIVITY_COLUMNS)
        )

    def source_index_scan(self) -> pl.LazyFrame:
        """Return the canonical source-outcome frame without collecting it."""
        return self.cache_scan().select(frames.SOURCE_INDEX_COLUMNS)

    def last_sync_stamp(self) -> datetime | None:
        """Persisted ``last_sync`` from ``metadata.json`` for cross-process throttling."""
        metadata = parquet.read_json(self.cache_dir / schemas.METADATA_FILE)
        last_sync = metadata.get("last_sync") if metadata else None
        if not isinstance(last_sync, str):
            return None
        try:
            stamp = datetime.fromisoformat(last_sync)
        except ValueError:
            return None
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=UTC)
        return stamp

    def load_sync_state(self) -> SyncState:
        """Load typed source outcomes and accepted activity records."""
        frame = parquet.read_parquet(self.cache_dir / schemas.CACHE_FILE)
        if frame.is_empty():
            return SyncState(index_rows=[], activities={})
        rows = frame.to_dicts()
        index_rows = [_source_entry_from_row(row) for row in rows]
        activities: dict[str, ActivityRecord] = {}
        for row in rows:
            activity_id = row.get("activity_id")
            if row.get("start_time") is not None and activity_id is not None:
                record = _activity_record_from_row(row)
                activities[str(activity_id)] = record
        return SyncState(index_rows=index_rows, activities=activities)

    def save_sync_snapshot(self, state: SyncState, *, last_sync: datetime) -> None:
        """Join accepted activities to their source rows and atomically persist."""
        ok_ids = {
            row.activity_id
            for row in state.index_rows
            if row.status == frames.STATUS_OK and row.activity_id is not None
        }
        activities = {key: row for key, row in state.activities.items() if key in ok_ids}
        rows = [
            _cache_row(
                index_row,
                activities.get(index_row.activity_id)
                if index_row.status == frames.STATUS_OK and index_row.activity_id is not None
                else None,
            )
            for index_row in state.index_rows
        ]
        parquet.write_parquet_atomic(_cache_frame(rows), self.cache_dir / schemas.CACHE_FILE)
        parquet.write_json_atomic(
            {"last_sync": last_sync.isoformat()},
            self.cache_dir / schemas.METADATA_FILE,
        )

    def _scan(self, filename: str, schema: dict[str, pl.DataType]) -> pl.LazyFrame:
        path = self.cache_dir / filename
        if not path.exists():
            return pl.DataFrame(schema=schema).lazy()
        return pl.scan_parquet(path)


def _source_entry_from_row(row: dict[str, object]) -> SourceIndexEntry:
    status_value = row.get("status")
    if status_value not in {frames.STATUS_OK, frames.STATUS_FAILED, frames.STATUS_DUPLICATE}:
        raise CacheError("cache row has an invalid source status")
    processed_at = row.get(frames.PROCESSED_AT)
    if not isinstance(processed_at, datetime):
        raise CacheError("cache row has an invalid processed_at value")
    return SourceIndexEntry(
        source_name=str(row.get("source_name", "")),
        source_path=str(row.get("source_path", "")),
        source_hash=_optional_str(row.get("source_hash")),
        file_size=_optional_int(row.get("file_size")),
        modified_at=_optional_datetime(row.get("modified_at")),
        activity_id=_optional_str(row.get("activity_id")),
        segment_index=_int(row.get("segment_index")),
        segment_count=_int(row.get("segment_count")),
        file_repair_flags=_string_tuple(row.get(frames.FILE_REPAIR_FLAGS)),
        discarded_segment_count=_int(row.get("discarded_segment_count")),
        processed_at=processed_at,
        source_analysis_version=str(row.get(frames.SOURCE_ANALYSIS_VERSION, "")),
        status=status_value,
        error_message=_optional_str(row.get("error_message")),
    )


def _activity_record_from_row(row: dict[str, object]) -> ActivityRecord:
    start_time = row.get("start_time")
    if not isinstance(start_time, datetime):
        raise CacheError("activity row has an invalid start_time value")
    return ActivityRecord(
        activity_id=str(row.get("activity_id", "")),
        start_time=start_time,
        sport=str(row.get("sport", "")),
        sub_sport=_optional_str(row.get("sub_sport")),
        duration_s=_optional_float(row.get("duration_s")),
        moving_time_s=_optional_float(row.get("moving_time_s")),
        paused_time_s=_optional_float(row.get("paused_time_s")),
        distance_m=_optional_float(row.get("distance_m")),
        elevation_gain_m=_optional_float(row.get("elevation_gain_m")),
        avg_hr=_optional_float(row.get("avg_hr")),
        max_hr=_optional_float(row.get("max_hr")),
        avg_speed_mps=_optional_float(row.get("avg_speed_mps")),
        max_speed_mps=_optional_float(row.get("max_speed_mps")),
        segment_index=_int(row.get("segment_index")),
        segment_count=_int(row.get("segment_count")),
        repair_flags=_string_tuple(row.get("repair_flags")),
        source_name=str(row.get("source_name", "")),
        source_path=str(row.get("source_path", "")),
        hr_zone_1_s=_float(row.get("hr_zone_1_s")),
        hr_zone_2_s=_float(row.get("hr_zone_2_s")),
        hr_zone_3_s=_float(row.get("hr_zone_3_s")),
        hr_zone_4_s=_float(row.get("hr_zone_4_s")),
        hr_zone_5_s=_float(row.get("hr_zone_5_s")),
        hr_coverage_ratio=_float(row.get("hr_coverage_ratio")),
        hr_histogram_s=_histogram(row.get("hr_histogram")),
        hr_10min_averages=_buckets(row.get("hr_10min_averages")),
        hr_drift_bpm_per_hour=_optional_float(row.get("hr_drift_bpm_per_hour")),
        load_value=_optional_float(row.get("load_value")),
        load_method=_optional_str(row.get("load_method")),
        load_confidence=_optional_str(row.get("load_confidence")),
        data_completeness=_float(row.get("data_completeness")),
        ef_first_half=_optional_float(row.get("ef_first_half")),
        ef_second_half=_optional_float(row.get("ef_second_half")),
        aerobic_decoupling_pct=_optional_float(row.get("aerobic_decoupling_pct")),
        decoupling_confidence=str(row.get("decoupling_confidence", "")),
        analysis_version=str(row.get(frames.ANALYSIS_VERSION, "")),
    )


def _cache_row(source: SourceIndexEntry, activity: ActivityRecord | None) -> dict[str, object]:
    row: dict[str, object] = {name: None for name in schemas.CACHE_SCHEMA}
    row.update(asdict(source))
    row[frames.FILE_REPAIR_FLAGS] = list(source.file_repair_flags)
    if activity is not None:
        row.update(asdict(activity))
        row["repair_flags"] = list(activity.repair_flags)
        row["hr_histogram"] = [
            {"bin": label, "seconds": seconds} for label, seconds in activity.hr_histogram_s.items()
        ]
        row["hr_10min_averages"] = [
            {"start_min": bucket.start_min, "avg_hr": bucket.avg_hr, "hr_time_s": bucket.hr_time_s}
            for bucket in activity.hr_10min_averages
        ]
    return row


def _cache_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=schemas.CACHE_SCHEMA)
    return pl.DataFrame(rows, schema=schemas.CACHE_SCHEMA)


def _histogram(value: object) -> dict[str, float]:
    if not isinstance(value, list):
        return {}
    return {
        str(row["bin"]): _float(row.get("seconds"))
        for row in value
        if isinstance(row, dict) and row.get("bin") is not None
    }


def _buckets(value: object) -> tuple[HrBucketAvg, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        HrBucketAvg(
            start_min=_int(row.get("start_min")),
            avg_hr=_float(row.get("avg_hr")),
            hr_time_s=_float(row.get("hr_time_s")),
        )
        for row in value
        if isinstance(row, dict)
    )


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(str(item) for item in value)


def _optional_datetime(value: object) -> datetime | None:
    return value if isinstance(value, datetime) else None


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _float(value: object) -> float:
    result = _optional_float(value)
    return 0.0 if result is None else result


def _int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return int(value)
