"""Repository abstracting Parquet access from the service layers.

Read side (``TrainingRepository``): lazy Polars scans of the cache plus row
mapping into domain models — no aggregation logic lives here. Aggregation
happens in :mod:`training_stats.analytics` over the lazy frame exposed by
:func:`ParquetTrainingRepository.activities_scan`, so every consumer decides
its own column projection. Write side (``SyncState`` load/save): the storage
boundary for ``cache/sync.py``, so the storage engine stays swappable behind
this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo

import polars as pl

from ..converters import (
    buckets_from_json,
    flags_from_json,
    histogram_from_json,
    optional_float,
    optional_str,
)
from ..errors import CacheError
from ..models import ActivityDetails, ActivitySummary, DataQuality
from . import parquet, schemas


@dataclass(frozen=True)
class SyncState:
    """Sync working set: index rows plus the merged activity store.

    The dict is mutated in place by the sync loop; ``index_rows`` is
    replaced wholesale via a new :class:`SyncState` when the snapshot is
    saved (rows of deleted files are simply not carried over).
    """

    index_rows: list[dict[str, object]]
    activities: dict[str, dict[str, object]]


class TrainingRepository(Protocol):
    """Storage interface used by the service layer."""

    def list_activities(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        sport: str | None = None,
        limit: int = 50,
    ) -> list[ActivitySummary]: ...

    def get_activity(self, activity_id: str) -> ActivityDetails | None: ...

    def data_quality(self) -> DataQuality: ...

    def activities_scan(self) -> pl.LazyFrame: ...

    def last_sync_stamp(self) -> datetime | None: ...


class ParquetTrainingRepository:
    """Parquet-backed repository. Reads are lazy scans; mapping lives here."""

    def __init__(self, cache_dir: Path, timezone: ZoneInfo) -> None:
        self.cache_dir = cache_dir
        self.tz = timezone

    def activities_scan(self) -> pl.LazyFrame:
        """Lazy scan of the merged activity table.

        Returns an empty frame (with the full schema) when the cache file
        does not exist yet; the consumer decides projection and collect.
        """
        return self._scan(schemas.ACTIVITIES_FILE, schemas.ACTIVITY_SCHEMA)

    def list_activities(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        sport: str | None = None,
        limit: int = 50,
    ) -> list[ActivitySummary]:
        frame = (
            self.activities_scan()
            .select(
                "activity_id",
                "start_time",
                "sport",
                "sub_sport",
                "duration_s",
                "moving_time_s",
                "distance_m",
                "elevation_gain_m",
                "avg_hr",
                "max_hr",
                "calories",
                "load_value",
            )
            .with_columns(
                pl.col("start_time")
                .dt.convert_time_zone(str(self.tz))
                .dt.date()
                .alias("local_date")
            )
        )
        if start_date is not None:
            frame = frame.filter(pl.col("local_date") >= start_date)
        if end_date is not None:
            frame = frame.filter(pl.col("local_date") <= end_date)
        if sport is not None:
            frame = frame.filter(pl.col("sport").str.to_lowercase() == sport.lower())
        rows = frame.sort("start_time", descending=True).head(limit).collect().to_dicts()
        return [_summary_row(row, self.tz) for row in rows]

    def get_activity(self, activity_id: str) -> ActivityDetails | None:
        rows = (
            self.activities_scan().filter(pl.col("activity_id") == activity_id).collect().to_dicts()
        )
        if not rows:
            return None
        return _details_row(rows[0], self.tz)

    def data_quality(self) -> DataQuality:
        metrics = self.activities_scan().select("hr_coverage_ratio", "data_completeness").collect()
        activity_count = metrics.height
        mean_coverage = _mean(metrics, "hr_coverage_ratio")
        mean_completeness = _mean(metrics, "data_completeness")
        index = parquet.read_parquet(self.cache_dir / schemas.SOURCE_INDEX_FILE)
        failed = 0
        duplicates = 0
        repaired_files = 0
        discarded_segments = 0
        if not index.is_empty():
            rows = index.to_dicts()
            files: dict[tuple[str, str], tuple[bool, int]] = {}
            for row in rows:
                status = str(row.get("status"))
                if status == schemas.STATUS_FAILED:
                    failed += 1
                elif status == schemas.STATUS_DUPLICATE:
                    duplicates += 1
                key = (str(row.get("source_name")), str(row.get("source_path")))
                repaired = status == schemas.STATUS_OK and bool(
                    flags_from_json(row.get("repair_flags_json"))
                )
                previous = files.get(key, (False, 0))
                files[key] = (
                    previous[0] or repaired,
                    _int(row.get("discarded_segment_count")),
                )
            repaired_files = sum(1 for repaired, _discarded in files.values() if repaired)
            discarded_segments = sum(discarded for _repaired, discarded in files.values())
        return DataQuality(
            activity_count=activity_count,
            mean_hr_coverage_ratio=mean_coverage,
            mean_data_completeness=mean_completeness,
            failed_file_count=failed,
            repaired_file_count=repaired_files,
            discarded_segment_count=discarded_segments,
            duplicate_activity_count=duplicates,
        )

    def last_sync_stamp(self) -> datetime | None:
        """Persisted ``last_sync`` from ``metadata.json`` (cross-process throttle)."""
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

    # -- sync write side ----------------------------------------------------

    def load_sync_state(self) -> SyncState:
        """Read the sync working set: source index plus the merged activity store."""
        index_frame = parquet.read_parquet(self.cache_dir / schemas.SOURCE_INDEX_FILE)
        activities_frame = parquet.read_parquet(self.cache_dir / schemas.ACTIVITIES_FILE)
        return SyncState(
            index_rows=[] if index_frame.is_empty() else index_frame.to_dicts(),
            activities=_store_dict(activities_frame),
        )

    def save_sync_snapshot(
        self, state: SyncState, *, last_sync: datetime, analysis_version: str
    ) -> None:
        """Atomically rewrite the whole cache from the sync working set.

        Prunes activity rows without an active ``ok`` index row (removed or
        re-specified files, historical orphans) so the store and the index
        stay consistent, then writes the merged activity table, the index
        and ``metadata.json``. Aggregates are derived on read (see
        :mod:`training_stats.analytics`), never persisted.
        """
        ok_ids = {
            str(row["activity_id"])
            for row in state.index_rows
            if row.get("status") == schemas.STATUS_OK and row.get("activity_id") is not None
        }
        activities = {key: row for key, row in state.activities.items() if key in ok_ids}

        parquet.write_parquet_atomic(
            _frame(activities, schemas.ACTIVITY_SCHEMA), self.cache_dir / schemas.ACTIVITIES_FILE
        )
        parquet.write_parquet_atomic(
            _index_frame(state.index_rows), self.cache_dir / schemas.SOURCE_INDEX_FILE
        )
        parquet.write_json_atomic(
            {
                "last_sync": last_sync.isoformat(),
                "analysis_version": analysis_version,
                "activity_count": len(activities),
                "failed_file_count": sum(
                    1 for row in state.index_rows if row.get("status") == schemas.STATUS_FAILED
                ),
            },
            self.cache_dir / schemas.METADATA_FILE,
        )

    # -- helpers -----------------------------------------------------------

    def _scan(self, filename: str, schema: dict[str, pl.DataType]) -> pl.LazyFrame:
        path = self.cache_dir / filename
        if not path.exists():
            return pl.DataFrame(schema=schema).lazy()
        return pl.scan_parquet(path)


# -- row mapping helpers ----------------------------------------------------


def _summary_row(row: dict[str, object], tz: ZoneInfo) -> ActivitySummary:
    start_time = row["start_time"]
    if not isinstance(start_time, datetime):
        msg = "activity start_time has unexpected type"
        raise CacheError(msg)
    return ActivitySummary(
        activity_id=str(row["activity_id"]),
        start_time=start_time,
        sport=str(row["sport"]),
        sub_sport=optional_str(row.get("sub_sport")),
        duration_s=optional_float(row.get("duration_s")),
        distance_m=optional_float(row.get("distance_m")),
        elevation_gain_m=optional_float(row.get("elevation_gain_m")),
        avg_hr=optional_float(row.get("avg_hr")),
        max_hr=optional_float(row.get("max_hr")),
        calories=optional_float(row.get("calories")),
        load_value=optional_float(row.get("load_value")),
        moving_time_s=optional_float(row.get("moving_time_s")),
        local_start_time=start_time.astimezone(tz).isoformat(),
    )


def _details_row(row: dict[str, object], tz: ZoneInfo) -> ActivityDetails:
    summary = _summary_row(row, tz)
    return ActivityDetails(
        activity_id=summary.activity_id,
        start_time=summary.start_time,
        sport=summary.sport,
        sub_sport=summary.sub_sport,
        duration_s=summary.duration_s,
        moving_time_s=summary.moving_time_s,
        distance_m=summary.distance_m,
        elevation_gain_m=summary.elevation_gain_m,
        avg_hr=summary.avg_hr,
        max_hr=summary.max_hr,
        avg_speed_mps=optional_float(row.get("avg_speed_mps")),
        max_speed_mps=optional_float(row.get("max_speed_mps")),
        calories=summary.calories,
        hr_zone_1_s=_float(row.get("hr_zone_1_s")),
        hr_zone_2_s=_float(row.get("hr_zone_2_s")),
        hr_zone_3_s=_float(row.get("hr_zone_3_s")),
        hr_zone_4_s=_float(row.get("hr_zone_4_s")),
        hr_zone_5_s=_float(row.get("hr_zone_5_s")),
        hr_coverage_ratio=_float(row.get("hr_coverage_ratio")),
        hr_histogram_s=histogram_from_json(row.get("hr_histogram_json")),
        hr_10min_averages=buckets_from_json(row.get("hr_10min_json")),
        hr_drift_bpm_per_hour=optional_float(row.get("hr_drift_bpm_per_hour")),
        load_value=summary.load_value,
        load_method=optional_str(row.get("load_method")),
        load_confidence=optional_str(row.get("load_confidence")),
        data_completeness=_float(row.get("data_completeness")),
        analysis_version=str(row.get(schemas.ANALYSIS_VERSION, "")),
        source_name=str(row.get("source_name", "")),
        source_path=str(row.get("source_path", "")),
        ef_first_half=optional_float(row.get("ef_first_half")),
        ef_second_half=optional_float(row.get("ef_second_half")),
        aerobic_decoupling_pct=optional_float(row.get("aerobic_decoupling_pct")),
        decoupling_confidence=optional_str(row.get("decoupling_confidence")),
        local_start_time=summary.local_start_time,
        paused_time_s=optional_float(row.get("paused_time_s")),
        repair_flags=flags_from_json(row.get("repair_flags_json")),
    )


def _store_dict(frame: pl.DataFrame) -> dict[str, dict[str, object]]:
    if frame.is_empty():
        return {}
    return {str(row["activity_id"]): row for row in frame.to_dicts()}


def _float(value: object) -> float:
    result = optional_float(value)
    return 0.0 if result is None else result


def _int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return int(value)


def _mean(frame: pl.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    series = frame[column].drop_nulls()
    if series.len() == 0:
        return None
    return float(series.mean())  # type: ignore[arg-type]


def _frame(rows: dict[str, dict[str, object]], schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(list(rows.values()), schema=schema)


def _index_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=schemas.SOURCE_INDEX_SCHEMA)
    return pl.DataFrame(rows, schema=schemas.SOURCE_INDEX_SCHEMA)
