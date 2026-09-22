"""Repository abstracting Parquet access from the service layers.

Read side (``TrainingRepository``): queries for the API and CLI. Write side
(``SyncState`` load/save): the storage boundary for ``cache/sync.py``, so the
storage engine stays swappable behind this module.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo

import polars as pl

from ..analytics.activities import ActivityMetrics, AnalyzedActivity
from ..analytics.daily import DailySummaryRow, summarize_daily
from ..analytics.fitness import (
    compute_days_since_60k,
    compute_rolling_distance,
    compute_training_form,
)
from ..analytics.weekly import WeeklySummaryRow, daily_to_weekly
from ..config import AthleteConfig
from ..converters import (
    buckets_from_json,
    flags_from_json,
    histogram_from_json,
    optional_float,
    optional_int,
    optional_str,
    required_float,
)
from ..errors import CacheError
from ..fit.types import ParsedActivity, empty_records
from ..models import (
    ActivityDetails,
    ActivitySummary,
    DataQuality,
    Overview,
    TrainingFormResult,
)
from . import parquet, schemas
from .locking import read_lock

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SyncState:
    """Sync working set: index rows plus the activity/metrics stores.

    The dicts are mutated in place by the sync loop; ``index_rows`` is
    replaced wholesale via a new :class:`SyncState` when the snapshot is
    saved (rows of deleted files are simply not carried over).
    """

    index_rows: list[dict[str, object]]
    activities: dict[str, dict[str, object]]
    metrics: dict[str, dict[str, object]]


class TrainingRepository(Protocol):
    """Read interface used by the API and CLI layers."""

    def list_activities(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        sport: str | None = None,
        limit: int = 50,
    ) -> list[ActivitySummary]: ...

    def get_activity(self, activity_id: str) -> ActivityDetails | None: ...

    def daily_summary(
        self, start_date: date | None = None, end_date: date | None = None, sport: str | None = None
    ) -> list[DailySummaryRow]: ...

    def weekly_summary(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        weeks: int = 8,
        sport: str | None = None,
    ) -> list[WeeklySummaryRow]: ...

    def training_form(self, days: int = 42, sport: str | None = None) -> TrainingFormResult: ...

    def data_quality(self) -> DataQuality: ...

    def overview(self, days: int = 28, sport: str | None = None) -> Overview: ...

    def last_sync_stamp(self) -> datetime | None: ...


class ParquetTrainingRepository:
    """Parquet-backed repository. All filtering/aggregation lives here."""

    def __init__(
        self,
        cache_dir: Path,
        timezone: ZoneInfo,
        athlete: AthleteConfig,
        lock_timeout_seconds: float = 30.0,
    ) -> None:
        self.cache_dir = cache_dir
        self.tz = timezone
        self.athlete = athlete
        self.lock_timeout_seconds = lock_timeout_seconds

    def _read_lock(self) -> AbstractContextManager[None]:
        return read_lock(self.cache_dir / schemas.LOCK_FILE, self.lock_timeout_seconds)

    # -- activities -------------------------------------------------------

    def list_activities(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        sport: str | None = None,
        limit: int = 50,
    ) -> list[ActivitySummary]:
        with self._read_lock():
            return self._list_activities(start_date, end_date, sport, limit)

    def _list_activities(
        self,
        start_date: date | None,
        end_date: date | None,
        sport: str | None,
        limit: int,
    ) -> list[ActivitySummary]:
        frame = self._activities_with_load()
        if frame.is_empty():
            return []
        frame = self._with_local_date(frame)
        frame = self._filter(frame, start_date, end_date, sport)
        frame = frame.sort("start_time", descending=True).head(limit)
        return [_summary_row(row, self.tz) for row in frame.to_dicts()]

    def get_activity(self, activity_id: str) -> ActivityDetails | None:
        with self._read_lock():
            activities = parquet.read_parquet(self.cache_dir / schemas.ACTIVITIES_FILE)
            metrics = parquet.read_parquet(self.cache_dir / schemas.ACTIVITY_METRICS_FILE)
            if activities.is_empty() or metrics.is_empty():
                return None
            joined = activities.join(metrics, on="activity_id", how="inner")
            selected = joined.filter(pl.col("activity_id") == activity_id)
            if selected.is_empty():
                return None
            return _details_row(selected.to_dicts()[0], self.tz)

    # -- aggregations -----------------------------------------------------

    def daily_summary(
        self, start_date: date | None = None, end_date: date | None = None, sport: str | None = None
    ) -> list[DailySummaryRow]:
        with self._read_lock():
            frame = parquet.read_parquet(self.cache_dir / schemas.DAILY_FILE)
            if frame.is_empty():
                return []
            rows = frame.to_dicts()
            filtered = [
                row
                for row in rows
                if _date_in_range(date.fromisoformat(str(row["date"])), start_date, end_date)
                and (sport is None or str(row["sport"]).lower() == sport.lower())
            ]
            if sport is None:
                filtered = _aggregate_daily(filtered)
            return [
                _daily_row(row)
                for row in sorted(filtered, key=lambda row: (str(row["date"]), str(row["sport"])))
            ]

    def weekly_summary(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        weeks: int = 8,
        sport: str | None = None,
    ) -> list[WeeklySummaryRow]:
        with self._read_lock():
            return self._weekly_summary(start_date, end_date, weeks, sport)

    def _weekly_summary(
        self,
        start_date: date | None,
        end_date: date | None,
        weeks: int,
        sport: str | None,
    ) -> list[WeeklySummaryRow]:
        frame = parquet.read_parquet(self.cache_dir / schemas.WEEKLY_FILE)
        if frame.is_empty():
            return []
        rows = frame.to_dicts()
        filtered = [
            row
            for row in rows
            if (sport is None or str(row["sport"]).lower() == sport.lower())
            and (start_date is None or date.fromisoformat(str(row["week_start"])) >= start_date)
            and (end_date is None or date.fromisoformat(str(row["week_start"])) <= end_date)
        ]
        if sport is None:
            filtered = _aggregate_weekly(filtered)
        filtered.sort(key=lambda row: (_int(row["iso_year"]), _int(row["iso_week"])), reverse=True)
        if start_date is None and end_date is None:
            filtered = filtered[:weeks]
        return [_weekly_row(row) for row in filtered]

    def training_form(self, days: int = 42, sport: str | None = None) -> TrainingFormResult:
        with self._read_lock():
            today = datetime.now(self.tz).date()

            # 1. Daily loads and distances
            daily_frame = parquet.read_parquet(self.cache_dir / schemas.DAILY_FILE)
            daily_loads: dict[date, float] = {}
            daily_distances: dict[date, float] = {}

            if not daily_frame.is_empty():
                rows = daily_frame.to_dicts()
                if sport is not None:
                    rows = [r for r in rows if str(r["sport"]).lower() == sport.lower()]
                for r in rows:
                    d = date.fromisoformat(str(r["date"]))
                    load = _float(r.get("load_value"))
                    dist = _float(r.get("distance_m"))
                    daily_loads[d] = daily_loads.get(d, 0.0) + load
                    daily_distances[d] = daily_distances.get(d, 0.0) + dist

            # 2. Rolling distances (meters -> km)
            dist_7d_m = compute_rolling_distance(daily_distances, today, 7)
            dist_28d_m = compute_rolling_distance(daily_distances, today, 28)
            dist_7d_km = round(dist_7d_m / 1000.0, 2)
            dist_28d_km = round(dist_28d_m / 1000.0, 2)

            # 3. Days since 60+ km ride
            activities_frame = parquet.read_parquet(self.cache_dir / schemas.ACTIVITIES_FILE)
            days_since_60k: int | None = None
            if not activities_frame.is_empty():
                act_rows = activities_frame.to_dicts()
                if sport is not None:
                    act_rows = [r for r in act_rows if str(r["sport"]).lower() == sport.lower()]
                qualifying: list[tuple[date, float]] = []
                for r in act_rows:
                    start_time = r["start_time"]
                    dist = _float(r.get("distance_m"))
                    if isinstance(start_time, datetime) and dist >= 60_000.0:
                        local_d = start_time.astimezone(self.tz).date()
                        qualifying.append((local_d, dist))
                days_since_60k = compute_days_since_60k(qualifying, today)

            # 4. Load confidence ratio
            metrics_frame = parquet.read_parquet(self.cache_dir / schemas.ACTIVITY_METRICS_FILE)
            load_conf_ratio = 1.0
            if not metrics_frame.is_empty():
                m_rows = metrics_frame.to_dicts()
                if sport is not None and not activities_frame.is_empty():
                    sport_act_ids = {
                        str(r["activity_id"])
                        for r in activities_frame.to_dicts()
                        if str(r.get("sport", "")).lower() == sport.lower()
                    }
                    m_rows = [r for r in m_rows if str(r["activity_id"]) in sport_act_ids]

                total_load = 0.0
                high_conf_load = 0.0
                for r in m_rows:
                    row_load = optional_float(r.get("load_value"))
                    if row_load is None:
                        continue
                    total_load += row_load
                    if r.get("load_confidence") == "high":
                        high_conf_load += row_load
                if total_load > 0:
                    load_conf_ratio = round(high_conf_load / total_load, 4)

            return compute_training_form(
                daily_loads=daily_loads,
                today=today,
                days=days,
                rolling_7d_distance_km=dist_7d_km,
                rolling_28d_distance_km=dist_28d_km,
                days_since_60k_ride=days_since_60k,
                load_confidence_ratio=load_conf_ratio,
            )

    # -- quality and context ----------------------------------------------

    def data_quality(self) -> DataQuality:
        with self._read_lock():
            metrics = parquet.read_parquet(self.cache_dir / schemas.ACTIVITY_METRICS_FILE)
            index = parquet.read_parquet(self.cache_dir / schemas.SOURCE_INDEX_FILE)
            activity_count = metrics.height
            mean_coverage = _mean(metrics, "hr_coverage_ratio") if not metrics.is_empty() else None
            mean_completeness = (
                _mean(metrics, "data_completeness") if not metrics.is_empty() else None
            )
            failed = 0
            duplicates = 0
            repaired_files = 0
            discarded_segments = 0
            if not index.is_empty():
                rows = index.to_dicts()
                files: dict[tuple[str, str], tuple[bool, int]] = {}
                for row in rows:
                    status = str(row.get("status"))
                    if status == "failed":
                        failed += 1
                    elif status == "duplicate":
                        duplicates += 1
                    key = (str(row.get("source_name")), str(row.get("source_path")))
                    repaired = str(row.get("status")) == "ok" and bool(
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

    def overview(self, days: int = 28, sport: str | None = None) -> Overview:
        from ..coach.overview import build_overview

        return build_overview(self, self.athlete, self.tz, days=days, sport=sport)

    # -- sync write side ----------------------------------------------------

    def load_sync_state(self) -> SyncState:
        """Read the sync working set: source index plus activity/metrics stores."""
        index_frame = parquet.read_parquet(self.cache_dir / schemas.SOURCE_INDEX_FILE)
        activities_frame = parquet.read_parquet(self.cache_dir / schemas.ACTIVITIES_FILE)
        metrics_frame = parquet.read_parquet(self.cache_dir / schemas.ACTIVITY_METRICS_FILE)
        return SyncState(
            index_rows=[] if index_frame.is_empty() else index_frame.to_dicts(),
            activities=_store_dict(activities_frame),
            metrics=_store_dict(metrics_frame),
        )

    def save_sync_snapshot(
        self, state: SyncState, *, last_sync: datetime, analysis_version: str
    ) -> None:
        """Atomically rewrite the whole cache from the sync working set.

        Prunes store rows without an active "ok" index row (removed or
        re-specified files, historical orphans) so activities, aggregates
        and the index stay consistent, rebuilds daily/weekly aggregates
        from the surviving rows and stamps ``metadata.json``.
        """
        ok_ids = {
            str(row["activity_id"])
            for row in state.index_rows
            if row.get("status") == schemas.STATUS_OK and row.get("activity_id") is not None
        }
        activities = {key: row for key, row in state.activities.items() if key in ok_ids}
        metrics = {key: row for key, row in state.metrics.items() if key in ok_ids}
        analyzed = [
            _analyzed_from_rows(
                activities[str(row["activity_id"])], metrics[str(row["activity_id"])]
            )
            for row in state.index_rows
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

    # -- helpers -----------------------------------------------------------

    def _activities_with_load(self) -> pl.DataFrame:
        activities = parquet.read_parquet(self.cache_dir / schemas.ACTIVITIES_FILE)
        if activities.is_empty():
            return activities
        metrics = parquet.read_parquet(self.cache_dir / schemas.ACTIVITY_METRICS_FILE)
        if metrics.is_empty():
            return activities.with_columns(pl.lit(None, dtype=pl.Float64).alias("load_value"))
        load = metrics.select(["activity_id", "load_value"])
        return activities.join(load, on="activity_id", how="left")

    def _with_local_date(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(
            pl.col("start_time").dt.convert_time_zone(str(self.tz)).dt.date().alias("local_date")
        )

    def _filter(
        self, frame: pl.DataFrame, start_date: date | None, end_date: date | None, sport: str | None
    ) -> pl.DataFrame:
        if start_date is not None:
            frame = frame.filter(pl.col("local_date") >= start_date)
        if end_date is not None:
            frame = frame.filter(pl.col("local_date") <= end_date)
        if sport is not None:
            frame = frame.filter(pl.col("sport").str.to_lowercase() == sport.lower())
        return frame


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


def _analyzed_from_rows(
    activity_row: dict[str, object], metrics_row: dict[str, object]
) -> AnalyzedActivity:
    """Hydrate an ``AnalyzedActivity`` from stored activity + metrics rows."""
    start_time = activity_row["start_time"]
    if not isinstance(start_time, datetime):
        msg = "activity start_time has unexpected type"
        raise CacheError(msg)
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


def _daily_row(row: dict[str, object]) -> DailySummaryRow:
    return DailySummaryRow(
        date=str(row["date"]),
        sport=str(row["sport"]),
        activity_count=_int(row["activity_count"]),
        duration_s=_float(row.get("duration_s")),
        distance_m=_float(row.get("distance_m")),
        elevation_gain_m=_float(row.get("elevation_gain_m")),
        hr_zone_1_s=_float(row.get("hr_zone_1_s")),
        hr_zone_2_s=_float(row.get("hr_zone_2_s")),
        hr_zone_3_s=_float(row.get("hr_zone_3_s")),
        hr_zone_4_s=_float(row.get("hr_zone_4_s")),
        hr_zone_5_s=_float(row.get("hr_zone_5_s")),
        load_value=_float(row.get("load_value")),
        hard_activity_count=_int(row["hard_activity_count"]),
        longest_activity_s=_float(row.get("longest_activity_s")),
        longest_activity_id=_optional_str(row.get("longest_activity_id")),
    )


def _weekly_row(row: dict[str, object]) -> WeeklySummaryRow:
    return WeeklySummaryRow(
        iso_year=_int(row["iso_year"]),
        iso_week=_int(row["iso_week"]),
        week_start=str(row["week_start"]),
        sport=str(row["sport"]),
        activity_count=_int(row["activity_count"]),
        duration_s=_float(row.get("duration_s")),
        distance_m=_float(row.get("distance_m")),
        elevation_gain_m=_float(row.get("elevation_gain_m")),
        hr_zone_1_s=_float(row.get("hr_zone_1_s")),
        hr_zone_2_s=_float(row.get("hr_zone_2_s")),
        hr_zone_3_s=_float(row.get("hr_zone_3_s")),
        hr_zone_4_s=_float(row.get("hr_zone_4_s")),
        hr_zone_5_s=_float(row.get("hr_zone_5_s")),
        load_value=_float(row.get("load_value")),
        hard_activity_count=_int(row["hard_activity_count"]),
        longest_activity_s=_float(row.get("longest_activity_s")),
        longest_activity_id=_optional_str(row.get("longest_activity_id")),
        load_value_prev=_optional_float(row.get("load_value_prev")),
        duration_s_prev=_optional_float(row.get("duration_s_prev")),
        distance_m_prev=_optional_float(row.get("distance_m_prev")),
    )


def _aggregate_daily(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate per-(date, sport) rows into per-date totals."""
    by_date: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_date.setdefault(str(row["date"]), []).append(row)
    aggregated: list[dict[str, object]] = []
    for day, group in by_date.items():
        longest = max(group, key=lambda row: _float(row.get("longest_activity_s")))
        aggregated.append(
            {
                "date": day,
                "sport": "all",
                "activity_count": sum(_int(row["activity_count"]) for row in group),
                "duration_s": sum(_float(row.get("duration_s")) for row in group),
                "distance_m": sum(_float(row.get("distance_m")) for row in group),
                "elevation_gain_m": sum(_float(row.get("elevation_gain_m")) for row in group),
                "hr_zone_1_s": sum(_float(row.get("hr_zone_1_s")) for row in group),
                "hr_zone_2_s": sum(_float(row.get("hr_zone_2_s")) for row in group),
                "hr_zone_3_s": sum(_float(row.get("hr_zone_3_s")) for row in group),
                "hr_zone_4_s": sum(_float(row.get("hr_zone_4_s")) for row in group),
                "hr_zone_5_s": sum(_float(row.get("hr_zone_5_s")) for row in group),
                "load_value": sum(_float(row.get("load_value")) for row in group),
                "hard_activity_count": sum(_int(row["hard_activity_count"]) for row in group),
                "longest_activity_s": _float(longest.get("longest_activity_s")),
                "longest_activity_id": longest.get("longest_activity_id"),
            }
        )
    return aggregated


def _aggregate_weekly(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate per-(week, sport) rows into per-week totals."""
    by_week: dict[tuple[int, int], list[dict[str, object]]] = {}
    for row in rows:
        by_week.setdefault((_int(row["iso_year"]), _int(row["iso_week"])), []).append(row)
    aggregated: list[dict[str, object]] = []
    for (_year, _week), group in sorted(by_week.items()):
        longest = max(group, key=lambda row: _float(row.get("longest_activity_s")))
        aggregated.append(
            {
                "iso_year": group[0]["iso_year"],
                "iso_week": group[0]["iso_week"],
                "week_start": min(str(row["week_start"]) for row in group),
                "sport": "all",
                "activity_count": sum(_int(row["activity_count"]) for row in group),
                "duration_s": sum(_float(row.get("duration_s")) for row in group),
                "distance_m": sum(_float(row.get("distance_m")) for row in group),
                "elevation_gain_m": sum(_float(row.get("elevation_gain_m")) for row in group),
                "hr_zone_1_s": sum(_float(row.get("hr_zone_1_s")) for row in group),
                "hr_zone_2_s": sum(_float(row.get("hr_zone_2_s")) for row in group),
                "hr_zone_3_s": sum(_float(row.get("hr_zone_3_s")) for row in group),
                "hr_zone_4_s": sum(_float(row.get("hr_zone_4_s")) for row in group),
                "hr_zone_5_s": sum(_float(row.get("hr_zone_5_s")) for row in group),
                "load_value": sum(_float(row.get("load_value")) for row in group),
                "hard_activity_count": sum(_int(row["hard_activity_count"]) for row in group),
                "longest_activity_s": _float(longest.get("longest_activity_s")),
                "longest_activity_id": longest.get("longest_activity_id"),
                "load_value_prev": _sum_optional(row.get("load_value_prev") for row in group),
                "duration_s_prev": _sum_optional(row.get("duration_s_prev") for row in group),
                "distance_m_prev": _sum_optional(row.get("distance_m_prev") for row in group),
            }
        )
    return aggregated


def _sum_optional(values: Iterable[object]) -> float | None:
    present = [
        float(value)
        for value in values
        if isinstance(value, int | float) and not isinstance(value, bool)
    ]
    return sum(present) if present else None


def _date_in_range(value: date, start_date: date | None, end_date: date | None) -> bool:
    if start_date is not None and value < start_date:
        return False
    return not (end_date is not None and value > end_date)


def _optional_str(value: object) -> str | None:
    return optional_str(value)


def _optional_float(value: object) -> float | None:
    return optional_float(value)


def _float(value: object) -> float:
    return _optional_float(value) or 0.0


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
