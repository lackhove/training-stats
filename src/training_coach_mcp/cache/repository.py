"""Read-side repository abstracting Parquet access from MCP tools."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo

import polars as pl

from ..analytics.daily import DailySummaryRow
from ..analytics.fitness import (
    compute_days_since_60k,
    compute_rolling_distance,
    compute_training_form,
)
from ..analytics.trends import TrendResult, compute_trend
from ..analytics.weekly import WeeklySummaryRow
from ..config import AthleteConfig
from ..errors import CacheError
from ..models import (
    ActivityDetails,
    ActivitySummary,
    CoachContext,
    DataQuality,
    TrainingFormResult,
)
from . import parquet, schemas

logger = logging.getLogger(__name__)


class TrainingRepository(Protocol):
    """Read interface used by the MCP server."""

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

    def training_trends(
        self, metric: str, days: int = 30, sport: str | None = None
    ) -> TrendResult: ...

    def training_form(self, days: int = 42, sport: str | None = None) -> TrainingFormResult: ...

    def data_quality(self) -> DataQuality: ...

    def coach_context(self, days: int = 28, sport: str | None = None) -> CoachContext: ...


class ParquetTrainingRepository:
    """Parquet-backed repository. All filtering/aggregation lives here."""

    def __init__(self, cache_dir: Path, timezone: ZoneInfo, athlete: AthleteConfig) -> None:
        self.cache_dir = cache_dir
        self.tz = timezone
        self.athlete = athlete

    # -- activities -------------------------------------------------------

    def list_activities(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        sport: str | None = None,
        limit: int = 50,
    ) -> list[ActivitySummary]:
        frame = self._activities_with_load()
        if frame.is_empty():
            return []
        frame = self._with_local_date(frame)
        frame = self._filter(frame, start_date, end_date, sport)
        frame = frame.sort("start_time", descending=True).head(limit)
        return [_summary_row(row, self.tz) for row in frame.to_dicts()]

    def get_activity(self, activity_id: str) -> ActivityDetails | None:
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

    def training_trends(self, metric: str, days: int = 30, sport: str | None = None) -> TrendResult:
        end_date = datetime.now(self.tz).date()
        start_date = end_date - timedelta(days=max(days, 1) - 1)
        rows = self.daily_summary(start_date=start_date, end_date=end_date, sport=sport)
        return compute_trend(rows, metric, sport)

    def training_form(self, days: int = 42, sport: str | None = None) -> TrainingFormResult:
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

            total_load = sum(_float(r.get("load_value")) for r in m_rows if r.get("load_value"))
            high_conf_load = sum(
                _float(r.get("load_value"))
                for r in m_rows
                if r.get("load_value") and r.get("load_confidence") == "high"
            )
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
        metrics = parquet.read_parquet(self.cache_dir / schemas.ACTIVITY_METRICS_FILE)
        index = parquet.read_parquet(self.cache_dir / schemas.SOURCE_INDEX_FILE)
        activity_count = metrics.height
        mean_coverage = _mean(metrics, "hr_coverage_ratio") if not metrics.is_empty() else None
        mean_completeness = _mean(metrics, "data_completeness") if not metrics.is_empty() else None
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
                    _flags(row.get("repair_flags_json"))
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

    def coach_context(self, days: int = 28, sport: str | None = None) -> CoachContext:
        from ..coach.context import build_coach_context

        return build_coach_context(self, self.athlete, self.tz, days=days, sport=sport)

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
        sub_sport=_optional_str(row.get("sub_sport")),
        duration_s=_optional_float(row.get("duration_s")),
        distance_m=_optional_float(row.get("distance_m")),
        elevation_gain_m=_optional_float(row.get("elevation_gain_m")),
        avg_hr=_optional_float(row.get("avg_hr")),
        max_hr=_optional_float(row.get("max_hr")),
        calories=_optional_float(row.get("calories")),
        load_value=_optional_float(row.get("load_value")),
        moving_time_s=_optional_float(row.get("moving_time_s")),
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
        avg_speed_mps=_optional_float(row.get("avg_speed_mps")),
        max_speed_mps=_optional_float(row.get("max_speed_mps")),
        calories=summary.calories,
        hr_zone_1_s=_float(row.get("hr_zone_1_s")),
        hr_zone_2_s=_float(row.get("hr_zone_2_s")),
        hr_zone_3_s=_float(row.get("hr_zone_3_s")),
        hr_zone_4_s=_float(row.get("hr_zone_4_s")),
        hr_zone_5_s=_float(row.get("hr_zone_5_s")),
        hr_coverage_ratio=_float(row.get("hr_coverage_ratio")),
        hr_histogram_s=_histogram(row.get("hr_histogram_json")),
        hr_10min_averages=_buckets(row.get("hr_10min_json")),
        hr_drift_bpm_per_hour=_optional_float(row.get("hr_drift_bpm_per_hour")),
        load_value=summary.load_value,
        load_method=_optional_str(row.get("load_method")),
        load_confidence=_optional_str(row.get("load_confidence")),
        data_completeness=_float(row.get("data_completeness")),
        analysis_version=str(row.get(schemas.ANALYSIS_VERSION, "")),
        source_name=str(row.get("source_name", "")),
        source_path=str(row.get("source_path", "")),
        ef_first_half=_optional_float(row.get("ef_first_half")),
        ef_second_half=_optional_float(row.get("ef_second_half")),
        aerobic_decoupling_pct=_optional_float(row.get("aerobic_decoupling_pct")),
        decoupling_confidence=_optional_str(row.get("decoupling_confidence")),
        local_start_time=summary.local_start_time,
        paused_time_s=_optional_float(row.get("paused_time_s")),
        repair_flags=_flags(row.get("repair_flags_json")),
    )


def _flags(value: object) -> tuple[str, ...]:
    """Parse the serialized repair flags; invalid input yields ``()``."""
    if not isinstance(value, str) or not value:
        return ()
    try:
        raw = json.loads(value)
    except json.JSONDecodeError:
        return ()
    if not isinstance(raw, list):
        return ()
    return tuple(str(flag) for flag in raw)


def _histogram(value: object) -> dict[str, float]:
    """Parse the serialized 5-bpm histogram; invalid input yields ``{}``."""
    if not isinstance(value, str) or not value:
        return {}
    try:
        raw = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(key): _float(seconds) for key, seconds in raw.items()}


def _buckets(value: object) -> tuple[dict[str, object], ...]:
    """Parse the serialized 10-minute averages; invalid input yields ``()``."""
    if not isinstance(value, str) or not value:
        return ()
    try:
        raw = json.loads(value)
    except json.JSONDecodeError:
        return ()
    if not isinstance(raw, list):
        return ()
    return tuple(dict(entry) for entry in raw if isinstance(entry, dict))


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
    return str(value) if value is not None else None


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


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
