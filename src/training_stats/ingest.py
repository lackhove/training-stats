"""Domain ingestion: parse FIT content and assemble Parquet cache rows.

``FileProcessor`` is the domain half of the sync pipeline: ``SyncService``
owns discovery, change detection and cache I/O, and delegates each changed
file to the processor injected via its constructor. Together with
``analytics/``, this module forms the domain layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from .analytics.activities import AnalyzedActivity, compute_activity_metrics
from .cache import schemas
from .config import AthleteConfig, QualityConfig
from .converters import buckets_to_json, flags_to_json, histogram_to_json
from .errors import FitParseError
from .fit.parser import FitParser
from .models import SourceFile

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Outcome:
    """Result of processing one file during a sync."""

    index_rows: list[dict[str, object]]
    ok_activity_rows: list[dict[str, object]]
    duplicates: int
    repaired: bool
    discarded_segments: int


def _empty_outcome(index_rows: list[dict[str, object]]) -> Outcome:
    return Outcome(
        index_rows=index_rows,
        ok_activity_rows=[],
        duplicates=0,
        repaired=False,
        discarded_segments=0,
    )


class FileProcessor:
    """Parse one FIT file's content into index, activity and metrics rows."""

    def __init__(
        self,
        *,
        parser: FitParser,
        athlete: AthleteConfig,
        quality: QualityConfig,
        analysis_version: str,
    ) -> None:
        self.parser = parser
        self.athlete = athlete
        self.quality = quality
        self.analysis_version = analysis_version

    def process(
        self,
        source_file: SourceFile,
        content: bytes,
        file_hash: str,
        existing_ids: set[str],
        processed_at: datetime,
    ) -> Outcome:
        """Analyze ``content``; mutates ``existing_ids`` with accepted ids."""
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
            "status": schemas.STATUS_FAILED,
            "error_message": None,
        }
        try:
            parsed_file = self.parser.parse(content, source_file)
        except FitParseError as exc:
            logger.error("cannot parse %s/%s: %s", source_file.source_name, source_file.path, exc)
            base_row["error_message"] = str(exc)
            return _empty_outcome([base_row])

        file_flags_json = flags_to_json(parsed_file.repair_flags)
        index_rows: list[dict[str, object]] = []
        ok_activity_rows: list[dict[str, object]] = []
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
                row["status"] = schemas.STATUS_DUPLICATE
                row["error_message"] = "duplicate activity id"
                duplicates += 1
                logger.info(
                    "duplicate %s/%s: activity %s already cached",
                    source_file.source_name,
                    source_file.path,
                    activity.activity_id,
                )
            else:
                row["status"] = schemas.STATUS_OK
                existing_ids.add(activity.activity_id)
                ok_activity_rows.append(_activity_row(analyzed, source_file.source_name))
            index_rows.append(row)
        return Outcome(
            index_rows=index_rows,
            ok_activity_rows=ok_activity_rows,
            duplicates=duplicates,
            repaired=bool(parsed_file.repair_flags),
            discarded_segments=parsed_file.discarded_segment_count,
        )


def _activity_row(item: AnalyzedActivity, source_name: str) -> dict[str, object]:
    """Build one merged activity row: session facts plus derived metrics."""
    activity = item.activity
    metric_values = item.metrics
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
        "repair_flags_json": flags_to_json(activity.repair_flags),
        "source_name": source_name,
        "source_path": activity.source_path,
        "hr_zone_1_s": metric_values.hr_zone_1_s,
        "hr_zone_2_s": metric_values.hr_zone_2_s,
        "hr_zone_3_s": metric_values.hr_zone_3_s,
        "hr_zone_4_s": metric_values.hr_zone_4_s,
        "hr_zone_5_s": metric_values.hr_zone_5_s,
        "hr_coverage_ratio": metric_values.hr_coverage_ratio,
        "hr_histogram_json": histogram_to_json(metric_values.hr_histogram_s),
        "hr_10min_json": buckets_to_json(metric_values.hr_10min_averages),
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


def _ensure_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
