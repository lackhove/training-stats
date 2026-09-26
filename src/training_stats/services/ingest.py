"""Per-file processing use case: parse, analyze, and return typed records."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from ..domain.analytics.activities import AnalyzedActivity
from ..domain.analytics.segments import analyze_segments
from ..domain.errors import FitParseError
from ..domain.fit.repair import repair_timeline
from ..domain.models import STATUS_DUPLICATE, STATUS_FAILED, STATUS_OK, SourceFile
from ..domain.settings import AthleteConfig, QualityConfig
from .ports import FitParser
from .state import ActivityRecord, SourceIndexEntry, SourceStatus

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Outcome:
    """Typed result of processing one FIT file."""

    index_rows: list[SourceIndexEntry]
    activities: list[ActivityRecord]
    duplicates: int
    repaired: bool
    discarded_segments: int


class FileProcessor:
    """Parse one FIT file and compute its domain activity records."""

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
        """Analyze ``content``; update ``existing_ids`` with accepted ids."""
        try:
            fit_data = self.parser.parse(content, source_file)
        except FitParseError as exc:
            logger.error("cannot parse %s/%s: %s", source_file.source_name, source_file.path, exc)
            return Outcome(
                index_rows=[
                    _source_entry(
                        source_file=source_file,
                        source_hash=file_hash,
                        processed_at=processed_at,
                        analysis_version=self.analysis_version,
                        status=STATUS_FAILED,
                        error_message=str(exc),
                    )
                ],
                activities=[],
                duplicates=0,
                repaired=False,
                discarded_segments=0,
            )

        raw_records = fit_data.frame.filter(fit_data.frame["message_type"] == "record").select(
            "timestamp"
        )
        repair_flags = (
            ["non_monotonic_timestamps"]
            if not raw_records.is_empty() and not raw_records["timestamp"].is_sorted()
            else []
        )
        repaired_segments = repair_timeline(
            fit_data,
            gap_threshold_s=self.quality.activity_gap_threshold_s,
            distance_gap_threshold_m=self.quality.activity_distance_gap_threshold_m,
        )
        analysis = analyze_segments(
            repaired_segments,
            source_path=source_file.path,
            athlete=self.athlete,
            analysis_version=self.analysis_version,
            quality=self.quality,
        )
        if analysis.discarded_segment_count:
            repair_flags.append("segments_discarded")

        index_rows: list[SourceIndexEntry] = []
        activities: list[ActivityRecord] = []
        duplicates = 0
        for analyzed in analysis.activities:
            activity = analyzed.activity
            if activity.activity_id in existing_ids:
                index_rows.append(
                    _source_entry(
                        source_file=source_file,
                        source_hash=file_hash,
                        processed_at=processed_at,
                        analysis_version=self.analysis_version,
                        activity_id=activity.activity_id,
                        segment_index=activity.segment_index,
                        segment_count=activity.segment_count,
                        file_repair_flags=tuple(repair_flags),
                        discarded_segment_count=analysis.discarded_segment_count,
                        status=STATUS_DUPLICATE,
                        error_message="duplicate activity id",
                    )
                )
                duplicates += 1
                logger.info(
                    "duplicate %s/%s: activity %s already cached",
                    source_file.source_name,
                    source_file.path,
                    activity.activity_id,
                )
                continue

            existing_ids.add(activity.activity_id)
            index_rows.append(
                _source_entry(
                    source_file=source_file,
                    source_hash=file_hash,
                    processed_at=processed_at,
                    analysis_version=self.analysis_version,
                    activity_id=activity.activity_id,
                    segment_index=activity.segment_index,
                    segment_count=activity.segment_count,
                    file_repair_flags=tuple(repair_flags),
                    discarded_segment_count=analysis.discarded_segment_count,
                    status=STATUS_OK,
                )
            )
            activities.append(
                _activity_record(analyzed, source_file.source_name, self.analysis_version)
            )

        if not index_rows:
            index_rows.append(
                _source_entry(
                    source_file=source_file,
                    source_hash=file_hash,
                    processed_at=processed_at,
                    analysis_version=self.analysis_version,
                    status=STATUS_OK,
                    file_repair_flags=tuple(repair_flags),
                    discarded_segment_count=analysis.discarded_segment_count,
                )
            )

        return Outcome(
            index_rows=index_rows,
            activities=activities,
            duplicates=duplicates,
            repaired=bool(repair_flags),
            discarded_segments=analysis.discarded_segment_count,
        )


def _source_entry(
    *,
    source_file: SourceFile,
    source_hash: str | None,
    processed_at: datetime,
    analysis_version: str,
    status: SourceStatus,
    error_message: str | None = None,
    activity_id: str | None = None,
    segment_index: int = 0,
    segment_count: int = 0,
    file_repair_flags: tuple[str, ...] = (),
    discarded_segment_count: int = 0,
) -> SourceIndexEntry:
    return SourceIndexEntry(
        source_name=source_file.source_name,
        source_path=source_file.path,
        source_hash=source_hash,
        file_size=source_file.size,
        modified_at=source_file.modified_at,
        activity_id=activity_id,
        segment_index=segment_index,
        segment_count=segment_count,
        file_repair_flags=file_repair_flags,
        discarded_segment_count=discarded_segment_count,
        processed_at=processed_at,
        source_analysis_version=analysis_version,
        status=status,
        error_message=error_message,
    )


def _activity_record(
    item: AnalyzedActivity, source_name: str, analysis_version: str
) -> ActivityRecord:
    activity = item.activity
    return ActivityRecord(
        activity_id=activity.activity_id,
        start_time=_ensure_utc(activity.start_time),
        sport=activity.sport,
        sub_sport=activity.sub_sport,
        segment_index=activity.segment_index,
        segment_count=activity.segment_count,
        repair_flags=activity.repair_flags,
        source_name=source_name,
        source_path=activity.source_path,
        analysis_version=analysis_version,
        metrics=item.metrics,
    )


def _ensure_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
