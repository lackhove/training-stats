"""Service layer exposing the training analytics domain operations.

Methods are thin, validating adapters over the repository and sync service;
all business logic lives in the analytics, ingest, cache and coach layers.
They return domain models — serialization to HTTP/CLI JSON happens only at
the edges via :mod:`training_stats.wire`. The FastAPI application
(:mod:`training_stats.api`) and the CLI (:mod:`training_stats.cli`) are both
wired over this class.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, date, datetime

from .analytics.weekly import WeeklySummaryRow
from .cache import schemas
from .cache.parquet import read_json
from .cache.repository import ParquetTrainingRepository, TrainingRepository
from .cache.sync import SyncService
from .config import AppConfig, LocalSourceConfig, WebDAVSourceConfig
from .errors import TrainingStatsError
from .fit.parser import FitDecodeParser, FitParser
from .ingest import FileProcessor
from .models import (
    MAX_DAYS,
    MAX_LIST_LIMIT,
    MAX_WEEKS,
    ActivityDetails,
    ActivitySummary,
    Overview,
    SyncResult,
    ThrottledSync,
)
from .sources.base import ActivitySource
from .sources.local import LocalDirectorySource
from .sources.webdav import WebDAVSource

logger = logging.getLogger(__name__)


class TrainingStatsService:
    """Domain-facing service; testable without HTTP or CLI wiring."""

    def __init__(
        self,
        config: AppConfig,
        sources: list[ActivitySource],
        parser: FitParser,
        sync_service: SyncService,
        repository: TrainingRepository,
    ) -> None:
        self.config = config
        self.sources = sources
        self.parser = parser
        self.sync_service = sync_service
        self.repository = repository
        self._last_sync_monotonic: float | None = None

    # -- service operations (directly testable) ----------------------------

    def sync_if_due(self) -> SyncResult | ThrottledSync:
        """Synchronize unless the last successful sync is recent.

        Two-layer throttle: an in-memory monotonic stamp covers repeated
        calls in one process (e.g. REST request bursts); when absent, the
        ``last_sync`` timestamp from ``cache/metadata.json`` covers fresh
        CLI processes. Stamps are only recorded when the cache was actually
        written, so failed syncs never block retries. A
        ``cache.sync_min_interval_seconds`` of ``0`` always synchronizes.
        """
        interval = self.config.cache.sync_min_interval_seconds
        now = time.monotonic()
        since = self._seconds_since_last_sync(now)
        if interval > 0 and since is not None and since < interval:
            return ThrottledSync(seconds_since_last_sync=round(max(since, 0.0), 1))
        result = self.sync_activities()
        if result.cache_updated_at is not None:
            self._last_sync_monotonic = now
        return result

    def _seconds_since_last_sync(self, now: float) -> float | None:
        if self._last_sync_monotonic is not None:
            return now - self._last_sync_monotonic
        metadata = read_json(self.config.cache.directory / schemas.METADATA_FILE)
        last_sync = metadata.get("last_sync") if metadata else None
        if not isinstance(last_sync, str):
            return None
        try:
            stamp = datetime.fromisoformat(last_sync)
        except ValueError:
            return None
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=UTC)
        return (datetime.now(UTC) - stamp).total_seconds()

    def sync_activities(self) -> SyncResult:
        """Synchronize new or changed FIT files unconditionally; never raises.

        Source-level failures (e.g. an unreachable WebDAV source) are reported
        in ``source_errors``; unexpected sync failures are logged as a warning
        so callers can keep serving the existing cache. Runs at server startup.
        """
        try:
            result = self.sync_service.sync()
        except Exception:
            logger.warning("activity sync failed; serving cached data", exc_info=True)
            return SyncResult(
                files_discovered=0,
                files_processed=0,
                files_skipped=0,
                files_failed=0,
            )
        if result.source_errors:
            logger.warning("activity sync reported source errors: %s", list(result.source_errors))
        if result.files_processed or result.files_failed:
            logger.info(
                "activity sync: processed=%d skipped=%d failed=%d",
                result.files_processed,
                result.files_skipped,
                result.files_failed,
            )
        return result

    def list_activities(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        sport: str | None = None,
        limit: int = 50,
        sync_first: bool = True,
    ) -> list[ActivitySummary]:
        """List compact activity summaries, newest first."""
        if sync_first:
            self.sync_if_due()
        _validate_int_range("limit", limit, 1, MAX_LIST_LIMIT)
        return self.repository.list_activities(
            start_date=start_date, end_date=end_date, sport=sport, limit=limit
        )

    def get_activity(self, activity_id: str) -> ActivityDetails:
        """Get one activity with metrics, zones, load and data quality."""
        details = self.repository.get_activity(activity_id)
        if details is None:
            msg = f"activity not found: {activity_id!r}"
            raise LookupError(msg)
        return details

    def weekly_summary(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        weeks: int = 8,
        sport: str | None = None,
        sync_first: bool = True,
    ) -> list[WeeklySummaryRow]:
        """Weekly training totals with previous-week comparisons."""
        if sync_first:
            self.sync_if_due()
        _validate_int_range("weeks", weeks, 1, MAX_WEEKS)
        return self.repository.weekly_summary(
            start_date=start_date, end_date=end_date, weeks=weeks, sport=sport
        )

    def overview(
        self, days: int = 28, sport: str | None = None, sync_first: bool = True
    ) -> Overview:
        """Compact overview: profile, daily series, weekly summaries and form.

        The daily series carries per-(date, sport) totals for every metric,
        so any metric trend is a filter and projection over it.
        """
        if sync_first:
            self.sync_if_due()
        _validate_int_range("days", days, 1, MAX_DAYS)
        return self.repository.overview(days=days, sport=sport)


def create_service(config: AppConfig) -> TrainingStatsService:
    """Instantiate sources, parser, sync service and repository from config."""
    from .config import resolve_webdav_credentials

    credentials = resolve_webdav_credentials(config)
    sources: list[ActivitySource] = []
    for source_config in config.sources:
        if isinstance(source_config, LocalSourceConfig):
            sources.append(
                LocalDirectorySource(
                    name=source_config.name,
                    path=source_config.path,
                    recursive=source_config.recursive,
                    follow_symlinks=source_config.follow_symlinks,
                )
            )
        elif isinstance(source_config, WebDAVSourceConfig):
            sources.append(
                WebDAVSource(
                    source_config,
                    credentials[source_config.name],
                )
            )
        else:  # pragma: no cover - exhaustive by construction
            msg = f"unsupported source type: {type(source_config).__name__}"
            raise TrainingStatsError(msg)

    parser: FitParser = FitDecodeParser(quality=config.quality)
    processor = FileProcessor(
        parser=parser,
        athlete=config.athlete,
        quality=config.quality,
        analysis_version=schemas.ANALYSIS_VERSION_VALUE,
    )
    sync_service = SyncService(
        sources=sources,
        processor=processor,
        cache_dir=config.cache.directory,
        timezone=config.application.zoneinfo,
        analysis_version=schemas.ANALYSIS_VERSION_VALUE,
        lock_timeout_seconds=config.cache.lock_timeout_seconds,
    )
    repository = ParquetTrainingRepository(
        cache_dir=config.cache.directory,
        timezone=config.application.zoneinfo,
        athlete=config.athlete,
        lock_timeout_seconds=config.cache.lock_timeout_seconds,
    )
    return TrainingStatsService(
        config=config,
        sources=sources,
        parser=parser,
        sync_service=sync_service,
        repository=repository,
    )


def _validate_int_range(name: str, value: int, low: int, high: int) -> None:
    if not low <= value <= high:
        msg = f"{name} must be between {low} and {high}, got {value}"
        raise ValueError(msg)
