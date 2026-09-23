"""Service layer exposing the training analytics domain operations.

Methods are thin, validating adapters that orchestrate the repository and
the pure analytics functions: the repository provides lazy cache scans,
analytics provides all filtering, aggregation, activity queries and
training-form mathematics. They return domain models — serialization to
HTTP/CLI JSON happens only at the edges via adapter wire models. HTTP and
CLI adapters are both wired over this class.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, date, datetime, timedelta

from ..domain.analytics import activities as activity_queries
from ..domain.analytics.daily import select_daily, summarize_daily
from ..domain.analytics.fitness import training_form_from
from ..domain.analytics.overview import build_overview
from ..domain.analytics.weekly import WeeklySummaryRow, daily_to_weekly, select_weekly
from ..domain.models import ActivityDetails, ActivitySummary, Overview
from .limits import MAX_DAYS, MAX_LIST_LIMIT, MAX_WEEKS
from .ports import TrainingRepository
from .results import SyncResult, ThrottledSync
from .settings import ServiceSettings
from .sync import SyncService

logger = logging.getLogger(__name__)

OVERVIEW_WEEKS = 4
OVERVIEW_RECENT_LIMIT = 5


class TrainingStatsService:
    """Domain-facing service; testable without HTTP or CLI wiring."""

    def __init__(
        self,
        settings: ServiceSettings,
        sync_service: SyncService,
        repository: TrainingRepository,
    ) -> None:
        self.settings = settings
        self.sources = sync_service.sources
        self.sync_service = sync_service
        self.repository = repository
        self.tz = settings.timezone
        self._last_sync_monotonic: float | None = None

    # -- service operations (directly testable) ----------------------------

    def sync_if_due(self) -> SyncResult | ThrottledSync:
        """Synchronize unless the last successful sync is recent.

        Two-layer throttle: an in-memory monotonic stamp covers repeated
        calls in one process (e.g. REST request bursts); when absent, the
        ``last_sync`` timestamp from ``cache/metadata.json`` covers fresh
        CLI processes. Stamps are only recorded when the cache was actually
        written, so failed syncs never block retries. A
        A zero minimum interval always synchronizes.
        """
        interval = self.settings.sync_min_interval_seconds
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
        stamp = self.repository.last_sync_stamp()
        if stamp is None:
            return None
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
        return activity_queries.activity_summaries(
            self.repository.activities_scan(),
            self.tz,
            start_date=start_date,
            end_date=end_date,
            sport=sport,
            limit=limit,
        )

    def get_activity(self, activity_id: str) -> ActivityDetails:
        """Get one activity with metrics, zones, load and data quality."""
        details = activity_queries.activity_details(
            self.repository.activities_scan(), activity_id, self.tz
        )
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
        daily = summarize_daily(self.repository.activities_scan(), self.tz)
        return select_weekly(
            daily_to_weekly(daily),
            start_date=start_date,
            end_date=end_date,
            weeks=weeks,
            sport=sport,
        )

    def overview(
        self, days: int = 28, sport: str | None = None, sync_first: bool = True
    ) -> Overview:
        """Compact overview: profile, daily series, weekly summaries and form.

        The daily series carries per-(date, sport) totals for every metric,
        so any metric trend is a filter and projection over it. All pieces
        are computed here; :func:`~training_stats.domain.analytics.overview.
        build_overview` is a pure assembler over them.
        """
        if sync_first:
            self.sync_if_due()
        _validate_int_range("days", days, 1, MAX_DAYS)
        today = datetime.now(self.tz).date()
        period_start = today - timedelta(days=max(days, 1) - 1)

        activities = self.repository.activities_scan()
        daily_all = summarize_daily(activities, self.tz)
        recent = activity_queries.activity_summaries(
            activities,
            self.tz,
            start_date=period_start,
            end_date=today,
            sport=sport,
            limit=OVERVIEW_RECENT_LIMIT,
        )
        latest_rows = activity_queries.activity_summaries(activities, self.tz, sport=sport, limit=1)
        latest_details = (
            activity_queries.activity_details(activities, latest_rows[0].activity_id, self.tz)
            if latest_rows
            else None
        )
        return build_overview(
            athlete=self.settings.athlete,
            period_start=period_start,
            period_end=today,
            sport=sport,
            recent=recent,
            weekly=select_weekly(daily_to_weekly(daily_all), weeks=OVERVIEW_WEEKS, sport=sport),
            daily_rows=select_daily(
                daily_all, start_date=period_start, end_date=today, sport=sport
            ),
            form_result=training_form_from(
                activities, tz=self.tz, days=days, sport=sport, today=today
            ),
            quality=activity_queries.data_quality(activities, self.repository.source_index_scan()),
            latest_summary=latest_rows[0] if latest_rows else None,
            latest_details=latest_details,
        )


def _validate_int_range(name: str, value: int, low: int, high: int) -> None:
    if not low <= value <= high:
        msg = f"{name} must be between {low} and {high}, got {value}"
        raise ValueError(msg)
