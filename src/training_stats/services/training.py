"""Service orchestration for sync and metric-report queries."""

from __future__ import annotations

import logging
import time
from datetime import UTC, date, datetime

from ..domain.analytics import (
    activity_list_report,
    activity_report,
    overview_report,
    weekly_summary,
)
from ..domain.metrics import MetricReport
from ..services.limits import MAX_DAYS, MAX_LIST_LIMIT, MAX_WEEKS
from .ports import TrainingRepository
from .results import SyncResult, ThrottledSync
from .settings import ServiceSettings
from .state import SyncState
from .sync import SyncService

logger = logging.getLogger(__name__)


class TrainingStatsService:
    """Application service: synchronize, validate, load records, call analytics."""

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

    def sync_if_due(self) -> SyncResult | ThrottledSync:
        """Synchronize unless the last successful sync is recent."""
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
        """Synchronize unconditionally; failures leave the existing cache usable."""
        try:
            result = self.sync_service.sync()
        except Exception:
            logger.warning("activity sync failed; serving cached data", exc_info=True)
            return SyncResult(
                files_discovered=0, files_processed=0, files_skipped=0, files_failed=0
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
    ) -> MetricReport:
        """Return trip metrics for selected activities, newest first."""
        if sync_first:
            self.sync_if_due()
        _validate_int_range("limit", limit, 1, MAX_LIST_LIMIT)
        state = self.repository.load_sync_state()
        return activity_list_report(
            tuple(state.activities.values()),
            self.tz,
            start_date=start_date,
            end_date=end_date,
            sport=sport,
            limit=limit,
        )

    def get_activity(self, activity_id: str) -> MetricReport:
        """Return one activity's cached metrics."""
        report = activity_report(
            tuple(self.repository.load_sync_state().activities.values()), activity_id
        )
        if report is None:
            msg = f"activity not found: {activity_id!r}"
            raise LookupError(msg)
        return report

    def weekly_summary(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        weeks: int = 8,
        sport: str | None = None,
        sync_first: bool = True,
    ) -> MetricReport:
        """Return weekly table metrics over selected activity records."""
        if sync_first:
            self.sync_if_due()
        _validate_int_range("weeks", weeks, 1, MAX_WEEKS)
        records = tuple(self.repository.load_sync_state().activities.values())
        return weekly_summary(
            records,
            self.tz,
            start_date=start_date,
            end_date=end_date,
            weeks=weeks,
            sport=sport,
        )

    def overview(
        self, days: int = 28, sport: str | None = None, sync_first: bool = True
    ) -> MetricReport:
        """Return a generic metric report for current training context."""
        if sync_first:
            self.sync_if_due()
        _validate_int_range("days", days, 1, MAX_DAYS)
        state: SyncState = self.repository.load_sync_state()
        return overview_report(
            tuple(state.activities.values()),
            tuple(state.index_rows),
            athlete=self.settings.athlete,
            tz=self.tz,
            days=days,
            sport=sport,
        )


def _validate_int_range(name: str, value: int, low: int, high: int) -> None:
    if not low <= value <= high:
        msg = f"{name} must be between {low} and {high}, got {value}"
        raise ValueError(msg)
