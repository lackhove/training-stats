"""FastMCP server exposing the training analytics domain tools.

Tools are thin, validating adapters over the repository and sync service; all
business logic lives in the analytics, cache and coach layers.
"""

from __future__ import annotations

import logging
from datetime import date, datetime

from fastmcp import FastMCP

from .cache import schemas
from .cache.repository import ParquetTrainingRepository, TrainingRepository
from .cache.sync import SyncService
from .config import AppConfig, LocalSourceConfig, WebDAVSourceConfig
from .errors import TrainingMCPError
from .fit.parser import FitDecodeParser, FitParser
from .models import SyncResult
from .sources.base import ActivitySource
from .sources.local import LocalDirectorySource
from .sources.webdav import WebDAVSource

logger = logging.getLogger(__name__)

_MAX_LIMIT = 500
_MAX_WEEKS = 104
_MAX_DAYS = 3650


class TrainingCoachServer:
    """Domain-facing server; also builds the FastMCP application."""

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

    # -- tool implementations (directly testable) -------------------------

    def sync_activities(self) -> dict[str, object]:
        """Synchronize new or changed FIT files; never raises.

        Source-level failures (e.g. an unreachable WebDAV source) are reported
        in ``source_errors``; unexpected sync failures are logged as a warning
        so callers can keep serving the existing cache.
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
            ).to_dict()
        if result.source_errors:
            logger.warning("activity sync reported source errors: %s", list(result.source_errors))
        if result.files_processed or result.files_failed:
            logger.info(
                "activity sync: processed=%d skipped=%d failed=%d",
                result.files_processed,
                result.files_skipped,
                result.files_failed,
            )
        return result.to_dict()

    def list_activities(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        sport: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, object]]:
        """Synchronize, then list compact activity summaries, newest first."""
        self.sync_activities()
        start = _parse_date_arg("start_date", start_date)
        end = _parse_date_arg("end_date", end_date)
        _validate_int_range("limit", limit, 1, _MAX_LIMIT)
        activities = self.repository.list_activities(
            start_date=start, end_date=end, sport=sport, limit=limit
        )
        return [item.to_dict() for item in activities]

    def get_activity(self, activity_id: str) -> dict[str, object]:
        """Get one activity with metrics, zones, load and data quality."""
        details = self.repository.get_activity(activity_id)
        if details is None:
            msg = f"activity not found: {activity_id!r}"
            raise LookupError(msg)
        return details.to_dict()

    def weekly_summary(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        weeks: int = 8,
        sport: str | None = None,
    ) -> list[dict[str, object]]:
        """Weekly training totals with previous-week comparisons."""
        start = _parse_date_arg("start_date", start_date)
        end = _parse_date_arg("end_date", end_date)
        _validate_int_range("weeks", weeks, 1, _MAX_WEEKS)
        rows = self.repository.weekly_summary(
            start_date=start, end_date=end, weeks=weeks, sport=sport
        )
        return [row.to_dict() for row in rows]

    def training_trends(
        self, metric: str, days: int = 30, sport: str | None = None
    ) -> dict[str, object]:
        """Daily trend for a metric: duration, distance, elevation, load or activities."""
        _validate_int_range("days", days, 1, _MAX_DAYS)
        trend = self.repository.training_trends(metric=metric, days=days, sport=sport)
        return trend.to_dict()

    def training_form(self, days: int = 42, sport: str | None = None) -> dict[str, object]:
        """Impulse-response training form (CTL, ATL, TSB) and tour readiness."""
        _validate_int_range("days", days, 1, _MAX_DAYS)
        result = self.repository.training_form(days=days, sport=sport)
        return result.to_dict()

    def get_coach_context(self, days: int = 28, sport: str | None = None) -> dict[str, object]:
        """Compact coaching context: profile, recent activities, trends, quality."""
        _validate_int_range("days", days, 1, _MAX_DAYS)
        return self.repository.coach_context(days=days, sport=sport).to_dict()

    # -- resource implementations ------------------------------------------

    def athlete_profile_resource(self) -> dict[str, object]:
        from .coach.profile import athlete_profile

        return athlete_profile(self.config.athlete)

    def zones_resource(self) -> dict[str, object]:
        from .coach.profile import zone_descriptions

        return {
            "zones": zone_descriptions(self.config.athlete),
            "source": "configuration (athlete.heart_rate_zones)",
        }

    def data_quality_resource(self) -> dict[str, object]:
        return self.repository.data_quality().to_dict()

    def current_week_resource(self) -> dict[str, object]:
        rows = self.repository.weekly_summary(weeks=1, sport=None)
        if not rows:
            return {"message": "no activities recorded yet"}
        return rows[0].to_dict()

    def training_form_resource(self) -> dict[str, object]:
        return self.training_form(days=42, sport=None)

    # -- FastMCP wiring ------------------------------------------------------

    def build_mcp(self) -> FastMCP:
        mcp = FastMCP(name=self.config.application.name)

        @mcp.tool
        def list_activities(
            start_date: str | None = None,
            end_date: str | None = None,
            sport: str | None = None,
            limit: int = 50,
        ) -> list[dict[str, object]]:
            """List compact activity summaries, newest first.

            Synchronizes new or changed FIT files from all configured sources
            before listing. Dates use YYYY-MM-DD format in the configured
            local timezone.
            """
            return self.list_activities(
                start_date=start_date, end_date=end_date, sport=sport, limit=limit
            )

        @mcp.tool
        def get_activity(activity_id: str) -> dict[str, object]:
            """Get one activity with metrics, zones, load and data quality."""
            return self.get_activity(activity_id)

        @mcp.tool
        def weekly_summary(
            start_date: str | None = None,
            end_date: str | None = None,
            weeks: int = 8,
            sport: str | None = None,
        ) -> list[dict[str, object]]:
            """Weekly training totals with previous-week comparisons."""
            return self.weekly_summary(
                start_date=start_date, end_date=end_date, weeks=weeks, sport=sport
            )

        @mcp.tool
        def training_trends(
            metric: str, days: int = 30, sport: str | None = None
        ) -> dict[str, object]:
            """Daily trend for one metric.

            Allowed metrics: duration, distance, elevation, load, activities.
            """
            return self.training_trends(metric=metric, days=days, sport=sport)

        @mcp.tool
        def training_form(days: int = 42, sport: str | None = None) -> dict[str, object]:
            """Impulse-response training form (CTL, ATL, TSB) and tour readiness."""
            return self.training_form(days=days, sport=sport)

        @mcp.tool
        def get_coach_context(days: int = 28, sport: str | None = None) -> dict[str, object]:
            """Compact coaching context for the last N days.

            Includes athlete profile, recent activities, weekly summaries,
            zone distribution, load trend, data quality and latest activity.
            """
            return self.get_coach_context(days=days, sport=sport)

        @mcp.resource("athlete://profile")
        def athlete_profile_resource() -> dict[str, object]:
            """Athlete profile from configuration."""
            return self.athlete_profile_resource()

        @mcp.resource("training://zones")
        def zones_resource() -> dict[str, object]:
            """Configured heart-rate zone model."""
            return self.zones_resource()

        @mcp.resource("training://data-quality")
        def data_quality_resource() -> dict[str, object]:
            """Aggregate data quality of the cached training data."""
            return self.data_quality_resource()

        @mcp.resource("training://current-week")
        def current_week_resource() -> dict[str, object]:
            """Current ISO week training totals."""
            return self.current_week_resource()

        @mcp.resource("training://form")
        def training_form_resource() -> dict[str, object]:
            """Current training form (CTL, ATL, TSB) and tour readiness snapshot."""
            return self.training_form_resource()

        return mcp


def create_server(config: AppConfig) -> TrainingCoachServer:
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
            raise TrainingMCPError(msg)

    parser: FitParser = FitDecodeParser(quality=config.quality)
    sync_service = SyncService(
        sources=sources,
        parser=parser,
        cache_dir=config.cache.directory,
        athlete=config.athlete,
        timezone=config.application.zoneinfo,
        analysis_version=schemas.ANALYSIS_VERSION_VALUE,
        quality=config.quality,
        lock_timeout_seconds=config.cache.lock_timeout_seconds,
    )
    repository = ParquetTrainingRepository(
        cache_dir=config.cache.directory,
        timezone=config.application.zoneinfo,
        athlete=config.athlete,
    )
    return TrainingCoachServer(
        config=config,
        sources=sources,
        parser=parser,
        sync_service=sync_service,
        repository=repository,
    )


def _parse_date_arg(name: str, value: str | None) -> date | None:
    if value is None:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        msg = f"{name} must use YYYY-MM-DD format, got {value!r}"
        raise ValueError(msg) from exc


def _validate_int_range(name: str, value: int, low: int, high: int) -> None:
    if not low <= value <= high:
        msg = f"{name} must be between {low} and {high}, got {value}"
        raise ValueError(msg)
