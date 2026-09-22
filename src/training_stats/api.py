"""FastAPI REST application exposing the training analytics domain.

Produces the OpenAPI schema consumed by Open WebUI (external OpenAPI tool
server) and other clients; route handlers are thin adapters over the service
layer. Coarse derived aggregates (5-bpm HR histogram, 10-minute weighted HR
averages) are returned; raw time-series records never are.

Response shapes are declared as Pydantic models in :mod:`.wire` and mapped
from domain models in the handlers (the anti-corruption layer), so the
OpenAPI schema and runtime validation stay in one place without coupling
the API to internal domain shapes.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import Depends, FastAPI, Query
from fastapi.responses import JSONResponse
from starlette.requests import Request

from . import __version__, wire
from .auth import BearerGuard
from .config import resolve_auth_token
from .errors import TrainingStatsError
from .models import MAX_DAYS, MAX_LIST_LIMIT, MAX_WEEKS
from .services import TrainingStatsService

StartDate = Annotated[date | None, Query(description="Inclusive lower bound (YYYY-MM-DD).")]
EndDate = Annotated[date | None, Query(description="Inclusive upper bound (YYYY-MM-DD).")]
SportQuery = Annotated[str | None, Query(description="Exact sport name, case-insensitive.")]

_SYNC_NOTE = (
    "Synchronizes new or changed FIT files from all configured sources first. "
    "The sync is throttled: it runs at most once per "
    "``cache.sync_min_interval_seconds`` (default 10 seconds); within that "
    "window data is served from the local cache without contacting sources."
)


# -- error handlers ------------------------------------------------------------


async def _value_error_handler(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, ValueError):
        raise exc
    return JSONResponse(status_code=400, content={"detail": str(exc)})


async def _lookup_error_handler(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, LookupError):
        raise exc
    return JSONResponse(status_code=404, content={"detail": str(exc)})


async def _training_stats_error_handler(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, TrainingStatsError):
        raise exc
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# -- application factory --------------------------------------------------------


def build_app(service: TrainingStatsService) -> FastAPI:
    """Build the FastAPI application over the given service."""
    app = FastAPI(
        title=service.config.application.name,
        version=__version__,
        description=(
            "REST API for FIT-based training analytics: activities, weekly "
            "summaries and a compact coaching overview (profile, daily series, "
            "form, data quality). Load is an hr_trimp estimate, never TSS."
        ),
    )
    app.add_exception_handler(ValueError, _value_error_handler)
    app.add_exception_handler(LookupError, _lookup_error_handler)
    app.add_exception_handler(TrainingStatsError, _training_stats_error_handler)

    token = resolve_auth_token(service.config)
    route_dependencies = [Depends(BearerGuard(token))] if token is not None else []

    @app.get(
        "/activities",
        response_model=list[wire.ActivitySummary],
        summary="List activity summaries, newest first",
        dependencies=route_dependencies,
        tags=["activities"],
        description=(_SYNC_NOTE + " Dates use YYYY-MM-DD format in the configured local timezone."),
    )
    def list_activities(
        start_date: StartDate = None,
        end_date: EndDate = None,
        sport: SportQuery = None,
        limit: Annotated[int, Query(gt=0, le=MAX_LIST_LIMIT)] = 50,
    ) -> list[wire.ActivitySummary]:
        return [
            wire.activity_summary(item)
            for item in service.list_activities(
                start_date=start_date, end_date=end_date, sport=sport, limit=limit
            )
        ]

    @app.get(
        "/activities/{activity_id}",
        response_model=wire.ActivityDetails,
        summary="Get one activity with metrics, zones, load and quality",
        description="Serves cached data; IDs come from /activities.",
        dependencies=route_dependencies,
        tags=["activities"],
    )
    def get_activity(activity_id: str) -> wire.ActivityDetails:
        return wire.activity_details(service.get_activity(activity_id))

    @app.get(
        "/weekly",
        response_model=list[wire.WeeklyRow],
        summary="Weekly training totals with previous-week comparisons",
        dependencies=route_dependencies,
        tags=["aggregates"],
        description=(
            _SYNC_NOTE + " Newest ISO weeks first. When no date range is "
            "given, the last ``weeks`` weeks are returned."
        ),
    )
    def weekly_summary(
        start_date: StartDate = None,
        end_date: EndDate = None,
        weeks: Annotated[int, Query(gt=0, le=MAX_WEEKS)] = 8,
        sport: SportQuery = None,
    ) -> list[wire.WeeklyRow]:
        return [
            wire.weekly_row(row)
            for row in service.weekly_summary(
                start_date=start_date, end_date=end_date, weeks=weeks, sport=sport
            )
        ]

    @app.get(
        "/overview",
        response_model=wire.Overview,
        summary="Compact coaching overview",
        dependencies=route_dependencies,
        tags=["aggregates"],
        description=(
            _SYNC_NOTE + " Athlete profile, recent activities, weekly "
            "summaries, daily series (per-(date, sport) totals for every "
            "metric), zone distribution, training form with history, tour "
            "readiness and data quality."
        ),
    )
    def overview(
        days: Annotated[int, Query(gt=0, le=MAX_DAYS)] = 28,
        sport: SportQuery = None,
    ) -> wire.Overview:
        return wire.overview(service.overview(days=days, sport=sport))

    return app
