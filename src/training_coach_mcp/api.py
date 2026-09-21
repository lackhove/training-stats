"""FastAPI REST application exposing the training analytics domain.

Produces the OpenAPI schema consumed by Open WebUI (external OpenAPI tool
server) and other clients; route handlers are thin adapters over the service
layer. Coarse derived aggregates (5-bpm HR histogram, 10-minute weighted HR
averages) are returned; raw time-series records never are.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from starlette.requests import Request

from . import __version__
from .server import TrainingCoachServer

_MAX_LIMIT = 500
_MAX_WEEKS = 104
_MAX_DAYS = 3650

StartDate = Annotated[date | None, Query(description="Inclusive lower bound (YYYY-MM-DD).")]
EndDate = Annotated[date | None, Query(description="Inclusive upper bound (YYYY-MM-DD).")]
SportQuery = Annotated[str | None, Query(description="Exact sport name, case-insensitive.")]


# -- response models (mirror the domain to_dict() payloads) -------------------


class TotalsModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    activity_count: int
    duration_s: float
    distance_m: float
    elevation_gain_m: float
    hr_zones_s: dict[str, float]
    load_value: float
    load_method: str
    hard_activity_count: int
    longest_activity_s: float
    longest_activity_id: str | None


class ComparisonModel(BaseModel):
    load_value: float | None
    duration_s: float | None
    distance_m: float | None


class ActivityCoreModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    activity_id: str
    start_time: str
    sport: str
    sub_sport: str | None
    duration_s: float | None
    moving_time_s: float | None
    distance_m: float | None
    elevation_gain_m: float | None
    avg_hr: float | None
    max_hr: float | None
    calories: float | None


class ActivitySummaryModel(ActivityCoreModel):
    local_start_time: str
    load_value: float | None


class WeeklyRowModel(TotalsModel):
    iso_year: int
    iso_week: int
    week_start: str
    sport: str
    comparison_vs_prev_week: ComparisonModel


class HrZonesModel(BaseModel):
    zone_1: float
    zone_2: float
    zone_3: float
    zone_4: float
    zone_5: float


class DecouplingModel(BaseModel):
    ef_first_half: float | None
    ef_second_half: float | None
    aerobic_decoupling_pct: float | None
    confidence: str | None


class LoadModel(BaseModel):
    value: float | None
    method: str | None
    confidence: str | None


class SourceModel(BaseModel):
    name: str
    path: str


class TenMinuteAverageModel(BaseModel):
    start_min: int
    avg_hr: float
    hr_time_s: float


class ActivityDetailsModel(ActivityCoreModel):
    local_start_time: str
    avg_speed_mps: float | None
    max_speed_mps: float | None
    hr_zones_s: HrZonesModel
    hr_coverage_ratio: float
    hr_histogram_s: dict[str, float]
    hr_10min_averages: list[TenMinuteAverageModel]
    hr_drift_bpm_per_hour: float | None
    decoupling: DecouplingModel | None
    load: LoadModel
    data_completeness: float
    analysis_version: str
    paused_time_s: float | None
    data_hygiene_flags: list[str]
    source: SourceModel


class PeriodModel(BaseModel):
    start: str
    end: str


class LatestActivityModel(ActivitySummaryModel):
    model_config = ConfigDict(extra="ignore")

    decoupling: DecouplingModel | None = None


class OverviewModel(BaseModel):
    """Coaching overview; series sections stay loose dicts so one response
    carries per-(date, sport) totals for every metric."""

    model_config = ConfigDict(extra="ignore")

    athlete_profile: dict[str, object]
    period: PeriodModel
    recent_activities: list[ActivitySummaryModel]
    weekly_summaries: list[WeeklyRowModel]
    daily_series: list[dict[str, object]]
    zone_distribution_s: dict[str, float]
    form: dict[str, object]
    form_history: list[dict[str, object]]
    tour_readiness: dict[str, object]
    latest_activity: LatestActivityModel | None
    data_quality: dict[str, object]


# -- error handlers ------------------------------------------------------------


async def _value_error_handler(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, ValueError):
        raise exc
    return JSONResponse(status_code=400, content={"detail": str(exc)})


async def _lookup_error_handler(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, LookupError):
        raise exc
    return JSONResponse(status_code=404, content={"detail": str(exc)})


# -- application factory --------------------------------------------------------


def build_app(service: TrainingCoachServer) -> FastAPI:
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

    @app.get(
        "/activities",
        response_model=list[ActivitySummaryModel],
        summary="List activity summaries, newest first",
        tags=["activities"],
        description=(
            "Synchronizes new or changed FIT files from all configured sources "
            "before listing. Dates use YYYY-MM-DD format in the configured "
            "local timezone."
        ),
    )
    def list_activities(
        start_date: StartDate = None,
        end_date: EndDate = None,
        sport: SportQuery = None,
        limit: Annotated[int, Query(gt=0, le=_MAX_LIMIT)] = 50,
    ) -> list[dict[str, object]]:
        return service.list_activities(
            start_date=start_date, end_date=end_date, sport=sport, limit=limit
        )

    @app.get(
        "/activities/{activity_id}",
        response_model=ActivityDetailsModel,
        summary="Get one activity with metrics, zones, load and quality",
        tags=["activities"],
    )
    def get_activity(activity_id: str) -> dict[str, object]:
        return service.get_activity(activity_id)

    @app.get(
        "/weekly",
        response_model=list[WeeklyRowModel],
        summary="Weekly training totals with previous-week comparisons",
        tags=["aggregates"],
        description=(
            "Newest ISO weeks first. When no date range is given, the last "
            "``weeks`` weeks are returned."
        ),
    )
    def weekly_summary(
        start_date: StartDate = None,
        end_date: EndDate = None,
        weeks: Annotated[int, Query(gt=0, le=_MAX_WEEKS)] = 8,
        sport: SportQuery = None,
    ) -> list[dict[str, object]]:
        return service.weekly_summary(
            start_date=start_date, end_date=end_date, weeks=weeks, sport=sport
        )

    @app.get(
        "/overview",
        response_model=OverviewModel,
        summary="Compact coaching overview",
        tags=["aggregates"],
        description=(
            "Athlete profile, recent activities, weekly summaries, daily series "
            "(per-(date, sport) totals for every metric), zone distribution, "
            "training form with history, tour readiness and data quality."
        ),
    )
    def overview(
        days: Annotated[int, Query(gt=0, le=_MAX_DAYS)] = 28,
        sport: SportQuery = None,
    ) -> dict[str, object]:
        return service.overview(days=days, sport=sport)

    return app
