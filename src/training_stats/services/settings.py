"""Runtime settings needed by application services."""

from __future__ import annotations

from dataclasses import dataclass
from zoneinfo import ZoneInfo

from ..domain.settings import AthleteConfig


@dataclass(frozen=True, slots=True)
class ServiceSettings:
    application_name: str
    timezone: ZoneInfo
    sync_min_interval_seconds: float
    athlete: AthleteConfig
