"""TOML-based configuration with environment-resolved secrets.

Non-secret configuration lives in a TOML file; secrets (WebDAV credentials)
are read exclusively from environment variables whose *names* are configured
in the TOML file. Secret values never appear in configuration objects, logs
or error messages.
"""

from __future__ import annotations

import tomllib
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)

from .errors import ConfigurationError

CONFIG_ENV_VAR = "TRAINING_MCP_CONFIG"


class ApplicationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "training-coach-mcp"
    log_level: str = "INFO"
    timezone: str = "UTC"

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = value.upper()
        if upper not in allowed:
            msg = f"application.log_level must be one of {sorted(allowed)}, got {value!r}"
            raise ValueError(msg)
        return upper

    @property
    def zoneinfo(self) -> ZoneInfo:
        try:
            return ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as exc:
            msg = f"application.timezone {self.timezone!r} is not a valid IANA timezone"
            raise ConfigurationError(msg) from exc


class HeartRateZonesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    z1_min: int
    z1_max: int
    z2_min: int
    z2_max: int
    z3_min: int
    z3_max: int
    z4_min: int
    z4_max: int
    z5_min: int
    z5_max: int

    @model_validator(mode="after")
    def _validate_ordering(self) -> HeartRateZonesConfig:
        zones = [
            (1, self.z1_min, self.z1_max),
            (2, self.z2_min, self.z2_max),
            (3, self.z3_min, self.z3_max),
            (4, self.z4_min, self.z4_max),
            (5, self.z5_min, self.z5_max),
        ]
        for number, low, high in zones:
            if low <= 0 or high < low:
                msg = (
                    f"athlete.heart_rate_zones: zone {number} bounds are invalid "
                    f"(min={low}, max={high}); require 0 < min <= max"
                )
                raise ValueError(msg)
        for current, nxt in pairwise(zones):
            if current[2] >= nxt[1]:
                msg = (
                    f"athlete.heart_rate_zones: zone {current[0]} max ({current[2]}) "
                    f"must be below zone {nxt[0]} min ({nxt[1]})"
                )
                raise ValueError(msg)
        return self

    def as_bounds(self) -> list[tuple[int, int, int]]:
        """Return ordered ``(zone, min_bpm, max_bpm)`` tuples."""
        return [
            (1, self.z1_min, self.z1_max),
            (2, self.z2_min, self.z2_max),
            (3, self.z3_min, self.z3_max),
            (4, self.z4_min, self.z4_max),
            (5, self.z5_min, self.z5_max),
        ]


class AthleteConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_heart_rate: int = Field(gt=0)
    resting_heart_rate: int = Field(gt=0)
    primary_goal: str | None = None
    weight_kg: float | None = Field(default=None, gt=0)
    ftp_watts: float | None = Field(default=None, gt=0)
    heart_rate_zones: HeartRateZonesConfig

    @model_validator(mode="after")
    def _validate_heart_rates(self) -> AthleteConfig:
        if self.resting_heart_rate >= self.max_heart_rate:
            msg = (
                "athlete.resting_heart_rate must be below "
                f"athlete.max_heart_rate (got {self.resting_heart_rate} "
                f">= {self.max_heart_rate})"
            )
            raise ValueError(msg)
        return self


class LocalSourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: Literal["local"] = "local"
    path: Path
    recursive: bool = True
    follow_symlinks: bool = False


class WebDAVSourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: Literal["webdav"] = "webdav"
    base_url: str
    path: str = ""
    username_env: str
    password_env: str
    verify_tls: bool = True
    timeout_seconds: float = Field(default=30.0, gt=0)
    recursive: bool = True


SourceConfig = Annotated[LocalSourceConfig | WebDAVSourceConfig, Field(discriminator="type")]


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)


class CacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    directory: Path = Path("./data/cache")
    lock_timeout_seconds: float = Field(default=30.0, gt=0)
    retain_time_series: bool = False


class QualityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    moving_speed_threshold_mps: float = Field(default=1.0, gt=0)
    record_gap_threshold_s: float = Field(default=120.0, gt=0)
    clock_jump_min_gap_s: float = Field(default=3600.0, gt=0)
    minimum_segment_duration_s: float = Field(default=600.0, gt=0)
    clock_shift_max_pre_moving_s: float = Field(default=60.0, ge=0)
    forgotten_stop_max_duration_s: float = Field(default=86400.0, gt=0)
    forgotten_stop_min_moving_ratio: float = Field(default=0.20, gt=0, le=1.0)
    decoupling_min_duration_s: float = Field(default=1200.0, gt=0)
    decoupling_min_coverage: float = Field(default=0.80, gt=0, le=1.0)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    application: ApplicationConfig = ApplicationConfig()
    cache: CacheConfig = CacheConfig()
    athlete: AthleteConfig
    quality: QualityConfig = Field(default_factory=QualityConfig)
    sources: list[SourceConfig] = []
    server: ServerConfig = ServerConfig()

    @model_validator(mode="after")
    def _validate_sources(self) -> AppConfig:
        names = [source.name for source in self.sources]
        if not names:
            msg = "at least one [[sources]] entry is required"
            raise ValueError(msg)
        if len(set(names)) != len(names):
            msg = f"source names must be unique, got: {names}"
            raise ValueError(msg)
        return self


class WebDAVCredentials:
    """Immutable container for resolved WebDAV secrets.

    Values are wrapped in :class:`SecretStr` so they never leak through
    ``repr``, logs or error messages.
    """

    __slots__ = ("_password", "_username")

    def __init__(self, username: SecretStr, password: SecretStr) -> None:
        self._username = username
        self._password = password

    @property
    def username(self) -> str:
        return self._username.get_secret_value()

    @property
    def password(self) -> str:
        return self._password.get_secret_value()

    def __repr__(self) -> str:
        return "WebDAVCredentials(username=***, password=***)"


def resolve_webdav_credentials(config: AppConfig) -> dict[str, WebDAVCredentials]:
    """Resolve environment credentials for all WebDAV sources.

    Raises :class:`ConfigurationError` if a referenced environment variable is
    unset or empty. Error messages contain variable names, never values.
    """
    import os

    resolved: dict[str, WebDAVCredentials] = {}
    for source in config.sources:
        if isinstance(source, WebDAVSourceConfig):
            username = os.environ.get(source.username_env)
            password = os.environ.get(source.password_env)
            if not username:
                msg = (
                    f"webdav source {source.name!r}: environment variable "
                    f"{source.username_env!r} is not set"
                )
                raise ConfigurationError(msg)
            if not password:
                msg = (
                    f"webdav source {source.name!r}: environment variable "
                    f"{source.password_env!r} is not set"
                )
                raise ConfigurationError(msg)
            resolved[source.name] = WebDAVCredentials(
                username=SecretStr(username), password=SecretStr(password)
            )
    return resolved


def config_path_from_env() -> Path | None:
    """Return the configuration path from ``TRAINING_MCP_CONFIG`` if set."""
    import os

    value = os.environ.get(CONFIG_ENV_VAR)
    return Path(value) if value else None


def load_config(path: str | Path) -> AppConfig:
    """Load, parse and validate the TOML configuration file.

    Fails early with actionable error messages for missing files, malformed
    TOML or invalid values.
    """
    config_file = Path(path)
    if not config_file.is_file():
        msg = (
            f"configuration file not found: {config_file}; pass a path via CLI "
            f"argument or set the {CONFIG_ENV_VAR} environment variable"
        )
        raise ConfigurationError(msg)
    try:
        with config_file.open("rb") as handle:
            raw = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        msg = f"configuration file {config_file} is not valid TOML: {exc}"
        raise ConfigurationError(msg) from exc
    try:
        return AppConfig.model_validate(raw)
    except ValidationError as exc:
        msg = f"configuration file {config_file} is invalid:\n{exc}"
        raise ConfigurationError(msg) from exc
