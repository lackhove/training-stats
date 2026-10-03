"""TOML-based configuration with environment-resolved secrets.

Non-secret configuration lives in a TOML file; secrets (WebDAV credentials)
are read exclusively from environment variables whose *names* are configured
in the TOML file. Secret values never appear in configuration objects, logs
or error messages.
"""

from __future__ import annotations

import os
import tomllib
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

from ..domain.settings import AthleteConfig, QualityConfig
from .errors import ConfigurationError

CONFIG_ENV_VAR = "TRAINING_STATS_CONFIG"


class ApplicationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "training-stats"
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


class AuthConfig(BaseModel):
    """Shared-secret bearer authentication for the REST server.

    Only the *name* of the environment variable holding the token is
    configured here; the secret itself is resolved via
    :func:`resolve_auth_token` and wrapped in a :class:`SecretStr`.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["bearer"] = "bearer"
    token_env: str

    @field_validator("token_env")
    @classmethod
    def _validate_token_env(cls, value: str) -> str:
        if not value or not value.strip():
            msg = "server.auth.token_env must name an environment variable"
            raise ValueError(msg)
        return value


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    auth: AuthConfig | None = None


class CacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    directory: Path = Path("./data/cache")
    lock_timeout_seconds: float = Field(default=30.0, gt=0)
    sync_min_interval_seconds: float = Field(default=10.0, ge=0)


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


def _require_env(name: str, context: str, missing: str) -> str:
    """Read a required environment variable for ``context``.

    ``context`` is the message prefix including its colon and ``missing``
    describes the unset/empty state. Error messages name the variable, never
    its value.
    """
    value = os.environ.get(name)
    if not value:
        msg = f"{context} environment variable {name!r} {missing}"
        raise ConfigurationError(msg)
    return value


def resolve_webdav_credentials(config: AppConfig) -> dict[str, WebDAVCredentials]:
    """Resolve environment credentials for all WebDAV sources.

    Raises :class:`ConfigurationError` if a referenced environment variable is
    unset or empty. Error messages contain variable names, never values.
    """
    resolved: dict[str, WebDAVCredentials] = {}
    for source in config.sources:
        if isinstance(source, WebDAVSourceConfig):
            context = f"webdav source {source.name!r}:"
            username = _require_env(source.username_env, context, "is not set")
            password = _require_env(source.password_env, context, "is not set")
            resolved[source.name] = WebDAVCredentials(
                username=SecretStr(username), password=SecretStr(password)
            )
    return resolved


def resolve_auth_token(config: AppConfig) -> SecretStr | None:
    """Resolve the bearer auth token from the environment, if auth is enabled.

    Raises :class:`ConfigurationError` if the referenced environment variable
    is unset or empty. Error messages contain variable names, never values.
    """
    auth = config.server.auth
    if auth is None:
        return None
    token = _require_env(auth.token_env, "server.auth:", "is not set or empty")
    return SecretStr(token)


def config_path_from_env() -> Path | None:
    """Return the configuration path from ``TRAINING_STATS_CONFIG`` if set."""
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
