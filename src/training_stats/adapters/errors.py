"""Errors raised by concrete adapters."""

from ..domain.errors import TrainingStatsError
from ..services.errors import SourceError


class ConfigurationError(TrainingStatsError):
    """Raised when configuration is missing, malformed or inconsistent."""


class AuthenticationError(SourceError):
    """Raised when credentials for a remote source are rejected."""


class RemoteFileError(SourceError):
    """Raised when a remote file cannot be accessed or downloaded."""


class CacheError(TrainingStatsError):
    """Raised when the JSON cache cannot be read or written."""
