"""Typed exceptions for training-stats."""


class TrainingStatsError(Exception):
    """Base class for all errors raised by this package."""


class ConfigurationError(TrainingStatsError):
    """Raised when configuration is missing, malformed or inconsistent."""


class SourceError(TrainingStatsError):
    """Raised when an activity source cannot be listed or read."""


class AuthenticationError(SourceError):
    """Raised when credentials for a remote source are rejected."""


class RemoteFileError(SourceError):
    """Raised when a remote file cannot be accessed or downloaded."""


class FitParseError(TrainingStatsError):
    """Raised when a FIT file cannot be parsed."""


class CacheError(TrainingStatsError):
    """Raised when the Parquet cache cannot be read or written."""


class AnalysisError(TrainingStatsError):
    """Raised when analytics computation fails unexpectedly."""
