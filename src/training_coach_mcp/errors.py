"""Typed exceptions for the training coach."""


class TrainingMCPError(Exception):
    """Base class for all errors raised by this package."""


class ConfigurationError(TrainingMCPError):
    """Raised when configuration is missing, malformed or inconsistent."""


class SourceError(TrainingMCPError):
    """Raised when an activity source cannot be listed or read."""


class AuthenticationError(SourceError):
    """Raised when credentials for a remote source are rejected."""


class RemoteFileError(SourceError):
    """Raised when a remote file cannot be accessed or downloaded."""


class FitParseError(TrainingMCPError):
    """Raised when a FIT file cannot be parsed."""


class CacheError(TrainingMCPError):
    """Raised when the Parquet cache cannot be read or written."""


class AnalysisError(TrainingMCPError):
    """Raised when analytics computation fails unexpectedly."""
