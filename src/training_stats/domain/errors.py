"""Errors that are part of the training domain contract."""


class TrainingStatsError(Exception):
    """Base class for expected training-stats failures."""


class FitParseError(TrainingStatsError):
    """Raised when a FIT file cannot be parsed into domain activity data."""
