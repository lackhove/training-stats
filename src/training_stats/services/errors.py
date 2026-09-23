"""Errors crossing application service ports."""

from ..domain.errors import TrainingStatsError


class SourceError(TrainingStatsError):
    """Raised when an activity source cannot be listed or read."""
