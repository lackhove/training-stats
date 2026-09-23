"""Application-service result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True, slots=True)
class SyncResult:
    """Outcome of one synchronization run."""

    files_discovered: int
    files_processed: int
    files_skipped: int
    files_failed: int
    cache_updated_at: datetime | None = field(default=None)
    source_errors: tuple[str, ...] = ()
    files_repaired: int = 0
    segments_discarded: int = 0
    duplicates_detected: int = 0


@dataclass(frozen=True, slots=True)
class ThrottledSync:
    """A skipped sync: the last successful one is still within the interval."""

    seconds_since_last_sync: float
