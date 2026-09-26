"""Typed synchronization records shared by services and storage adapters."""

from __future__ import annotations

from dataclasses import dataclass

from ..domain.models import ActivityRecord, ActivitySummary, SourceIndexEntry, SourceStatus

__all__ = ["ActivityRecord", "ActivitySummary", "SourceIndexEntry", "SourceStatus", "SyncState"]


@dataclass(frozen=True)
class SyncState:
    """Source outcomes plus accepted activity facts for an incremental sync."""

    index_rows: list[SourceIndexEntry]
    activities: dict[str, ActivityRecord]
