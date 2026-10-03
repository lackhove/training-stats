"""Protocols and state contracts used by application services."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from ..domain.fit.types import FitData
from ..domain.models import SourceFile
from .state import SyncState


class ActivitySource(Protocol):
    """Port for discovering and reading activity files."""

    name: str

    def list_files(self) -> list[SourceFile]: ...

    def read_file(self, source_file: SourceFile) -> bytes: ...


class FitParser(Protocol):
    """Port for decoding FIT bytes into normalized transient inputs."""

    def parse(self, content: bytes, source_file: SourceFile) -> FitData: ...


class TrainingRepository(Protocol):
    """Read-side storage port for decoded domain cache records."""

    def load_sync_state(self) -> SyncState: ...

    def last_sync_stamp(self) -> datetime | None: ...


class SyncRepository(TrainingRepository, Protocol):
    """Write-side storage port used by incremental synchronization."""

    def save_sync_snapshot(self, state: SyncState, *, last_sync: datetime) -> None: ...

    def save_sync_stamp(self, state: SyncState, *, last_sync: datetime) -> None: ...
