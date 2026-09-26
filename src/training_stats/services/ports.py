"""Protocols and state contracts used by application services."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Protocol

from ..domain.fit.types import FitData
from ..domain.models import SourceFile
from .state import SyncState


class ActivitySource(ABC):
    """Port for discovering and reading activity files."""

    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def list_files(self) -> list[SourceFile]: ...

    @abstractmethod
    def read_file(self, source_file: SourceFile) -> bytes: ...


class FitParser(Protocol):
    """Port for decoding FIT bytes into normalized transient inputs."""

    def parse(self, content: bytes, source_file: SourceFile) -> FitData: ...


class TrainingRepository(Protocol):
    """Read-side storage port for decoded domain cache records."""

    def load_sync_state(self) -> SyncState: ...

    def last_sync_stamp(self) -> datetime | None: ...


class SyncRepository(Protocol):
    """Write-side storage port used by incremental synchronization."""

    def load_sync_state(self) -> SyncState: ...

    def save_sync_snapshot(self, state: SyncState, *, last_sync: datetime) -> None: ...

    def save_sync_stamp(self, *, last_sync: datetime) -> None: ...
