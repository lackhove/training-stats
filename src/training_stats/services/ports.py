"""Protocols and state contracts used by application services."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Protocol

import polars as pl

from ..domain.fit.types import ParsedFile
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
    """Port for decoding raw FIT bytes into normalized domain activities."""

    def parse(self, content: bytes, source_file: SourceFile) -> ParsedFile: ...


class TrainingRepository(Protocol):
    """Read-side storage port. Frames use the domain activity-frame contract."""

    def activities_scan(self) -> pl.LazyFrame: ...

    def source_index_scan(self) -> pl.LazyFrame: ...

    def last_sync_stamp(self) -> datetime | None: ...


class SyncRepository(Protocol):
    """Write-side storage port used by incremental synchronization."""

    def load_sync_state(self) -> SyncState: ...

    def save_sync_snapshot(self, state: SyncState, *, last_sync: datetime) -> None: ...
