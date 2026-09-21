"""Common interface for activity sources."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import SourceFile


class ActivitySource(ABC):
    """Abstract base class for FIT file sources.

    Concrete sources provide a stable list of :class:`SourceFile` metadata
    objects and binary access to file content. Paths returned by
    ``list_files`` must be accepted by ``read_file`` unchanged.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def list_files(self) -> list[SourceFile]:
        """Return metadata for all available FIT files."""

    @abstractmethod
    def read_file(self, source_file: SourceFile) -> bytes:
        """Return the binary content of ``source_file``."""

    @staticmethod
    def _is_fit_file(path: str) -> bool:
        return path.lower().endswith(".fit")
