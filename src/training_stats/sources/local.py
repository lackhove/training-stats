"""Local filesystem activity source."""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from ..errors import SourceError
from ..models import SourceFile
from .base import ActivitySource, is_fit_file

logger = logging.getLogger(__name__)


class LocalDirectorySource(ActivitySource):
    """Reads FIT files from a configurable local directory."""

    def __init__(
        self, *, name: str, path: Path, recursive: bool = True, follow_symlinks: bool = False
    ) -> None:
        super().__init__(name)
        self.root = path
        self.recursive = recursive
        self.follow_symlinks = follow_symlinks

    def _require_directory(self) -> Path:
        if not self.root.exists():
            msg = f"local source {self.name!r}: directory does not exist: {self.root}"
            raise SourceError(msg)
        if not self.root.is_dir():
            msg = f"local source {self.name!r}: path is not a directory: {self.root}"
            raise SourceError(msg)
        return self.root

    def list_files(self) -> list[SourceFile]:
        root = self._require_directory()
        files: list[SourceFile] = []
        if self.recursive:
            walker = os.walk(root, followlinks=self.follow_symlinks)
            for dirpath, _dirnames, filenames in walker:
                base = Path(dirpath)
                if not self.follow_symlinks:
                    _dirnames[:] = [d for d in _dirnames if not (base / d).is_symlink()]
                for filename in filenames:
                    candidate = base / filename
                    self._maybe_add(root, candidate, files)
        else:
            for candidate in sorted(root.iterdir()):
                self._maybe_add(root, candidate, files)
        logger.debug("local source %r discovered %d FIT files", self.name, len(files))
        return files

    def _maybe_add(self, root: Path, candidate: Path, files: list[SourceFile]) -> None:
        if not is_fit_file(candidate.name):
            return
        if candidate.is_symlink() and not self.follow_symlinks:
            return
        if not candidate.is_file():
            return
        relative = candidate.relative_to(root).as_posix()
        try:
            stat_result = candidate.stat()
        except OSError:
            logger.warning("local source %r: cannot stat %s", self.name, relative)
            return
        files.append(
            SourceFile(
                source_name=self.name,
                path=relative,
                size=stat_result.st_size,
                modified_at=datetime.fromtimestamp(stat_result.st_mtime, tz=UTC),
            )
        )

    def read_file(self, source_file: SourceFile) -> bytes:
        self._require_directory()
        target = (self.root / source_file.path).resolve()
        if not target.is_relative_to(self.root.resolve()):
            msg = (
                f"local source {self.name!r}: refusing to read path outside "
                f"the configured directory: {source_file.path}"
            )
            raise SourceError(msg)
        try:
            return target.read_bytes()
        except OSError as exc:
            msg = f"local source {self.name!r}: cannot read {source_file.path}: {exc}"
            raise SourceError(msg) from exc
