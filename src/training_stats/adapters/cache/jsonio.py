"""Atomic JSON cache access helpers."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

from ..errors import CacheError

logger = logging.getLogger(__name__)


def write_json_atomic(payload: dict[str, object], path: Path) -> None:
    """Write a JSON object atomically using a same-directory temporary file."""
    _ensure_parent(path)
    tmp_path = _write_tmp(path, lambda target: _write_json(payload, target))
    os.replace(tmp_path, path)
    logger.debug("wrote %s", path)


def read_json(path: Path) -> dict[str, object] | None:
    """Read a JSON object file; return ``None`` when missing or malformed."""
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("cannot read %s: %s", path, exc)
        return None
    return payload if isinstance(payload, dict) else None


def _ensure_parent(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        msg = f"cannot create cache directory {path.parent}: {exc}"
        raise CacheError(msg) from exc


def _write_tmp(path: Path, writer: Callable[[Path], None]) -> Path:
    tmp_path: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        os.close(descriptor)
        tmp_path = Path(name)
        writer(tmp_path)
        _fsync_file(tmp_path)
    except OSError as exc:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        msg = f"cannot write cache file {path}: {exc}"
        raise CacheError(msg) from exc
    except Exception as exc:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        msg = f"cannot write cache file {path}: {type(exc).__name__}"
        raise CacheError(msg) from exc
    return tmp_path


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(payload: dict[str, object], target: Path) -> None:
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
