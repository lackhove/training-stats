"""Atomic Parquet/JSON file access helpers."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import polars as pl

from ..errors import CacheError

logger = logging.getLogger(__name__)


def write_parquet_atomic(df: pl.DataFrame, path: Path) -> None:
    """Write ``df`` to ``path`` atomically (tmp file + ``os.replace``)."""
    _ensure_parent(path)
    tmp_path = _write_tmp(path, lambda target: df.write_parquet(target))
    os.replace(tmp_path, path)
    logger.debug("wrote %s (%d rows)", path, df.height)


def write_json_atomic(payload: dict[str, object], path: Path) -> None:
    """Write a JSON object to ``path`` atomically."""
    _ensure_parent(path)
    tmp_path = _write_tmp(path, lambda target: _write_json(payload, target))
    os.replace(tmp_path, path)
    logger.debug("wrote %s", path)


def read_parquet(path: Path) -> pl.DataFrame:
    """Read a Parquet file; return an empty frame when it does not exist."""
    if not path.exists():
        return pl.DataFrame()
    try:
        return pl.read_parquet(path)
    except Exception as exc:
        msg = f"cannot read cache file {path}: {type(exc).__name__}"
        raise CacheError(msg) from exc


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
    parent = path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        msg = f"cannot create cache directory {parent}: {exc}"
        raise CacheError(msg) from exc


def _write_tmp(path: Path, writer: Callable[[Path], None]) -> Path:
    """Write via a temporary file in the target directory and return its path."""
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        os.close(descriptor)
        tmp_path = Path(name)
        writer(tmp_path)
        _fsync_file(tmp_path)
    except OSError as exc:
        msg = f"cannot write cache file {path}: {exc}"
        raise CacheError(msg) from exc
    except Exception as exc:
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
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
