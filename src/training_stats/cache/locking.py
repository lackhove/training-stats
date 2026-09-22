"""Inter-process file locking for sync operations (POSIX)."""

from __future__ import annotations

import fcntl
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ..errors import CacheError


@contextmanager
def sync_lock(path: Path, timeout_seconds: float = 30.0) -> Iterator[None]:
    """Hold an exclusive advisory lock on ``path`` for the duration.

    Raises :class:`CacheError` when the lock cannot be acquired in time.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    msg = (
                        f"could not acquire sync lock {path} within "
                        f"{timeout_seconds:.0f}s; another sync may be running"
                    )
                    raise CacheError(msg) from None
                time.sleep(0.1)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        os.close(handle)


@contextmanager
def read_lock(path: Path, timeout_seconds: float = 30.0) -> Iterator[None]:
    """Hold a shared advisory lock on ``path`` for the duration.

    Multiple readers may hold the lock concurrently; the lock waits for an
    in-progress exclusive sync to finish so multi-file cache reads never
    observe a torn write. Nested shared locks in the same process are
    re-entrant on POSIX (shared with shared is compatible).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    msg = (
                        f"could not acquire read lock {path} within "
                        f"{timeout_seconds:.0f}s; a sync may be running"
                    )
                    raise CacheError(msg) from None
                time.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        os.close(handle)
