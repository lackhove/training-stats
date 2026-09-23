"""Application-wide cache locking (POSIX flock).

Exactly one process — the running CLI command or the server — may operate
on the cache directory at a time. The lock is acquired once per process and
held until the process exits; the kernel releases flock automatically when
the process dies, so no stale locks can remain. Reads inside the running
process need no lock of their own: every cache file is replaced atomically,
so each read observes one consistent file version.
"""

from __future__ import annotations

import fcntl
import os
import time
from pathlib import Path

from ..errors import CacheError
from .schemas import LOCK_FILE

_HANDLES: dict[str, int] = {}


def acquire_app_lock(cache_dir: Path, timeout_seconds: float = 30.0) -> None:
    """Acquire the application-wide exclusive cache lock (idempotent).

    Repeated calls within the same process are no-ops; the lock is kept
    until process exit. Raises :class:`CacheError` when another instance
    (a server or another CLI command) holds the lock and it cannot be
    acquired within ``timeout_seconds``.
    """
    path = cache_dir / LOCK_FILE
    key = str(path)
    if key in _HANDLES:
        return
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
                        f"cache {path.parent} is in use by another instance "
                        f"(server or CLI); waited {timeout_seconds:.0f}s — "
                        "use the running instance's API instead"
                    )
                    raise CacheError(msg) from None
                time.sleep(0.1)
    except BaseException:
        os.close(handle)
        raise
    _HANDLES[key] = handle
