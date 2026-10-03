"""JSON-backed cache adapter for activity records and source outcomes."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from ...domain.models import ActivityRecord, SourceIndexEntry
from ...services.ports import SyncRepository
from ...services.state import SyncState
from ..errors import CacheError
from . import jsonio, schemas

_ACTIVITIES_ADAPTER = TypeAdapter(list[ActivityRecord])
_SOURCE_INDEX_ADAPTER = TypeAdapter(list[SourceIndexEntry])

# Metadata key holding the snapshot stat fingerprint written with each stamp.
SNAPSHOT_FINGERPRINT_KEY = "snapshot_fingerprint"


class JsonTrainingRepository(SyncRepository):
    """Atomically persist one metric-bearing record per accepted activity."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self._cached_sync_state: tuple[tuple[int, int, int] | None, SyncState] | None = None

    def load_sync_state(self) -> SyncState:
        """Load the cache snapshot; invalid data rebuilds from configured sources."""
        while True:
            fingerprint = self._snapshot_stat_key()
            cached = self._cached_sync_state
            if cached is not None and cached[0] == fingerprint:
                return _copy_sync_state(cached[1])

            payload = jsonio.read_json(self.cache_dir / schemas.CACHE_FILE)
            if fingerprint != self._snapshot_stat_key():
                continue
            state = _decode_sync_state(payload)
            self._cached_sync_state = (fingerprint, state)
            return _copy_sync_state(state)

    def save_sync_snapshot(self, state: SyncState, *, last_sync: datetime) -> None:
        """Write records and sync stamp atomically at their respective boundaries."""
        self._write_snapshot(state)
        self._write_sync_metadata_stamp(last_sync=last_sync)

    def save_sync_stamp(self, state: SyncState, *, last_sync: datetime) -> None:
        """Record a successful sync stamp without rewriting an unchanged snapshot.

        Metadata alone is rewritten while the stamp on disk still covers the
        snapshot by fingerprint, which is written immediately after every
        snapshot write and therefore proves the file is that valid snapshot.
        Metadata that is missing, legacy, or out of version, and a snapshot
        that is gone or changed on disk, mean ``state`` is written first: a
        stamp must never cover a snapshot it did not vouch for. The caller
        passes the state its sync produced and stays responsible for having
        established that it matches what was loaded (the sync's no-op path).
        """
        if not self._snapshot_matches_last_stamp():
            self.save_sync_snapshot(state, last_sync=last_sync)
            return
        self._write_sync_metadata_stamp(last_sync=last_sync)

    def last_sync_stamp(self) -> datetime | None:
        """Read the throttle stamp from metadata alone; never open the snapshot.

        The stamp is honoured only while the snapshot still matches the
        fingerprint recorded with it; metadata without a fingerprint, or a
        snapshot that is gone or changed on disk, forces a sync.
        """
        metadata = jsonio.read_json(self.cache_dir / schemas.METADATA_FILE)
        if metadata is None or metadata.get("format_version") != schemas.CACHE_FORMAT_VERSION:
            return None
        value = metadata.get("last_sync")
        if not isinstance(value, str):
            return None
        try:
            stamp = datetime.fromisoformat(value)
        except ValueError:
            return None
        if not self._fingerprint_matches(metadata.get(SNAPSHOT_FINGERPRINT_KEY)):
            return None
        return stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=UTC)

    def _snapshot_matches_last_stamp(self) -> bool:
        """True when the stamp on disk still covers the snapshot by fingerprint.

        Reads only the small metadata file and stats the snapshot; the
        multi-megabyte snapshot is never parsed on this path.
        """
        metadata = jsonio.read_json(self.cache_dir / schemas.METADATA_FILE)
        if metadata is None or metadata.get("format_version") != schemas.CACHE_FORMAT_VERSION:
            return False
        return self._fingerprint_matches(metadata.get(SNAPSHOT_FINGERPRINT_KEY))

    def _write_snapshot(self, state: SyncState) -> None:
        try:
            payload: dict[str, object] = {
                "format_version": schemas.CACHE_FORMAT_VERSION,
                "activities": _ACTIVITIES_ADAPTER.dump_python(
                    list(state.activities.values()), mode="json"
                ),
                "source_index": _SOURCE_INDEX_ADAPTER.dump_python(state.index_rows, mode="json"),
            }
        except (TypeError, ValueError) as exc:
            raise CacheError("cannot serialize cache domain records") from exc
        jsonio.write_json_atomic(payload, self.cache_dir / schemas.CACHE_FILE)
        self._cached_sync_state = (self._snapshot_stat_key(), _copy_sync_state(state))

    def _write_sync_metadata_stamp(self, *, last_sync: datetime) -> None:
        jsonio.write_json_atomic(
            {
                "format_version": schemas.CACHE_FORMAT_VERSION,
                "last_sync": last_sync.isoformat(),
                SNAPSHOT_FINGERPRINT_KEY: self._snapshot_fingerprint(),
            },
            self.cache_dir / schemas.METADATA_FILE,
        )

    def _snapshot_fingerprint(self) -> dict[str, int] | None:
        """Stat the snapshot for identity; ``None`` when it does not exist."""
        key = self._snapshot_stat_key()
        if key is None:
            return None
        size, mtime_ns, inode = key
        return {"size": size, "mtime_ns": mtime_ns, "inode": inode}

    def _snapshot_stat_key(self) -> tuple[int, int, int] | None:
        """Return the atomic snapshot's size, nanosecond mtime, and inode."""
        try:
            info = (self.cache_dir / schemas.CACHE_FILE).stat()
        except OSError:
            return None
        return info.st_size, info.st_mtime_ns, info.st_ino

    def _fingerprint_matches(self, stored: object) -> bool:
        if not isinstance(stored, dict):
            return False
        current = self._snapshot_fingerprint()
        if current is None:
            return False
        return all(stored.get(key) == value for key, value in current.items())


def _decode_sync_state(payload: object) -> SyncState:
    if (
        not isinstance(payload, dict)
        or payload.get("format_version") != schemas.CACHE_FORMAT_VERSION
    ):
        return SyncState(index_rows=[], activities={})

    raw_activities = payload.get("activities", [])
    raw_index = payload.get("source_index", [])
    if not isinstance(raw_activities, list) or not isinstance(raw_index, list):
        return SyncState(index_rows=[], activities={})
    try:
        activities = _ACTIVITIES_ADAPTER.validate_python(raw_activities)
        index_rows = _SOURCE_INDEX_ADAPTER.validate_python(raw_index)
    except ValidationError:
        return SyncState(index_rows=[], activities={})
    return SyncState(
        index_rows=index_rows,
        activities={activity.activity_id: activity for activity in activities},
    )


def _copy_sync_state(state: SyncState) -> SyncState:
    """Copy mutable containers so repository callers cannot mutate the memo."""
    return SyncState(index_rows=list(state.index_rows), activities=dict(state.activities))
