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


class JsonTrainingRepository(SyncRepository):
    """Atomically persist one metric-bearing record per accepted activity."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir

    def load_sync_state(self) -> SyncState:
        """Load the cache snapshot; an absent or obsolete format rebuilds empty."""
        payload = jsonio.read_json(self.cache_dir / schemas.CACHE_FILE)
        if payload is None or payload.get("format_version") != schemas.CACHE_FORMAT_VERSION:
            return SyncState(index_rows=[], activities={})

        raw_activities = payload.get("activities", [])
        raw_index = payload.get("source_index", [])
        if not isinstance(raw_activities, list) or not isinstance(raw_index, list):
            raise CacheError("cache JSON has invalid activity/source-index collections")
        try:
            activities = _ACTIVITIES_ADAPTER.validate_python(raw_activities)
            index_rows = _SOURCE_INDEX_ADAPTER.validate_python(raw_index)
        except ValidationError as exc:
            raise CacheError("cache JSON contains an invalid domain record") from exc
        return SyncState(
            index_rows=index_rows,
            activities={activity.activity_id: activity for activity in activities},
        )

    def save_sync_snapshot(self, state: SyncState, *, last_sync: datetime) -> None:
        """Write records and sync stamp atomically at their respective boundaries."""
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
        self.save_sync_stamp(last_sync=last_sync)

    def save_sync_stamp(self, *, last_sync: datetime) -> None:
        """Record a successful sync stamp without serializing the snapshot."""
        jsonio.write_json_atomic(
            {
                "format_version": schemas.CACHE_FORMAT_VERSION,
                "last_sync": last_sync.isoformat(),
            },
            self.cache_dir / schemas.METADATA_FILE,
        )

    def last_sync_stamp(self) -> datetime | None:
        """Read the throttle stamp from metadata alone; never open the snapshot."""
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
        return stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=UTC)
