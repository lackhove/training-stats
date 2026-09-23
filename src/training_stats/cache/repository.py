"""Repository abstracting Parquet access from the service layers.

Read side (``TrainingRepository``): lazy Polars scans of the unified cache
file — one row per accepted activity enriched with its source-index fields,
plus rows for failed and duplicate source outcomes. No filtering, aggregation
or row mapping lives here: :mod:`training_stats.analytics` turns the frames
into domain models. Write side (``SyncState`` load/save): the storage
boundary for ``cache/sync.py``, so the storage engine stays swappable behind
this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import polars as pl

from . import parquet, schemas


@dataclass(frozen=True)
class SyncState:
    """Sync working set: index rows plus the merged activity store.

    The dict is mutated in place by the sync loop; ``index_rows`` is
    replaced wholesale via a new :class:`SyncState` when the snapshot is
    saved (rows of deleted files are simply not carried over).
    """

    index_rows: list[dict[str, object]]
    activities: dict[str, dict[str, object]]


class TrainingRepository(Protocol):
    """Storage interface used by the service layer."""

    def activities_scan(self) -> pl.LazyFrame: ...

    def source_index_scan(self) -> pl.LazyFrame: ...

    def last_sync_stamp(self) -> datetime | None: ...


class ParquetTrainingRepository:
    """Parquet-backed repository. Reads are lazy scans of the unified cache."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir

    def cache_scan(self) -> pl.LazyFrame:
        """Lazy scan of the unified cache file (activity and source-only rows)."""
        return self._scan(schemas.CACHE_FILE, schemas.CACHE_SCHEMA)

    def activities_scan(self) -> pl.LazyFrame:
        """Lazy scan of accepted activities (rows with a non-null ``start_time``).

        Returns an empty frame (with the full activity schema) when the cache
        file does not exist yet; the consumer decides projection and collect.
        """
        return (
            self.cache_scan()
            .filter(pl.col("start_time").is_not_null())
            .select(schemas.ACTIVITY_COLUMNS)
        )

    def source_index_scan(self) -> pl.LazyFrame:
        """Lazy scan of source-index fields for all rows (ok, failed, duplicate)."""
        return self.cache_scan().select(schemas.INDEX_COLUMNS)

    def last_sync_stamp(self) -> datetime | None:
        """Persisted ``last_sync`` from ``metadata.json`` (cross-process throttle)."""
        metadata = parquet.read_json(self.cache_dir / schemas.METADATA_FILE)
        last_sync = metadata.get("last_sync") if metadata else None
        if not isinstance(last_sync, str):
            return None
        try:
            stamp = datetime.fromisoformat(last_sync)
        except ValueError:
            return None
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=UTC)
        return stamp

    # -- sync write side ----------------------------------------------------

    def load_sync_state(self) -> SyncState:
        """Read the sync working set: source-index rows plus the activity store."""
        frame = parquet.read_parquet(self.cache_dir / schemas.CACHE_FILE)
        if frame.is_empty():
            return SyncState(index_rows=[], activities={})
        rows = frame.to_dicts()
        return SyncState(
            index_rows=[{name: row.get(name) for name in schemas.INDEX_COLUMNS} for row in rows],
            activities={
                str(row["activity_id"]): row
                for row in rows
                if row.get("start_time") is not None and row.get("activity_id") is not None
            },
        )

    def save_sync_snapshot(
        self,
        state: SyncState,
        *,
        last_sync: datetime,
    ) -> None:
        """Atomically rewrite the unified cache from the sync working set.

        Accepted activities (``ok`` index rows with a matching activity row)
        are enriched with their source-index fields; failed and duplicate
        outcomes are kept as rows with null activity fields so change
        detection and data quality keep working. ``last_sync`` goes to
        ``metadata.json``. Aggregates are derived on read (see
        :mod:`training_stats.analytics`), never persisted.
        """
        ok_ids = {
            str(row["activity_id"])
            for row in state.index_rows
            if row.get("status") == schemas.STATUS_OK and row.get("activity_id") is not None
        }
        activities = {key: row for key, row in state.activities.items() if key in ok_ids}

        rows: list[dict[str, object]] = []
        for index_row in state.index_rows:
            activity_id = index_row.get("activity_id")
            activity_row = (
                activities.get(str(activity_id))
                if activity_id is not None and index_row.get("status") == schemas.STATUS_OK
                else None
            )
            row: dict[str, object] = {name: index_row.get(name) for name in schemas.CACHE_SCHEMA}
            if activity_row is not None:
                for name, value in activity_row.items():
                    if name in schemas.CACHE_SCHEMA:
                        row[name] = value
            rows.append(row)

        parquet.write_parquet_atomic(_cache_frame(rows), self.cache_dir / schemas.CACHE_FILE)
        parquet.write_json_atomic(
            {"last_sync": last_sync.isoformat()},
            self.cache_dir / schemas.METADATA_FILE,
        )

    # -- helpers -----------------------------------------------------------

    def _scan(self, filename: str, schema: dict[str, pl.DataType]) -> pl.LazyFrame:
        path = self.cache_dir / filename
        if not path.exists():
            return pl.DataFrame(schema=schema).lazy()
        return pl.scan_parquet(path)


def _cache_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=schemas.CACHE_SCHEMA)
    return pl.DataFrame(rows, schema=schemas.CACHE_SCHEMA)
