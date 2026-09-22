# Plan: Move Direct Parquet Access Out of `sync.py` Into the Repository Layer

> Status: implemented and overtaken by later work — the write side
> (`SyncState`/`load_sync_state`/`save_sync_snapshot`) lives in
> `cache/repository.py` today, derived tables were dropped, and the
> layering moved further (lazy-frame analytics, coach merged into
> analytics, app-wide lock). See `AGENTS.md` for the current architecture.

## Goal

`cache/sync.py` should contain orchestration only (source iteration, change
detection, locking, counters). All direct Parquet I/O — reads, writes, frame
building, and row↔domain hydration — moves to `cache/repository.py`. This
establishes the repository as the storage boundary so a later engine swap
(e.g. SQLite) stays a repo-internal change.

No SQLite/DuckDB migration in this plan. This is a pure layering refactor.

## Context

- Commit `ee31499` already extracted `FileProcessor` into `ingest.py` and
  injected it into `SyncService` (`sync.py` is ~486 lines, down from 645).
- `sync.py` still owns all cache I/O: loading index/stores, writing 5 Parquet
  files + `metadata.json`, hydrating `AnalyzedActivity` from rows, building
  Polars frames.
- `repository.py` is currently read-only (`ParquetTrainingRepository` +
  `TrainingRepository` Protocol used by `services.py`/`coach/`).
- One leak outside the cache package: `services.py:85` calls
  `read_json(metadata.json)` directly for the sync throttle.

## What Moves to `repository.py`

| Piece | Lines (sync.py) | Nature |
|---|---|---|
| `_load_index` / `_load_store` | 228–253 | Direct Parquet reads |
| `_write_cache` | 312–370 | Orphan prune + hydrate + aggregate + 5× `write_parquet_atomic` + metadata |
| `_rebuild_analyzed` | 372–423 | Row→`AnalyzedActivity` hydration (mirror of repo's `_details_row`) |
| `_frame` / `_index_frame` / `_daily_records` / `_weekly_records` | 426–486 | Schema-aware frame builders |
| `services.py` metadata read | 85 | `read_json` for `last_sync` throttle |

## Proposed Repository API

```python
def load_sync_state(self, *, reset: bool = False) -> SyncState: ...


def save_sync_snapshot(
    self, state: SyncState, *, last_sync: datetime, analysis_version: str
) -> None: ...  # prune orphans → hydrate ok rows → daily/weekly → atomic writes


def last_sync_stamp(self) -> datetime | None: ...  # replaces services' read_json
```

With a small frozen dataclass for the sync working set:

```python
@dataclass
class SyncState:
    index_rows: list[dict[str, object]]
    activities: dict[str, dict[str, object]]
    metrics: dict[str, dict[str, object]]
```

`save_sync_snapshot` absorbs the whole `_write_cache` body including
`summarize_daily`/`daily_to_weekly` — the repo already owns the timezone and
sits next to the aggregate files.

## What Stays in `sync.py`

- Orchestration: `sync()`, `rebuild_cache()`, `_run_sync()`
- Two-tier change detection: `_previous_rows`, `_unchanged`,
  `_metadata_unchanged`, `_same_size`, `_same_time`
- `_purge_previous`, `_failure_row`, `_read_content`
- Locking (`sync_lock` around the whole operation) — needs `cache_dir` +
  `lock_timeout_seconds`
- Counters and `SyncResult` building

## Dependency Injection

- One shared `ParquetTrainingRepository` instance wired into both
  `SyncService` and `TrainingStatsService` (`services.py:create_service`).
- `SyncService` keeps `cache_dir` + `lock_timeout_seconds` only for
  `sync_lock`; gains a `repository` parameter, drops nothing else if the
  aggregate recompute moves with the save path.
- Narrow protocol (e.g. `SyncCacheRepository`) for sync writes so
  `TrainingRepository` stays read-only for `service/coach`:

```python
class SyncCacheRepository(Protocol):
    def load_sync_state(self, *, reset: bool = False) -> SyncState: ...
    def save_sync_snapshot(
        self, state: SyncState, *, last_sync: datetime, analysis_version: str
    ) -> None: ...
```

Import direction is safe: `sync.py` importing `repository.py` creates no
cycle (repository never imports sync).

## Net Effect

- Zero `parquet.*` calls outside `repository.py` (and `parquet.py` itself).
- `sync.py` drops ~200 lines of I/O/serialization.
- Storage engine becomes swappable behind the repository.

## Open Questions

1. **Write-path scope:**
   - **A (recommended)** — full `save_sync_snapshot`: repo also hydrates and
     recomputes daily/weekly. Sync just mutates state and calls save.
     Maximum simplification; puts aggregate orchestration in the repo.
   - **B (thinner)** — repo only does prune + raw frame writes; sync still
     hydrates via repo and calls `summarize_daily`/`daily_to_weekly` itself,
     passing daily/weekly rows in. Slightly clearer layering (analytics stays
     out of repo), sync keeps ~30 more lines.
2. **Include the `services.py` metadata read** (`last_sync_stamp` on repo) in
   this same change, or leave it for a follow-up?

## Test Impact

- `tests/test_cache.py` `make_sync()` must construct a repository and pass it
  into `SyncService`.
- `tests/test_service.py` metadata/`last_sync` throttle tests: update to use
  the repository method (test setup may still write `metadata.json` directly).
- Tests reading Parquet files directly for assertions may keep doing so
  (test-level access is fine).
- All quality gates must pass:
  `uv run ruff format --check . && uv run ruff check . && uv run mypy --strict src && uv run pytest`

## Docs to Update

- `doc/how-it-works.md` §4.1 (sync flow) and §4.2 (cache files).
- `AGENTS.md` architecture sketch (repository gains write side; sync no
  longer touches Parquet directly).

## Explicitly Out of Scope

- SQLite/any storage engine migration.
- Changes to change-detection semantics.
- Further changes to `ingest.py`/`FileProcessor`.