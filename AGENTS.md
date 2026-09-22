# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project Overview

`training-stats` is a Python CLI and FastAPI REST server that:

1. Reads FIT activity files from local directories or Nextcloud via WebDAV
2. Detects new/changed files incrementally (file size + modification time)
3. Parses FIT sessions and records via `fitdecode`
4. Computes derived analytics (HR zones, estimated load, daily/weekly summaries)
5. Caches per-activity facts + derived metrics as Parquet (never time-series
   data; period aggregates are derived on read, not persisted)
6. Exposes verb-based CLI commands (Rich tables on a TTY, JSON otherwise)
   and a typed REST API with
   OpenAPI schema (consumable as an Open WebUI external OpenAPI tool server)

Architecture details and design decisions are captured below; user-facing
setup instructions in `README.md`.

## Commands

All commands run through `uv`. Quality gates (run all four before finishing
any task; the code is only complete when all of them pass):

```bash
uv run ruff format --check .   # formatting (run `uv run ruff format .` to fix)
uv run ruff check .            # linting (run with `--fix` to auto-fix)
uv run mypy --strict src       # static typing (src only, not tests)
uv run pytest                  # tests (~285 tests, <25 s)
```

Combined one-liner:

```bash
uv run ruff format --check . && uv run ruff check . && uv run mypy --strict src && uv run pytest
```

Other useful commands:

```bash
uv sync                        # install/sync dependencies after pyproject changes
uv run training-stats list --limit 5    # CLI smoke test (needs config.toml + secrets)
uv run training-stats server            # run the REST API locally
uv run pytest tests/test_cache.py -x -q   # single test file
```

## Architecture

Package lives in `src/training_stats/` (uv build backend, `uv_build`).

```
sources/   ActivitySource ABC -> LocalDirectorySource, WebDAVSource (httpx)
fit/       FitParser protocol (returns ParsedFile), FitDecodeParser,
           repair (clock shifts + segment splits), normalize helpers,
           RECORD_SCHEMA (records as transient Polars frame)
analytics/ activities (metrics from records), heart_rate (vectorized zones),
           load (hr_trimp), endurance, daily (frame group-by + row selection),
           weekly, fitness (training form), overview + profile (pure
           Overview assembly — the former coach/ package)
ingest.py  FileProcessor: parse one file, compute metrics, build merged
           Parquet row dicts (domain half of sync, injected into cache/sync.py)
cache/     schemas (Polars), parquet (atomic I/O), locking (app-wide flock),
           sync (orchestration + change detection), repository (storage
           boundary: lazy activities_scan + row mapping + sync state
           load/save; the only module touching Parquet directly)
services.py  service layer orchestrating repository reads and pure analytics
             calls; methods on TrainingStatsService are directly testable
wire.py    Pydantic wire/response models + domain→wire mappers, shared by
           api.py (response_model) and cli.py (--json) so REST and CLI JSON
           stay identical (anti-corruption layer at the edges)
api.py     FastAPI application with wire response models (OpenAPI schema)
cli.py     cyclopts CLI (TTY detection, global --config/--json); maps domain
           → wire, dispatches to cli_output (Rich) or plain JSON
cli_output.py  Rich renderers taking wire dicts (model_dump) + injected
               Console (the only module with rendering logic)
config.py  Pydantic TOML models, env-based secret resolution
models.py  shared frozen dataclasses (SourceFile, SyncResult, summaries,
           deep Overview, ThrottledSync, ...)
errors.py  typed exception hierarchy (TrainingStatsError base)
```

Data flow: `sources -> ingest (fit.parser + analytics) -> cache.sync -> cache.repository -> services (analytics over activities_scan) -> (wire -> api, cli)`.

The read boundary: the repository exposes `activities_scan() -> pl.LazyFrame`
plus row-mapped queries; **all** aggregation lives in `analytics/`, which the
service layer is the only module allowed to call. Analytics functions are
pure (`LazyFrame → dataclass rows` / `rows → rows`) — no I/O, no locking.

**Serialization convention:** services return domain models; serialize only
at the edges. `api.py` and `cli.py` map domain → `wire.py` Pydantic models
(the anti-corruption layer for HTTP/CLI JSON). Never leak domain shapes into
REST/CLI payloads, and never pass wire models into the service layer.

## Critical Invariants

These are core design decisions — do not violate them casually:

- **Session headers are never trusted blindly.** Start time and duration are
  re-derived from the repaired record timeline (`fit/repair.py`): clock
  jumps shift static prefixes onto the post-fix timeline, recording breaks
  split into segment activities, and implausible session elapsed time falls
  back to the record span.

- **No raw time-series persistence.** Record data lives as a transient Polars
  frame (`fit/types.py` `RECORD_SCHEMA`) during processing and is discarded.
  Raw records must never appear in Parquet files, REST/CLI responses, or logs.
  Coarse derived aggregates (5-bpm HR histogram, 10-minute weighted HR
  averages) are allowed in `activities.parquet` and API responses.
  Schemas in `cache/schemas.py` enforce this.
- **No stored period aggregates.** Daily/weekly summaries and training form
  are derived on read by `analytics/` over the lazy activity frame. Only
  per-activity rows (`activities.parquet`) and the source index are
  persisted; `metadata.json` holds the sync throttle stamp.
- **Secrets only from environment variables.** The TOML file stores the
  *names* of env vars (`username_env`/`password_env`). Values are wrapped in
  `SecretStr`/`WebDAVCredentials` and must never leak into logs, `repr`,
  exceptions, or cache files.
- **Atomic cache writes only.** All Parquet/JSON writes go through
  `cache/parquet.py` (tmp file + fsync + `os.replace`). Exactly one process
  may operate on the cache: `cache/locking.py`'s app-wide exclusive flock is
  acquired once per process at the CLI entry point (`cli._service`) and held
  until process exit. Repository, sync, analytics and services code never
  locks; within a process, atomic `os.replace` makes every read see one
  consistent file version. Don't write cache files directly.
- **Incremental sync via file metadata.** Unchanged files are skipped
  without being read when `file_size` + `modified_at` match the previous
  sync and a content hash is stored (the prior read succeeded; stored in
  `source_index.parquet`); downloaded candidates are reprocessed
  unconditionally — FIT files are only ever added, deleted or rewritten,
  and a rewrite changes size or modification time. Rows without a stored
  hash (failed reads) always retry. Bumping the internal
  `ANALYSIS_VERSION_VALUE` (`cache/schemas.py`) forces full reprocessing.
- **Load is an estimate, never "TSS".** The method is `hr_trimp` (Banister),
  stored with version and confidence. Don't rename it or drop the metadata.
- **Deterministic analytics.** HR zones and athlete values come from
  configuration, never hardcoded.
- **Typed errors.** Use the exception hierarchy in `errors.py`; per-file FIT
  failures are recorded in the source index, they don't abort the sync.

## Code Style

- Python 3.12+, fully type-annotated; `mypy --strict` must stay green.
- Line length 100 (Ruff-configured); double quotes; Ruff handles formatting —
  never hand-format.
- Ruff rule set: E, F, I, UP, B, SIM, N, ANN, RUF (ANN401 ignored).
- No comments unless they explain *why*; docstrings for public API.
- Prefer stdlib (`tomllib`, `dataclasses`, `itertools.pairwise`, `zoneinfo`).

## Testing

- Fixtures and helpers live in `tests/conftest.py`:
  `StubSource` (in-memory source), `StubParser` (deterministic, parses a
  `start|sport|duration|distance[,hr;hr;...]` text grammar via
  `stub_activity_content`; one activity per line for multi-segment files),
  `make_record_frame` (builds record frames), `make_app_config`.
- FIT fixtures in `tests/fixtures/` (`cycling.fit`, `activity_2.fit`) are
  synthetic, privacy-safe files (no GPS, serials, or body data). Do not add
  real athlete FIT files.
- WebDAV tests use `httpx.MockTransport` — never a live server.
- Trend/current-week tests must anchor fixture dates relative to
  `datetime.now()` so time-window tools stay valid over time.
- Cache tests create isolated `tmp_path` directories; the app-lock conflict
  test requires a POSIX OS.
- Query operations synchronize "if due": a real sync runs only when the
  last successful one is older than `cache.sync_min_interval_seconds`
  (default 10 s; two-layer stamp: in-memory monotonic + persisted
  `last_sync` in `metadata.json`; failed syncs never stamp). Applies to
  `list`/`weekly`/`overview` and `GET /activities`, `/weekly`, `/overview`;
  `show`/`GET /activities/{id}` serve the cache as-is. `--no-sync` opts
  out; startup sync in `training-stats server` is unconditional. CLI tests
  must exercise weekly/overview against an empty cache directly (they
  self-synchronize).
- New behavior needs tests; bug fixes need a regression test first.
- `test_fit_parser.py` contains golden zone values for both fixtures —
  they lock the vectorized `time_in_zones` to the original loop semantics
  (intervals belong to their *starting* record).

## Type Checking Notes

- `mypy --strict` is configured with one narrow override: `fitdecode` has no
  type stubs (`ignore_missing_imports`). Keep fitdecode usage isolated to
  `fit/parser.py`; keep cyclopts usage isolated to `cli.py` (cyclopts and
  uvicorn ship type information, no override needed).
- Polars is used both for transient record frames (`fit/types.py`) and for
  cache/storage (`cache/`); it is intentional that `fit/` and `analytics/`
  depend on it. Aggregation entry points take `pl.LazyFrame` so consumers
  decide column projection; the repository returns lazy scans.
- Polars `to_dicts()` returns `dict[str, object]`; use the converters
  (`optional_float`, `optional_str`, `histogram_from_json`, …) and the
  module-level `_float`/`_int` helpers in `cache/repository.py` and
  `analytics/daily.py` when reading values back. Same pattern for
  `Series.sum()`/`min()`/`max()` results — guard with `isinstance` checks
  (see `analytics/heart_rate.py`).

## Environment

- Python >= 3.12; managed and run exclusively via `uv` (no pip, no venv
  activation — use `uv run ...`).
- WebDAV secrets in tests come from `monkeypatch.setenv`, never from a real
  `.env` file.
- `data/` (cache output) and `config.toml` are gitignored; never commit
  caches, credentials, or personal FIT data.
