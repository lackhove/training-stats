# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project Overview

`training-stats` is a Python CLI and FastAPI REST server that:

1. Reads FIT activity files from local directories or Nextcloud via WebDAV
2. Detects new/changed files incrementally (file size + modification time)
3. Decodes FIT records, timer events, and sport metadata via `fitdecode`
4. Repairs record gaps and computes derived analytics (HR zones, estimated load,
   daily/weekly summaries)
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
uv run pytest                  # tests (~300 tests, <25 s)
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
domain/       models and settings; analytics; normalized FIT types and
              timeline repair; canonical Polars frame contracts
services/     use-case orchestration (TrainingStatsService, SyncService,
              FileProcessor), typed state and ports (source/parser/repository)
adapters/     cache/ (Parquet, serialization, locking), fit/ (fitdecode),
              sources/ (local, WebDAV), config loading, cli/, http/, wire,
              and bootstrap composition root
__main__.py   delegates to adapters.cli.app
```

Dependency direction: `domain <- services <- adapters`; only the adapter
bootstrap wires concrete implementations into service ports. Domain analytics
may use Polars, but has no imports from adapters or services.

The cache adapter exposes lazy scans using the domain frame contract; analytics
owns user-facing filtering, aggregation, and mapping into domain models.
Parquet schemas and nested-value serialization stay inside `adapters/cache`.
Domain analytics are pure (`LazyFrame → dataclass rows` / `rows → rows`) — no
I/O, no locking.

**Serialization convention:** services return domain models; serialize only
at the edges. `adapters/http` and `adapters/cli` map domain → `adapters/wire.py` Pydantic models
(the anti-corruption layer for HTTP/CLI JSON). Never leak domain shapes into
REST/CLI payloads, and never pass wire models into the service layer.

## Critical Invariants

These are core design decisions — do not violate them casually:

- **FIT parsing is decode-only; analysis owns statistics.** The parser returns
  a thin `FitData` wrapper around normalized raw messages. Session aggregates
  are ignored; only raw sport/sub-sport metadata is retained. The service calls
  pure `domain/fit/repair.py` gap segmentation, then `domain/analytics/` derives
  all activity statistics. Gaps split at configurable thresholds (default
  `dt > 900 s` or adjacent distance change `> 500 m`); pauses are separately
  identified from timer events or speed, count toward elapsed duration, and are
  excluded from statistics. Segments under 300 s moving time are discarded.
- **Calories are not part of the product.** Do not add calorie fields to FIT
  parsing, analysis, caches, domain models, or REST/CLI payloads.

- **No raw time-series persistence.** Record data lives as a transient Polars
  frame (`domain/fit/types.py` `FitData.SCHEMA`) during processing and is discarded.
  Raw records must never appear in Parquet files, REST/CLI responses, or logs.
  Coarse derived aggregates (5-bpm HR histogram, 10-minute weighted HR
  averages) are allowed in `training_cache.parquet` and API responses.
  Schemas in `adapters/cache/schemas.py` enforce this.
- **No stored period aggregates.** Daily/weekly summaries and training form
  are derived on read by `domain/analytics/` over the lazy activity frame. Only the
  unified cache file is persisted (`training_cache.parquet`: one row per
  accepted activity, enriched with its source-index fields, plus rows for
  failed/duplicate/source-only outcomes); `metadata.json` holds the sync
  throttle stamp.
- **Secrets only from environment variables.** The TOML file stores the
  *names* of env vars (`username_env`/`password_env`). Values are wrapped in
  `SecretStr`/`WebDAVCredentials` and must never leak into logs, `repr`,
  exceptions, or cache files.
- **Atomic cache writes only.** All Parquet/JSON writes go through
  `adapters/cache/parquet.py` (tmp file + fsync + `os.replace`). Exactly one
  process may operate on the cache: `adapters/cache/locking.py`'s app-wide
  exclusive flock is acquired once per process at the CLI entry point
  (`adapters/cli/app.py`) and held
  until process exit. Repository, sync, analytics and services code never
  locks; within a process, atomic `os.replace` makes every read see one
  consistent file version. Don't write cache files directly.
- **Incremental sync via file metadata.** Unchanged files are skipped
  without being read when `file_size` + `modified_at` match the previous
  sync and a content hash is stored (the prior read succeeded; stored on
  the cache row); downloaded candidates are reprocessed
  unconditionally — FIT files are only ever added, deleted or rewritten,
  and a rewrite changes size or modification time. Rows without a stored
  hash (failed reads) always retry. Bumping the internal
  `ANALYSIS_VERSION_VALUE` (`adapters/cache/schemas.py`) forces full reprocessing.
- **Load is an estimate, never "TSS".** The method is `hr_trimp` (Banister),
  stored with version and confidence. Don't rename it or drop the metadata.
- **Deterministic analytics.** HR zones and athlete values come from
  configuration, never hardcoded.
- **Typed errors.** Use the layer-owned exception types under `domain/errors.py`,
  `services/errors.py`, and `adapters/errors.py`; per-file FIT failures are
  recorded as source-only rows in the cache, they don't abort the sync.

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
  `adapters/fit/parser.py`; keep cyclopts usage isolated to
  `adapters/cli/app.py` (cyclopts and
  uvicorn ship type information, no override needed).
- Polars is used both for transient record frames (`domain/fit/types.py`) and
  for cache/storage (`adapters/cache/`); it is intentional that `domain/fit/`
  and `domain/analytics/`
  depend on it. Aggregation entry points take `pl.LazyFrame` so consumers
  decide column projection; the repository returns lazy scans.
- Polars `to_dicts()` returns `dict[str, object]`; use the safe converters
  (`optional_float`, `optional_str`, …) and module-level `_float`/`_int`
  helpers in `adapters/cache/repository.py` and `domain/analytics/daily.py`
  when reading values back. Same pattern for
  `Series.sum()`/`min()`/`max()` results — guard with `isinstance` checks
  (see `domain/analytics/heart_rate.py`).

## Environment

- Python >= 3.12; managed and run exclusively via `uv` (no pip, no venv
  activation — use `uv run ...`).
- WebDAV secrets in tests come from `monkeypatch.setenv`, never from a real
  `.env` file.
- `data/` (cache output) and `config.toml` are gitignored; never commit
  caches, credentials, or personal FIT data.
