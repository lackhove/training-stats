# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project Overview

`training-stats` is a Python CLI and FastAPI REST server that:

1. Reads FIT activity files from local directories or Nextcloud via WebDAV
2. Detects new/changed files incrementally (SHA-256 content hashes)
3. Parses FIT sessions and records via `fitdecode`
4. Computes derived analytics (HR zones, estimated load, daily/weekly summaries)
5. Caches aggregates as Parquet files (never time-series data)
6. Exposes verb-based CLI commands (JSON output) and a typed REST API with
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
uv run pytest                  # tests (~224 tests, <15 s)
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
analytics/ activities, heart_rate (vectorized zones), load (hr_trimp),
           daily, weekly
cache/     schemas (Polars), parquet (atomic I/O), locking (flock), sync, repository
coach/     athlete profile (from TOML), overview assembly
server.py  service layer; methods on TrainingCoachServer are directly testable
api.py     FastAPI application with Pydantic response models (OpenAPI schema)
cli.py     cyclopts CLI (JSON output); app.meta.default consumes global --config
config.py  Pydantic TOML models, env-based secret resolution
models.py  shared frozen dataclasses (SourceFile, SyncResult, summaries, Overview, ...)
errors.py  typed exception hierarchy (TrainingStatsError base)
```

Data flow: `sources -> fit.parser -> analytics -> cache.sync -> cache.repository -> server -> (api, cli)`.

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
  averages) are allowed in `activity_metrics.parquet` and API responses.
  Schemas in `cache/schemas.py` enforce this.
- **Secrets only from environment variables.** The TOML file stores the
  *names* of env vars (`username_env`/`password_env`). Values are wrapped in
  `SecretStr`/`WebDAVCredentials` and must never leak into logs, `repr`,
  exceptions, or cache files.
- **Atomic cache writes only.** All Parquet/JSON writes go through
  `cache/parquet.py` (tmp file + fsync + `os.replace`). Sync runs under
  `cache/locking.py`'s flock. Don't write cache files directly.
- **Incremental sync via hashes.** Change detection uses SHA-256 hashes in
  `source_index.parquet`, not modification times. Bumping the internal
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
- Cache tests create isolated `tmp_path` directories; the flock test
  requires a POSIX OS.
- `weekly`/`overview` never synchronize; only `list`/`GET /activities` do
  (`--no-sync` opts out) plus startup sync in `training-stats server`. CLI tests
  must populate the cache via `list` before exercising weekly/overview.
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
  depend on it for record data.
- Polars `to_dicts()` returns `dict[str, object]`; use the module-level
  `_optional_float`/`_int`/`_str` converter helpers in `cache/sync.py` and
  `cache/repository.py` when reading values back. Same pattern for
  `Series.sum()`/`min()`/`max()` results — guard with `isinstance` checks
  (see `analytics/heart_rate.py`).

## Environment

- Python >= 3.12; managed and run exclusively via `uv` (no pip, no venv
  activation — use `uv run ...`).
- WebDAV secrets in tests come from `monkeypatch.setenv`, never from a real
  `.env` file.
- `data/` (cache output) and `config.toml` are gitignored; never commit
  caches, credentials, or personal FIT data.
