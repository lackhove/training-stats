# How training-stats works

Documentation of the architecture, the data-cleaning pipeline, processing
steps and all metric definitions.

- Sources: local directories and Nextcloud via WebDAV (read FIT files only)
- Language: Python 3.12+, everything managed through `uv`
- Cache: atomic JSON records at ActivityRecord grain (per-activity metrics,
  never raw time series; period aggregates derived on read)
- Interface: verb-based CLI (Rich tables on a TTY, JSON when piped or via
  `--json`) and FastAPI REST API with OpenAPI schema

---

## 1. Architecture & Data Flow

```
adapters/sources + adapters/fit -> services -> domain analytics
                 adapters/cache <- services
adapters/bootstrap wires services to adapters; adapters/http and adapters/cli
serialize generic MetricReport values at the output boundary.
```

| Package | Responsibility |
|---|---|
| `domain/` | Domain dataclasses/settings, normalized FIT types, timeline repair, Metric types, and pure analytics |
| `services/` | Ports and use cases (`TrainingStatsService`, `SyncService`, `FileProcessor`); no concrete adapters |
| `adapters/cache/` | Domain-record JSON codec, atomic I/O, repository implementation and app-wide lock |
| `adapters/fit/` | `FitDecodeParser` and raw FIT value normalization |
| `adapters/sources/` | `LocalDirectorySource` and `WebDAVSource` implementations |
| `adapters/config.py` | TOML/environment loading and secret resolution; maps domain athlete/quality settings |
| `adapters/http/` | FastAPI application and auth |
| `adapters/cli/` | cyclopts CLI; generic shape-based Rich rendering and stderr diagnostics |
| `adapters/wire.py` | Generic Pydantic metric/report transport models shared by HTTP and CLI JSON |
| `adapters/bootstrap.py` | Composition root wiring concrete adapters to application-service ports |
| `errors.py` | Typed exception hierarchy (`TrainingStatsError` base) |

**Core invariants**

1. Session headers are never trusted blindly — start time and duration are
   re-derived from the repaired record timeline (see §3).
2. No raw time-series persistence — record samples live in a transient
   Polars frame during processing and are discarded. Only coarse derived
   aggregates are cached (5-bpm HR histogram, 10-min weighted HR averages,
   binned heatmaps). Period aggregates (daily/weekly, training form) are
   **never persisted**; they are derived on read by `domain/analytics/`.
3. Secrets only from environment variables (`SecretStr`, never logged).
4. Atomic JSON cache writes only (tmp file + fsync + `os.replace`); exactly one
   process operates on the cache at a time via the app-wide flock
    (`adapters/cache/locking.py`), held for the process lifetime.
5. Incremental sync: size + mtime skip re-reads; downloaded candidates are
   reprocessed (FIT files are only ever added, deleted or rewritten, and a
   rewrite changes size or mtime).
6. Load is an estimate (`hr_trimp`), stored with method, version and confidence.
7. Deterministic analytics — HR zones and athlete values come from configuration.
8. Typed errors; per-file FIT failures are recorded in the source index and
   never abort a sync.

---

## 2. Configuration

Everything non-secret lives in a TOML file (see `config.example.toml`):

- `[application]` — name, log level, IANA timezone (all local-date grouping uses it)
- `[athlete]` — max/resting heart rate, optional weight/FTP/goal, and
  `heart_rate_zones` (five validated, non-overlapping `(min, max)` bpm bounds)
- `[[sources]]` — one or more `local` or `webdav` sources; WebDAV
  credentials are referenced by *environment variable name*
- `[quality]` — repair, sensor-coverage, and speed-threshold settings
- `[cache]` — directory and lock timeout

The analysis version (see §10) is owned by the code, not the config: it is
the reprocessing switch — when parsing or metric semantics change, the
version is bumped and the next sync reprocesses every file (current: `"6"`).

---

## 3. Ingestion and Data Cleaning

### 3.1 Discovery & Change Detection

1. Each source adapter lists FIT files (`*.fit`, case-insensitive; symlinks are
   skipped unless `follow_symlinks`).
2. Files whose `file_size` and `modified_at` match the previous sync are
   skipped without being read at all (metadata short-circuit).
3. Remaining files are read and reprocessed unconditionally; the SHA-256
   content hash and the source-level `analysis_version` are recorded on the
   cache row (the hash marks successful reads).
4. Files whose read previously failed (no stored hash) always retry.
5. Bumping the internal `ANALYSIS_VERSION_VALUE` (`adapters/cache/schemas.py`)
   forces full reprocessing of every file.

### 3.2 Why Cleaning Is Necessary

The FIT adapter decodes normalized raw record samples, timer events, and
sport/sub-sport metadata into a thin `FitData(frame)` wrapper. It ignores all
session aggregate fields.

### 3.3 Repair and segmentation (`domain/fit/repair.py`)

`FileProcessor` calls the pure repair function after parsing. Repair stably
sorts record rows and splits the frame when adjacent record timestamps differ
by **more than 15 minutes** (`activity_gap_threshold_s = 900`) or their
cumulative distance readings differ by **more than 500 m**
(`activity_distance_gap_threshold_m = 500`). Both thresholds are configurable.
Repair returns a list of `FitData` wrappers, one per candidate segment, and
does not alter timestamps. A pause under 15 minutes remains within a segment.

Analysis identifies pauses from FIT timer stop/start events; if timer events
are unavailable, it uses record speed below `moving_speed_threshold_mps` as
the fallback. Pauses remain part of elapsed duration but do not contribute to
statistics. Analysis drops every segment with less than
`minimum_segment_duration_s = 300` seconds of moving time, including when all
segments are short. Such files are indexed as successful source-only outcomes
and are not retried on every sync.

The `forgotten_stop` diagnostic remains: an activity is flagged when duration
exceeds 24 hours, or moving-time ratio is below the configured minimum for
activities longer than two hours. Flagged activities are retained.

**Repair/hygiene flags:** `segments_discarded` and
`non_monotonic_timestamps` are file-level; `forgotten_stop` is per activity.
They are exposed in activity details and data-quality summaries.

### 3.4 Statistics (`domain/analytics/`)

Analytics is a collection of stateless functions. Per-trip functions accept a
`ParsedActivity` (including its transient Polars record frame) and return one
or more immutable `Metric` objects. Public summary functions compose these
metric functions and return `MetricReport`; services provide cache records and
configuration and perform sync/validation only. FIT session aggregates are not
used as fallbacks. Start time is the first record timestamp; duration is the
record span and therefore includes pauses. Moving time excludes pauses, and
`paused_time_s = duration_s − moving_time_s`. Distance and elevation gain are
derived from active record intervals. Sport and sub-sport come from raw
session metadata.

Time-weighted averages, HR zones, histograms, HR buckets, load, aerobic
decoupling, and maxima use active intervals/samples only. HR time buckets use
an active-time axis, so pauses do not shift later active samples into later
statistical windows. The 30-second per-record interval cap guards against
sparse samples.

### 3.5 Active Time from Records

When timer events are present, their stop/start pairs define pause intervals.
Otherwise, speed readings classify intervals below
`moving_speed_threshold_mps` as paused when at least half of intervals have a
speed reading. If neither source provides pause evidence, the segment is
treated as active for its full record span.

### 3.6 Activity Identity and Duplicates

`activity_id = sha256(canonical-UTC start | sport | duration_s | segment_index)[:16]`

- IDs are deterministic across sources, re-downloads and reprocessing.
- **Exact duplicate detection:** when two files produce the same
  `activity_id` (typical: the same ride exported by both watch and app),
  the later file is marked `duplicate` in the source index and its rows are
  excluded from the activity store and API
  responses. Fuzzy near-duplicate detection (different clocks producing
  slightly different IDs) is explicitly out of scope.

---

## 4. Processing: Sync & Cache

### 4.1 Sync Flow (`services/sync.py` + `services/ingest.py` + `adapters/cache/repository.py`)

1. Load source index + per-activity metric records (in memory) via
   `repository.load_sync_state()`.
2. List all files from all enabled sources.
3. Skip files matching size + mtime + version from the previous sync
   (no read); read + reprocess the rest.
4. On change: **purge all cached rows of that file** (every segment), then
   delegate to `ingest.FileProcessor` — parse, repair, compute metrics,
   index one row per segment.
5. Duplicate segments are marked `status="duplicate"` and retained as
   source-index outcomes without an ActivityRecord.
6. `repository.save_sync_snapshot()` prunes orphan activity records (no `ok`
   index row) and writes the JSON cache snapshot (`training_cache.json`) and
   `metadata.json` atomically under the app-wide cache lock.
   Daily/weekly aggregates are **never written**: they are derived on read
   (§6) strictly from rows with activity data.

Failure isolation: an unreadable/unparseable file gets a `failed` index row
with the error message; the sync continues. A failing *source* (e.g. WebDAV
down) is reported in `source_errors` without aborting other sources.

### 4.2 Cache Files

| File | Content |
|---|---|
| `training_cache.json` | Snapshot with one ActivityRecord and per-trip Metric collection per accepted activity, plus source-index outcomes |
| `metadata.json` | Last successful sync time |

Daily and weekly summaries and the training-form curve are **not stored**;
the analytics report functions derive them from cached ActivityRecords on
read. Cache serialization uses domain dataclasses and is independent from the
Pydantic API wire models.

Statuses: `ok`, `duplicate`, `failed`. Row key: `(source_name, source_path, segment_index)`.

---

## 5. Metric Definitions (per activity)

### 5.1 Interval Attribution (shared basis)

Almost all record-derived metrics share one rule: each interval between
consecutive records **belongs to its starting record**, contributes
`counted_s = min(dt, 30 s)`, requires a plausible HR value (`> 0`) where HR
is involved, and never crosses segment boundaries. The 30 s cap prevents
recording gaps from inflating time-in-zone.

### 5.2 Heart-Rate Zones (`time_in_zones`)

- Each starting record's HR sample assigns its interval to a zone via the
  configured bounds (inclusive both ends); samples outside all zones (e.g.
  below zone 1) contribute to coverage but to no zone.
- `zone_1_s … zone_5_s`: summed counted seconds per zone.
- `valid_hr_time_s`: summed counted seconds with valid HR.
- `hr_coverage_ratio = valid_hr_time_s / record_span` (clamped to 1.0).

### 5.3 HR Histogram (5-bpm bins)

Time-weighted histogram over *all* plausible HR samples; labels are
inclusive ranges (`"135-139"`), ascending. Allowed to persist (coarse
aggregate).

### 5.4 10-Minute Weighted HR Averages

Buckets of 600 s aligned to the first record timestamp ("first 10 minutes of
the activity"). Each bucket stores `start_min`, time-weighted `avg_hr`
(`Σ(hr·t)/Σt`) and `hr_time_s`. Buckets without HR data are omitted —
the surviving `start_min` values reveal recording gaps.

### 5.5 HR Drift (`hr_drift_bpm_per_hour`)

Time-weighted least-squares slope over bucket centers (bucket start + 300 s),
reported in bpm per hour. Positive = cardiac drift. `None` for < 2 buckets
or zero variance.

### 5.6 Estimated Load (`load.py`, method `hr_trimp`)

Banister-style TRIMP from time-in-zone using configured `max_heart_rate`
and `resting_heart_rate`:

```
HRr_k     = (HR_mid_k − HRrest) / (HRmax − HRrest)      # zone midpoint ratio, clamped to [0, 1.2]
weight_k  = 0.64 · exp(1.92 · HRr_k)                     # male coefficients
load      = Σ_k (minutes_in_zone_k × weight_k)
```

- `load_method = "hr_trimp"`, `LOAD_VERSION = "1"`; the analysis version is
  persisted as `"<analysis_version>:<load_version>"`.
- `load_confidence` from HR coverage: ≥ 0.9 → `high`, ≥ 0.5 → `medium`, else `low`.
- `None` without valid HR data. This is an **estimate**, never labeled "TSS".

### 5.7 Aerobic Decoupling (`endurance.py`)

Compares two **equal moving-time halves** of an activity:

```
EF_half          = time-weighted mean speed / time-weighted mean HR
decoupling_pct   = (EF_1 − EF_2) / EF_1 × 100
```

Quality gates (clean `None` + confidence on failure):

| Gate | Value |
|---|---|
| Minimum moving time with HR | ≥ 1200 s (`decoupling_min_duration_s`) |
| HR coverage of moving time | ≥ 0.80 (`decoupling_min_coverage`) |
| Moving intervals | speed ≥ 1.0 m/s, interval ≤ 30 s, same segment |
| Moving-time basis | Records and pause intervals only |

Confidence values: `high`, `insufficient_duration`, `insufficient_coverage`,
`invalid_ef`.

### 5.8 Data Completeness

`data_completeness` = fraction of 7 key fields present: duration, moving
time, distance, elevation gain, avg HR, max HR, and avg speed.

---

## 6. Aggregations

### 6.1 Daily (`daily.py`)

`daily_summary` projects cached records into a transient Polars frame and
returns one table metric per measure, grouped by **local calendar date**
(configured timezone) × sport. Totals include activity count, duration,
distance, elevation, zone seconds, load, `hard_activity_count` (load ≥ 150.0),
and longest activity.

### 6.2 Weekly (`weekly.py`)

Weekly metrics roll daily totals up by **ISO week** (Monday start) × sport.
Previous-week comparisons are emitted as table metrics with missing cells when
there is no prior week.

### 6.3 Trends

Daily and weekly trend outputs are `TableValue` metrics with date/week axes.
The same report shape supports metrics added later without a trend-specific
wire or renderer model.

---

## 7. Training Form & Tour Readiness (`fitness.py`)

Banister/Coggan impulse-response model over **daily** TRIMP loads in local
calendar days (rest days contribute 0):

```
CTL_{t+1} = CTL_t + (TRIMP_t − CTL_t) · (1 − e^(−1/42))   # fitness, 42-day time constant
ATL_{t+1} = ATL_t + (TRIMP_t − ATL_t) · (1 − e^(−1/7))    # fatigue, 7-day constant
TSB_t     = CTL_t − ATL_t                                  # form
```

Initial conditions CTL = ATL = 0 on the day before the first activity.
`is_warmup = history_days < 42`.

**Form classification (TSB):**

| Band | Classification |
|---|---|
| > +5 | `fresh` |
| −10 … +5 | `optimal` |
| −30 … −10 | `high_fatigue` |
| < −30 | `overtraining_risk` |

**Tour readiness / endurance context:**

- `rolling_7d_distance_km`, `rolling_28d_distance_km` (sum over inclusive windows)
- `days_since_60k_ride` — calendar days since the last ride ≥ 60 km
- `load_confidence_ratio` — share of total load computed from
  high-confidence (`hr_coverage ≥ 0.9`) activities

---

## 8. Data Quality Signals

The overview report exposes data-quality metrics over all cached
activity/source history:

| Signal | Meaning |
|---|---|
| `activity_count` | Cached segment activities |
| `mean_hr_coverage_ratio`, `mean_data_completeness` | Averages over activities |
| `failed_file_count` | Source-index rows with parse/read errors |
| `repaired_file_count` | Files whose parsed activities carry repair flags |
| `discarded_segment_count` | Short segments dropped by segmentation |
| `duplicate_activity_count` | Exact-duplicate files excluded from counting |

Wire names are namespaced, for example `data_quality.activity_count` and
`data_quality.mean_hr_coverage_ratio`.

Per-activity repair flags are exposed as metrics so clients can reason about
data trustworthiness.

---

## 9. CLI Commands & REST Endpoints

**CLI** (`training-stats`, Rich tables on a TTY / JSON when piped or via
`--json`; `training-stats server` runs the REST API)

| Command | Purpose |
|---|---|
| `list [--start? --end? --sport? --limit? --no-sync]` | Generic trip-metric report, newest first |
| `show <activity-id>` | One activity's metric report |
| `weekly [--start? --end? --weeks? --sport? --no-sync]` | ISO-week table metrics |
| `overview [--days? --sport? --no-sync]` | Generic coaching report: profile, trends, form, readiness, and quality metrics |

**REST** (same service layer; OpenAPI at `/openapi.json`)

| Endpoint | Purpose |
|---|---|
| `GET /activities` | As `list`; synchronizes if due |
| `GET /activities/{activity_id}` | As `show`; serves cached data |
| `GET /weekly` | As `weekly`; synchronizes if due |
| `GET /overview` | As `overview`; synchronizes if due |

All metric commands return one generic MetricReport envelope. Each Metric is
self-describing through its stable name, period, shape, unit, missing reason,
kind, and optional calculation provenance. Adding a scalar/table/matrix metric
does not require route or Rich renderer changes.

`list`, `weekly` and `overview` synchronize *if due*: a real sync runs
only when the last successful one is older than
`cache.sync_min_interval_seconds` (default 10 s, `0` disables the
throttle). The gate is two-layer: an in-memory monotonic stamp covers
repeated calls in one process (REST request bursts); without it, the
persisted `last_sync` timestamp in `metadata.json` covers fresh CLI
processes. Stamps are only recorded when the cache was actually written,
so failed syncs never block retries. `show` and `GET
/activities/{activity_id}` serve the cache as-is.

Date filters use `YYYY-MM-DD` in the configured local timezone; limits are
validated (max 500 activities / 104 weeks / 3650 days). Responses never
contain raw time-series data.

---

## 10. Cache Format & Versioning Notes

- The JSON cache is written atomically (temporary file + fsync + replace).
- Cache records are ActivityRecord-grain; metrics are a typed domain collection
  serialized inside each record. Source outcomes are stored separately in the
  snapshot. Period reports are derived on read.
- There is no backward migration path. Delete `training_cache.json` and re-sync
  to build a fresh snapshot after a cache-format or analysis-version change.
- Bumping the internal `ANALYSIS_VERSION_VALUE` in `adapters/cache/schemas.py`
  forces full reprocessing on the next sync; activity IDs are re-derived
  deterministically.
- History: v2 (histogram + 10-min averages + drift), v3 (clock-correction
  start times), v4 (timeline repair, segment splits, records-derived moving
  time, UTC-canonical activity IDs).
