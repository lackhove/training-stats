# How the Training Coach Works

Documentation of the architecture, the data-cleaning pipeline, processing
steps and all metric definitions.

- Sources: local directories and Nextcloud via WebDAV (read FIT files only)
- Language: Python 3.12+, everything managed through `uv`
- Cache: Polars DataFrames persisted as Parquet (aggregates only, never raw time series)
- Interface: verb-based CLI (JSON output) and FastAPI REST API with OpenAPI schema

---

## 1. Architecture & Data Flow

```
sources -> fit.parser -> fit.repair -> analytics -> cache.sync -> cache.repository -> server
```

| Package | Responsibility |
|---|---|
| `sources/` | `ActivitySource` ABC; `LocalDirectorySource` (filesystem walk) and `WebDAVSource` (httpx + env-resolved secrets) |
| `fit/` | `FitParser` protocol returning a `ParsedFile`; `FitDecodeParser` (fitdecode boundary); `repair.py` (timeline repair, segmentation, moving time); `normalize.py` (coercion helpers); `types.py` (`RECORD_SCHEMA`, `ParsedActivity`, `ParsedFile`) |
| `analytics/` | Per-activity metrics (`activities`, `heart_rate`, `load`, `endurance`), aggregations (`daily`, `weekly`), training form (`fitness`) |
| `cache/` | Parquet schemas (`schemas.py`), atomic I/O (`parquet.py`), flock locking (`locking.py`), incremental sync (`sync.py`), read-side repository (`repository.py`) |
| `coach/` | Athlete profile from TOML, overview assembly |
| `server.py` | Service layer; methods on `TrainingCoachServer` are directly testable |
| `api.py` | FastAPI application with Pydantic response models (OpenAPI schema) |
| `cli.py` | cyclopts CLI (JSON output) |
| `config.py` | Pydantic TOML models; WebDAV secrets only via environment variables |
| `models.py` | Shared frozen dataclasses (`SourceFile`, `SyncResult`, summaries, …) |
| `errors.py` | Typed exception hierarchy (`TrainingMCPError` base) |

**Core invariants**

1. Session headers are never trusted blindly — start time and duration are
   re-derived from the repaired record timeline (see §3).
2. No raw time-series persistence — record samples live in a transient
   Polars frame during processing and are discarded. Only coarse derived
   aggregates are cached (5-bpm HR histogram, 10-min weighted HR averages).
3. Secrets only from environment variables (`SecretStr`, never logged).
4. Atomic cache writes only (tmp file + fsync + `os.replace` under flock).
5. Incremental sync via SHA-256 content hashes, not modification times.
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
- `[quality]` — repair and metric thresholds (defaults below, §3.2)
- `[cache]` — directory and lock timeout

The analysis version (see §10) is owned by the code, not the config: it is
the reprocessing switch — when parsing or metric semantics change, the
version is bumped and the next sync reprocesses every file (current: `"4"`).

---

## 3. Ingestion and Data Cleaning

### 3.1 Discovery & Change Detection

1. Each source lists FIT files (`*.fit`, case-insensitive; symlinks are
   skipped unless `follow_symlinks`).
2. Every file is read and hashed with SHA-256 during a sync.
3. A file is **skipped** only when all of its cached segment rows match
   both `source_hash` and `analysis_version` in `source_index.parquet`.
4. `rebuild_cache()` ignores cached derived data and reprocesses everything.

### 3.2 Why Cleaning Is Necessary

Real device files exhibit three failure modes:

1. **Clock corrections during recording** — e.g. a Sigma-style device with
   an unsynced RTC records minutes/hours on a wrong clock (even a 2021
   epoch) until GPS fixes the time mid-ride; `total_elapsed_time` then
   includes the jump (observed: 82 h "rides").
2. **Recording breaks / appended blips** — a device restarted days later
   appends a seconds-long static blip to the same file; elapsed time spans
   the gap (observed: 43 h gaps between a 12 s blip and the real ride).
3. **Day-tour rides with long pauses** — multiple substantial legs separated
   by gaps; these are legitimate multi-segment activities.

### 3.3 The Repair Pipeline (`fit/repair.py`)

The repair layer operates exclusively on the transient record frame
(timestamp, heart_rate, distance_m, altitude_m, speed_mps).

1. **Sort & validate.** Records are stably sorted by timestamp. If the
   original file order was not monotonic, the flag
   `non_monotonic_timestamps` is set.
2. **Discontinuity classification.** For every gap between consecutive
   records (config: `record_gap_threshold_s = 120`, `clock_jump_min_gap_s = 3600`):
   - *Gap ≥ 3600 s → clock-jump candidate.* The repair computes the
     **moving time of the pre-gap records** (speed ≥ 1.0 m/s, capped at
     30 s per interval). The shift is accepted when pre-gap movement is
     negligible (`clock_shift_max_pre_moving_s = 60` s) — or unmeasurable
     (no speed data, typical for pre-GPS static recordings). Accepted shifts
     set `clock_corrected`.
     Rationale: a static prefix is a device waiting for a GPS fix; a moving
     prefix is a real ride leg. This distinguishes f957-style clock
     corrections (correct) from cca0-style resume blips (rejected → split).
   - *Gap ≥ 120 s (and < 3600 s, or a rejected shift) → recording break.*
     The timeline splits into separate segments.
3. **Shift mechanics.** Only the pre-jump records are shifted, by exactly
   the jump offset, so the boundary becomes seamless and the merged
   timeline is strictly continuous. Records after the correction keep
   their (trusted) timestamps. The activity *start time* is the first
   record of the repaired timeline — the true start.
4. **Segmentation & discard.** Segments shorter than
   `minimum_segment_duration_s = 600` (10 min) are **discarded** when at
   least one longer segment exists in the same file; if all segments are
   short, the longest is retained so a file is never completely lost.
   Discards increment `discarded_segment_count` and set
   `segments_discarded`. Each retained segment becomes its own
   `ParsedActivity` (with `segment_index`/`segment_count`).
5. **Forgotten stop (diagnostic).** An activity is flagged
   `forgotten_stop` when duration > 86,400 s (24 h), or when moving
   ratio < 0.20 for activities longer than 2 h. Flagged activities are
   retained.

**Repair/hygiene flags:** `clock_corrected`, `segments_discarded`,
`non_monotonic_timestamps`, `forgotten_stop` — persisted per activity and
per file, exposed in activity details and the data-quality resource.

### 3.4 Session Header Re-derivation (per segment)

| Field | Rule |
|---|---|
| `start_time` | First record of the (repaired) segment timeline; falls back to session `start_time` only when no records exist |
| `duration_s` | Record-timeline span; the session `total_elapsed_time` is kept **only when coherent** (`|elapsed − span| ≤ record_gap_threshold_s`), otherwise the span wins. Session-less files fall back to `total_timer_time` |
| `moving_time_s` | Derived from record speeds (§4.2); falls back to session `total_moving_time` (allocated proportionally across segments), then to duration |
| `paused_time_s` | `duration_s − moving_time_s` when both exist |
| `distance_m` | Record distance delta (`last − first`); falls back to session `total_distance` (proportional per segment) |
| `elevation_gain_m` | Session `total_ascent` (proportional per segment); records-derived gain as fallback |
| `avg_hr` / `max_hr` | Per-segment record HR; single-segment files prefer session values with records fallback |
| `avg_speed_mps` / `max_speed_mps` | Session weighted values (single segment); per-segment `distance/span` and record max for multi-segment |
| `calories` | Session `total_calories`, allocated proportionally by segment span |
| `sport` / `sub_sport` | From the session |

### 3.5 Moving Time from Records

Intervals between consecutive records are assigned to their **starting**
record. An interval counts when the starting record's speed ≥
`moving_speed_threshold_mps` (1.0 m/s) and contributes
`min(dt, 30 s)`. When fewer than 50% of intervals carry a speed reading,
the derivation is rejected (fallback chain above). This is what makes a
static pre-ride prefix show up as *paused* instead of *moving* time.

### 3.6 Activity Identity and Duplicates

`activity_id = sha256(canonical-UTC start | sport | duration_s | segment_index)[:16]`

- IDs are deterministic across sources, re-downloads and reprocessing.
- **Exact duplicate detection:** when two files produce the same
  `activity_id` (typical: the same ride exported by both watch and app),
  the later file is marked `duplicate` in the source index and its rows are
  excluded from activities, metrics, daily/weekly aggregates and MCP
  responses. Fuzzy near-duplicate detection (different clocks producing
  slightly different IDs) is explicitly out of scope.

---

## 4. Processing: Sync & Cache

### 4.1 Sync Flow (`cache/sync.py`)

1. Load source index + activity/metrics stores (in memory).
2. List, read and hash all files from all enabled sources.
3. Skip unchanged files (all segment rows matching hash + version).
4. On change: **purge all cached rows of that file** (every segment), parse,
   repair, compute metrics, index one row per segment.
5. Duplicate segments are marked `status="duplicate"` with no store rows.
6. Rebuild daily/weekly aggregates **strictly from `status == "ok"` rows**.
7. Write all Parquet files + `metadata.json` atomically under `sync.lock`.

Failure isolation: an unreadable/unparseable file gets a `failed` index row
with the error message; the sync continues. A failing *source* (e.g. WebDAV
down) is reported in `source_errors` without aborting other sources.

### 4.2 Cache Files

| File | Content |
|---|---|
| `activities.parquet` | One row per segment activity: identity, start time, sport, duration/moving/paused, distance, elevation, HR, speeds, calories, segment fields, repair flags |
| `activity_metrics.parquet` | Derived per-activity analytics (§5): zone seconds, coverage, histogram JSON, 10-min buckets JSON, drift, load (+method/confidence), decoupling, completeness, analysis version |
| `daily_metrics.parquet` | Per local-date & sport totals |
| `weekly_metrics.parquet` | Per ISO week & sport totals + previous-week comparison |
| `source_index.parquet` | File ↔ activity mapping, hashes, statuses, segment fields, per-file repair flags |
| `metadata.json` | last sync time, analysis version, activity/failed counts |

Statuses: `ok`, `duplicate`, `failed`. Index key: `(source_name, source_path, segment_index)`.

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
| Moving-time basis | `max(session moving time, records-derived moving time)` |

Confidence values: `high`, `insufficient_duration`, `insufficient_coverage`,
`invalid_ef`.

### 5.8 Data Completeness

`data_completeness` = fraction of 8 key fields present: duration, moving
time, distance, elevation gain, avg HR, max HR, avg speed, calories.

---

## 6. Aggregations

### 6.1 Daily (`daily.py`)

Grouped by **local calendar date** (configured timezone) × sport. Totals:
activity count, duration, distance, elevation, zone seconds, load,
`hard_activity_count` (load ≥ 150.0), longest activity (duration + id).
The repository can aggregate across sports on request (`sport="all"`).

### 6.2 Weekly (`weekly.py`)

Daily rows rolled up by **ISO week** (Monday start) × sport, with a
`comparison_vs_prev_week` block (load, duration, distance of the previous
ISO week, `None` when absent).

### 6.3 Trends (`trends.py`)

`training_trends(metric, days, sport)` builds a daily series for one of
`duration | distance | elevation | load | activities` plus `total`, `mean`,
and a direction classified by comparing first-half vs second-half means:
`change_pct > +5% → increasing`, `< −5% → decreasing`, else `stable`.

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

`training://data-quality` and `DataQuality`:

| Signal | Meaning |
|---|---|
| `activity_count` | Cached segment activities |
| `mean_hr_coverage_ratio`, `mean_data_completeness` | Averages over activities |
| `failed_file_count` | Source-index rows with parse/read errors |
| `repaired_file_count` | Files whose parsed activities carry repair flags |
| `discarded_segment_count` | Short segments dropped by segmentation |
| `duplicate_activity_count` | Exact-duplicate files excluded from counting |

Per-activity `data_hygiene_flags` (in `get_activity`) expose the specific
repair outcomes so the LLM can reason about data trustworthiness.

---

## 9. CLI Commands & REST Endpoints

**CLI** (`coach`, JSON output; `coach server` runs the REST API)

| Command | Purpose |
|---|---|
| `list [--start? --end? --sport? --limit? --no-sync]` | Syncs sources first, then compact summaries (newest first), incl. `local_start_time`, `moving_time_s` |
| `show <activity-id>` | Full record: zones, histogram, 10-min buckets, drift, decoupling, load, paused time, hygiene flags |
| `weekly [--start? --end? --weeks? --sport?]` | ISO-week totals with previous-week comparison |
| `overview [--days? --sport?]` | Compact coaching bundle: profile, recent activities, weeklies, daily series, zone distribution, form + form history, quality, latest activity |

**REST** (same service layer; OpenAPI at `/openapi.json`)

| Endpoint | Purpose |
|---|---|
| `GET /activities` | As `list`; synchronizes sources first |
| `GET /activities/{activity_id}` | As `show` |
| `GET /weekly` | As `weekly` |
| `GET /overview` | As `overview` |

The overview's daily series carries per-(date, sport) totals for every
metric (duration, distance, elevation, load, activities, zone seconds), so
any metric trend is a filter and projection over it.

Date filters use `YYYY-MM-DD` in the configured local timezone; limits are
validated (max 500 activities / 104 weeks / 3650 days). Responses never
contain raw time-series data.

---

## 10. Cache Schemas & Versioning Notes

- All Parquet writes are atomic; reads of missing files return empty frames.
- Old cache files missing newly added columns are filled with nulls on
  read/rewrite (schema evolution), so upgrades don't crash mid-sync.
- Bumping the internal `ANALYSIS_VERSION_VALUE` in `cache/schemas.py`
  (currently `"4"`) forces full reprocessing on the next sync; activity IDs
  are re-derived deterministically.
- History: v2 (histogram + 10-min averages + drift), v3 (clock-correction
  start times), v4 (timeline repair, segment splits, records-derived moving
  time, UTC-canonical activity IDs).
