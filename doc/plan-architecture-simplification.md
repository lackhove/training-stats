# Architecture Simplification Roadmap

## Goal

Reduce repeated computation, repeated orchestration, and duplicated aggregation
logic while keeping the current HTTP, CLI, OpenAPI, and cache behavior stable.
Make changes in small steps so that behavior-preserving simplifications remain
separate from correctness fixes and product decisions.

This is a follow-up to `doc/plan-analytics-simplification.md`. It does not
repeat work already completed by that plan. In particular, keep the existing
domain/service/adapter boundaries, typed wire layer, independent cache codec,
and explicit metric producers.

## Guardrails

- Preserve public JSON and OpenAPI contracts, including metric ordering, names,
  values, units, descriptions, periods, missing reasons, provenance, axes, and
  null cells.
- Preserve all metrics currently emitted by the activity-detail response.
- Preserve the cache format and analysis version unless a separately reviewed
  correctness change requires reprocessing.
- Keep FIT record frames transient. Any shared per-activity computation state
  must be scoped to processing that activity; do not retain record frames in a
  process-wide cache.
- Preserve interval attribution, pause handling, validity filters, and time
  axes. Similar-looking metrics may intentionally use different rules.
- Preserve sync behavior for failed sources, duplicate activities, orphan
  promotion, and unchanged snapshots.
- Preserve atomic cache writes, the app-wide lock, secret handling, and sync
  throttling.

## Findings and proposed work

### 1. Reuse facts within `activity_statistics.py`

**Scope:** `domain/analytics/activity_statistics.py` and its tests.

The composer already builds one `Interval` sequence and passes it to the
statistics families. Other activity-level facts are still derived repeatedly:
available windows, efficiency facts, cadence coverage, and short-activity
checks. The families also carry repeated optional `intervals=` plumbing.

**Plan:**

1. Add a private, per-activity context (or equivalent explicit bundle) that
   owns the interval sequence and reusable derived facts.
2. Build each shared fact once in the composer and let the relevant families
   consume it.
3. Retain standalone family calculations where tests or internal callers need
   them, without causing the composed path to recompute the same facts.
4. Keep the context local to one activity calculation; do not use
   `functools.lru_cache` over `ParsedActivity` or Polars frames.

**Acceptance:** Existing metric parity and order tests pass. Add or retain tests
showing that the composer and standalone family calls emit the same outputs.
Profile a fixed synthetic activity before and after to confirm shared work is
performed once.

### 2. Reuse the Polars interval frame across analytics modules

**Scope:** `domain/analytics/segments.py`, `activities.py`, `heart_rate.py`,
`endurance.py`, `intervals.py`, and relevant tests. `statistics_records.py` may
also be involved if the shared context is introduced there.

One activity currently builds an interval frame while deriving segment facts,
then rebuilds equivalent frames for heart-rate zones, histogram, bucket
averages, and aerobic decoupling. The repeated table construction scans the
same records multiple times.

**Plan:**

1. Identify the earliest point where records and pauses are both available.
2. Build the frame once and pass it through a private transient analysis
   context to its consumers.
3. Keep consumer-specific filtering and transformations at each consumer.
   Sharing the frame must not imply sharing HR-validity thresholds, pause
   merging behavior, or bucket/window axes.
4. Keep the frame out of persisted models and cache data. Do not introduce a
   global cache keyed by activity ID or object identity.

**Acceptance:** Golden FIT zone tests, activity-statistics wire digests, and
aggregate compatibility tests remain unchanged. Profile the same fixture and
record whether interval-frame construction is reduced from multiple builds to
one per activity.

### 3. Consolidate daily and weekly total accumulation

**Scope:** `domain/analytics/daily.py`, `weekly.py`, a small shared helper or
constants module if warranted, and aggregate tests.

Daily and weekly aggregation repeat accumulation of activity count, duration,
distance, elevation, five HR zones, load, hard-activity count, and longest
activity. Sport merging repeats much of the same field-by-field logic.

**Plan:**

1. Introduce one focused totals accumulator for the shared fields.
2. Leave daily and weekly bucketing, date calculations, and previous-period
   handling in their current owners.
3. Centralize the five zone names/count if that reduces repeated literals
   without obscuring the wire field names.

**Acceptance:** Preserve input-order floating-point summation, first-wins
longest-activity ties, sport-merging order, and `None` when every previous-week
value is absent. Run the exact aggregate compatibility tests.

### 4. Remove repeated sync scans and share file-processing flow

**Scope:** `services/sync.py` and sync/cache/service tests.

The sync loop rescans all index rows for each listed file and rebuilds the set
of activity IDs owned by other files for each changed file. The normal ingest
path and orphan-promotion path also repeat much of the same read, hash, purge,
process, and state-update sequence.

**Plan:**

1. Add regression coverage for shared activity IDs, failed listings, read
   failures, changed files, and orphan promotion.
2. Group prior index rows by `(source_name, source_path)` once per sync.
3. Maintain activity ownership information as rows are kept, purged, replaced,
   or promoted.
4. Extract a focused helper for common file read/process/state-update steps;
   keep promotion-specific decisions in the promotion driver.
5. Keep `_state_unchanged` order-insensitive behavior and the current snapshot
   write/stamp decision.

**Acceptance:** The new tests and all existing sync/cache tests pass. Compare
cache snapshots and sync counters for the same fixtures before and after.
Retain SHA-256 source hashes unless the documented successful-read/hash
invariant is explicitly changed in a separate decision.

### 5. Avoid the second full-cache parse on sync-due queries only if justified

**Scope:** Likely `services/training.py`, `sync.py`, and possibly service result
or repository types, depending on the selected design.

A query that triggers sync can load and validate the same complete cache once
for sync and again for the query. A reviewed local cache took about 236 ms for
one load; the benefit will depend on actual cache size and query latency.

**Plan:**

1. Measure cache load time and query latency with representative cache sizes.
2. If material, compare returning/reusing the post-sync state with a
   repository-level read cache. Prefer the option that adds the least lifecycle
   and invalidation complexity.
3. Keep invalid-cache recovery behavior: malformed cache data must still
   trigger the existing recovery path.

**Acceptance:** Add a test proving the same snapshot is not parsed twice on the
selected path, while throttled and failed-sync behavior remains unchanged.
Skip this phase if measurements do not justify the extra state plumbing.

### 6. Small conversion and maintenance reductions

These may be done as small, independent changes alongside the phases above:

- In `adapters/wire.py::weekly_report`, check shared periods on the domain
  metrics before converting each metric once to its weekly wire form.
- Consolidate duplicate metric-reading helpers in `daily.py` and
  `activities.py` where their semantics match.
- Make `analysis_version` have one owner rather than injecting independent
  copies into both `SyncService` and `FileProcessor`.
- Remove the redundant UTC normalization in `services/ingest.py` after
  retaining the parser's required FIT timestamp normalization.
- Remove explicit `None` padding from FIT record/event dictionaries where the
  declared Polars schema supplies the same null columns; keep parser parity
  tests.

Each item should be a separate, behavior-preserving change with focused tests.

## Separate decisions—not part of the internal refactor

### Activity statistics product scope

The `activity.statistics.*` family is large and overlaps conceptually with
other metrics. It is also emitted by the current activity-detail response.
Do not delete or rename these metrics as an internal cleanup. If reducing the
public metric set is desired, first agree on API compatibility/versioning and
which metrics remain supported.

### Segment metadata correctness

`domain/fit/repair.py` copies all non-record messages into every segment. A
multi-session file can therefore attribute later segments to the first
session's sport metadata. Fixing metadata scoping is a separate correctness
change: specify how session rows map to segments, add multi-session regression
fixtures, and decide whether to bump the analysis version so cached activities
are reprocessed.

### Cache and OpenAPI model cleanup

Some source-index fields appear write-only, and some enum/schema branches appear
unused in current responses. Removing them changes the cache format or
published OpenAPI schema. Revisit only as explicit compatibility changes, not
as part of these behavior-preserving phases.

## Recommended implementation order

1. Capture the current test and profile baselines.
2. Apply the small wire conversion simplification.
3. Reuse facts inside `activity_statistics.py`.
4. Share the Polars interval frame across analytics modules.
5. Consolidate daily/weekly accumulation.
6. Simplify sync indexing and file-processing flow.
7. Measure whether duplicate cache parsing warrants a design change.
8. Apply the remaining small maintenance reductions and update architecture
   documentation.
9. Handle public metric pruning and the multi-session metadata bug only through
   their separate decisions above.

Keep each numbered phase independently reviewable. If public metric parity
changes during a behavior-preserving phase, stop and investigate the semantic
difference rather than updating golden outputs to accept it.

## Quality gates

Run the repository's required gates after each significant implementation
phase and all four before declaring the work complete:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy --strict src
uv run pytest
```
