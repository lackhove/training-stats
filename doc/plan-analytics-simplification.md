# Analytics Simplification Plan

## Goal

Make analytics a small collection of pure functions with explicit inputs and
outputs, remove unnecessary conversions and dead paths, and preserve the
existing HTTP and CLI JSON contracts.

The required data flows are:

```text
ParsedActivity + explicit configuration -> per-activity Metric(s)
cached ActivityRecord(s) + explicit query context -> aggregate Metric(s)
```

Analytics must not perform I/O, access repositories, depend on adapters, or
reconstruct Polars frames from cached metrics. Polars may still be used while
processing transient FIT records when it is useful.

## Compatibility requirements

- Keep HTTP paths, response-model schemas, OpenAPI components, CLI JSON shapes,
  metric order, names, values, axes, descriptions, units, periods, missing
  reasons, provenance, and null-cell behavior unchanged for the same inputs.
- Keep all metrics currently emitted through `/activities/{activity_id}`.
  That endpoint serializes every metric in the cached `ActivityRecord`; there is
  no per-name API filter today. Pruning must remove internal/non-API work, not
  silently shrink this response.
- Keep weekly period hoisting and the overview period mix unchanged.
- Do not persist raw time-series data or period aggregates.
- Do not change the cache format or analysis version unless a cache payload
  change is unavoidable. This plan does not require one.
- Preserve locking, atomic cache writes, secret handling, and sync throttling.
- The mixed-source-listing data-loss fix is an intentional correctness fix:
  retain cached activities for sources that fail to list. Its response schema
  remains unchanged.

## Ordered implementation steps

Each numbered step is a separate major commit. Do not commit generated CLI
output, local cache files, configuration, or FIT files.

### 1. Document the design and this plan

- Update `AGENTS.md` with the functional analytics contract:
  - per-activity analytics take `ParsedActivity` and explicit settings and
    return one or more `Metric`s;
  - aggregate analytics take `ActivityRecord`s and explicit query context and
    return metrics / `MetricReport`;
  - cached records are never projected back into Polars frames;
  - analytics have no I/O or adapter/service dependencies.
- Clarify that Polars remains available for transient FIT record processing.
- Add this ordered plan and its compatibility requirements.

### 2. Replace cached-record-to-Polars aggregation

- Change daily aggregation to consume `ActivityRecord`s and read only the
  cached metrics it needs.
- Change training-form calculation to consume records directly.
- Update weekly, overview, and the exported daily-summary path to use the
  record-based aggregators.
- Preserve local-time bucketing, sorting/tie-breaks, missing-value defaults,
  sport merging/filtering, full-history previous-week and training-form
  calculations, periods, and report ordering.
- Delete `domain/analytics/record_frames.py` and the activity-frame schemas and
  dtypes from `domain/frames.py`.
- Move `SourceStatus` and `STATUS_OK` / `STATUS_FAILED` / `STATUS_DUPLICATE`
  beside `SourceIndexEntry`; then delete `domain/frames.py` if no uses remain.
- Replace Polars-frame test fixtures with metric-bearing `ActivityRecord`s.
- Add exact expected public aggregate values for representative fixtures.

### 3. Replace the statistics dictionary/walker with explicit Metric producers

- Remove the `dict[str, object]` statistics result, its separately maintained
  empty-data schema, the recursive metric walker, and inferred metadata
  helpers.
- Add focused pure metric-producing functions for coherent output families
  (for example, speed, heart rate/zones, cadence, time windows, relationships,
  heatmaps, and sensor quality). Each function constructs its `Metric`s
  explicitly; related metrics may be returned together.
- Keep a thin composer for the per-activity metric families. Share small,
  typed numerical helpers where they avoid duplicated algorithms.
- Preserve the current public metric set and ordering, including its
  data-dependent omissions, null table cells, names, and metadata.
- Update `tests/test_activity_statistics.py` to assert `Metric` outputs rather
  than private nested dictionaries. Cover missing/sparse cadence, pauses,
  short/empty activity data, windows, relationships, and heatmaps.
- Keep an end-to-end metric inventory/parity test for a deterministic synthetic
  `ParsedActivity`.

### 4. Remove analytics entry points and helpers with no API consumer

- Remove `activity_report`, `activity_list_report`, and standalone
  `daily_summary` if they remain unused by routes; retain the daily metric
  calculations included in `/overview`.
- Remove test-only or unused helpers, including the pass-through `report()`
  helper, unused `HrBucketAvg.to_dict()`, unused converter functions, and
  unused frame/schema constants.
- Remove or consolidate duplicated coercion helpers.
- Remove the unreachable segment-specific branch in endurance analytics.
- Keep `MetricReport` if it remains the stable analytics/API report contract.

### 5. Consolidate adapter and service detours without changing wire output

- Keep `adapters/wire.py` as the API/CLI anti-corruption layer.
- Share duplicated metric fields between `MetricWire` and
  `MetricWithoutPeriodWire` without changing component names or properties.
- Replace dump-then-validate conversions with direct domain-to-wire validation
  where parity tests confirm identical JSON. Keep explicit schema-parity tests
  because attribute validation does not enforce `extra="forbid"` identically.
- Choose one owner for sync-before-query and remove the unused `sync_first`
  branches/parameters.
- Use the in-scope source when reading its file; avoid rebuilding activity-ID
  sets per file.
- Simplify duplicated CLI sync formatting and the CLI serializer callback.
- Remove unreachable `isinstance` checks in registered HTTP exception
  handlers.
- Consolidate duplicated repository protocols and the service-state model
  re-export shim where this does not weaken the port boundary.

### 6. Fix sync/cache work and remaining low-risk duplication

- Preserve prior index/activity rows for a source whose listing failed, even if
  other sources listed successfully; add a regression test.
- Avoid rewriting the full snapshot when a successful sync finds no data
  changes; still update the successful-sync stamp.
- Avoid parsing the full cache file only to read `last_sync_stamp()`; preserve
  format-version and throttle semantics.
- Consolidate duplicate WebDAV status mapping and environment-variable
  resolution where the resulting errors remain identical.
- Remove FIT parser null padding when the declared Polars schema fills absent
  fields, with parser regression tests.
- Simplify `jsonio` duplicate exception handling. Retain the durability fsync
  required by the atomic-write invariant.

### 7. Finish test/dependency cleanup and run all gates

- Consolidate repeated test builders and assertion helpers; remove
  cross-test-module imports where practical.
- Verify direct dependencies such as `pyarrow` and `pytest-cov` are not needed
  by supported workflows before removing them.
- Retain the documented `--json` compatibility option unless a CLI-breaking
  removal is separately approved.
- Run `ruff format --check`, `ruff check`, `mypy --strict src`, and the full
  pytest suite.
- Inspect `git status`, `git diff`, and recent history before each commit;
  stage only the files for that step.

## Explicitly deferred

- Compacting cache JSON by omitting null fields or storing a shared activity
  period once. This needs a codec/cache-version design and adds rehydration
  behavior; it should be considered only after the analytics refactor.
- Removing the wire layer, source adapters, authentication, process-wide cache
  lock, or atomic-write guarantees. These are intentional boundaries/invariants,
  not needless conversions.
