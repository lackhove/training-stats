# Analytics and Metric API Rewrite Plan

## Goals

- Replace the heterogeneous analytics outputs with immutable domain `Metric`
  dataclasses.
- Keep analytics mostly functional: pure functions accept existing domain or
  Polars inputs and return one or more metrics.
- Make public summary functions compose metric functions and return a generic
  `MetricReport`.
- Make the API and Rich renderer generic over metric names and the supported
  value shapes.
- Store one activity record with its metric collection in a JSON cache. A cache
  rebuild is sufficient when the cache format or analysis semantics change;
  there is no backwards cache migration requirement.

## Domain metric contract

Create immutable domain types, independent of Pydantic and adapters:

- `Metric`: stable name, required human-readable description, period, value or
  missing reason, kind, optional unit, and optional method, algorithm version,
  confidence, formula, coefficient set, and time basis.
- Metric value variants: scalar, one-axis table, or two-axis matrix. A heatmap
  is two axes plus a cell measure.
- Period variants: trip, date range, point-in-time snapshot, and all-time.
- Axis labels and numeric bins for table and matrix values.
- `MetricReport`: an ordered collection of metrics.

Use string enums for finite semantic vocabularies and `Literal` discriminators
for period and value variants. Enforce a non-empty metric description and that
a missing metric has a reason while a present metric has a value. Zero is a
present value. The metric unit applies to table/matrix cells; each axis may
have its own unit.

For `hr_trimp`, retain the estimated kind, `hr_trimp` method, algorithm version,
confidence, formula, and coefficient set where applicable.

## Functional analytics design

`ParsedActivity` is the per-trip input containing transient `records:
pl.DataFrame`; `FitData.frame` contains normalized raw FIT messages. Neither raw
record series nor `FitData` is persisted. Read-side analytics consume cached
activity records (and may use transient Polars frames internally).

Leaf metric functions are stateless and return `Metric` or tuples of `Metric`.
Summary functions compose leaf functions, own analytic selection/aggregation,
and return `MetricReport`. These summary functions are the public
`domain.analytics` API used by the service layer. There are no mutable analytics
services or endpoint-specific output dataclasses. Pure intermediate helpers
and algorithm result types may remain private where useful.

Rewrite the current per-activity analytics, activity statistics, daily/weekly
aggregation, fitness/load, and overview report boundaries to produce metrics
rather than exposing `ActivityMetrics`, row dataclasses, `TrainingFormResult`,
`Overview`, or untyped nested dictionaries. Small typed intermediate results
may remain inside pure calculations, but service and adapter callers receive
only MetricReports. Preserve current formulas and behavior, including
configured athlete zones, pause/gap rules, data-quality semantics, and the
starting-record interval convention for heart-rate zones.

Proposed public summary operations include activity listing/detail reports,
weekly summaries, and overview reports. Activity metrics use trip periods;
weekly and daily table metrics use date-range periods and date/week axes;
current form metrics use snapshots. Period aggregates are calculated on read
and are not stored in the cache.

## JSON cache and ingestion

Cache grain remains one accepted `ActivityRecord` per activity. Each record
contains its activity/source metadata and a collection of persisted per-trip
metrics. Keep source outcome entries for failed, duplicate, and source-only
files in the cache snapshot, separate from accepted activity records. Do not
cache `MetricReport` objects or period summaries.

Use a JSON cache snapshot with top-level activity and source-index collections.
The cache adapter serializes/validates domain records using an explicit codec or
a Pydantic `TypeAdapter` over domain dataclasses. API wire Pydantic models are
not the cache schema. Preserve atomic temporary-file writes, `fsync`,
`os.replace`, and the process-wide cache lock. Keep sync-throttle metadata
separate as appropriate. Because descriptions are required fields in cached
metrics, changing that record shape bumps the cache format and triggers a fresh
source sync rather than attempting a migration.

The ingest path remains: parse FIT, repair/segment into `ParsedActivity`,
compute metrics, and persist the resulting activity record. New metrics can be
added to the record's JSON metric collection without changing a fixed column
schema. When changed metric semantics require
recalculation, invalidate/rebuild the cache; no migration loader is needed.
Only coarse derived values may persist. Raw high-frequency records remain
transient.

## Service layer

Keep sync, request validation, and repository orchestration in services. Load
typed cached activity/source records and call the public analytics summary
functions. Services do not construct metric payloads field-by-field. Ingest
services pass `ParsedActivity` to activity metric functions and hand results to
the cache boundary.

## Wire and HTTP API

Define generic Pydantic wire models/TypeAdapters for `MetricReport`, `Metric`,
the three value shapes, and period variants. FastAPI endpoints share the
generic report response contract. Route code retains paths, query parameters,
authentication, and sync behavior but does not enumerate metric names or map
individual metric fields.

Validate JSON round-trips and inspect generated OpenAPI for all tagged variants.
The wire layer stays at the HTTP/JSON boundary; domain code has no Pydantic
dependency.

## Rich output and CLI

Replace field- and endpoint-specific metric renderers with one generic
`render_report` function that dispatches only on metric value shape:

- Scalars render in a summary table.
- Tables render from their axis and values.
- Matrices render as heatmaps with axis labels and cell units.
- Missing metrics remain visible with their missing reason.
- Descriptions are displayed from the metric; names are humanized automatically.
  Known units receive generic formatting,
  with a value-plus-unit fallback for unfamiliar units.
- Period, kind, confidence, and provenance render from metric metadata.

Rich consumes domain `MetricReport` dataclasses. CLI JSON uses the generic wire
adapter. Adding a metric using an existing value shape must not require Rich or
API-module changes. Non-metric process diagnostics such as sync status and
errors remain diagnostic output.

## Migration of existing code and tests

- Replace the persisted `ActivityMetrics` output with metric tuples; retain
  parsed-segment pairing only as an ingestion intermediate.
- Replace `activity_statistics.py` dictionary output with metric functions.
- Keep daily/weekly/form intermediates inside analytics; expose their results
  only as MetricReports and remove obsolete output models once unused.
- Consolidate legacy adapter wire mappings into the generic metric wire layer.
- Replace CLI list/show/weekly/overview field-specific renderers for metric
  reports.
- Keep FIT parsing, timeline repair, source handling, typed service errors,
  synchronization, and cache locking in their existing architectural layers.
- Update `AGENTS.md` and user documentation to describe the new domain/API/cache
  boundaries and the JSON cache rebuild behavior.

## Verification

Add tests for metric invariants, all period/value variants, cache JSON
round-trips, analytics outputs and periods, missing reasons, and generic API
and Rich behavior with arbitrary metric names and unknown units. Preserve
regression coverage for current analytics results and FIT timing semantics.
Verify cache rebuilds, incremental sync, source outcomes, atomic writes, and
that raw records never appear in the cache.

Run all repository quality gates:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy --strict src
uv run pytest
```
