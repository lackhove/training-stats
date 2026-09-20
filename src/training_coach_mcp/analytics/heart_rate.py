"""Heart-rate zone computation based on configured, validated zone bounds."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import polars as pl

# Intervals longer than this are capped to avoid over-counting time-in-zone
# when a device pauses recording or produces sparse smart recordings.
MAX_RECORD_INTERVAL_S = 30.0

# Bucket size for time-weighted heart-rate averages (drift detection).
HR_BUCKET_S = 600.0


@dataclass(frozen=True, slots=True)
class ZoneBounds:
    zone: int
    min_bpm: int
    max_bpm: int


@dataclass(frozen=True, slots=True)
class ZoneResult:
    """Time in each heart-rate zone plus HR data coverage."""

    zone_seconds: tuple[float, float, float, float, float]
    valid_hr_time_s: float
    hr_coverage_ratio: float


@dataclass(frozen=True, slots=True)
class HrBucketAvg:
    """Time-weighted heart-rate average for one bucket of an activity."""

    start_min: int
    avg_hr: float
    hr_time_s: float

    def to_dict(self) -> dict[str, object]:
        return {"start_min": self.start_min, "avg_hr": self.avg_hr, "hr_time_s": self.hr_time_s}


def build_zone_bounds(bpm_bounds: list[tuple[int, int, int]]) -> list[ZoneBounds]:
    """Create :class:`ZoneBounds` from validated ``(zone, min, max)`` tuples."""
    return [ZoneBounds(zone=zone, min_bpm=low, max_bpm=high) for zone, low, high in bpm_bounds]


def _zone_tuple(values: list[float]) -> tuple[float, float, float, float, float]:
    padded = (values + [0.0] * 5)[:5]
    return (padded[0], padded[1], padded[2], padded[3], padded[4])


def _empty_zone_result() -> ZoneResult:
    return ZoneResult(
        zone_seconds=(0.0, 0.0, 0.0, 0.0, 0.0), valid_hr_time_s=0.0, hr_coverage_ratio=0.0
    )


def _zone_expression(zones: list[ZoneBounds]) -> pl.Expr:
    """Build an expression assigning each HR sample its zone number.

    Zones are validated non-overlapping, so coalescing one matcher per zone
    is equivalent to a when/then chain.
    """
    heart_rate = pl.col("heart_rate")
    matchers = [
        pl.when(heart_rate.is_between(zone.min_bpm, zone.max_bpm, closed="both")).then(
            pl.lit(zone.zone, dtype=pl.UInt32)
        )
        for zone in zones
    ]
    return pl.coalesce(matchers)


def _interval_frame(records: pl.DataFrame, max_interval_s: float) -> pl.DataFrame:
    """Per-interval time weights shared by zones, histogram and buckets.

    Each interval belongs to its *starting* record: the row carries that
    record's timestamp and heart rate, and ``counted_s`` is the time until the
    next record, capped at ``max_interval_s``. Non-positive intervals collapse
    to zero; rows without heart rate are dropped.
    """
    return records.select(
        pl.col("timestamp"),
        pl.col("heart_rate"),
        _interval_seconds().clip(0.0, max_interval_s).alias("counted_s"),
    ).filter(pl.col("heart_rate").is_not_null() & (pl.col("heart_rate") > 0))


def _interval_seconds() -> pl.Expr:
    """Seconds to the *next* record; ``None`` on the last row.

    The interval belongs to its starting record so that the row's heart rate
    is the interval's zone-defining sample.
    """
    return (
        pl.col("timestamp").shift(-1) - pl.col("timestamp")
    ).dt.total_microseconds() / 1_000_000.0


def time_in_zones(
    records: pl.DataFrame,
    zones: list[ZoneBounds],
    max_interval_s: float = MAX_RECORD_INTERVAL_S,
) -> ZoneResult:
    """Compute elapsed time per heart-rate zone from record samples.

    Each interval between consecutive timestamped records contributes
    ``min(dt, max_interval_s)`` seconds when the interval's starting record has
    a plausible heart-rate value. ``hr_coverage_ratio`` is the share of the
    sampled time span with HR data.

    ``records`` follows :data:`training_coach_mcp.fit.types.RECORD_SCHEMA` and
    is never mutated.
    """
    if records.height < 2 or not zones:
        return _empty_zone_result()

    timestamps = records["timestamp"]
    first, last = timestamps.min(), timestamps.max()
    if not isinstance(first, datetime) or not isinstance(last, datetime):
        return _empty_zone_result()
    span_s = (last - first).total_seconds()
    if span_s <= 0:
        return _empty_zone_result()

    intervals = _interval_frame(records, max_interval_s)
    valid_hr_time_s = _sum_column(intervals, "counted_s")

    zone_seconds = [0.0, 0.0, 0.0, 0.0, 0.0]
    with_zone = intervals.select(
        pl.col("counted_s"),
        _zone_expression(zones).alias("zone"),
    ).filter(pl.col("zone").is_not_null())
    per_zone = with_zone.group_by("zone").agg(pl.col("counted_s").sum()).sort("zone")
    for row in per_zone.iter_rows(named=True):
        index = int(row["zone"]) - 1
        if 0 <= index < 5:
            zone_seconds[index] = _as_float(row["counted_s"])

    coverage = valid_hr_time_s / span_s
    return ZoneResult(
        zone_seconds=_zone_tuple(zone_seconds),
        valid_hr_time_s=valid_hr_time_s,
        hr_coverage_ratio=min(coverage, 1.0),
    )


def hr_histogram(
    records: pl.DataFrame,
    bin_bpm: int = 5,
    max_interval_s: float = MAX_RECORD_INTERVAL_S,
) -> dict[str, float]:
    """Time-weighted histogram of heart-rate samples in ``bin_bpm``-wide bins.

    Covers all plausible heart rates (not only in-zone ones); bin labels are
    inclusive integer ranges like ``"135-139"``, returned in ascending order.
    Interval handling matches :func:`time_in_zones`.
    """
    if records.height < 2 or bin_bpm <= 0:
        return {}

    heart_rate = pl.col("heart_rate")
    intervals = _interval_frame(records, max_interval_s).with_columns(
        ((heart_rate // bin_bpm) * bin_bpm).alias("bin_start")
    )
    per_bin = intervals.group_by("bin_start").agg(pl.col("counted_s").sum()).sort("bin_start")
    histogram: dict[str, float] = {}
    for row in per_bin.iter_rows(named=True):
        low = int(row["bin_start"])
        seconds = _as_float(row["counted_s"])
        if seconds > 0:
            histogram[f"{low}-{low + bin_bpm - 1}"] = seconds
    return histogram


def hr_bucket_averages(
    records: pl.DataFrame,
    bucket_s: float = HR_BUCKET_S,
    max_interval_s: float = MAX_RECORD_INTERVAL_S,
) -> list[HrBucketAvg]:
    """Time-weighted heart-rate averages per ``bucket_s`` window.

    Buckets are aligned to the first record timestamp so they read as "first
    10 minutes of the activity". Buckets without HR data are omitted; the
    remaining ``start_min`` values reveal gaps. Irregular recording is handled
    by weighting each sample with its valid interval time.
    """
    if records.height < 2 or bucket_s <= 0:
        return []

    first = records["timestamp"].min()
    if not isinstance(first, datetime):
        return []

    intervals = _interval_frame(records, max_interval_s).with_columns(
        ((pl.col("timestamp") - pl.lit(first)).dt.total_microseconds() / 1_000_000.0 // bucket_s)
        .cast(pl.UInt32)
        .alias("bucket")
    )
    per_bucket = (
        intervals.group_by("bucket")
        .agg(
            pl.col("counted_s").sum().alias("hr_time_s"),
            (pl.col("counted_s") * pl.col("heart_rate")).sum().alias("hr_weighted"),
        )
        .sort("bucket")
    )

    buckets: list[HrBucketAvg] = []
    for row in per_bucket.iter_rows(named=True):
        hr_time_s = _as_float(row["hr_time_s"])
        weighted = _as_float(row["hr_weighted"])
        if hr_time_s <= 0:
            continue
        buckets.append(
            HrBucketAvg(
                start_min=int(row["bucket"]) * int(bucket_s // 60),
                avg_hr=weighted / hr_time_s,
                hr_time_s=hr_time_s,
            )
        )
    return buckets


def hr_drift_bpm_per_hour(buckets: list[HrBucketAvg]) -> float | None:
    """Time-weighted least-squares HR slope over buckets, in bpm per hour.

    Positive values indicate cardiac drift; ``None`` when fewer than two
    buckets or no variance in bucket timing.
    """
    if len(buckets) < 2:
        return None
    points = [(b.start_min * 60.0 + HR_BUCKET_S / 2.0, b.avg_hr, b.hr_time_s) for b in buckets]
    total_weight = sum(weight for _, _, weight in points)
    if total_weight <= 0:
        return None
    mean_x = sum(x * weight for x, _, weight in points) / total_weight
    mean_y = sum(y * weight for _, y, weight in points) / total_weight
    variance = sum(weight * (x - mean_x) ** 2 for x, _, weight in points)
    if variance <= 0:
        return None
    covariance = sum(weight * (x - mean_x) * (y - mean_y) for x, y, weight in points)
    slope_per_second = covariance / variance
    return slope_per_second * 3600.0


def _sum_column(frame: pl.DataFrame, column: str) -> float:
    series = frame[column].drop_nulls()
    if series.len() == 0:
        return 0.0
    return _as_float(series.sum())


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return float(value)
