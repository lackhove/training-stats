"""Transient FIT timeline repair.

Repairs device-clock corrections (e.g. a pre-GPS RTC recorded while the
device was waiting for a GPS fix) and splits recording breaks into
independent segments. Operates exclusively on the transient record frame;
raw samples are consumed and produced in memory only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import polars as pl

# After a shift repair the boundary between the shifted prefix and the first
# record after the jump must look like normal recording cadence.
_MAX_BOUNDARY_INTERVAL_S = 30.0

# Minimum share of record intervals with a speed reading before moving time
# is derived from records instead of falling back to session fields.
_MIN_SPEED_COVERAGE = 0.5


@dataclass(frozen=True, slots=True)
class RepairSegment:
    """One retained recording segment on a coherent, shift-repaired timeline."""

    records: pl.DataFrame
    segment_index: int
    flags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RepairResult:
    """Repair outcome for one FIT file."""

    segments: tuple[RepairSegment, ...]
    flags: tuple[str, ...]
    discarded_segment_count: int


def repair_timeline(
    records: pl.DataFrame,
    *,
    gap_threshold_s: float,
    clock_jump_min_gap_s: float,
    minimum_segment_duration_s: float,
    clock_shift_max_pre_moving_s: float,
    moving_speed_threshold_mps: float,
) -> RepairResult:
    """Split and repair the record timeline into coherent segments.

    Discontinuities of at least ``clock_jump_min_gap_s`` are treated as a
    device-clock correction: the preceding records are shifted forward by the
    jump offset so the recording continues seamlessly. The shift is accepted
    only while the preceding records show negligible movement, which
    distinguishes a static pre-GPS/appended blip from a genuine recording
    break after a real ride leg. All other gaps at least
    ``gap_threshold_s`` long split segments; segments shorter than
    ``minimum_segment_duration_s`` are discarded when a longer segment
    exists.
    """
    if records.is_empty():
        return RepairResult(segments=(), flags=(), discarded_segment_count=0)

    sorted_records = records.sort("timestamp", maintain_order=True)
    file_flags: list[str] = []
    if not records["timestamp"].is_sorted():
        file_flags.append("non_monotonic_timestamps")

    segment_ranges = _split_pieces(
        sorted_records,
        gap_threshold_s=gap_threshold_s,
        clock_jump_min_gap_s=clock_jump_min_gap_s,
        clock_shift_max_pre_moving_s=clock_shift_max_pre_moving_s,
        moving_speed_threshold_mps=moving_speed_threshold_mps,
    )
    candidates = [_build_segment_frame(sorted_records, ranges) for ranges in segment_ranges]
    if any(shifted for _frame, _span, shifted in candidates):
        file_flags.append("clock_corrected")
    retained, discarded = _apply_minimum_duration(candidates, minimum_segment_duration_s)
    if discarded:
        file_flags.append("segments_discarded")

    segments = tuple(
        RepairSegment(
            records=frame,
            segment_index=index,
            flags=("clock_corrected",) if shifted else (),
        )
        for index, (frame, _span, shifted) in enumerate(retained)
    )
    return RepairResult(
        segments=segments,
        flags=tuple(file_flags),
        discarded_segment_count=discarded,
    )


def moving_time_from_records(
    records: pl.DataFrame,
    *,
    threshold_mps: float,
    max_interval_s: float = _MAX_BOUNDARY_INTERVAL_S,
    min_coverage: float = _MIN_SPEED_COVERAGE,
) -> float | None:
    """Sum speed-gated interval durations, each interval capped.

    Intervals are assigned to their starting record and only count when that
    record's speed reaches ``threshold_mps``. Returns ``None`` when fewer
    than ``min_coverage`` of the intervals carry a speed reading, signalling
    the caller to fall back to session fields.
    """
    if records.height < 2:
        return None
    dt_s = records["timestamp"].diff().dt.total_seconds()
    pairs = pl.DataFrame(
        {
            "dt_s": dt_s.slice(1),
            "speed_mps": records["speed_mps"].slice(0, records.height - 1),
        }
    )
    valid = pairs.filter(pl.col("dt_s") > 0)
    if valid.is_empty():
        return None
    coverage = valid["speed_mps"].is_not_null().mean()
    if not isinstance(coverage, float) or coverage < min_coverage:
        return None
    moving = valid.filter(pl.col("speed_mps") >= threshold_mps).select(
        pl.col("dt_s").clip(0, max_interval_s).sum()
    )
    value = moving.item()
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


@dataclass(frozen=True, slots=True)
class _Range:
    """Contiguous row range with a uniform timestamp shift applied."""

    start: int
    end: int
    shift_s: float


def _split_pieces(
    sorted_records: pl.DataFrame,
    *,
    gap_threshold_s: float,
    clock_jump_min_gap_s: float,
    clock_shift_max_pre_moving_s: float,
    moving_speed_threshold_mps: float,
) -> list[list[_Range]]:
    """Walk the sorted timeline into segments of shifted row ranges.

    Each segment is a list of ranges: an accepted clock jump closes the
    pre-jump range with the jump offset applied and starts a new unshifted
    range, so the shift never touches records recorded after the correction.
    """
    timestamps = sorted_records["timestamp"]
    gaps = timestamps.diff().dt.total_seconds().to_list()
    segments: list[list[_Range]] = []
    ranges: list[_Range] = []
    sub_start = 0
    sub_shift = 0.0

    def close(end: int) -> None:
        nonlocal sub_start, sub_shift, ranges
        if end > sub_start:
            ranges.append(_Range(start=sub_start, end=end, shift_s=sub_shift))
        segments.append(ranges)
        ranges = []
        sub_start = end
        sub_shift = 0.0

    for index, gap in enumerate(gaps):
        if gap is None:
            continue
        if gap >= clock_jump_min_gap_s:
            prefix = sorted_records.slice(sub_start, index - sub_start)
            pre_moving = moving_time_from_records(
                prefix,
                threshold_mps=moving_speed_threshold_mps,
                max_interval_s=_MAX_BOUNDARY_INTERVAL_S,
            )
            # A prefix without speed readings (or too few records) cannot show
            # movement; pre-GPS recordings typically lack GPS-derived speed.
            if pre_moving is None or pre_moving <= clock_shift_max_pre_moving_s:
                if index > sub_start:
                    ranges.append(_Range(start=sub_start, end=index, shift_s=sub_shift + gap))
                sub_start = index
                sub_shift = 0.0
                continue
            close(index)
        elif gap >= gap_threshold_s:
            close(index)
    close(len(timestamps))
    return segments


def _build_segment_frame(
    sorted_records: pl.DataFrame, ranges: list[_Range]
) -> tuple[pl.DataFrame, float, bool]:
    """Materialize one segment with all shifts applied; return (frame, span, shifted)."""
    parts: list[pl.DataFrame] = []
    for rng in ranges:
        if rng.end <= rng.start:
            continue
        part = sorted_records.slice(rng.start, rng.end - rng.start)
        if rng.shift_s:
            part = part.with_columns(pl.col("timestamp") + timedelta(seconds=rng.shift_s))
        parts.append(part)
    if not parts:
        msg = "empty segment range"
        raise ValueError(msg)
    frame = pl.concat(parts) if len(parts) > 1 else parts[0]
    first = frame["timestamp"][0]
    last = frame["timestamp"][-1]
    if not isinstance(first, datetime) or not isinstance(last, datetime):
        msg = "record timestamps have unexpected type"
        raise ValueError(msg)
    return frame, (last - first).total_seconds(), any(rng.shift_s for rng in ranges)


def _apply_minimum_duration(
    candidates: list[tuple[pl.DataFrame, float, bool]], minimum_s: float
) -> tuple[list[tuple[pl.DataFrame, float, bool]], int]:
    """Drop short candidate segments; always keep at least the longest."""
    if len(candidates) <= 1:
        return candidates, 0
    substantial = [candidate for candidate in candidates if candidate[1] >= minimum_s]
    if substantial:
        return substantial, len(candidates) - len(substantial)
    longest = max(candidates, key=lambda candidate: candidate[1])
    return [longest], len(candidates) - 1
