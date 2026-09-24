"""Pure FIT record ordering and activity-gap segmentation."""

from __future__ import annotations

import polars as pl

from .types import FitData


def repair_timeline(
    fit_data: FitData,
    *,
    gap_threshold_s: float,
    distance_gap_threshold_m: float,
) -> list[FitData]:
    """Split raw FIT messages where adjacent records show an activity gap."""
    frame = fit_data.frame
    if frame.is_empty():
        return []
    records = frame.filter(pl.col("message_type") == "record").sort(
        "timestamp", maintain_order=True
    )
    if records.is_empty():
        return []

    sorted_records = records
    timestamps = sorted_records["timestamp"].to_list()
    distances = sorted_records["distance_m"].to_list()
    boundaries: list[int] = []

    for index in range(1, sorted_records.height):
        dt = (timestamps[index] - timestamps[index - 1]).total_seconds()
        previous_distance = distances[index - 1]
        distance = distances[index]
        distance_gap = (
            isinstance(previous_distance, int | float)
            and not isinstance(previous_distance, bool)
            and isinstance(distance, int | float)
            and not isinstance(distance, bool)
            and abs(float(distance) - float(previous_distance)) > distance_gap_threshold_m
        )
        if dt > gap_threshold_s or distance_gap:
            boundaries.append(index)

    ranges = list(zip([0, *boundaries], [*boundaries, sorted_records.height], strict=True))
    file_messages = frame.filter(pl.col("message_type") != "record")
    result: list[FitData] = []
    for start, end in ranges:
        rows = pl.concat([file_messages, sorted_records.slice(start, end - start)])
        result.append(
            FitData(
                frame=rows.sort("timestamp", maintain_order=True).cast(pl.Schema(FitData.SCHEMA))
            )
        )
    return result
