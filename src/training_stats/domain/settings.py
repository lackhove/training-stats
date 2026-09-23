"""Framework-independent athlete and analysis settings."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise


@dataclass(frozen=True, slots=True)
class HeartRateZonesConfig:
    z1_min: int
    z1_max: int
    z2_min: int
    z2_max: int
    z3_min: int
    z3_max: int
    z4_min: int
    z4_max: int
    z5_min: int
    z5_max: int

    def __post_init__(self) -> None:
        zones = [
            (1, self.z1_min, self.z1_max),
            (2, self.z2_min, self.z2_max),
            (3, self.z3_min, self.z3_max),
            (4, self.z4_min, self.z4_max),
            (5, self.z5_min, self.z5_max),
        ]
        for number, low, high in zones:
            if low <= 0 or high < low:
                msg = (
                    f"athlete.heart_rate_zones: zone {number} bounds are invalid "
                    f"(min={low}, max={high}); require 0 < min <= max"
                )
                raise ValueError(msg)
        for current, nxt in pairwise(zones):
            if current[2] >= nxt[1]:
                msg = (
                    f"athlete.heart_rate_zones: zone {current[0]} max ({current[2]}) "
                    f"must be below zone {nxt[0]} min ({nxt[1]})"
                )
                raise ValueError(msg)

    def as_bounds(self) -> list[tuple[int, int, int]]:
        """Return ordered ``(zone, min_bpm, max_bpm)`` tuples."""
        return [
            (1, self.z1_min, self.z1_max),
            (2, self.z2_min, self.z2_max),
            (3, self.z3_min, self.z3_max),
            (4, self.z4_min, self.z4_max),
            (5, self.z5_min, self.z5_max),
        ]


@dataclass(frozen=True, slots=True)
class AthleteConfig:
    max_heart_rate: int
    resting_heart_rate: int
    heart_rate_zones: HeartRateZonesConfig
    primary_goal: str | None = None
    weight_kg: float | None = None
    ftp_watts: float | None = None

    def __post_init__(self) -> None:
        if self.max_heart_rate <= 0:
            raise ValueError("athlete.max_heart_rate must be positive")
        if self.resting_heart_rate <= 0:
            raise ValueError("athlete.resting_heart_rate must be positive")
        if self.resting_heart_rate >= self.max_heart_rate:
            msg = (
                "athlete.resting_heart_rate must be below "
                f"athlete.max_heart_rate (got {self.resting_heart_rate} "
                f">= {self.max_heart_rate})"
            )
            raise ValueError(msg)
        if self.weight_kg is not None and self.weight_kg <= 0:
            raise ValueError("athlete.weight_kg must be positive")
        if self.ftp_watts is not None and self.ftp_watts <= 0:
            raise ValueError("athlete.ftp_watts must be positive")


@dataclass(frozen=True, slots=True)
class QualityConfig:
    moving_speed_threshold_mps: float = 1.0
    record_gap_threshold_s: float = 120.0
    clock_jump_min_gap_s: float = 3600.0
    minimum_segment_duration_s: float = 600.0
    clock_shift_max_pre_moving_s: float = 60.0
    forgotten_stop_max_duration_s: float = 86400.0
    forgotten_stop_min_moving_ratio: float = 0.20
    decoupling_min_duration_s: float = 1200.0
    decoupling_min_coverage: float = 0.80

    def __post_init__(self) -> None:
        positive = (
            "moving_speed_threshold_mps",
            "record_gap_threshold_s",
            "clock_jump_min_gap_s",
            "minimum_segment_duration_s",
            "forgotten_stop_max_duration_s",
            "decoupling_min_duration_s",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"quality.{name} must be positive")
        if self.clock_shift_max_pre_moving_s < 0:
            raise ValueError("quality.clock_shift_max_pre_moving_s must be non-negative")
        for name in ("forgotten_stop_min_moving_ratio", "decoupling_min_coverage"):
            value = getattr(self, name)
            if not 0 < value <= 1:
                raise ValueError(f"quality.{name} must be in (0, 1]")
