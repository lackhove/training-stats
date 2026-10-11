"""Availability of measured quantities in transient FIT record data."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .types import ParsedActivity


@dataclass(frozen=True, slots=True)
class QuantityAvailability:
    """Whether an activity contains at least one sample for each FIT quantity."""

    heart_rate: bool
    cadence: bool
    power: bool
    speed: bool
    distance: bool
    elevation: bool

    @property
    def present(self) -> frozenset[str]:
        """Names of the quantities represented by at least one record sample."""
        quantities = {
            "heart_rate": self.heart_rate,
            "cadence": self.cadence,
            "power": self.power,
            "speed": self.speed,
            "distance": self.distance,
            "elevation": self.elevation,
        }
        return frozenset(name for name, available in quantities.items() if available)

    @classmethod
    def from_activity(cls, activity: ParsedActivity) -> QuantityAvailability:
        """Inspect raw record columns; zero-valued samples count as present."""
        records = activity.records
        return cls(
            heart_rate=_has_sample(records, "heart_rate"),
            cadence=_has_sample(records, "cadence") or _has_sample(records, "fractional_cadence"),
            power=_has_sample(records, "power_w"),
            speed=_has_sample(records, "speed_mps"),
            distance=_has_sample(records, "distance_m"),
            elevation=_has_sample(records, "altitude_m"),
        )


def _has_sample(records: pl.DataFrame, column: str) -> bool:
    return any(value is not None for value in records.get_column(column).to_list())
