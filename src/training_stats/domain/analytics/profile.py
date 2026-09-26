"""Metric functions for configuration-backed athlete profile values."""

from __future__ import annotations

from ..metrics import AllTimePeriod, Axis, Metric, MetricKind, MissingReason
from ..settings import AthleteConfig
from .metric_helpers import scalar_metric, table_metric


def athlete_profile(athlete: AthleteConfig) -> tuple[Metric, ...]:
    """Return stable athlete/profile metrics and configured heart-rate zones."""
    period = AllTimePeriod()
    metrics = [
        scalar_metric(
            "athlete_profile.max_heart_rate",
            period,
            athlete.max_heart_rate,
            description="Maximum heart-rate value used for configured zones and athlete metrics.",
            unit="bpm",
            kind=MetricKind.MEASURED,
        ),
        scalar_metric(
            "athlete_profile.resting_heart_rate",
            period,
            athlete.resting_heart_rate,
            description="Resting heart-rate value used for training-load estimates.",
            unit="bpm",
            kind=MetricKind.MEASURED,
        ),
        scalar_metric(
            "athlete_profile.primary_goal",
            period,
            athlete.primary_goal,
            description="Athlete's configured primary training goal.",
            kind=MetricKind.INTERPRETED,
            missing_reason=MissingReason.NO_DATA,
        ),
        scalar_metric(
            "athlete_profile.weight",
            period,
            athlete.weight_kg,
            description="Athlete body mass from configuration.",
            unit="kg",
            kind=MetricKind.MEASURED,
        ),
        scalar_metric(
            "athlete_profile.functional_threshold_power",
            period,
            athlete.ftp_watts,
            description="Functional threshold power configured for the athlete.",
            unit="W",
            kind=MetricKind.MEASURED,
        ),
    ]
    zones = athlete.heart_rate_zones.as_bounds()
    axis = Axis(name="heart_rate_zone", values=tuple(f"zone_{zone}" for zone, _, _ in zones))
    metrics.extend(
        [
            table_metric(
                "athlete_profile.heart_rate_zone_minimum",
                period,
                axis,
                tuple(low for _, low, _ in zones),
                description="Lower heart-rate boundary for each configured zone.",
                unit="bpm",
                kind=MetricKind.MEASURED,
            ),
            table_metric(
                "athlete_profile.heart_rate_zone_maximum",
                period,
                axis,
                tuple(high for _, _, high in zones),
                description="Upper heart-rate boundary for each configured zone.",
                unit="bpm",
                kind=MetricKind.MEASURED,
            ),
        ]
    )
    return tuple(metrics)
