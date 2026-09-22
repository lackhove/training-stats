"""Athlete profile serialization for the overview bundle."""

from __future__ import annotations

from ..config import AthleteConfig


def athlete_profile(athlete: AthleteConfig) -> dict[str, object]:
    """Return the athlete profile including configuration-dependent metric info."""
    profile: dict[str, object] = {
        "max_heart_rate": athlete.max_heart_rate,
        "resting_heart_rate": athlete.resting_heart_rate,
        "heart_rate_zones": zone_descriptions(athlete),
    }
    if athlete.primary_goal is not None:
        profile["primary_goal"] = athlete.primary_goal
    if athlete.weight_kg is not None:
        profile["weight_kg"] = athlete.weight_kg
    if athlete.ftp_watts is not None:
        profile["ftp_watts"] = athlete.ftp_watts
    return profile


def zone_descriptions(athlete: AthleteConfig) -> list[dict[str, object]]:
    """Describe the configured five-zone heart-rate model."""
    return [
        {"zone": zone, "min_bpm": low, "max_bpm": high}
        for zone, low, high in athlete.heart_rate_zones.as_bounds()
    ]
