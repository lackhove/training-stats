"""Shared helpers for concrete activity-source adapters."""

from ...services.ports import ActivitySource


def is_fit_file(path: str) -> bool:
    """Return whether ``path`` refers to a FIT file by extension."""
    return path.lower().endswith(".fit")


__all__ = ["ActivitySource", "is_fit_file"]
