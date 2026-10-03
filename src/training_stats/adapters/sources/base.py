"""Shared helpers for concrete activity-source adapters."""


def is_fit_file(path: str) -> bool:
    """Return whether ``path`` refers to a FIT file by extension."""
    return path.lower().endswith(".fit")


__all__ = ["is_fit_file"]
