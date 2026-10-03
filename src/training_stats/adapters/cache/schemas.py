"""Versioned JSON cache identifiers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from ...domain.settings import QualityConfig

CACHE_FILE = "training_cache.json"
METADATA_FILE = "metadata.json"
LOCK_FILE = "cache.lock"
CACHE_FORMAT_VERSION = 2

# Cache-invalidation key for per-activity metric semantics.
ANALYSIS_VERSION_VALUE = "8"


def effective_analysis_version(quality: QualityConfig) -> str:
    """Return a stable analysis identity that includes quality settings.

    The default settings retain the code-only version for backwards-compatible
    default outputs. Any non-default quality configuration receives a
    deterministic suffix, so changing analysis settings forces file
    reprocessing and is visible in the per-activity ``analysis.version`` metric.
    """
    values = asdict(quality)
    if values == asdict(QualityConfig()):
        return ANALYSIS_VERSION_VALUE
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"{ANALYSIS_VERSION_VALUE}+quality-{digest}"
