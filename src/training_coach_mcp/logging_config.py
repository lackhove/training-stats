"""Logging setup with privacy-conscious defaults.

Secrets must never be passed to loggers; this module only centralizes
formatting and level configuration.
"""

import logging

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging with a compact, structured format."""
    logging.basicConfig(level=level.upper(), format=_LOG_FORMAT, force=True)
