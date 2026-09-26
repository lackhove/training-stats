"""JSON cache adapter and application-wide lock."""

from .jsonio import read_json, write_json_atomic
from .locking import acquire_app_lock
from .repository import JsonTrainingRepository
from .schemas import ANALYSIS_VERSION_VALUE, CACHE_FILE, CACHE_FORMAT_VERSION, METADATA_FILE

__all__ = [
    "ANALYSIS_VERSION_VALUE",
    "CACHE_FILE",
    "CACHE_FORMAT_VERSION",
    "METADATA_FILE",
    "JsonTrainingRepository",
    "acquire_app_lock",
    "read_json",
    "write_json_atomic",
]
