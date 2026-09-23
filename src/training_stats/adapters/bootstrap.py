"""Composition root wiring application services to concrete adapters."""

from __future__ import annotations

from ..domain.errors import TrainingStatsError
from ..services.ingest import FileProcessor
from ..services.ports import ActivitySource, FitParser
from ..services.settings import ServiceSettings
from ..services.sync import SyncService
from ..services.training import TrainingStatsService
from .cache.repository import ParquetTrainingRepository
from .cache.schemas import ANALYSIS_VERSION_VALUE
from .config import (
    AppConfig,
    LocalSourceConfig,
    WebDAVSourceConfig,
    resolve_webdav_credentials,
)
from .fit.parser import FitDecodeParser
from .sources.local import LocalDirectorySource
from .sources.webdav import WebDAVSource


def create_service(config: AppConfig) -> TrainingStatsService:
    """Build a service graph from parsed application configuration."""
    credentials = resolve_webdav_credentials(config)
    sources: list[ActivitySource] = []
    for source_config in config.sources:
        if isinstance(source_config, LocalSourceConfig):
            sources.append(
                LocalDirectorySource(
                    name=source_config.name,
                    path=source_config.path,
                    recursive=source_config.recursive,
                    follow_symlinks=source_config.follow_symlinks,
                )
            )
        elif isinstance(source_config, WebDAVSourceConfig):
            sources.append(WebDAVSource(source_config, credentials[source_config.name]))
        else:  # pragma: no cover - exhaustive by discriminated union
            msg = f"unsupported source type: {type(source_config).__name__}"
            raise TrainingStatsError(msg)

    parser: FitParser = FitDecodeParser(quality=config.quality)
    processor = FileProcessor(
        parser=parser,
        athlete=config.athlete,
        quality=config.quality,
        analysis_version=ANALYSIS_VERSION_VALUE,
    )
    repository = ParquetTrainingRepository(cache_dir=config.cache.directory)
    sync_service = SyncService(
        sources=sources,
        processor=processor,
        repository=repository,
        analysis_version=ANALYSIS_VERSION_VALUE,
    )
    settings = ServiceSettings(
        application_name=config.application.name,
        timezone=config.application.zoneinfo,
        sync_min_interval_seconds=config.cache.sync_min_interval_seconds,
        athlete=config.athlete,
    )
    return TrainingStatsService(settings=settings, sync_service=sync_service, repository=repository)
