"""Coach command line interface.

Query commands print their service payload as JSON on stdout, regardless of
whether stdout is a TTY. Diagnostics (sync feedback, errors) go to stderr.
Exit codes: 0 success, 1 domain error (unknown activity, invalid range),
2 configuration or usage error.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter
from cyclopts.exceptions import CycloptsError

from ... import __version__
from ...domain.errors import TrainingStatsError
from ...domain.models import ActivitySummary
from ...services.results import SyncResult, ThrottledSync
from ...services.training import TrainingStatsService
from .. import wire
from ..bootstrap import create_service
from ..cache.locking import acquire_app_lock
from ..config import (
    CONFIG_ENV_VAR,
    AppConfig,
    config_path_from_env,
    load_config,
    resolve_auth_token,
)
from ..errors import ConfigurationError
from ..logging_config import setup_logging

logger = logging.getLogger(__name__)

app = App(
    name="training-stats",
    help="Training stats for FIT activity data. Outputs JSON.",
    version=__version__,
    result_action="return_none",
)
SportArg = Annotated[
    str | None, Parameter(name=["--sport"], help="Exact sport name, case-insensitive.")
]
StartArg = Annotated[
    date | None, Parameter(name=["--start"], help="Start date (YYYY-MM-DD), inclusive.")
]
EndArg = Annotated[date | None, Parameter(name=["--end"], help="End date (YYYY-MM-DD), inclusive.")]
LimitArg = Annotated[
    int, Parameter(name=["--limit"], help="Maximum number of activities to return.")
]
SyncFlag = Annotated[
    bool,
    Parameter(
        name=["--sync-first"],
        negative=["--no-sync"],
        help=(
            "Synchronize sources unless the last sync is younger than "
            "cache.sync_min_interval_seconds (default: enabled)."
        ),
    ),
]
ConfigArg = Annotated[
    Path | None,
    Parameter(
        name=["--config", "-c"],
        show=False,
        help=(
            f"Path to the TOML configuration file (default: ${CONFIG_ENV_VAR} or ./config.toml)."
        ),
    ),
]
GlobalConfigArg = Annotated[
    Path | None,
    Parameter(
        name=["--config", "-c"],
        help=(
            f"Path to the TOML configuration file (default: ${CONFIG_ENV_VAR} or ./config.toml)."
        ),
    ),
]
JsonFlag = Annotated[
    bool,
    Parameter(
        name=["--json"],
        show=False,
        help="Accepted for compatibility; commands always output JSON.",
    ),
]
GlobalJsonFlag = Annotated[
    bool,
    Parameter(
        name=["--json"],
        help="Accepted for compatibility; commands always output JSON.",
    ),
]


def _load_config(config_path: Path | None) -> AppConfig:
    if config_path is not None:
        return load_config(config_path)
    from_env = config_path_from_env()
    if from_env is not None:
        return load_config(from_env)
    return load_config(Path("config.toml"))


def _service_context(config_path: Path | None) -> tuple[AppConfig, TrainingStatsService]:
    config = _load_config(config_path)
    setup_logging(config.application.log_level)
    # One instance operates on the cache: the exclusive lock is held until
    # the process exits (a CLI command ends, the server keeps serving).
    acquire_app_lock(config.cache.directory, config.cache.lock_timeout_seconds)
    return config, create_service(config)


def _service(config_path: Path | None) -> TrainingStatsService:
    return _service_context(config_path)[1]


def _emit(payload: object) -> None:
    """Serialize one JSON payload on stdout."""
    print(json.dumps(payload, indent=2))


def _emit_summaries(summaries: tuple[ActivitySummary, ...]) -> None:
    """Serialize activity summaries as a JSON array on stdout."""
    _emit(
        [
            summary.model_dump(mode="json", exclude_none=True)
            for summary in wire.activity_summaries(summaries)
        ]
    )


def _emit_wire(
    payload: wire.ActivityRecordWire | wire.MetricReportWire | wire.WeeklyReportWire,
) -> None:
    """Serialize one wire model on stdout."""
    _emit(payload.model_dump(mode="json", exclude_none=True))


@app.meta.default
def _launcher(
    *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
    config: GlobalConfigArg = None,
    json_output: GlobalJsonFlag = False,
) -> None:
    """Consume global options so they may appear before the command."""
    forwarded: list[str] = []
    if config is not None:
        forwarded += ["--config", str(config)]
    if json_output:
        forwarded += ["--json"]
    app(
        list(tokens) + forwarded,
        exit_on_error=False,
        print_error=False,
        result_action="return_none",
    )


def _print_sync_summary(result: SyncResult) -> None:
    """Print one sync progress/summary line plus error details to stderr.

    Stdout stays machine-readable: JSON payloads go there and only there.
    """
    print("syncing sources...", file=sys.stderr)
    print(
        "sync: "
        f"{result.files_discovered} found, "
        f"{result.files_processed} new, "
        f"{result.files_skipped} unchanged, "
        f"{result.files_failed} failed",
        file=sys.stderr,
    )
    extras: list[str] = []
    if result.files_repaired:
        extras.append(f"{result.files_repaired} repaired")
    if result.segments_discarded:
        extras.append(f"{result.segments_discarded} segments discarded")
    if result.duplicates_detected:
        count = result.duplicates_detected
        extras.append(f"{count} duplicate" if count == 1 else f"{count} duplicates")
    if extras:
        print("sync: " + ", ".join(extras), file=sys.stderr)
    for error in result.source_errors:
        print(f"sync source error: {error}", file=sys.stderr)


def _print_sync_feedback(result: SyncResult | ThrottledSync) -> None:
    """Report the outcome of a (possibly throttled) sync on stderr."""
    if isinstance(result, ThrottledSync):
        print(
            f"sync: skipped (last sync {result.seconds_since_last_sync}s ago)",
            file=sys.stderr,
        )
        return
    _print_sync_summary(result)


@app.command(name="list")
def list_activities(
    config: ConfigArg = None,
    start: StartArg = None,
    end: EndArg = None,
    sport: SportArg = None,
    limit: LimitArg = 50,
    sync_first: SyncFlag = True,
    json_output: JsonFlag = False,
) -> None:
    """List compact activity summaries, newest first."""
    service = _service(config)
    if sync_first:
        _print_sync_feedback(service.sync_if_due())
    _emit_summaries(
        service.list_activities(start_date=start, end_date=end, sport=sport, limit=limit)
    )


@app.command
def show(activity_id: str, config: ConfigArg = None, json_output: JsonFlag = False) -> None:
    """Show one activity record with its metrics."""
    _emit_wire(wire.activity_record(_service(config).get_activity(activity_id)))


@app.command
def weekly(
    config: ConfigArg = None,
    weeks: Annotated[int, Parameter(help="Number of past ISO weeks.")] = 8,
    sport: SportArg = None,
    start: StartArg = None,
    end: EndArg = None,
    sync_first: SyncFlag = True,
    json_output: JsonFlag = False,
) -> None:
    """Show weekly training totals with previous-week comparisons."""
    service = _service(config)
    if sync_first:
        _print_sync_feedback(service.sync_if_due())
    _emit_wire(
        wire.weekly_report(
            service.weekly_summary(start_date=start, end_date=end, weeks=weeks, sport=sport)
        )
    )


@app.command
def overview(
    config: ConfigArg = None,
    days: Annotated[int, Parameter(help="Length of the period ending today.")] = 28,
    sport: SportArg = None,
    sync_first: SyncFlag = True,
    json_output: JsonFlag = False,
) -> None:
    """Show the coaching overview: profile, daily series, form, quality."""
    service = _service(config)
    if sync_first:
        _print_sync_feedback(service.sync_if_due())
    _emit_wire(wire.metric_report(service.overview(days=days, sport=sport)))


def _startup_sync(service: TrainingStatsService) -> SyncResult:
    """Synchronize sources once before the server starts; never aborts startup."""
    result = service.sync_activities()
    logger.info(
        "startup sync: discovered=%d processed=%d skipped=%d failed=%d%s",
        result.files_discovered,
        result.files_processed,
        result.files_skipped,
        result.files_failed,
        "" if not result.source_errors else f" source_errors={list(result.source_errors)}",
    )
    return result


_OPEN_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _warn_unauthenticated_bind(host: str, has_auth: bool) -> None:
    """Log a warning for non-loopback binds on an unauthenticated server."""
    if not has_auth and host not in _OPEN_HOSTS:
        logger.warning(
            "server binds non-loopback host %r without [server.auth]; the REST "
            "API accepts unauthenticated requests — configure bearer auth to "
            "protect it",
            host,
        )


@app.command
def server(
    config: ConfigArg = None,
    host: Annotated[
        str | None, Parameter(help="Bind address (default from [server].host).")
    ] = None,
    port: Annotated[int | None, Parameter(help="Port (default from [server].port).")] = None,
) -> None:
    """Run the FastAPI REST server (OpenAPI at /openapi.json)."""
    import uvicorn

    from ..http.api import build_app

    app_config, service = _service_context(config)
    logger.info("starting %s with config %s", app_config.application.name, config or "config.toml")
    _startup_sync(service)
    bind_host = host if host is not None else app_config.server.host
    _warn_unauthenticated_bind(bind_host, app_config.server.auth is not None)
    uvicorn.run(
        build_app(service, auth_token=resolve_auth_token(app_config)),
        host=bind_host,
        port=port if port is not None else app_config.server.port,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the training-stats CLI; returns the process exit code."""
    args = list(argv) if argv is not None else sys.argv[1:]
    try:
        app.meta(args, exit_on_error=False, print_error=False, result_action="return_none")
    except CycloptsError as exc:
        print(f"usage error: {exc}", file=sys.stderr)
        return 2
    except ConfigurationError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except TrainingStatsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (LookupError, ValueError) as exc:
        print(json.dumps({"detail": str(exc)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
