"""Coach command line interface.

Interactive terminals (stdout is a TTY) get Rich tables and panels; piped or
redirected output — or an explicit ``--json`` — prints the service payload as
JSON on stdout and nothing else. Diagnostics (sync feedback, errors) go to
stderr. Exit codes: 0 success, 1 domain error (unknown activity, invalid
range), 2 configuration or usage error.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter
from cyclopts.exceptions import CycloptsError
from rich.console import Console

from . import __version__, cli_output, converters
from .config import CONFIG_ENV_VAR, AppConfig, config_path_from_env, load_config
from .errors import ConfigurationError, TrainingStatsError
from .logging_config import setup_logging
from .server import TrainingCoachServer, create_server

logger = logging.getLogger(__name__)

app = App(
    name="training-stats",
    help="Training stats for FIT activity data. Pretty tables on a TTY; JSON otherwise.",
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
        help="Print the payload as JSON on stdout even when attached to a terminal.",
    ),
]
GlobalJsonFlag = Annotated[
    bool,
    Parameter(
        name=["--json"],
        help="Print payloads as JSON on stdout even when attached to a terminal.",
    ),
]


def _load_config(config_path: Path | None) -> AppConfig:
    if config_path is not None:
        return load_config(config_path)
    from_env = config_path_from_env()
    if from_env is not None:
        return load_config(from_env)
    return load_config(Path("config.toml"))


def _service(config_path: Path | None) -> TrainingCoachServer:
    config = _load_config(config_path)
    setup_logging(config.application.log_level)
    return create_server(config)


def _emit(
    payload: object,
    renderer: Callable[[object, Console], None],
    *,
    force_json: bool,
) -> None:
    """Render a payload as a Rich view when stdout is a TTY, else as JSON."""
    if not force_json and sys.stdout.isatty():
        renderer(payload, Console())
    else:
        print(json.dumps(payload, indent=2))


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


def _print_sync_summary(result: dict[str, object]) -> None:
    """Print one sync progress/summary line plus error details to stderr.

    Stdout stays machine-readable: JSON payloads go there and only there.
    """
    print("syncing sources...", file=sys.stderr)
    print(
        "sync: "
        f"{converters.int_value(result['files_discovered'])} found, "
        f"{converters.int_value(result['files_processed'])} new, "
        f"{converters.int_value(result['files_skipped'])} unchanged, "
        f"{converters.int_value(result['files_failed'])} failed",
        file=sys.stderr,
    )
    extras: list[str] = []
    repaired = converters.int_value(result["files_repaired"])
    discarded = converters.int_value(result["segments_discarded"])
    duplicates = converters.int_value(result["duplicates_detected"])
    if repaired:
        extras.append(f"{repaired} repaired")
    if discarded:
        extras.append(f"{discarded} segments discarded")
    if duplicates:
        extras.append(f"{duplicates} duplicate" if duplicates == 1 else f"{duplicates} duplicates")
    if extras:
        print("sync: " + ", ".join(extras), file=sys.stderr)
    source_errors = result["source_errors"]
    if isinstance(source_errors, list):
        for error in source_errors:
            print(f"sync source error: {error}", file=sys.stderr)


def _print_sync_feedback(result: dict[str, object]) -> None:
    """Report the outcome of a (possibly throttled) sync on stderr."""
    if result.get("throttled"):
        seconds = result["seconds_since_last_sync"]
        print(f"sync: skipped (last sync {seconds}s ago)", file=sys.stderr)
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
    _emit(
        service.list_activities(
            start_date=start, end_date=end, sport=sport, limit=limit, sync_first=False
        ),
        cli_output.render_list,
        force_json=json_output,
    )


@app.command
def show(activity_id: str, config: ConfigArg = None, json_output: JsonFlag = False) -> None:
    """Show one activity with metrics, zones, load and data quality."""
    _emit(
        _service(config).get_activity(activity_id),
        cli_output.render_show,
        force_json=json_output,
    )


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
    _emit(
        service.weekly_summary(
            start_date=start, end_date=end, weeks=weeks, sport=sport, sync_first=False
        ),
        cli_output.render_weekly,
        force_json=json_output,
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
    _emit(
        service.overview(days=days, sport=sport, sync_first=False),
        cli_output.render_overview,
        force_json=json_output,
    )


def _startup_sync(service: TrainingCoachServer) -> dict[str, object]:
    """Synchronize sources once before the server starts; never aborts startup."""
    result = service.sync_activities()
    logger.info(
        "startup sync: discovered=%d processed=%d skipped=%d failed=%d%s",
        result["files_discovered"],
        result["files_processed"],
        result["files_skipped"],
        result["files_failed"],
        "" if not result["source_errors"] else f" source_errors={result['source_errors']}",
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

    from .api import build_app

    service = _service(config)
    app_config = service.config
    logger.info("starting %s with config %s", app_config.application.name, config or "config.toml")
    _startup_sync(service)
    bind_host = host if host is not None else app_config.server.host
    _warn_unauthenticated_bind(bind_host, app_config.server.auth is not None)
    uvicorn.run(
        build_app(service),
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
