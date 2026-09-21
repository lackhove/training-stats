"""Coach command line interface (JSON-only output).

Every verb prints the service payload as JSON on stdout and nothing else;
log output goes to stderr. Exit codes: 0 success, 1 domain error (unknown
activity, invalid range), 2 configuration or usage error.
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

from . import __version__
from .config import CONFIG_ENV_VAR, AppConfig, config_path_from_env, load_config
from .errors import ConfigurationError
from .logging_config import setup_logging
from .server import TrainingCoachServer, create_server

logger = logging.getLogger(__name__)

app = App(
    name="coach",
    help="Training coach for FIT activity data. All output is JSON.",
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


def _load_config(config_path: Path | None) -> AppConfig:
    if config_path is not None:
        return load_config(config_path)
    from_env = config_path_from_env()
    if from_env is not None:
        return load_config(from_env)
    return load_config(Path("config.toml"))


def _service(config_path: Path | None) -> TrainingCoachServer:
    return create_server(_load_config(config_path))


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2))


@app.meta.default
def _launcher(
    *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
    config: GlobalConfigArg = None,
) -> None:
    """Consume global options so they may appear before the command."""
    forwarded: list[str] = []
    if config is not None:
        forwarded += ["--config", str(config)]
    app(
        list(tokens) + forwarded,
        exit_on_error=False,
        print_error=False,
        result_action="return_none",
    )


@app.command(name="list")
def list_activities(
    config: ConfigArg = None,
    start: StartArg = None,
    end: EndArg = None,
    sport: SportArg = None,
    limit: LimitArg = 50,
    sync_first: Annotated[
        bool,
        Parameter(
            name=["--sync-first"],
            negative=["--no-sync"],
            help="Synchronize sources before listing (default: enabled).",
        ),
    ] = True,
) -> None:
    """List compact activity summaries, newest first."""
    service = _service(config)
    _print(
        service.list_activities(
            start_date=start, end_date=end, sport=sport, limit=limit, sync_first=sync_first
        )
    )


@app.command
def show(activity_id: str, config: ConfigArg = None) -> None:
    """Show one activity with metrics, zones, load and data quality."""
    _print(_service(config).get_activity(activity_id))


@app.command
def weekly(
    config: ConfigArg = None,
    weeks: Annotated[int, Parameter(help="Number of past ISO weeks.")] = 8,
    sport: SportArg = None,
    start: StartArg = None,
    end: EndArg = None,
) -> None:
    """Show weekly training totals with previous-week comparisons."""
    _print(
        _service(config).weekly_summary(start_date=start, end_date=end, weeks=weeks, sport=sport)
    )


@app.command
def overview(
    config: ConfigArg = None,
    days: Annotated[int, Parameter(help="Length of the period ending today.")] = 28,
    sport: SportArg = None,
) -> None:
    """Show the coaching overview: profile, daily series, form, quality."""
    _print(_service(config).overview(days=days, sport=sport))


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

    app_config = _load_config(config)
    setup_logging(app_config.application.log_level)
    logger.info("starting %s with config %s", app_config.application.name, config or "config.toml")
    service = create_server(app_config)
    _startup_sync(service)
    uvicorn.run(
        build_app(service),
        host=host if host is not None else app_config.server.host,
        port=port if port is not None else app_config.server.port,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the coach CLI; returns the process exit code."""
    try:
        app.meta(argv, exit_on_error=False, print_error=False, result_action="return_none")
    except CycloptsError as exc:
        print(f"usage error: {exc}", file=sys.stderr)
        return 2
    except ConfigurationError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except (LookupError, ValueError) as exc:
        print(json.dumps({"detail": str(exc)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
