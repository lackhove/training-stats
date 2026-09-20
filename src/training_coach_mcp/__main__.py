"""Command line entry point for the training coach MCP server."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import CONFIG_ENV_VAR, config_path_from_env, load_config
from .errors import ConfigurationError
from .logging_config import setup_logging
from .server import TrainingCoachServer, create_server

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="training-coach-mcp", description="MCP server for FIT-based training analytics"
    )
    parser.add_argument(
        "config",
        nargs="?",
        help=f"path to the TOML configuration file (default: ${CONFIG_ENV_VAR} or ./config.toml)",
    )
    return parser


def _resolve_config_path(args: argparse.Namespace) -> Path:
    if args.config is not None:
        return Path(args.config)
    from_env = config_path_from_env()
    if from_env is not None:
        return from_env
    return Path("config.toml")


def _startup_sync(server: TrainingCoachServer) -> None:
    """Synchronize sources once before the server starts; never aborts startup."""
    result = server.sync_activities()
    logger.info(
        "startup sync: discovered=%d processed=%d skipped=%d failed=%d%s",
        result["files_discovered"],
        result["files_processed"],
        result["files_skipped"],
        result["files_failed"],
        "" if not result["source_errors"] else f" source_errors={result['source_errors']}",
    )


def main(argv: list[str] | None = None) -> int:
    """Run the MCP server with the configured transport."""
    args = _build_parser().parse_args(argv)
    config_path = _resolve_config_path(args)
    try:
        config = load_config(config_path)
    except ConfigurationError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    setup_logging(config.application.log_level)
    logger.info("starting %s with config %s", config.application.name, config_path)

    server = create_server(config)
    _startup_sync(server)
    mcp = server.build_mcp()
    transport = config.server.transport
    if transport in ("http", "sse", "streamable-http"):
        mcp.run(transport=transport, host=config.server.host, port=config.server.port)
    else:
        mcp.run(transport=transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
