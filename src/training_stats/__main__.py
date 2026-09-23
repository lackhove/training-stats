"""Console entry point shim for the training-stats CLI."""

from .adapters.cli.app import main

if __name__ == "__main__":
    raise SystemExit(main())
