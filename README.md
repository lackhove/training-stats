# training-coach-mcp

**100% AI genereated, use at your own risk!**

A Python [MCP](https://modelcontextprotocol.io) server that reads FIT activity
files (Garmin & friends) from local directories or Nextcloud via WebDAV,
derives training analytics (heart-rate zones, estimated load, weekly
summaries, trends) and exposes them to LLM clients through
[FastMCP](https://gofastmcp.com).

Key properties:

- **Incremental sync** — new/changed files are detected via SHA-256 content
  hashes; unchanged files are never reprocessed.
- **No time-series retention** — high-frequency FIT records are used for
  analytics and then discarded; only aggregates are cached as Parquet files.
- **Deterministic analytics** — HR zones and load are computed from TOML
  configuration, never invented by an LLM. The load method (`hr_trimp`, a
  Banister-style TRIMP estimate) is stored alongside its version and
  confidence.
- **Secret hygiene** — configuration comes from TOML, credentials exclusively
  from environment variables.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
git clone <this-repo> && cd training-coach-mcp
uv sync

# 1. Create a configuration
cp config.example.toml config.toml   # edit paths and athlete values

# 2. Start the MCP server (stdio); it synchronizes your sources automatically
uv run training-coach-mcp config.toml
```

Synchronization runs automatically at server startup and before every
`list_activities` tool call; only new or changed files are processed.

## Configuration

Copy `config.example.toml` to `config.toml`. The configuration path is
resolved in this order:

1. CLI argument: `uv run training-coach-mcp /path/to/config.toml`
2. `TRAINING_MCP_CONFIG` environment variable
3. `./config.toml`

### Local-only setup

```toml
[[sources]]
name = "garmin-local"
type = "local"
path = "/data/activities"
enabled = true
recursive = true
```

No environment variables are required for local sources.

### WebDAV (Nextcloud) setup

1. In Nextcloud, go to **Settings → Security → Devices & sessions** and create
   an **app password** (needed because WebDAV may reject your login password).
2. Point the source at your DAV endpoint, typically
   `https://cloud.example.com/remote.php/dav/files/USERNAME/`.
3. Reference the *names* of environment variables in the config and export the
   values (see `.env.example`):

   ```bash
   export NEXTCLOUD_USERNAME="your-user"
   NEXTCLOUD_APP_PASSWORD="xxxxx-xxxxx-xxxxx-xxxxx-xxxxx"
   ```

4. Set `type = "webdav"` and `enabled = true`.

Combined local + WebDAV configurations are supported: list both `[[sources]]`
blocks; `enabled = false` skips a source without deleting its config.

### Server transport

For local MCP clients (Claude Desktop, Open WebUI desktop, …) keep
`transport = "stdio"`. To serve over HTTP for remote clients such as
[Open WebUI](https://docs.openwebui.com/openapi-server/), set:

```toml
[server]
transport = "streamable-http"   # or "http", "sse"
host = "0.0.0.0"
port = 8000
```

and connect Open WebUI (Admin Panel → Settings → Tools) to
`http://your-host:8000/mcp`.

## Podman deployment (Quadlet)

On servers running [podman](https://podman.io), a Quadlet unit manages the
container as a rootless user service:

```bash
# 1. Build the image (repeat after code or dependency changes)
podman build -t localhost/training-coach-mcp:latest .

# 2. Install the unit; <repo-root> placeholders become this repo's absolute
#    path (run the command from the repository root)
mkdir -p ~/.config/containers/systemd
sed "s|<repo-root>|$PWD|" training-coach-mcp.container \
    > ~/.config/containers/systemd/training-coach-mcp.container

# 3. Start, and start automatically on login
systemctl --user daemon-reload
systemctl --user start training-coach-mcp.service
```

Logs are available via `journalctl --user -u training-coach-mcp -f`; after
rebuilding the image, restart with
`systemctl --user restart training-coach-mcp.service`. The FIT source volume
(`/path/to/activities:/data/activities`) must match the `path` of the local
source in `config.toml`.

### Reaching the MCP from another container

A web UI in its own container cannot use `localhost` — inside that container
localhost is the web UI itself. Both containers join a shared podman network
so they reach each other by container DNS name; the MCP is not published to
the host or LAN:

```bash
cp training-mcp.network ~/.config/containers/systemd/
# add "Network=training-mcp.network" to the web UI's .container file too
systemctl --user daemon-reload
systemctl --user restart training-mcp-network.service training-coach-mcp.service
```

The web UI then connects to `http://training-coach-mcp:<port>/mcp` (container
DNS name; requires `[server] host = "0.0.0.0"` in `config.toml` so the port
is reachable from outside the MCP container).

If a same-host client ever needs direct access, add a loopback-only
`PublishPort=127.0.0.1:<port>:<port>` to the unit — never a plain
`PublishPort=<port>:<port>`, which would expose the MCP to the whole
network.

## MCP tools

| Tool | Purpose |
| --- | --- |
| `list_activities(start_date, end_date, sport, limit)` | Compact activity list (synchronizes sources first) |
| `get_activity(activity_id)` | One activity incl. zones, load, data quality |
| `weekly_summary(start_date, end_date, weeks, sport)` | Weekly totals + previous-week comparison |
| `training_trends(metric, days, sport)` | Daily trend of `duration`/`distance`/`elevation`/`load`/`activities` |
| `get_coach_context(days, sport)` | Compact bundle for coaching conversations |

Read-only resources: `athlete://profile`, `training://zones`,
`training://data-quality`, `training://current-week`.

## Cache layout

```text
data/cache/
├── activities.parquet          # one row per activity
├── activity_metrics.parquet    # derived metrics (zones, load, quality)
├── daily_metrics.parquet       # one row per local date and sport
├── weekly_metrics.parquet      # one row per ISO week and sport
├── source_index.parquet        # per-file hash, status, error diagnostics
└── metadata.json               # last sync, counts, analysis version
```

All writes are atomic (temporary file + rename) and guarded by a lock file.
Synchronization runs at server startup and before every `list_activities`
call; a failed source (e.g. an unreachable WebDAV server) is logged and does
not abort the remaining sources or the tool response. A full rebuild happens
when the internal analysis version in `cache/schemas.py` invalidates derived
data on metric semantics changes.

## Development

```bash
uv run ruff format --check .   # formatting
uv run ruff check .            # linting
uv run mypy --strict src       # static typing
uv run pytest                  # tests
```

## License

MIT — see `LICENSE`.
