# training-stats

**100% AI genereated, use at your own risk!**

A Python command line tool and REST API that reads FIT activity files (Garmin &
friends) from local directories or Nextcloud via WebDAV and derives training
analytics: heart-rate zones, estimated load, weekly summaries and a compact
coaching overview.

Key properties:

- **Incremental sync** — new/changed files are detected via file size and
  modification time; unchanged files are never re-read or reprocessed.
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
git clone <this-repo> && cd training-stats
uv sync

# 1. Create a configuration
cp config.example.toml config.toml   # edit paths and athlete values

# 2. Use the CLI; queries synchronize your sources automatically
uv run training-stats list --limit 10
```

Synchronization runs automatically at server startup and before every query
(`list`, `weekly`, `overview`; `show` serves the cache as-is). It is
throttled: a real sync runs at most once per
`cache.sync_min_interval_seconds` (default 10 s), so bursts of commands or
API requests contact your sources only once — set the interval to `0` to
always synchronize, `--no-sync` to never synchronize for one call. On an
interactive terminal commands render Rich tables and panels; when stdout is
piped or redirected — or `--json` is given — they print JSON on stdout and
nothing else, while progress and diagnostics (sync summary, source errors)
go to stderr, so piping stays clean: `training-stats list | jq '.[].sport'`.

## Commands

| Command | Purpose |
| --- | --- |
| `training-stats list [--sport --start --end --limit --no-sync]` | Compact activity summaries, newest first (synchronizes if due) |
| `training-stats show <activity-id>` | One activity incl. zones, load, data quality (serves cached data) |
| `training-stats weekly [--weeks --sport --start --end --no-sync]` | Weekly totals + previous-week comparison (synchronizes if due) |
| `training-stats overview [--days --sport --no-sync]` | Compact bundle: profile, daily series, weekly summaries, form, quality (synchronizes if due) |
| `training-stats server [--host --port]` | Run the REST API (OpenAPI at `/openapi.json`, docs at `/docs`) |

Global options: `--config/-c` and `--json` (before or after the verb),
`--help`, `--version`.
The overview's daily series carries per-(date, sport) totals for every metric,
so any metric trend is a simple filter and projection over it.

## Configuration

Copy `config.example.toml` to `config.toml`. The configuration path is
resolved in this order:

1. CLI option: `uv run training-stats -c /path/to/config.toml list`
2. `TRAINING_STATS_CONFIG` environment variable
3. `./config.toml`

### Local-only setup

```toml
[[sources]]
name = "garmin-local"
type = "local"
path = "/data/activities"
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
   export NEXTCLOUD_APP_PASSWORD="xxxxx-xxxxx-xxxxx-xxxxx-xxxxx"
   ```

4. Set `type = "webdav"` in the source block.

Combined local + WebDAV configurations are supported: list both `[[sources]]`
blocks; every entry is active.

## REST API

```bash
uv run training-stats server            # binds [server] host/port from config.toml
uv run training-stats server --host 0.0.0.0 --port 8000
```

| Endpoint | Purpose |
| --- | --- |
| `GET /activities` | Compact activity list (synchronizes if due); `start_date`, `end_date`, `sport`, `limit` |
| `GET /activities/{activity_id}` | One activity incl. zones, load, data quality (serves cached data) |
| `GET /weekly` | Weekly totals + previous-week comparison (synchronizes if due); `weeks`, `sport`, dates |
| `GET /overview` | Coaching overview (synchronizes if due); `days`, `sport` |

The OpenAPI schema is served at `/openapi.json` (interactive docs at
`/docs`); response payloads are fully typed, and each syncing endpoint's
description documents the throttled synchronization.

### Auth (Bearer token)

Once the server is reachable beyond localhost, add a shared-secret bearer
token so only clients that know the secret may call it. The token lives in
an environment variable; `config.toml` only names the variable:

```toml
[server.auth]
token_env = "TRAINING_STATS_AUTH_TOKEN"
```

```bash
export TRAINING_STATS_AUTH_TOKEN="$(openssl rand -hex 32)"
```

Without this section the server accepts unauthenticated requests (starting
`training-stats server` on a non-loopback host without auth logs a warning); with it,
every data request must carry `Authorization: Bearer <token>` while
`/openapi.json` and `/docs` stay reachable for schema discovery. Rotate the
token by changing the env value and restarting the server. The CLI verbs
are unaffected — they read the cache directly and never use the token.

For Open WebUI: add the server as an external OpenAPI tool server
(**Admin Panel → Settings → Tools**), enter the base URL, choose
**Auth: Bearer** and paste the token into the key field — leave it empty
only if the server runs without `[server.auth]`. A plain `curl` passes the
same header: `-H "Authorization: Bearer <token>"`.

### Open WebUI integration

[Open WebUI](https://docs.openwebui.com/openapi-server/) can consume the API
as an external OpenAPI tool server: add `http://your-host:8000` (the server
base URL, not a special path) as an OpenAPI tool server in
**Admin Panel → Settings → Tools**. Open WebUI reads `/openapi.json` and maps
the endpoints to callable tools.

## Podman deployment (Quadlet)

On servers running [podman](https://podman.io), a Quadlet unit manages the
container as a rootless user service:

```bash
# 1. Pull the image (repeat to pick up new versions)
podman pull ghcr.io/lackhove/training-stats:latest

# 2. Install the unit; <repo-root> placeholders become this repo's absolute
#    path (run the command from the repository root)
mkdir -p ~/.config/containers/systemd
sed "s|<repo-root>|$PWD|" training-stats.container \
    > ~/.config/containers/systemd/training-stats.container

# 3. Start, and start automatically on login
systemctl --user daemon-reload
systemctl --user start training-stats.service
```

Logs are available via `journalctl --user -u training-stats -f`; after
pulling a new image, restart with
`systemctl --user restart training-stats.service`. The FIT source volume
(`/path/to/activities:/data/activities`) must match the `path` of the local
source in `config.toml`.

### Reaching the API from another container

A web UI in its own container cannot use `localhost` — inside that container
localhost is the web UI itself. Both containers join a shared podman network
so they reach each other by container DNS name; the API is not published to
the host or LAN:

```bash
cp training-stats.network ~/.config/containers/systemd/
# add "Network=training-stats.network" to the web UI's .container file too
systemctl --user daemon-reload
systemctl --user restart training-stats-network.service training-stats.service
```

The web UI then connects to `http://training-stats:<port>` (container
DNS name; requires `[server] host = "0.0.0.0"` in `config.toml` so the port
is reachable from outside the API container).

If a same-host client ever needs direct access, add a loopback-only
`PublishPort=127.0.0.1:<port>:<port>` to the unit — never a plain
`PublishPort=<port>:<port>`, which would expose the API to the whole
network.

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
Synchronization runs at server startup and before every query command or
endpoint (`show`/`GET /activities/{id}` serve the cache as-is), throttled
by `cache.sync_min_interval_seconds` (default 10 s, anchored at the
`last_sync` timestamp in `metadata.json`); a failed source (e.g. an
unreachable WebDAV server) is logged and does not abort the remaining
sources or the response, and a failed sync is never recorded as "recent".
A full rebuild happens when the internal analysis version in
`cache/schemas.py` invalidates derived data on metric semantics changes.

## Development

```bash
uv run ruff format --check .   # formatting
uv run ruff check .            # linting
uv run mypy --strict src       # static typing
uv run pytest                  # tests
```

## License

MIT — see `LICENSE`.
