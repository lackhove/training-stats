FROM python:3.12-slim

# uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first for better layer caching
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

COPY config.example.toml /app/config/config.example.toml
COPY data/.gitkeep /app/data/.gitkeep

ENV TRAINING_MCP_CONFIG=/app/config/config.toml \
    UV_COMPILE_BYTECODE=1

# Mount your config at /app/config/config.toml and FIT sources somewhere readable.
VOLUME ["/app/config", "/app/data"]

EXPOSE 8000

# WebDAV credentials are injected via environment variables.
CMD ["uv", "run", "--no-dev", "coach", "server"]
