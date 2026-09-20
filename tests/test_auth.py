"""Tests for bearer token auth: verifier, config, wiring and HTTP behavior."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr
from starlette.testclient import TestClient

from tests.conftest import make_app_config
from training_coach_mcp.auth import SharedSecretBearerVerifier
from training_coach_mcp.config import AuthConfig, ServerConfig, load_config, resolve_auth_token
from training_coach_mcp.errors import ConfigurationError
from training_coach_mcp.server import create_server

TOKEN_ENV = "TRAINING_MCP_AUTH_TOKEN"
TOKEN = "test-token-0123456789abcdef"


def _write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(content, encoding="utf-8")
    return path


def _local_config() -> str:
    return """
[athlete]
max_heart_rate = 180
resting_heart_rate = 55

[athlete.heart_rate_zones]
z1_min = 90
z1_max = 108
z2_min = 109
z2_max = 126
z3_min = 127
z3_max = 144
z4_min = 145
z4_max = 162
z5_min = 163
z5_max = 180

[[sources]]
name = "local"
type = "local"
path = "/srv/activities"
"""


class TestSharedSecretBearerVerifier:
    def test_accepts_matching_token(self) -> None:
        verifier = SharedSecretBearerVerifier(SecretStr(TOKEN))
        access = asyncio.run(verifier.verify_token(TOKEN))
        assert access is not None
        assert access.client_id == "bearer-client"
        assert access.scopes == []

    def test_rejects_wrong_token(self) -> None:
        verifier = SharedSecretBearerVerifier(SecretStr(TOKEN))
        assert asyncio.run(verifier.verify_token("wrong-token")) is None

    def test_rejects_empty_token(self) -> None:
        verifier = SharedSecretBearerVerifier(SecretStr(TOKEN))
        assert asyncio.run(verifier.verify_token("")) is None

    def test_repr_masks_secret(self) -> None:
        verifier = SharedSecretBearerVerifier(SecretStr(TOKEN))
        assert TOKEN not in repr(verifier)


class TestAuthConfig:
    def test_auth_with_stdio_rejected(self) -> None:
        with pytest.raises(ValueError, match="HTTP transports"):
            ServerConfig(transport="stdio", auth=AuthConfig(token_env=TOKEN_ENV))

    @pytest.mark.parametrize("transport", ["http", "sse", "streamable-http"])
    def test_auth_with_http_transports_allowed(self, transport: str) -> None:
        server_config = ServerConfig(transport=transport, auth=AuthConfig(token_env=TOKEN_ENV))
        assert server_config.auth is not None
        assert server_config.auth.token_env == TOKEN_ENV

    def test_blank_token_env_rejected(self) -> None:
        with pytest.raises(ValueError, match="token_env"):
            AuthConfig(token_env="   ")


class TestResolveAuthToken:
    def test_no_auth_returns_none(self, tmp_path: Path) -> None:
        config = load_config(_write_config(tmp_path, _local_config()))
        assert config.server.auth is None
        assert resolve_auth_token(config) is None

    def test_resolves_from_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        content = _local_config() + (
            f'[server]\ntransport = "streamable-http"\n[server.auth]\ntoken_env = "{TOKEN_ENV}"\n'
        )
        config = load_config(_write_config(tmp_path, content))
        monkeypatch.setenv(TOKEN_ENV, TOKEN)
        token = resolve_auth_token(config)
        assert token is not None
        assert token.get_secret_value() == TOKEN

    def test_missing_env_raises_with_var_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        content = _local_config() + (
            f'[server]\ntransport = "streamable-http"\n[server.auth]\ntoken_env = "{TOKEN_ENV}"\n'
        )
        config = load_config(_write_config(tmp_path, content))
        monkeypatch.delenv(TOKEN_ENV, raising=False)
        with pytest.raises(ConfigurationError, match=TOKEN_ENV):
            resolve_auth_token(config)

    def test_empty_env_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        content = _local_config() + (
            f'[server]\ntransport = "streamable-http"\n[server.auth]\ntoken_env = "{TOKEN_ENV}"\n'
        )
        config = load_config(_write_config(tmp_path, content))
        monkeypatch.setenv(TOKEN_ENV, "")
        with pytest.raises(ConfigurationError, match=TOKEN_ENV):
            resolve_auth_token(config)

    def test_invalid_transport_toml(self, tmp_path: Path) -> None:
        content = _local_config() + (
            f'[server]\ntransport = "stdio"\n[server.auth]\ntoken_env = "{TOKEN_ENV}"\n'
        )
        with pytest.raises(ConfigurationError, match="HTTP transports"):
            load_config(_write_config(tmp_path, content))


def _auth_server_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    config = make_app_config(tmp_path / "cache", tmp_path / "src")
    config.server = ServerConfig(transport="streamable-http", auth=AuthConfig(token_env=TOKEN_ENV))
    server = create_server(config)
    mcp = server.build_mcp()
    assert mcp.auth is not None
    return mcp.http_app(transport="streamable-http")


_INIT_REQUEST = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"},
    },
}
_HEADERS = {"Accept": "application/json, text/event-stream"}


class TestBearerAuthHttp:
    def test_missing_and_wrong_tokens_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = _auth_server_app(tmp_path, monkeypatch)
        with TestClient(app) as client:
            response = client.post("/mcp", json=_INIT_REQUEST, headers=_HEADERS)
            assert response.status_code == 401
            assert response.headers["www-authenticate"] == "Bearer"

            response = client.post(
                "/mcp",
                json=_INIT_REQUEST,
                headers=_HEADERS | {"Authorization": "Bearer wrong-token"},
            )
            assert response.status_code == 401

            response = client.post(
                "/mcp",
                json=_INIT_REQUEST,
                headers=_HEADERS | {"Authorization": f"Bearer {TOKEN}"},
            )
            assert response.status_code == 200

    def test_open_bearer_header_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = _auth_server_app(tmp_path, monkeypatch)
        with TestClient(app) as client:
            response = client.post(
                "/mcp", json=_INIT_REQUEST, headers=_HEADERS | {"Authorization": "Bearer"}
            )
            assert response.status_code == 401

    def test_unauthenticated_server_stays_open(self, tmp_path: Path) -> None:
        config = make_app_config(tmp_path / "cache", tmp_path / "src")
        server = create_server(config)
        mcp = server.build_mcp()
        assert mcp.auth is None
        app = mcp.http_app(transport="streamable-http")
        with TestClient(app) as client:
            response = client.post("/mcp", json=_INIT_REQUEST, headers=_HEADERS)
            assert response.status_code == 200
