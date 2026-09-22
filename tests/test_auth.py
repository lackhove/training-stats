"""Tests for shared-secret bearer auth: config, dependency and HTTP behavior."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from tests.conftest import StubSource, make_app_config, make_processor, stub_activity_content
from training_stats.api import build_app
from training_stats.auth import BearerGuard
from training_stats.config import (
    AuthConfig,
    ServerConfig,
    resolve_auth_token,
)
from training_stats.errors import ConfigurationError
from training_stats.services import TrainingStatsService, create_service

TOKEN_ENV = "TRAINING_STATS_AUTH_TOKEN"
TOKEN = "test-token-0123456789abcdef"


class TestAuthConfig:
    def test_default_is_none(self) -> None:
        assert ServerConfig().auth is None

    def test_auth_accepted(self) -> None:
        server_config = ServerConfig(auth=AuthConfig(token_env=TOKEN_ENV))
        assert server_config.auth is not None
        assert server_config.auth.token_env == TOKEN_ENV

    def test_blank_token_env_rejected(self) -> None:
        with pytest.raises(ValueError, match="token_env"):
            AuthConfig(token_env="   ")


class TestResolveAuthToken:
    def test_no_auth_returns_none(self) -> None:
        config = make_app_config(Path("/tmp/cache"), Path("/tmp/activities"))
        assert config.server.auth is None
        assert resolve_auth_token(config) is None

    def test_resolves_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        config = make_app_config(Path("/tmp/cache"), Path("/tmp/activities"))
        config.server.auth = AuthConfig(token_env=TOKEN_ENV)
        monkeypatch.setenv(TOKEN_ENV, TOKEN)
        token = resolve_auth_token(config)
        assert token is not None
        assert token.get_secret_value() == TOKEN

    @pytest.mark.parametrize("missing", ["unset", "empty"])
    def test_unset_env_raises(self, monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
        config = make_app_config(Path("/tmp/cache"), Path("/tmp/activities"))
        config.server.auth = AuthConfig(token_env=TOKEN_ENV)
        if missing == "empty":
            monkeypatch.setenv(TOKEN_ENV, "")
        else:
            monkeypatch.delenv(TOKEN_ENV, raising=False)
        with pytest.raises(ConfigurationError, match=TOKEN_ENV):
            resolve_auth_token(config)


# -- HTTP behavior --------------------------------------------------------------


def _make_server(tmp_path: Path, with_auth: bool) -> TrainingStatsService:
    config = make_app_config(tmp_path / "cache", tmp_path / "src")
    if with_auth:
        config.server.auth = AuthConfig(token_env=TOKEN_ENV)
    server = create_service(config)
    server.sync_service.sources = [
        StubSource(
            files={
                "easy.fit": stub_activity_content(
                    datetime.now(UTC).replace(tzinfo=None, microsecond=0),
                    "cycling",
                    3600.0,
                )
            }
        )
    ]
    server.sync_service.processor = make_processor(
        athlete=config.athlete, analysis_version=server.sync_service.analysis_version
    )
    return server


@pytest.fixture
def token_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TOKEN_ENV, TOKEN)


class TestHttpBehavior:
    def test_accepts_matching_token(self, tmp_path: Path, token_env: None) -> None:
        with TestClient(build_app(_make_server(tmp_path, with_auth=True))) as client:
            response = client.get("/activities", headers={"Authorization": f"Bearer {TOKEN}"})
        assert response.status_code == 200

    def test_rejects_missing_header(self, tmp_path: Path, token_env: None) -> None:
        with TestClient(build_app(_make_server(tmp_path, with_auth=True))) as client:
            response = client.get("/activities")
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"

    def test_rejects_wrong_token(self, tmp_path: Path, token_env: None) -> None:
        with TestClient(build_app(_make_server(tmp_path, with_auth=True))) as client:
            response = client.get("/activities", headers={"Authorization": "Bearer wrong"})
        assert response.status_code == 401

    def test_rejects_empty_token(self, tmp_path: Path, token_env: None) -> None:
        with TestClient(build_app(_make_server(tmp_path, with_auth=True))) as client:
            response = client.get("/activities", headers={"Authorization": "Bearer "})
        assert response.status_code == 401

    def test_rejects_non_bearer_scheme(self, tmp_path: Path, token_env: None) -> None:
        with TestClient(build_app(_make_server(tmp_path, with_auth=True))) as client:
            response = client.get("/activities", headers={"Authorization": f"Basic {TOKEN}"})
        assert response.status_code == 401

    def test_open_server_without_auth_config(self, tmp_path: Path) -> None:
        with TestClient(build_app(_make_server(tmp_path, with_auth=False))) as client:
            assert client.get("/activities").status_code == 200
            assert client.get("/openapi.json").status_code == 200


class TestOpenApiSecurityScheme:
    def test_scheme_declared_when_auth_configured(self, tmp_path: Path, token_env: None) -> None:
        with TestClient(build_app(_make_server(tmp_path, with_auth=True))) as client:
            schema = client.get("/openapi.json").json()
        assert "HTTPBearer" in schema["components"]["securitySchemes"]
        assert schema["paths"]["/activities"]["get"]["security"] == [{"HTTPBearer": []}]
        assert client.get("/openapi.json").status_code == 200

    def test_no_security_requirement_without_auth(self, tmp_path: Path) -> None:
        with TestClient(build_app(_make_server(tmp_path, with_auth=False))) as client:
            schema = client.get("/openapi.json").json()
        assert "securitySchemes" not in schema["components"]
        assert "security" not in schema["paths"]["/activities"]["get"]


class TestGuardRepr:
    def test_repr_masks_secret(self) -> None:
        guard = BearerGuard(SecretStr(TOKEN))
        assert TOKEN not in repr(guard)


class TestOpenBindWarning:
    def test_warns_for_non_loopback_without_auth(self, caplog: pytest.LogCaptureFixture) -> None:
        from training_stats.cli import _warn_unauthenticated_bind

        with caplog.at_level("WARNING", logger="training_stats.cli"):
            _warn_unauthenticated_bind("0.0.0.0", has_auth=False)
        assert any("non-loopback" in record.message for record in caplog.records)

    @pytest.mark.parametrize(
        ("host", "has_auth"),
        [("0.0.0.0", True), ("127.0.0.1", False), ("localhost", False), ("::1", False)],
    )
    def test_no_warning_otherwise(
        self, caplog: pytest.LogCaptureFixture, host: str, has_auth: bool
    ) -> None:
        from training_stats.cli import _warn_unauthenticated_bind

        with caplog.at_level("WARNING", logger="training_stats.cli"):
            _warn_unauthenticated_bind(host, has_auth)
        assert caplog.records == []
