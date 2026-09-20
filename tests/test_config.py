"""Tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from training_coach_mcp.config import (
    load_config,
    resolve_webdav_credentials,
)
from training_coach_mcp.errors import ConfigurationError

VALID_LOCAL = """
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
name = "garmin-local"
type = "local"
path = "/srv/activities"
recursive = true
"""

VALID_WEBDAV = (
    VALID_LOCAL
    + """
[[sources]]
name = "nextcloud"
type = "webdav"
base_url = "https://cloud.example.com/remote.php/dav/files/user/"
path = "Garmin/Activities"
username_env = "NEXTCLOUD_USERNAME"
password_env = "NEXTCLOUD_APP_PASSWORD"
"""
)


def _write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(content, encoding="utf-8")
    return path


class TestValidConfigs:
    def test_local_only(self, tmp_path: Path) -> None:
        config = load_config(_write_config(tmp_path, VALID_LOCAL))
        assert len(config.sources) == 1
        assert config.sources[0].name == "garmin-local"
        assert config.sources[0].type == "local"

    def test_multiple_sources(self, tmp_path: Path) -> None:
        config = load_config(_write_config(tmp_path, VALID_WEBDAV))
        assert len(config.sources) == 2
        assert config.sources[0].name == "garmin-local"

    def test_defaults(self, tmp_path: Path) -> None:
        config = load_config(_write_config(tmp_path, VALID_LOCAL))
        assert config.application.log_level == "INFO"
        assert config.server.transport == "stdio"
        assert config.cache.retain_time_series is False
        assert config.quality.moving_speed_threshold_mps == 1.0
        assert config.quality.decoupling_min_duration_s == 1200.0
        assert config.quality.decoupling_min_coverage == 0.80

    def test_quality_config_custom(self, tmp_path: Path) -> None:
        content = (
            VALID_LOCAL
            + """
[quality]
moving_speed_threshold_mps = 1.5
decoupling_min_duration_s = 600.0
decoupling_min_coverage = 0.70
"""
        )
        config = load_config(_write_config(tmp_path, content))
        assert config.quality.moving_speed_threshold_mps == 1.5
        assert config.quality.decoupling_min_duration_s == 600.0
        assert config.quality.decoupling_min_coverage == 0.70


class TestInvalidConfigs:
    def test_malformed_toml(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="not valid TOML"):
            load_config(_write_config(tmp_path, "[athlete\nmax_heart_rate = "))

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="not found"):
            load_config(tmp_path / "missing.toml")

    def test_invalid_hr_zone_overlap(self, tmp_path: Path) -> None:
        content = VALID_LOCAL.replace("z2_min = 109", "z2_min = 100")
        with pytest.raises(ConfigurationError, match="zone 1 max"):
            load_config(_write_config(tmp_path, content))

    def test_invalid_hr_zone_bounds(self, tmp_path: Path) -> None:
        content = VALID_LOCAL.replace("z3_min = 127", "z3_min = 200")
        with pytest.raises(ConfigurationError, match="zone 3"):
            load_config(_write_config(tmp_path, content))

    def test_resting_above_max(self, tmp_path: Path) -> None:
        content = VALID_LOCAL.replace("resting_heart_rate = 55", "resting_heart_rate = 190")
        with pytest.raises(ConfigurationError, match="resting_heart_rate"):
            load_config(_write_config(tmp_path, content))

    def test_duplicate_source_names(self, tmp_path: Path) -> None:
        content = (
            VALID_LOCAL
            + """
[[sources]]
name = "garmin-local"
type = "local"
path = "/other"
"""
        )
        with pytest.raises(ConfigurationError, match="unique"):
            load_config(_write_config(tmp_path, content))

    def test_no_sources(self, tmp_path: Path) -> None:
        content = VALID_LOCAL.split("[[sources]]")[0]
        with pytest.raises(ConfigurationError, match="at least one"):
            load_config(_write_config(tmp_path, content))

    def test_invalid_log_level(self, tmp_path: Path) -> None:
        content = '[application]\nlog_level = "LOUD"\n' + VALID_LOCAL
        with pytest.raises(ConfigurationError, match="log_level"):
            load_config(_write_config(tmp_path, content))

    def test_invalid_transport(self, tmp_path: Path) -> None:
        content = VALID_LOCAL + '[server]\ntransport = "carrier-pigeon"\n'
        with pytest.raises(ConfigurationError, match="transport"):
            load_config(_write_config(tmp_path, content))

    def test_unknown_key(self, tmp_path: Path) -> None:
        content = VALID_LOCAL + "[cache]\nsurprise = true\n"
        with pytest.raises(ConfigurationError, match="surprise"):
            load_config(_write_config(tmp_path, content))


class TestSecrets:
    def test_webdav_secret_resolution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NEXTCLOUD_USERNAME", "user@example.com")
        monkeypatch.setenv("NEXTCLOUD_APP_PASSWORD", "app-pass")
        config = load_config(_write_config(tmp_path, VALID_WEBDAV))
        resolved = resolve_webdav_credentials(config)
        assert resolved["nextcloud"].username == "user@example.com"
        assert resolved["nextcloud"].password == "app-pass"

    def test_missing_username_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NEXTCLOUD_USERNAME", raising=False)
        monkeypatch.delenv("NEXTCLOUD_APP_PASSWORD", raising=False)
        config = load_config(_write_config(tmp_path, VALID_WEBDAV))
        with pytest.raises(ConfigurationError, match="NEXTCLOUD_USERNAME"):
            resolve_webdav_credentials(config)

    def test_missing_password_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEXTCLOUD_USERNAME", "user@example.com")
        monkeypatch.delenv("NEXTCLOUD_APP_PASSWORD", raising=False)
        config = load_config(_write_config(tmp_path, VALID_WEBDAV))
        with pytest.raises(ConfigurationError, match="NEXTCLOUD_APP_PASSWORD"):
            resolve_webdav_credentials(config)

    def test_unknown_source_key_rejected(self, tmp_path: Path) -> None:
        content = VALID_WEBDAV + "enabled = false\n"
        with pytest.raises(ConfigurationError, match="enabled"):
            load_config(_write_config(tmp_path, content))

    def test_credentials_repr_masks_values(self) -> None:
        from pydantic import SecretStr

        from training_coach_mcp.config import WebDAVCredentials

        credentials = WebDAVCredentials(username=SecretStr("hunter2"), password=SecretStr("s3cret"))
        assert "hunter2" not in repr(credentials)
        assert "s3cret" not in repr(credentials)
        assert credentials.username == "hunter2"
