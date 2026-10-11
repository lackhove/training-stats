"""Tests for the cyclopts-based training-stats CLI (JSON output, exit codes)."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from tests.conftest import (
    FIXTURES,
    T0,
    assert_no_none_fields,
    json_activity_ids,
    json_metric,
    json_metrics,
    make_server,
    stub_activity_content,
)
from training_stats.adapters.cli import app as cli_app
from training_stats.adapters.cli.app import main


def _write_config(tmp_path: Path, interval_s: float) -> Path:
    """Real FIT fixtures as local source; isolated cache directory."""
    activities = tmp_path / "activities"
    activities.mkdir()
    shutil.copy(FIXTURES / "cycling.fit", activities / "cycling.fit")
    shutil.copy(FIXTURES / "activity_2.fit", activities / "activity_2.fit")
    cache_dir = tmp_path / "cache"
    config = tmp_path / "config.toml"
    config.write_text(
        "\n".join(
            [
                "[application]",
                'name = "training-stats-test"',
                'timezone = "Europe/Berlin"',
                "",
                "[athlete]",
                "max_heart_rate = 180",
                "resting_heart_rate = 55",
                "",
                "[athlete.heart_rate_zones]",
                "z1_min = 90",
                "z1_max = 108",
                "z2_min = 109",
                "z2_max = 126",
                "z3_min = 127",
                "z3_max = 144",
                "z4_min = 145",
                "z4_max = 162",
                "z5_min = 163",
                "z5_max = 180",
                "",
                "[[sources]]",
                'name = "local"',
                'type = "local"',
                f'path = "{activities}"',
                "",
                "[cache]",
                f'directory = "{cache_dir}"',
                f"sync_min_interval_seconds = {interval_s}",
            ]
        )
    )
    return config


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    """Config with the sync throttle disabled (interval 0)."""
    return _write_config(tmp_path, interval_s=0.0)


@pytest.fixture
def throttled_config_path(tmp_path: Path) -> Path:
    """Config with the default sync throttle (10 s)."""
    return _write_config(tmp_path, interval_s=10.0)


def _stdout_payload(capsys: pytest.CaptureFixture[str]) -> object:
    return json.loads(capsys.readouterr().out)


class TestListCommand:
    def test_list_and_repeat_sync(
        self, config_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["--config", str(config_path), "list"]) == 0
        captured = capsys.readouterr()
        activities = json.loads(captured.out)
        assert_no_none_fields(activities)
        assert len(json_activity_ids(activities)) == 2
        assert all("metrics" not in activity for activity in activities)
        assert set(activities[0]) == {
            "activity_id",
            "start_time",
            "sport",
            "sub_sport",
        }
        assert {activity["sport"] for activity in activities} == {"cycling"}
        # Sync feedback goes to stderr; stdout stays pure JSON.
        assert "syncing sources..." in captured.err
        assert "sync: 2 found, 2 new, 0 unchanged, 0 failed" in captured.err

        # Second run: unchanged files are skipped via content hashes.
        assert main(["--config", str(config_path), "list"]) == 0
        captured = capsys.readouterr()
        assert len(json_activity_ids(json.loads(captured.out))) == 2
        assert "sync: 2 found, 0 new, 2 unchanged, 0 failed" in captured.err

    def test_config_before_command(
        self, config_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["-c", str(config_path), "list"]) == 0
        assert len(json_activity_ids(_stdout_payload(capsys))) == 2

    def test_no_sync_serves_empty_cache(
        self, config_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["-c", str(config_path), "list", "--no-sync"]) == 0
        captured = capsys.readouterr()
        assert json.loads(captured.out) == []
        assert "sync" not in captured.err

    def test_sport_filter(self, config_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["-c", str(config_path), "list", "--sport", "cycling"]) == 0
        activities = _stdout_payload(capsys)
        assert len(json_activity_ids(activities)) == 2

    def test_limit_type_error(self, config_path: Path) -> None:
        assert main(["-c", str(config_path), "list", "--limit", "abc"]) == 2


class TestShowCommand:
    def test_show_known_activity(
        self, config_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["-c", str(config_path), "list", "--limit", "1"]) == 0
        activity_id = next(iter(json_activity_ids(_stdout_payload(capsys))))

        assert main(["-c", str(config_path), "show", activity_id]) == 0
        details = _stdout_payload(capsys)
        assert_no_none_fields(details)
        assert details["activity_id"] == activity_id
        assert "start_time" in details
        assert "sport" in details
        assert "metrics" in details
        assert all("period" not in metric for metric in details["metrics"])
        assert json_metric(details, "training_load.hr_trimp")["method"] == "hr_trimp"
        assert "records" not in details

    def test_show_unknown_activity(
        self, config_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["-c", str(config_path), "show", "does-not-exist"]) == 1
        captured = capsys.readouterr()
        assert json.loads(captured.err)["detail"].startswith("activity not found")
        assert_no_none_fields(json.loads(captured.err))
        assert captured.out == ""

    def test_show_omits_analysis_for_unmeasured_quantities(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server = make_server(
            tmp_path,
            {"no-sensors.fit": stub_activity_content(T0, "running", 1800.0, 8000.0)},
        )
        server.sync_service.sync()
        activity_id = next(iter(server.repository.load_sync_state().activities))
        monkeypatch.setattr(cli_app, "_service", lambda _config: server)

        assert main(["show", activity_id]) == 0

        report = json.loads(capsys.readouterr().out)
        names = {metric["name"] for metric in report["metrics"]}
        assert "heart_rate.zone_time" not in names
        assert "training_load.hr_trimp" not in names
        assert "activity.statistics.cadence_statistics.overall.mean" not in names
        assert "activity.statistics.power_statistics.overall.mean" not in names
        assert json_metric(report, "heart_rate.coverage")["value"]["value"] == 0.0
        assert "present_quantities" not in report
        assert "records" not in report


class TestWeeklyCommand:
    def test_weekly_syncs_first(
        self, config_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Weekly synchronizes on its own; no prior list needed.
        assert main(["-c", str(config_path), "weekly", "--weeks", "4"]) == 0
        captured = capsys.readouterr()
        assert "syncing sources..." in captured.err
        report = json.loads(captured.out)
        assert_no_none_fields(report)
        assert report["period"]["kind"] == "date_range"
        assert all("period" not in metric for metric in report["metrics"])
        assert json_metric(report, "activity.distance_by_week")["value"]["shape"] == "table"

    def test_weekly_no_sync_serves_empty_cache(
        self, config_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["-c", str(config_path), "weekly", "--no-sync"]) == 0
        captured = capsys.readouterr()
        report = json.loads(captured.out)
        assert_no_none_fields(report)
        assert report["period"]["kind"] == "date_range"
        assert all("period" not in metric for metric in report["metrics"])
        assert all("value" not in metric for metric in json_metrics(report))
        assert "sync" not in captured.err


class TestOverviewCommand:
    def test_overview_sections(self, config_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        # Overview synchronizes on its own; no prior list needed.
        assert main(["-c", str(config_path), "overview", "--days", "14"]) == 0
        overview = _stdout_payload(capsys)
        assert_no_none_fields(overview)
        assert all(metric["period"]["kind"] != "trip" for metric in overview["metrics"])
        assert json_metric(overview, "athlete_profile.max_heart_rate")["value"]["value"] == 180
        assert json_metric(overview, "data_quality.activity_count")["value"]["value"] == 2
        assert json_metric(overview, "training_form.ctl_by_day")["value"]["shape"] == "table"


class TestErrorExits:
    def test_missing_config_file(self, tmp_path: Path) -> None:
        assert main(["--config", str(tmp_path / "nope.toml"), "list"]) == 2

    def test_unknown_option(self, config_path: Path) -> None:
        assert main(["-c", str(config_path), "list", "--nonsense"]) == 2

    def test_unknown_command(self, config_path: Path) -> None:
        assert main(["-c", str(config_path), "frobnicate"]) == 2


class TestSyncThrottleFeedback:
    def test_each_query_command_checks_sync_exactly_once(
        self, config_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from training_stats.services.training import TrainingStatsService

        checks: list[bool] = []
        original = TrainingStatsService.sync_if_due

        def counting_sync_if_due(self: TrainingStatsService) -> object:
            checks.append(True)
            return original(self)

        monkeypatch.setattr(TrainingStatsService, "sync_if_due", counting_sync_if_due)

        assert main(["-c", str(config_path), "list", "--no-sync"]) == 0
        capsys.readouterr()
        assert checks == []  # --no-sync skips the check entirely

        assert main(["-c", str(config_path), "list"]) == 0
        capsys.readouterr()
        assert len(checks) == 1

        assert main(["-c", str(config_path), "show", "does-not-exist"]) == 1
        capsys.readouterr()
        assert len(checks) == 1  # show serves the cache without syncing

        assert main(["-c", str(config_path), "weekly"]) == 0
        capsys.readouterr()
        assert len(checks) == 2

        assert main(["-c", str(config_path), "overview"]) == 0
        capsys.readouterr()
        assert len(checks) == 3

    def test_second_command_within_interval_is_skipped(
        self, throttled_config_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # First run performs a real sync and stamps metadata.json.
        assert main(["-c", str(throttled_config_path), "list"]) == 0
        first = capsys.readouterr()
        assert "sync: 2 found" in first.err

        # Fresh process: the persisted last_sync timestamp throttles the sync.
        assert main(["-c", str(throttled_config_path), "list"]) == 0
        second = capsys.readouterr()
        assert "sync: skipped (last sync" in second.err
        assert "syncing sources..." not in second.err
        assert len(json_activity_ids(json.loads(second.out))) == 2

    def test_zero_interval_reports_real_sync(
        self, config_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["-c", str(config_path), "list"]) == 0
        captured = capsys.readouterr()
        assert "sync: 2 found, 2 new, 0 unchanged, 0 failed" in captured.err
        assert "sync: skipped" not in captured.err


class TestSyncSummaryFeedback:
    def test_extras_and_source_errors(self, capsys: pytest.CaptureFixture[str]) -> None:
        from training_stats.adapters.cli.app import _print_sync_summary
        from training_stats.services.results import SyncResult

        _print_sync_summary(
            SyncResult(
                files_discovered=12,
                files_processed=2,
                files_skipped=10,
                files_failed=0,
                files_repaired=1,
                segments_discarded=3,
                duplicates_detected=1,
                source_errors=("nextcloud: connection refused",),
            )
        )
        err = capsys.readouterr().err
        assert "syncing sources..." in err
        assert "sync: 12 found, 2 new, 10 unchanged, 0 failed" in err
        assert "sync: 1 repaired, 3 segments discarded, 1 duplicate" in err
        assert "sync source error: nextcloud: connection refused" in err

    def test_no_extras_lines_when_clean(self, capsys: pytest.CaptureFixture[str]) -> None:
        from training_stats.adapters.cli.app import _print_sync_summary
        from training_stats.services.results import SyncResult

        _print_sync_summary(
            SyncResult(
                files_discovered=4,
                files_processed=0,
                files_skipped=4,
                files_failed=0,
            )
        )
        err = capsys.readouterr().err.splitlines()
        assert len(err) == 2  # progress + summary, no extras/source lines
        assert all("repaired" not in line for line in err)


@pytest.fixture
def interactive(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend stdout is attached to a terminal."""
    monkeypatch.setattr(type(sys.stdout), "isatty", lambda self: True)


class TestJsonOnlyOutput:
    def test_list_emits_json_when_stdout_is_a_tty(
        self,
        config_path: Path,
        capsys: pytest.CaptureFixture[str],
        interactive: None,
    ) -> None:
        assert main(["-c", str(config_path), "list"]) == 0
        captured = capsys.readouterr()
        activities = json.loads(captured.out)
        assert len(json_activity_ids(activities)) == 2
        assert {activity["sport"] for activity in activities} == {"cycling"}
        assert "sync: 2 found" in captured.err

    def test_show_emits_json_when_stdout_is_a_tty(
        self,
        config_path: Path,
        capsys: pytest.CaptureFixture[str],
        interactive: None,
    ) -> None:
        assert main(["-c", str(config_path), "--json", "list", "--limit", "1"]) == 0
        activity_id = next(iter(json_activity_ids(_stdout_payload(capsys))))

        assert main(["-c", str(config_path), "show", activity_id]) == 0
        details = _stdout_payload(capsys)
        assert details["activity_id"] == activity_id
        assert json_metric(details, "training_load.hr_trimp")["method"] == "hr_trimp"

    def test_overview_emits_json_when_stdout_is_a_tty(
        self,
        config_path: Path,
        capsys: pytest.CaptureFixture[str],
        interactive: None,
    ) -> None:
        assert main(["-c", str(config_path), "overview", "--days", "14"]) == 0
        overview = _stdout_payload(capsys)
        assert json_metric(overview, "athlete_profile.max_heart_rate")["value"]["value"] == 180

    def test_json_flag_before_and_after_command(
        self,
        config_path: Path,
        capsys: pytest.CaptureFixture[str],
        interactive: None,
    ) -> None:
        assert main(["-c", str(config_path), "--json", "list"]) == 0
        assert len(json_activity_ids(json.loads(capsys.readouterr().out))) == 2

        assert main(["-c", str(config_path), "list", "--json"]) == 0
        assert len(json_activity_ids(json.loads(capsys.readouterr().out))) == 2

    def test_domain_error_is_json_even_interactive(
        self,
        config_path: Path,
        capsys: pytest.CaptureFixture[str],
        interactive: None,
    ) -> None:
        assert main(["-c", str(config_path), "show", "does-not-exist"]) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert json.loads(captured.err)["detail"].startswith("activity not found")

    def test_domain_error_stays_json_with_json_flag(
        self,
        config_path: Path,
        capsys: pytest.CaptureFixture[str],
        interactive: None,
    ) -> None:
        assert main(["-c", str(config_path), "--json", "show", "does-not-exist"]) == 1
        captured = capsys.readouterr()
        assert json.loads(captured.err)["detail"].startswith("activity not found")
