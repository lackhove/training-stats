"""Tests for training form (CTL, ATL, TSB) and tour-readiness metrics."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from training_stats.domain.analytics.fitness import (
    ATL_DECAY,
    CLASSIFICATION_FRESH,
    CLASSIFICATION_HIGH_FATIGUE,
    CLASSIFICATION_OPTIMAL,
    CLASSIFICATION_OVERTRAINING_RISK,
    CTL_DECAY,
    classify_form,
    compute_days_since_60k,
    compute_form_curves,
    compute_rolling_distance,
    compute_training_form,
    training_form_from,
)
from training_stats.domain.metrics import (
    Confidence,
    Metric,
    MetricKind,
    MetricMethod,
    MissingReason,
    ScalarValue,
    TripPeriod,
)
from training_stats.domain.models import ActivityRecord


def _form_record(
    start: datetime,
    *,
    sport: str = "cycling",
    distance_m: float | None = 0.0,
    load_value: float | None = None,
    load_confidence: str | None = None,
    activity_id: str | None = None,
) -> ActivityRecord:
    """Build a metric-bearing cached record for training-form tests."""
    identifier = activity_id or f"act-{start.strftime('%Y%m%dT%H%M%S')}-{sport}"
    period = TripPeriod(activity_id=identifier, start=start)
    metrics = (
        Metric(
            name="activity.distance",
            description="Cached distance for training-form tests.",
            period=period,
            value=ScalarValue(distance_m) if distance_m is not None else None,
            kind=MetricKind.CALCULATED,
            unit="m",
            missing_reason=None if distance_m is not None else MissingReason.NO_DATA,
        ),
        Metric(
            name="training_load.hr_trimp",
            description="Cached Banister load for training-form tests.",
            period=period,
            value=ScalarValue(load_value) if load_value is not None else None,
            kind=MetricKind.ESTIMATED,
            unit="TRIMP",
            missing_reason=None if load_value is not None else MissingReason.NO_DATA,
            method=MetricMethod.HR_TRIMP,
            confidence=Confidence(load_confidence) if load_confidence is not None else None,
        ),
    )
    return ActivityRecord(
        activity_id=identifier,
        start_time=start,
        sport=sport,
        sub_sport=None,
        segment_index=1,
        segment_count=1,
        repair_flags=(),
        source_name="local",
        source_path=f"{identifier}.fit",
        analysis_version="1",
        metrics=metrics,
    )


class TestTrainingFormRecursion:
    def test_analytical_reference(self) -> None:
        # Day 1: 100 TRIMP
        # Day 2: 0 TRIMP (rest day)
        today = date(2026, 6, 2)
        day1 = date(2026, 6, 1)
        loads = {day1: 100.0}

        curves = compute_form_curves(loads, day1, today)
        assert len(curves) == 2

        # Day 1 checks
        expected_ctl_1 = 100.0 * CTL_DECAY
        expected_atl_1 = 100.0 * ATL_DECAY
        expected_tsb_1 = expected_ctl_1 - expected_atl_1
        assert curves[0].ctl == pytest.approx(expected_ctl_1)
        assert curves[0].atl == pytest.approx(expected_atl_1)
        assert curves[0].tsb == pytest.approx(expected_tsb_1)

        # Day 2 rest-day decay checks
        expected_ctl_2 = expected_ctl_1 * math.exp(-1.0 / 42.0)
        expected_atl_2 = expected_atl_1 * math.exp(-1.0 / 7.0)
        expected_tsb_2 = expected_ctl_2 - expected_atl_2
        assert curves[1].ctl == pytest.approx(expected_ctl_2)
        assert curves[1].atl == pytest.approx(expected_atl_2)
        assert curves[1].tsb == pytest.approx(expected_tsb_2)
        assert curves[1].daily_load == 0.0

    def test_tsb_consistency_and_zero_filling(self) -> None:
        start = date(2026, 1, 1)
        end = date(2026, 2, 20)  # 51 days
        # Training only on odd days
        loads = {
            start + timedelta(days=i): 80.0 for i in range((end - start).days + 1) if i % 2 == 1
        }

        curves = compute_form_curves(loads, start, end)
        assert len(curves) == (end - start).days + 1

        for point in curves:
            # TSB must strictly equal CTL - ATL
            assert point.tsb == pytest.approx(point.ctl - point.atl)

    def test_warmup_window(self) -> None:
        today = date(2026, 6, 1)
        # 41 days of history -> warmup
        start_41 = today - timedelta(days=40)
        loads_41 = {start_41: 50.0}
        result_41 = compute_training_form(loads_41, today=today)
        assert result_41.history_days == 41
        assert result_41.is_warmup is True

        # 42 days of history -> not warmup
        start_42 = today - timedelta(days=41)
        loads_42 = {start_42: 50.0}
        result_42 = compute_training_form(loads_42, today=today)
        assert result_42.history_days == 42
        assert result_42.is_warmup is False

    def test_empty_loads(self) -> None:
        today = date(2026, 6, 1)
        result = compute_training_form({}, today=today)
        assert result.current_ctl == 0.0
        assert result.current_atl == 0.0
        assert result.current_tsb == 0.0
        assert result.form_classification == CLASSIFICATION_OPTIMAL
        assert result.is_warmup is True
        assert result.history_days == 0
        assert result.trend_points == ()


class TestTrainingFormFrom:
    """The record-based composition entry point."""

    TZ = ZoneInfo("Europe/Berlin")

    def test_ctl_atl_and_rolling_distance(self) -> None:
        today = date(2026, 6, 10)
        records = [
            # 20 km + load today, 30 km + load 5 days ago
            _form_record(
                datetime(2026, 6, 10, 10, tzinfo=UTC),
                load_value=80.0,
                load_confidence="high",
                distance_m=20_000.0,
            ),
            _form_record(
                datetime(2026, 6, 5, 10, tzinfo=UTC),
                load_value=60.0,
                load_confidence="high",
                distance_m=30_000.0,
            ),
            # 50 km ride 10 days ago, outside the 7-day window
            _form_record(
                datetime(2026, 5, 31, 10, tzinfo=UTC), load_value=None, distance_m=50_000.0
            ),
        ]
        result = training_form_from(records, tz=self.TZ, today=today)
        assert result.current_ctl > 0.0
        assert result.current_atl > 0.0
        assert result.rolling_7d_distance_km == 50.0
        assert result.rolling_28d_distance_km == 100.0
        assert result.days_since_60k_ride is None
        assert result.load_confidence_ratio == 1.0

    def test_days_since_60k(self) -> None:
        today = date(2026, 6, 10)
        records = [
            _form_record(
                datetime(2026, 6, 7, 10, tzinfo=UTC), distance_m=65_000.0, load_value=100.0
            )
        ]
        result = training_form_from(records, tz=self.TZ, today=today)
        assert result.days_since_60k_ride == 3

    def test_load_confidence_ratio(self) -> None:
        today = date(2026, 6, 10)
        records = [
            _form_record(
                datetime(2026, 6, 9, 10, tzinfo=UTC), load_value=200.0, load_confidence="high"
            ),
            _form_record(
                datetime(2026, 6, 8, 10, tzinfo=UTC), load_value=100.0, load_confidence="low"
            ),
        ]
        result = training_form_from(records, tz=self.TZ, today=today)
        assert result.load_confidence_ratio == pytest.approx(0.6667)  # rounded to 4 decimals

    def test_sport_filter(self) -> None:
        today = date(2026, 6, 10)
        records = [
            _form_record(
                datetime(2026, 6, 9, 10, tzinfo=UTC),
                load_value=100.0,
                load_confidence="high",
                distance_m=70_000.0,
            ),
            _form_record(
                datetime(2026, 6, 8, 10, tzinfo=UTC),
                sport="running",
                load_value=50.0,
                load_confidence="high",
                distance_m=10_000.0,
            ),
        ]
        result = training_form_from(records, tz=self.TZ, today=today, sport="running")
        # Cycling's 70 km ride must not count toward running readiness.
        assert result.days_since_60k_ride is None
        assert result.rolling_7d_distance_km == 10.0

    def test_empty(self) -> None:
        result = training_form_from([], tz=self.TZ, today=date(2026, 6, 10))
        assert result.current_ctl == 0.0
        assert result.is_warmup is True
        assert result.history_days == 0
        assert result.days_since_60k_ride is None
        assert result.load_confidence_ratio == 1.0

    def test_sport_filter_is_case_insensitive(self) -> None:
        today = date(2026, 6, 10)
        records = [
            _form_record(
                datetime(2026, 6, 9, 10, tzinfo=UTC),
                load_value=100.0,
                load_confidence="high",
                distance_m=70_000.0,
            ),
            _form_record(
                datetime(2026, 6, 8, 10, tzinfo=UTC),
                sport="running",
                load_value=50.0,
                load_confidence="high",
                distance_m=10_000.0,
            ),
        ]
        lower = training_form_from(records, tz=self.TZ, today=today, sport="running")
        upper = training_form_from(records, tz=self.TZ, today=today, sport="RUNNING")
        assert (upper.current_ctl, upper.rolling_7d_distance_km) == (
            lower.current_ctl,
            lower.rolling_7d_distance_km,
        )

    def test_local_day_bucketing(self) -> None:
        today = date(2026, 6, 15)
        records = [
            _form_record(
                datetime(2026, 6, 1, 23, 40, tzinfo=UTC),
                load_value=100.0,
                load_confidence="high",
            )
        ]
        result = training_form_from(records, tz=self.TZ, today=today)
        # 2026-06-01 23:40 UTC is already 2026-06-02 in Berlin.
        assert result.history_days == 14
        daily = {point.date: point.daily_load for point in result.trend_points}
        assert date(2026, 6, 1) not in daily
        assert daily[date(2026, 6, 2)] == 100.0


class TestClassificationBands:
    def test_exact_transitions(self) -> None:
        # TSB > +5.0 -> fresh
        assert classify_form(10.0) == CLASSIFICATION_FRESH
        assert classify_form(5.0001) == CLASSIFICATION_FRESH

        # -10.0 <= TSB <= +5.0 -> optimal
        assert classify_form(5.0) == CLASSIFICATION_OPTIMAL
        assert classify_form(0.0) == CLASSIFICATION_OPTIMAL
        assert classify_form(-10.0) == CLASSIFICATION_OPTIMAL

        # -30.0 <= TSB < -10.0 -> high_fatigue
        assert classify_form(-10.0001) == CLASSIFICATION_HIGH_FATIGUE
        assert classify_form(-20.0) == CLASSIFICATION_HIGH_FATIGUE
        assert classify_form(-30.0) == CLASSIFICATION_HIGH_FATIGUE

        # TSB < -30.0 -> overtraining_risk
        assert classify_form(-30.0001) == CLASSIFICATION_OVERTRAINING_RISK
        assert classify_form(-45.0) == CLASSIFICATION_OVERTRAINING_RISK


class TestTourReadiness:
    def test_rolling_distances(self) -> None:
        today = date(2026, 6, 10)
        # Distances on various days:
        # Today: 20 km
        # 5 days ago (within 7d): 30 km
        # 10 days ago (outside 7d, within 28d): 50 km
        # 30 days ago (outside 28d): 100 km
        distances = {
            today: 20_000.0,
            today - timedelta(days=5): 30_000.0,
            today - timedelta(days=10): 50_000.0,
            today - timedelta(days=30): 100_000.0,
        }

        dist_7d = compute_rolling_distance(distances, today, 7)
        dist_28d = compute_rolling_distance(distances, today, 28)

        assert dist_7d == 50_000.0
        assert dist_28d == 100_000.0

    def test_days_since_60k_ride(self) -> None:
        today = date(2026, 6, 10)

        # Ride today returns 0
        acts_today = [(today, 62_000.0)]
        assert compute_days_since_60k(acts_today, today) == 0

        # Ride 3 days ago returns 3
        acts_3d = [
            (today - timedelta(days=3), 65_000.0),
            (today - timedelta(days=10), 80_000.0),
        ]
        assert compute_days_since_60k(acts_3d, today) == 3

        # Excludes activities < 60k
        acts_short = [
            (today, 59_999.0),
            (today - timedelta(days=1), 40_000.0),
        ]
        assert compute_days_since_60k(acts_short, today) is None

        # Empty history returns None
        assert compute_days_since_60k([], today) is None

    def test_timezone_boundary(self) -> None:
        # Late evening in UTC: 2026-06-01 23:30:00 UTC
        # In Europe/Berlin (UTC+2 in summer), this is 2026-06-02 01:30:00
        tz = ZoneInfo("Europe/Berlin")
        start_utc = datetime(2026, 6, 1, 23, 30, tzinfo=UTC)
        local_d = start_utc.astimezone(tz).date()
        assert local_d == date(2026, 6, 2)

        # Days elapsed relative to June 5 in Berlin
        today_berlin = date(2026, 6, 5)
        days = compute_days_since_60k([(local_d, 70_000.0)], today_berlin)
        assert days == 3  # June 5 - June 2 = 3 days (not 4)
