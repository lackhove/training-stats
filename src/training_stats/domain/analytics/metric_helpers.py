"""Small constructors shared by pure metric-producing functions."""

from __future__ import annotations

from ..metrics import (
    Axis,
    Confidence,
    Metric,
    MetricKind,
    MetricMethod,
    MetricScalar,
    MissingReason,
    Period,
    ScalarValue,
    TableValue,
    TimeBasis,
)


def scalar_metric(
    name: str,
    period: Period,
    value: MetricScalar | None,
    *,
    description: str,
    kind: MetricKind = MetricKind.MEASURED,
    unit: str | None = None,
    missing_reason: MissingReason = MissingReason.NO_DATA,
    method: MetricMethod | None = None,
    algorithm_version: str | None = None,
    confidence: Confidence | None = None,
    formula: str | None = None,
    coefficient_set: str | None = None,
    time_basis: TimeBasis | None = None,
) -> Metric:
    """Build a scalar metric, converting ``None`` to an explicit missing value."""
    return Metric(
        name=name,
        description=description,
        period=period,
        value=ScalarValue(value) if value is not None else None,
        kind=kind,
        unit=unit,
        missing_reason=missing_reason if value is None else None,
        method=method,
        algorithm_version=algorithm_version,
        confidence=confidence,
        formula=formula,
        coefficient_set=coefficient_set,
        time_basis=time_basis,
    )


def table_metric(
    name: str,
    period: Period,
    axis: Axis,
    values: tuple[MetricScalar | None, ...],
    *,
    description: str,
    kind: MetricKind = MetricKind.CALCULATED,
    unit: str | None = None,
    missing_reason: MissingReason | None = None,
    time_basis: TimeBasis | None = None,
    method: MetricMethod | None = None,
    algorithm_version: str | None = None,
    confidence: Confidence | None = None,
    formula: str | None = None,
    coefficient_set: str | None = None,
) -> Metric:
    """Build an axis/value table metric or an explicit missing metric."""
    if missing_reason is not None:
        return Metric(
            name=name,
            description=description,
            period=period,
            value=None,
            kind=kind,
            unit=unit,
            missing_reason=missing_reason,
            time_basis=time_basis,
            method=method,
            algorithm_version=algorithm_version,
            confidence=confidence,
            formula=formula,
            coefficient_set=coefficient_set,
        )
    return Metric(
        name=name,
        description=description,
        period=period,
        value=TableValue(axis=axis, values=values),
        kind=kind,
        unit=unit,
        time_basis=time_basis,
        method=method,
        algorithm_version=algorithm_version,
        confidence=confidence,
        formula=formula,
        coefficient_set=coefficient_set,
    )
