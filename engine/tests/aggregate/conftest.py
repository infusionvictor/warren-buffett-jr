"""Shared fixtures for `wbj.aggregate` tests.

Builds minimal, directly-constructed `SpecialistOutput` fixtures with known
category points (per the task-21 brief: "Build SpecialistOutput fixtures
with known category points") rather than running a full specialist
`run()` against a synthetic packet -- Task 21 tests the aggregation layer
in isolation from packet/provider plumbing.

Each `make_*` builder produces a category whose `category.awarded_points`
reproduces exactly from a single synthetic `Dimension` (weight 1.0, a
manufactured `score10`), so `wbj.aggregate.overrides.validate_handoff`'s
"category points must reproduce from dimensions" check passes by
construction unless a test deliberately breaks it.
"""

from __future__ import annotations

from typing import Any

import pytest

from wbj.core.nullstates import EvidenceClass, NullState, Value
from wbj.core.scoring import Dimension
from wbj.schemas.levels import (
    AVWAPLevel,
    EarningsGap,
    LevelsOutput,
    MovingAverageLevel,
    Touch,
    Zone,
)
from wbj.specialists import business, financial, market, risk, technical, valuation
from wbj.specialists.common import CategoryStats, MetricRow, SecurityRef

TICKER = "TEST"
KNOWLEDGE_TS = "2026-07-16T21:00:00+00:00"


def _security() -> SecurityRef:
    return SecurityRef(ticker=TICKER, exchange="NASDAQ", currency="USD")


def _dim(name: str, max_points: float, points: float) -> Dimension:
    score10 = (points / max_points * 10.0) if max_points else 0.0
    return Dimension(name=name, max_points=max_points, metric_scores=[(1.0, Value.of(score10, unit="score", evidence_class=EvidenceClass.C))])


def _category(max_points: float, points: float, confidence: float | None) -> CategoryStats:
    score10 = (points / max_points * 10.0) if max_points else 0.0
    return CategoryStats(max_points=max_points, awarded_points=points, score_10=score10, confidence=confidence)


def _base_kwargs(
    *,
    agent_id: str,
    max_points: float,
    points: float,
    confidence: float | None,
    coverage: float | None,
    mandatory_flags: list[str] | None,
    metrics: list[MetricRow] | None,
    knowledge_timestamp: str,
    dimensions: list[Dimension] | None,
) -> dict[str, Any]:
    return dict(
        agent_id=agent_id,
        status="COMPLETE",
        security=_security(),
        knowledge_timestamp=knowledge_timestamp,
        category=_category(max_points, points, confidence),
        coverage=coverage,
        dimensions=dimensions if dimensions is not None else [_dim(agent_id, max_points, points)],
        metrics=metrics or [],
        mandatory_flags=mandatory_flags or [],
    )


def make_business(
    *, points: float = 16.0, confidence: float = 90.0, coverage: float = 0.90,
    mandatory_flags: list[str] | None = None, metrics: list[MetricRow] | None = None,
    dimensions: list[Dimension] | None = None, three_thesis_killers: list[str] | None = None,
    knowledge_timestamp: str = KNOWLEDGE_TS,
) -> business.BusinessOutput:
    kwargs = _base_kwargs(
        agent_id=business.AGENT_ID, max_points=business.MAX_POINTS, points=points, confidence=confidence,
        coverage=coverage, mandatory_flags=mandatory_flags, metrics=metrics,
        knowledge_timestamp=knowledge_timestamp, dimensions=dimensions,
    )
    return business.BusinessOutput(three_thesis_killers=three_thesis_killers or [], **kwargs)


def make_financial(
    *, points: float = 10.5, confidence: float = 90.0, coverage: float = 0.90,
    mandatory_flags: list[str] | None = None, metrics: list[MetricRow] | None = None,
    dimensions: list[Dimension] | None = None, knowledge_timestamp: str = KNOWLEDGE_TS,
) -> financial.FinancialOutput:
    kwargs = _base_kwargs(
        agent_id=financial.AGENT_ID, max_points=financial.MAX_POINTS, points=points, confidence=confidence,
        coverage=coverage, mandatory_flags=mandatory_flags, metrics=metrics,
        knowledge_timestamp=knowledge_timestamp, dimensions=dimensions,
    )
    core27 = financial.Core27Summary(valid_count=0, points=0.0, maximum_valid_points=0.0, percent=0.0, score_10=0.0, rows=[])
    return financial.FinancialOutput(core_27_metrics=core27, **kwargs)


def make_market(
    *, points: float = 18.0, confidence: float = 90.0, coverage: float = 0.90,
    mandatory_flags: list[str] | None = None, metrics: list[MetricRow] | None = None,
    dimensions: list[Dimension] | None = None, three_growth_thesis_killers: list[str] | None = None,
    knowledge_timestamp: str = KNOWLEDGE_TS,
) -> market.MarketOutput:
    kwargs = _base_kwargs(
        agent_id=market.AGENT_ID, max_points=market.MAX_POINTS, points=points, confidence=confidence,
        coverage=coverage, mandatory_flags=mandatory_flags, metrics=metrics,
        knowledge_timestamp=knowledge_timestamp, dimensions=dimensions,
    )
    return market.MarketOutput(three_growth_thesis_killers=three_growth_thesis_killers or [], **kwargs)


def make_technical(
    *, points: float = 16.0, confidence: float = 90.0, coverage: float = 0.90,
    mandatory_flags: list[str] | None = None, metrics: list[MetricRow] | None = None,
    dimensions: list[Dimension] | None = None, important_levels: LevelsOutput | None = None,
    breakouts_and_failures: list[dict[str, Any]] | None = None, knowledge_timestamp: str = KNOWLEDGE_TS,
) -> technical.TechnicalOutput:
    kwargs = _base_kwargs(
        agent_id=technical.AGENT_ID, max_points=technical.MAX_POINTS, points=points, confidence=confidence,
        coverage=coverage, mandatory_flags=mandatory_flags, metrics=metrics,
        knowledge_timestamp=knowledge_timestamp, dimensions=dimensions,
    )
    return technical.TechnicalOutput(
        important_levels=important_levels or LevelsOutput(),
        breakouts_and_failures=breakouts_and_failures or [],
        **kwargs,
    )


def make_risk(
    *, points: float = 9.0, confidence: float = 90.0, coverage: float = 0.90,
    mandatory_flags: list[str] | None = None, metrics: list[MetricRow] | None = None,
    dimensions: list[Dimension] | None = None, thesis_killers: list[dict[str, Any]] | None = None,
    knowledge_timestamp: str = KNOWLEDGE_TS,
) -> risk.RiskOutput:
    kwargs = _base_kwargs(
        agent_id=risk.AGENT_ID, max_points=risk.MAX_POINTS, points=points, confidence=confidence,
        coverage=coverage, mandatory_flags=mandatory_flags, metrics=metrics,
        knowledge_timestamp=knowledge_timestamp, dimensions=dimensions,
    )
    return risk.RiskOutput(thesis_killers=thesis_killers or [], **kwargs)


def make_valuation(
    *, points: float = 7.0, confidence: float = 90.0, coverage: float = 0.90,
    mandatory_flags: list[str] | None = None, metrics: list[MetricRow] | None = None,
    dimensions: list[Dimension] | None = None, reference_bands: valuation.ReferenceBands | None = None,
    reverse_dcf: valuation.ReverseDCFSummary | None = None, scenarios: list[valuation.ScenarioSummary] | None = None,
    knowledge_timestamp: str = KNOWLEDGE_TS,
) -> valuation.ValuationOutput:
    kwargs = _base_kwargs(
        agent_id=valuation.AGENT_ID, max_points=valuation.MAX_POINTS, points=points, confidence=confidence,
        coverage=coverage, mandatory_flags=mandatory_flags, metrics=metrics,
        knowledge_timestamp=knowledge_timestamp, dimensions=dimensions,
    )
    return valuation.ValuationOutput(
        reference_bands=reference_bands or valuation.ReferenceBands(),
        reverse_dcf=reverse_dcf or valuation.ReverseDCFSummary(),
        scenarios=scenarios or [],
        **kwargs,
    )


@pytest.fixture
def outputs_main002():
    """MAIN-002's own numbers: 16 + 10.5 + 18 + 16 + 9 + 7 = 76.5."""
    return {
        "business": make_business(points=16.0),
        "financial": make_financial(points=10.5),
        "market": make_market(points=18.0),
        "technical": make_technical(points=16.0),
        "risk": make_risk(points=9.0),
        "valuation": make_valuation(points=7.0),
    }
