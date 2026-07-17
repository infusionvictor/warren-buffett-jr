"""Tests for `wbj.schemas.final_report` (Task 21): the `FinalReport`
pydantic schema and `build_final_report`.

Source of truth: `Cerebro/00_main_agent/FINAL_REPORT_SCHEMA.md`.
"""

from __future__ import annotations

import pytest

from wbj.aggregate.contradiction import CategoryScore10s, contradictions
from wbj.aggregate.gates import CategoryConfidences, CategoryPoints, apply_gates, raw_total
from wbj.aggregate.overrides import AggregateInputs, apply_overrides
from wbj.aggregate.synthesis import synthesize_levels
from wbj.schemas.final_report import REPORT_VERSION, ExecutiveThesis, FinalReport, build_final_report

from .conftest import (
    make_business,
    make_financial,
    make_market,
    make_risk,
    make_technical,
    make_valuation,
)


def _executive_thesis() -> ExecutiveThesis:
    return ExecutiveThesis(
        business_quality="The company sells enterprise software on a subscription basis.",
        value_creation_durability="Returns on invested capital have stayed above the cost of capital for five years.",
        growth_engine="Growth is funded from operating cash flow, not external capital.",
        market_validation="Relative strength versus the sector has been positive for two quarters.",
        valuation_message="The current price requires high-teens revenue growth to sustain.",
        key_levels_summary="Nearest support and resistance zones bracket the current price within one ATR.",
        primary_risk="A slowdown in enterprise IT spending would compress the growth assumption fastest.",
    )


def _aggregate_inputs() -> AggregateInputs:
    return AggregateInputs(
        business=make_business(points=16.0, three_thesis_killers=["Customer concentration above 30%"]),
        financial=make_financial(points=10.5),
        market=make_market(points=18.0, three_growth_thesis_killers=["TAM estimate unverified"]),
        technical=make_technical(points=16.0),
        risk=make_risk(points=9.0, thesis_killers=[{"description": "Covenant breach risk within 12 months"}]),
        valuation=make_valuation(points=7.0),
    )


def test_final_report_round_trips_through_build_final_report():
    inputs = _aggregate_inputs()
    overrides = apply_overrides(inputs)
    cats = CategoryPoints(business=16.0, financial=10.5, market=18.0, technical=16.0, risk=9.0, valuation=7.0)
    confidences = CategoryConfidences(business=90, financial=90, market=90, technical=90, risk=90, valuation=90)
    raw = raw_total(cats)
    profile = apply_gates(raw, cats, confidences, overrides)

    score10s = CategoryScore10s(
        business=cats.business / 20 * 10, financial=cats.financial / 15 * 10, market=cats.market / 20 * 10,
        technical=cats.technical / 20 * 10, risk=cats.risk / 15 * 10, valuation=cats.valuation / 10 * 10,
    )
    contras = contradictions(score10s, raw)

    levels = synthesize_levels(inputs.technical, inputs.valuation, price=100.0, atr=2.0)

    report = build_final_report(
        inputs=inputs,
        profile=profile,
        contradictions=contras,
        levels=levels,
        executive_thesis=_executive_thesis(),
        exchange="NASDAQ",
        currency="USD",
        analysis_timestamp="2026-07-17T12:00:00+00:00",
        packet_hashes={"packet": "abc123"},
        formula_versions=["2.0.0"],
    )

    assert isinstance(report, FinalReport)
    assert report.report_version == REPORT_VERSION
    assert report.security.ticker == "TEST"
    assert report.profile.raw_score == raw
    assert report.category_scorecard.business.points == 16.0
    assert report.category_scorecard.business.max == 20.0
    assert len(report.executive_thesis.as_sentences()) == 7
    assert "Customer concentration above 30%" in report.thesis_killers
    assert "TAM estimate unverified" in report.thesis_killers
    assert "Covenant breach risk within 12 months" in report.thesis_killers
    assert report.audit.packet_hashes == {"packet": "abc123"}
    assert report.audit.formula_versions == ["2.0.0"]
    assert len(report.important_levels) == len(levels.levels)


def test_final_report_rejects_wrong_report_version():
    with pytest.raises(Exception):
        FinalReport(
            report_version="1.0.0",
            security=dict(
                ticker="X", exchange="NASDAQ", currency="USD",
                analysis_timestamp="2026-01-01T00:00:00+00:00", knowledge_timestamp="2026-01-01T00:00:00+00:00",
            ),
            profile=dict(label="Speculative", raw_score=50.0, total_confidence=50.0),
            category_scorecard=dict(
                business=dict(points=10, max=20, confidence=50),
                financial=dict(points=7, max=15, confidence=50),
                market=dict(points=10, max=20, confidence=50),
                technical=dict(points=10, max=20, confidence=50),
                risk=dict(points=7, max=15, confidence=50),
                valuation=dict(points=5, max=10, confidence=50),
            ),
            executive_thesis=_executive_thesis(),
        )


def test_executive_thesis_requires_all_seven_sentences():
    with pytest.raises(Exception):
        ExecutiveThesis(business_quality="x")  # missing the other 6 required fields
