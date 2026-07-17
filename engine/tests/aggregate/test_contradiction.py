"""Tests for `wbj.aggregate.contradiction` (Task 21): the 6-row lookup
table from `Cerebro/00_main_agent/CONTRADICTION_RESOLUTION.md`.
"""

from __future__ import annotations

from wbj.aggregate.contradiction import (
    CategoryScore10s,
    ReverseDCFContext,
    contradictions,
)

_NEUTRAL = dict(business=5.5, financial=5.5, market=5.5, technical=5.5, risk=5.5, valuation=5.5)


def _cats(**overrides) -> CategoryScore10s:
    kwargs = dict(_NEUTRAL)
    kwargs.update(overrides)
    return CategoryScore10s(**kwargs)


def test_no_contradiction_when_everything_is_neutral():
    assert contradictions(_cats(), raw_total=55.0) == []


def test_row_1_strong_business_weak_technical():
    result = contradictions(_cats(business=8.0, technical=3.0), raw_total=60.0)
    labels = [c.label for c in result]
    assert "Quality watch / wait for confirmation" in labels


def test_row_2_weak_business_strong_technical():
    result = contradictions(_cats(business=3.0, technical=8.0), raw_total=60.0)
    labels = [c.label for c in result]
    assert "Speculative momentum only" in labels


def test_row_3_strong_valuation_weak_technical():
    result = contradictions(_cats(valuation=8.0, technical=3.0), raw_total=60.0)
    labels = [c.label for c in result]
    assert "Value watch" in labels


def test_row_4_expensive_valuation_strong_growth_and_technical():
    result = contradictions(_cats(valuation=3.0, market=8.0, technical=8.0), raw_total=60.0)
    labels = [c.label for c in result]
    assert "Momentum candidate if gates pass" in labels


def test_row_5_strong_total_low_risk_score():
    result = contradictions(_cats(risk=3.0), raw_total=85.0)
    labels = [c.label for c in result]
    assert "Apply risk override" in labels


def test_row_5_not_triggered_when_raw_total_is_not_strong():
    result = contradictions(_cats(risk=3.0), raw_total=60.0)
    labels = [c.label for c in result]
    assert "Apply risk override" not in labels


def test_row_6_dcf_high_reverse_dcf_demanding():
    ctx = ReverseDCFContext(base_case_upside_pct=0.30, reverse_dcf_implied_growth=0.20, reference_growth=0.05)
    result = contradictions(_cats(), raw_total=55.0, reverse_dcf=ctx)
    labels = [c.label for c in result]
    assert "Lower valuation confidence" in labels


def test_row_6_not_triggered_without_context():
    result = contradictions(_cats(), raw_total=55.0)
    labels = [c.label for c in result]
    assert "Lower valuation confidence" not in labels


def test_row_6_not_triggered_when_reverse_dcf_is_not_demanding():
    ctx = ReverseDCFContext(base_case_upside_pct=0.30, reverse_dcf_implied_growth=0.06, reference_growth=0.05)
    result = contradictions(_cats(), raw_total=55.0, reverse_dcf=ctx)
    labels = [c.label for c in result]
    assert "Lower valuation confidence" not in labels


def test_multiple_rows_can_match_simultaneously():
    cats = _cats(business=8.0, technical=3.0, valuation=8.0)
    result = contradictions(cats, raw_total=60.0)
    labels = {c.label for c in result}
    assert "Quality watch / wait for confirmation" in labels
    assert "Value watch" in labels


def test_contradictions_never_mutates_input_scores():
    cats = _cats(business=8.0, technical=3.0)
    before = (cats.business, cats.technical, cats.market, cats.financial, cats.risk, cats.valuation)
    contradictions(cats, raw_total=60.0)
    after = (cats.business, cats.technical, cats.market, cats.financial, cats.risk, cats.valuation)
    assert before == after
