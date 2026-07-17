"""Tests for `wbj.aggregate.gates` (Task 21): `raw_total`, `descriptive_band`,
`total_confidence`, and `apply_gates`'s profile-gate table.

Sources of truth: `Cerebro/00_main_agent/SCORING_AND_GATES.md`,
`Cerebro/00_main_agent/VALIDATION_TESTS.md` (MAIN-002..007).
"""

from __future__ import annotations

from wbj.aggregate.gates import (
    GATE_AVOID,
    GATE_CONDITIONAL,
    GATE_MOMENTUM,
    GATE_QUALITY,
    GATE_SPECULATIVE,
    GATE_VALUE,
    GATE_WEAK,
    CategoryConfidences,
    CategoryPoints,
    apply_gates,
    descriptive_band,
    raw_total,
    total_confidence,
)
from wbj.aggregate.overrides import AggregateInputs, apply_overrides

from .conftest import (
    make_business,
    make_financial,
    make_market,
    make_risk,
    make_technical,
    make_valuation,
)


# ============================================================================
# raw_total (MAIN-002)
# ============================================================================


def test_MAIN_002_raw_total():
    assert raw_total([16, 10.5, 18, 16, 9, 7]) == 76.5


def test_raw_total_accepts_category_points():
    cats = CategoryPoints(business=16, financial=10.5, market=18, technical=16, risk=9, valuation=7)
    assert raw_total(cats) == 76.5


# ============================================================================
# descriptive_band
# ============================================================================


def test_descriptive_band_boundaries():
    assert descriptive_band(90.0) == "Elite raw score"
    assert descriptive_band(89.99) == "Strong raw score"
    assert descriptive_band(80.0) == "Strong raw score"
    assert descriptive_band(79.99) == "Conditional raw score"
    assert descriptive_band(70.0) == "Conditional raw score"
    assert descriptive_band(69.99) == "Mixed / wait"
    assert descriptive_band(60.0) == "Mixed / wait"
    assert descriptive_band(59.99) == "Weak"
    assert descriptive_band(50.0) == "Weak"
    assert descriptive_band(49.99) == "Avoid on raw score"


# ============================================================================
# total_confidence
# ============================================================================


def test_total_confidence_is_weighted_by_fixed_category_maxima():
    # All categories at confidence=100 -> total confidence is 100 regardless
    # of awarded points (the formula is max-points-weighted, not
    # awarded-points-weighted).
    confidences = CategoryConfidences(business=100, financial=100, market=100, technical=100, risk=100, valuation=100)
    assert total_confidence(confidences) == 100.0

    confidences2 = CategoryConfidences(business=100, financial=0, market=0, technical=0, risk=0, valuation=0)
    # business max_points=20 -> 20*100/100 = 20
    assert total_confidence(confidences2) == 20.0


# ============================================================================
# apply_gates: helper
# ============================================================================


def _overrides_for(**kwargs) -> list:
    base = dict(
        business=make_business(), financial=make_financial(), market=make_market(),
        technical=make_technical(), risk=make_risk(), valuation=make_valuation(),
    )
    base.update(kwargs)
    return apply_overrides(AggregateInputs(**base))


_FULL_CONF = CategoryConfidences(business=90, financial=90, market=90, technical=90, risk=90, valuation=90)


# ============================================================================
# MAIN-003: Risk=4/15, total=90 -> Speculative
# ============================================================================


def test_MAIN_003_risk_cap():
    # MAIN-003 fixes raw_total=90 directly (risk=4/15 alone already caps
    # the profile at Speculative regardless of the other five categories'
    # exact split, so the fixture below need not itself sum to 90 -- only
    # `risk=4` and the explicit `raw_total=90.0` argument matter here).
    cats = CategoryPoints(business=20, financial=15, market=20, technical=16, risk=4, valuation=10)
    raw = 90.0
    overrides = _overrides_for(risk=make_risk(points=4.0))
    result = apply_gates(raw, cats, _FULL_CONF, overrides)
    assert result.label == GATE_SPECULATIVE


# ============================================================================
# MAIN-004: Valuation=3, Technical=7 -> Wait/Avoid breakdown override
# ============================================================================


def test_MAIN_004_premium_breakdown_override():
    cats = CategoryPoints(business=18, financial=14, market=18, technical=7, risk=12, valuation=3)
    raw = raw_total(cats)
    overrides = _overrides_for(valuation=make_valuation(points=3.0), technical=make_technical(points=7.0))
    result = apply_gates(raw, cats, _FULL_CONF, overrides)
    assert result.label == GATE_AVOID


# ============================================================================
# MAIN-005: ROIC<WACC, total=92 -> no Elite/Quality label
# ============================================================================


def test_MAIN_005_no_quality_label_when_roic_below_wacc():
    # VALIDATION_TESTS.md MAIN-005 fixes total=92. This combination sums to
    # exactly 92 and clears every Quality Opportunity threshold (raw>=80,
    # business>=16, financial>=11, risk>=10, valuation>=5, technical>=12),
    # while deliberately failing Momentum (market<16) and Value
    # (valuation<8) so the gate genuinely under test is Quality.
    cats = CategoryPoints(business=20, financial=15, market=15, technical=20, risk=15, valuation=7)
    raw = raw_total(cats)
    assert raw == 92.0

    # Without the override: Quality Opportunity passes.
    clean_overrides = _overrides_for()
    clean_result = apply_gates(raw, cats, _FULL_CONF, clean_overrides)
    assert clean_result.label == GATE_QUALITY

    # With ROIC<WACC: Quality Opportunity must not pass, and no Elite label.
    fin = make_financial(points=15.0, mandatory_flags=["OVERRIDE_2_ROIC_BELOW_WACC"])
    override_overrides = _overrides_for(financial=fin)
    result = apply_gates(raw, cats, _FULL_CONF, override_overrides)
    assert result.label != GATE_QUALITY
    assert "Elite" not in result.label
    assert any("OVERRIDE_2_ROIC_BELOW_WACC" in f for f in result.failed_gates)


# ============================================================================
# MAIN-006: interest coverage=1.2x -> solvency warning always present
# ============================================================================


def test_MAIN_006_solvency_warning_always_present():
    cats = CategoryPoints(business=16, financial=10.5, market=18, technical=16, risk=9, valuation=7)
    raw = raw_total(cats)
    rk = make_risk(points=9.0, mandatory_flags=["SOLVENCY_WARNING"])
    overrides = _overrides_for(risk=rk)
    result = apply_gates(raw, cats, _FULL_CONF, overrides)
    assert any("solvency" in w.lower() for w in result.warnings)


# ============================================================================
# MAIN-007: any category coverage=0.65 -> no profile gate may pass
# ============================================================================


def test_MAIN_007_low_coverage_blocks_gates():
    # A combination that clears every numeric gate on paper.
    cats = CategoryPoints(business=20, financial=15, market=20, technical=20, risk=15, valuation=10)
    raw = raw_total(cats)
    assert raw == 100.0
    fin = make_financial(points=15.0, coverage=0.65)
    overrides = _overrides_for(financial=fin)
    result = apply_gates(raw, cats, _FULL_CONF, overrides)
    assert result.label not in (GATE_MOMENTUM, GATE_QUALITY, GATE_VALUE)
    assert result.passed_gates == []


# ============================================================================
# Momentum gate: exact thresholds
# ============================================================================


def test_momentum_gate_exact_thresholds_pass():
    cats = CategoryPoints(business=15, financial=13, market=16, technical=17, risk=8, valuation=2)
    confidences = CategoryConfidences(business=90, financial=90, market=90, technical=70, risk=90, valuation=90)
    overrides = _overrides_for(
        technical=make_technical(points=17.0, confidence=70.0),
        market=make_market(points=16.0),
        risk=make_risk(points=8.0),
    )
    result = apply_gates(78.0, cats, confidences, overrides)
    assert result.label == GATE_MOMENTUM
    assert GATE_MOMENTUM in result.passed_gates


def test_momentum_gate_fails_just_below_raw_threshold():
    cats = CategoryPoints(business=15, financial=13, market=16, technical=17, risk=8, valuation=2)
    confidences = CategoryConfidences(business=90, financial=90, market=90, technical=70, risk=90, valuation=90)
    overrides = _overrides_for(
        technical=make_technical(points=17.0, confidence=70.0),
        market=make_market(points=16.0),
        risk=make_risk(points=8.0),
    )
    result = apply_gates(77.9, cats, confidences, overrides)
    assert result.label != GATE_MOMENTUM
    assert any("raw_total<78" in r for r in result.failed_gates)


def test_momentum_gate_fails_on_technical_confidence():
    cats = CategoryPoints(business=15, financial=13, market=16, technical=17, risk=8, valuation=2)
    confidences = CategoryConfidences(business=90, financial=90, market=90, technical=69.9, risk=90, valuation=90)
    overrides = _overrides_for(
        technical=make_technical(points=17.0, confidence=69.9),
        market=make_market(points=16.0),
        risk=make_risk(points=8.0),
    )
    result = apply_gates(78.0, cats, confidences, overrides)
    assert result.label != GATE_MOMENTUM
    assert any("technical_confidence<70" in r for r in result.failed_gates)


# ============================================================================
# Quality / Value gates: exact thresholds
# ============================================================================


def test_quality_gate_exact_thresholds_pass():
    cats = CategoryPoints(business=16, financial=11, market=16, technical=12, risk=10, valuation=5)
    overrides = _overrides_for(
        business=make_business(points=16.0), financial=make_financial(points=11.0),
        risk=make_risk(points=10.0), valuation=make_valuation(points=5.0),
        technical=make_technical(points=12.0),
    )
    result = apply_gates(80.0, cats, _FULL_CONF, overrides)
    assert result.label == GATE_QUALITY


def test_value_gate_exact_thresholds_pass():
    cats = CategoryPoints(business=13, financial=10, market=15, technical=9, risk=10, valuation=8)
    overrides = _overrides_for(
        business=make_business(points=13.0), valuation=make_valuation(points=8.0),
        risk=make_risk(points=10.0), technical=make_technical(points=9.0),
    )
    result = apply_gates(75.0, cats, _FULL_CONF, overrides)
    assert result.label == GATE_VALUE


# ============================================================================
# Conditional / Watch and Avoid / Wait fallbacks
# ============================================================================


def test_conditional_watch_when_raw_is_60_but_no_gate_passes():
    cats = CategoryPoints(business=10, financial=8, market=10, technical=10, risk=12, valuation=10)
    raw = raw_total(cats)
    assert raw == 60.0
    overrides = _overrides_for(risk=make_risk(points=12.0))
    result = apply_gates(raw, cats, _FULL_CONF, overrides)
    assert result.label == GATE_CONDITIONAL
    assert result.passed_gates == []


def test_avoid_wait_when_raw_below_50():
    cats = CategoryPoints(business=8, financial=6, market=8, technical=8, risk=10, valuation=5)
    raw = raw_total(cats)
    assert raw == 45.0
    overrides = _overrides_for()
    result = apply_gates(raw, cats, _FULL_CONF, overrides)
    assert result.label == GATE_AVOID


def test_speculative_when_total_confidence_below_60():
    cats = CategoryPoints(business=16, financial=10.5, market=18, technical=16, risk=9, valuation=7)
    raw = raw_total(cats)
    low_conf = CategoryConfidences(business=40, financial=40, market=40, technical=40, risk=40, valuation=40)
    overrides = _overrides_for()
    result = apply_gates(raw, cats, low_conf, overrides)
    assert result.label == GATE_SPECULATIVE


def test_speculative_when_pre_profit_low_confidence_terminal_value():
    """SCORING_AND_GATES.md Speculative bullet: 'pre-profit valuation
    depends on a low-confidence terminal value'. A high-raw name with the
    valuation HIGH_TERMINAL_SENSITIVITY flag AND pre-profit is forced
    Speculative despite an otherwise-strong score."""
    # A raw high enough to otherwise clear Quality (raw=90).
    cats = CategoryPoints(business=20, financial=15, market=15, technical=20, risk=15, valuation=5)
    raw = raw_total(cats)
    assert raw == 90.0
    overrides = _overrides_for()
    result = apply_gates(
        raw, cats, _FULL_CONF, overrides,
        pre_profit=True, valuation_mandatory_flags=["HIGH_TERMINAL_SENSITIVITY"],
    )
    assert result.label == GATE_SPECULATIVE
    assert any("low-confidence terminal value" in w for w in result.warnings)


def test_not_speculative_when_terminal_flag_present_but_not_pre_profit():
    """The bullet is scoped to 'pre-profit valuation' -- a profitable
    company with HIGH_TERMINAL_SENSITIVITY is NOT forced Speculative on
    this bullet alone."""
    cats = CategoryPoints(business=20, financial=15, market=15, technical=20, risk=15, valuation=5)
    raw = raw_total(cats)
    overrides = _overrides_for()
    result = apply_gates(
        raw, cats, _FULL_CONF, overrides,
        pre_profit=False, valuation_mandatory_flags=["HIGH_TERMINAL_SENSITIVITY"],
    )
    assert result.label != GATE_SPECULATIVE


def test_weak_wait_fallback_label_for_raw_in_50_to_60_no_gate():
    """The documented [50,60) gap: raw in this band with no override and no
    Speculative trigger returns the GATE_WEAK constant."""
    cats = CategoryPoints(business=10, financial=8, market=10, technical=10, risk=12, valuation=5)
    raw = raw_total(cats)
    assert raw == 55.0
    overrides = _overrides_for(risk=make_risk(points=12.0))
    result = apply_gates(raw, cats, _FULL_CONF, overrides)
    assert result.label == GATE_WEAK
    assert result.passed_gates == []
