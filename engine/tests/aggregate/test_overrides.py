"""Tests for `wbj.aggregate.overrides` (Task 21): `validate_handoff` and
`apply_overrides`'s all-7 mandatory overrides.

Sources of truth: `Cerebro/shared/HANDOFF_CONTRACT.md`,
`Cerebro/00_main_agent/SCORING_AND_GATES.md` "Mandatory overrides",
`Cerebro/00_main_agent/VALIDATION_TESTS.md` (MAIN-009, MAIN-010).
"""

from __future__ import annotations

from wbj.aggregate.overrides import (
    OVERRIDE_1_CAPITAL_DEPENDENCE,
    OVERRIDE_2_ROIC_BELOW_WACC,
    OVERRIDE_3_SOLVENCY_WARNING,
    OVERRIDE_4_RISK_FLOOR,
    OVERRIDE_5_PREMIUM_BREAKDOWN,
    OVERRIDE_6_COVERAGE_GATE_INELIGIBLE,
    OVERRIDE_7_DATA_CONFLICT_SUPPRESS_PER_SHARE,
    AggregateInputs,
    apply_overrides,
    is_handoff_valid,
    validate_handoff,
)
from wbj.core.nullstates import NullState, Value
from wbj.core.scoring import Dimension
from wbj.schemas.levels import LevelsOutput, Touch, Zone
from wbj.specialists.common import MetricRow

from .conftest import (
    make_business,
    make_financial,
    make_market,
    make_risk,
    make_technical,
    make_valuation,
)


def _inputs(**overrides) -> AggregateInputs:
    base = dict(
        business=make_business(), financial=make_financial(), market=make_market(),
        technical=make_technical(), risk=make_risk(), valuation=make_valuation(),
    )
    base.update(overrides)
    return AggregateInputs(**base)


# ============================================================================
# validate_handoff (HANDOFF_CONTRACT.md)
# ============================================================================


def test_validate_handoff_accepts_a_well_formed_output():
    assert validate_handoff(make_business()) == []
    assert is_handoff_valid(make_business())


def test_validate_handoff_rejects_category_points_not_reproducing_from_dimensions():
    biz = make_business(points=16.0)
    # Corrupt the envelope's claimed awarded_points without touching the
    # dimensions that actually produced it.
    tampered = biz.model_copy(update={"category": biz.category.model_copy(update={"awarded_points": 19.0})})
    reasons = validate_handoff(tampered)
    assert any("CATEGORY_POINTS_DO_NOT_REPRODUCE" in r for r in reasons)


def test_validate_handoff_rejects_missing_formula_id():
    bad_row = MetricRow(
        metric_id="FIN-XX-000", value=1.0, unit="usd", formula_id="", formula_version="2.0.0",
        score=5.0, confidence=80.0,
    )
    fin = make_financial(metrics=[bad_row])
    reasons = validate_handoff(fin)
    assert any("METRIC_MISSING_FORMULA_ID" in r for r in reasons)


def test_validate_handoff_rejects_missing_knowledge_timestamp():
    biz = make_business(knowledge_timestamp="")
    reasons = validate_handoff(biz)
    assert "MISSING_KNOWLEDGE_TIMESTAMP" in reasons


def test_validate_handoff_rejects_missing_confidence_or_coverage():
    biz_no_conf = make_business(confidence=None)
    assert "MISSING_CONFIDENCE" in validate_handoff(biz_no_conf)

    biz_no_cov = make_business(coverage=None)
    assert "MISSING_COVERAGE" in validate_handoff(biz_no_cov)


def test_validate_handoff_rejects_confirmed_zone_missing_touches():
    zone = Zone(
        zone_id="daily-support-1.00", type="support", lower=0.95, center=1.0, upper=1.05,
        timeframe="daily", status="confirmed", strength_0_100=80.0, touches=[],
        distance_atr=1.0, confirmation_rule="c", invalidation_rule="i",
    )
    tech = make_technical(important_levels=LevelsOutput(nearest_support=[zone]))
    reasons = validate_handoff(tech)
    assert any("ZONE_MISSING_TOUCHES" in r for r in reasons)


def test_validate_handoff_rejects_confirmed_zone_missing_atr_distance():
    zone = Zone(
        zone_id="daily-support-1.00", type="support", lower=0.95, center=1.0, upper=1.05,
        timeframe="daily", status="confirmed", strength_0_100=80.0,
        touches=[Touch(date="2026-01-05", pivot_price=1.0, rejection_atr=0.6, volume_ratio=1.3)],
        distance_atr=None, confirmation_rule="c", invalidation_rule="i",
    )
    tech = make_technical(important_levels=LevelsOutput(nearest_support=[zone]))
    reasons = validate_handoff(tech)
    assert any("ZONE_MISSING_ATR_DISTANCE" in r for r in reasons)


def test_validate_handoff_ignores_candidate_zones_missing_touches():
    """A bare `candidate` zone doesn't yet claim confirmation, so it isn't
    held to the touch/ATR-distance requirement."""
    zone = Zone(
        zone_id="daily-support-1.00", type="support", lower=0.95, center=1.0, upper=1.05,
        timeframe="daily", status="candidate", strength_0_100=10.0, touches=[],
        distance_atr=None, confirmation_rule="c", invalidation_rule="i",
    )
    tech = make_technical(important_levels=LevelsOutput(nearest_support=[zone]))
    assert validate_handoff(tech) == []


# ============================================================================
# apply_overrides: none triggered
# ============================================================================


def test_apply_overrides_none_triggered_on_a_clean_bundle():
    assert apply_overrides(_inputs()) == []


# ============================================================================
# Override 1: capital dependence -> caps Avoid/Speculative
# ============================================================================


def test_override_1_capital_dependence():
    fin = make_financial(mandatory_flags=["OVERRIDE_1_LOSS_NEGATIVE_FCF_EXTERNAL_DEPENDENCE"])
    triggered = apply_overrides(_inputs(financial=fin))
    ids = {o.id for o in triggered}
    assert OVERRIDE_1_CAPITAL_DEPENDENCE in ids


# ============================================================================
# Override 2: ROIC < WACC -> no Elite/Quality (MAIN-005)
# ============================================================================


def test_override_2_roic_below_wacc_from_financial_flag():
    fin = make_financial(mandatory_flags=["OVERRIDE_2_ROIC_BELOW_WACC"])
    triggered = apply_overrides(_inputs(financial=fin))
    ids = {o.id for o in triggered}
    assert OVERRIDE_2_ROIC_BELOW_WACC in ids


def test_override_2_roic_below_wacc_from_business_flag():
    biz = make_business(mandatory_flags=["VALUE_DESTRUCTION"])
    triggered = apply_overrides(_inputs(business=biz))
    ids = {o.id for o in triggered}
    assert OVERRIDE_2_ROIC_BELOW_WACC in ids


# ============================================================================
# Override 3: interest coverage < 1.5x -> solvency warning, always (MAIN-006)
# ============================================================================


def test_override_3_solvency_warning_from_risk_flag():
    rk = make_risk(mandatory_flags=["SOLVENCY_WARNING"])
    triggered = apply_overrides(_inputs(risk=rk))
    ids = {o.id for o in triggered}
    assert OVERRIDE_3_SOLVENCY_WARNING in ids


def test_override_3_solvency_warning_from_financial_metric_row():
    row = MetricRow(
        metric_id="FIN-BS-020", value=1.2, unit="ratio", formula_id="FIN-BS-020", formula_version="2.0.0",
        score=0.0, confidence=80.0, warnings=["SOLVENCY_WARNING"],
    )
    fin = make_financial(metrics=[row])
    triggered = apply_overrides(_inputs(financial=fin))
    ids = {o.id for o in triggered}
    assert OVERRIDE_3_SOLVENCY_WARNING in ids


# ============================================================================
# Override 4: Risk 0-4/15 -> caps Speculative (MAIN-003)
# ============================================================================


def test_override_4_risk_floor():
    rk = make_risk(points=4.0)
    triggered = apply_overrides(_inputs(risk=rk))
    ids = {o.id for o in triggered}
    assert OVERRIDE_4_RISK_FLOOR in ids


def test_override_4_risk_floor_not_triggered_above_4():
    rk = make_risk(points=4.01)
    triggered = apply_overrides(_inputs(risk=rk))
    ids = {o.id for o in triggered}
    assert OVERRIDE_4_RISK_FLOOR not in ids


# ============================================================================
# Override 5: Valuation<=4/10 AND Technical<=8/20 -> Wait/Avoid (MAIN-004)
# ============================================================================


def test_override_5_premium_breakdown():
    val = make_valuation(points=3.0)
    tech = make_technical(points=7.0)
    triggered = apply_overrides(_inputs(valuation=val, technical=tech))
    ids = {o.id for o in triggered}
    assert OVERRIDE_5_PREMIUM_BREAKDOWN in ids


def test_override_5_not_triggered_unless_both_conditions_hold():
    val = make_valuation(points=3.0)
    tech = make_technical(points=9.0)  # above the 8/20 breakdown floor
    triggered = apply_overrides(_inputs(valuation=val, technical=tech))
    ids = {o.id for o in triggered}
    assert OVERRIDE_5_PREMIUM_BREAKDOWN not in ids


# ============================================================================
# Override 6: any core category coverage < 0.70 -> gate-ineligible (MAIN-007)
# ============================================================================


def test_override_6_coverage_gate_ineligible():
    fin = make_financial(coverage=0.65)
    triggered = apply_overrides(_inputs(financial=fin))
    matches = [o for o in triggered if o.id == OVERRIDE_6_COVERAGE_GATE_INELIGIBLE]
    assert len(matches) == 1
    assert "financial" in matches[0].reason


def test_override_6_not_triggered_at_exactly_070():
    fin = make_financial(coverage=0.70)
    triggered = apply_overrides(_inputs(financial=fin))
    ids = {o.id for o in triggered}
    assert OVERRIDE_6_COVERAGE_GATE_INELIGIBLE not in ids


# ============================================================================
# Override 7: unresolved material facts-table conflict -> suppress
# per-share valuation (MAIN-009, MAIN-010)
# ============================================================================


def test_override_7_data_conflict_from_facts_table():
    facts_table = {
        "diluted_shares": Value.null(NullState.CONFLICTED, unit="shares", warnings=["FMP=1.0e9 EDGAR=1.2e9 diff=20% CONFLICTED"]),
    }
    inputs = _inputs(facts_table=facts_table)
    triggered = apply_overrides(inputs)
    matches = [o for o in triggered if o.id == OVERRIDE_7_DATA_CONFLICT_SUPPRESS_PER_SHARE]
    assert len(matches) == 1
    assert "diluted_shares" in matches[0].reason


def test_override_7_data_conflict_from_valuation_metric_fallback():
    """MAIN-010: 'Same metric has material source conflict' -> 'Mark
    conflicted and rerun affected agents'. When no `facts_table` is
    passed through, `apply_overrides` still detects a CONFLICTED metric
    row surfaced by the valuation specialist itself."""
    row = MetricRow(
        metric_id="VAL-NORM-001", value=None, state=NullState.CONFLICTED, unit="usd",
        formula_id="VAL-NORM-001", formula_version="2.0.0", score="NOT_SCORABLE", confidence=0.0,
    )
    val = make_valuation(metrics=[row])
    triggered = apply_overrides(_inputs(valuation=val))
    matches = [o for o in triggered if o.id == OVERRIDE_7_DATA_CONFLICT_SUPPRESS_PER_SHARE]
    assert len(matches) == 1


def test_override_7_not_triggered_without_a_conflict():
    facts_table = {"diluted_shares": Value.of(1.0e9, unit="shares")}
    triggered = apply_overrides(_inputs(facts_table=facts_table))
    ids = {o.id for o in triggered}
    assert OVERRIDE_7_DATA_CONFLICT_SUPPRESS_PER_SHARE not in ids


# ============================================================================
# Multiple overrides simultaneously
# ============================================================================


def test_multiple_overrides_can_trigger_together():
    rk = make_risk(points=2.0, mandatory_flags=["SOLVENCY_WARNING"])
    val = make_valuation(points=2.0)
    tech = make_technical(points=5.0)
    triggered = apply_overrides(_inputs(risk=rk, valuation=val, technical=tech))
    ids = {o.id for o in triggered}
    assert OVERRIDE_3_SOLVENCY_WARNING in ids
    assert OVERRIDE_4_RISK_FLOOR in ids
    assert OVERRIDE_5_PREMIUM_BREAKDOWN in ids
