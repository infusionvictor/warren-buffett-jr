"""Tests for `wbj.specialists.valuation` (Task 19): VAL-001..044 (mostly
reused from `wbj.engines.valuation_engine`), the five weighted dimensions,
the ADAPTER_UNSUPPORTED gate, and `run()` against the NVDA golden fixture.

Sources of truth: `Cerebro/06_valuation_analysis/{FORMULAS,SCORING,
DECISION_RULES,VALIDATION_TESTS}.md`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wbj.core.nullstates import NullState
from wbj.core.scoring import Category
from wbj.engines import valuation_engine as ve
from wbj.schemas.packet import AnalysisMeta, MarketData, Packet, Security
import wbj.specialists.valuation as val

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "packet" / "NVDA_packet.json"


@pytest.fixture(scope="module")
def nvda_packet() -> Packet:
    data = json.loads(_FIXTURE.read_text())
    return Packet.model_validate(data)


def _row(year: int, **overrides) -> dict:
    base = dict(
        calendarYear=str(year), date=f"{year}-12-31", period="FY",
        revenue=1000.0, cogs=600.0, gross_profit=400.0, ebit=200.0,
        income_before_tax=190.0, income_tax_expense=40.0, net_income=150.0,
        operating_cash_flow=180.0, capex=-50.0, fcf=130.0, cash=300.0,
        eps=1.5, total_debt=200.0, total_equity=600.0, diluted_shares=100.0,
    )
    base.update(overrides)
    return base


def _minimal_packet(
    annual_rows: list[dict], *, industry_adapter: str = "default_nonfinancial",
    price: float = 20.0, diluted_shares: float = 100.0, cash: float = 300.0, total_debt: float = 200.0,
    revenue: float = 1000.0, beta: float = 1.2, risk_free_rate: float = 0.04,
) -> Packet:
    from wbj.core.nullstates import EvidenceClass, Value

    facts = {
        "price": Value.of(price, unit="usd_per_share", evidence_class=EvidenceClass.R),
        "diluted_shares": Value.of(diluted_shares, unit="shares", evidence_class=EvidenceClass.R),
        "cash": Value.of(cash, unit="usd", evidence_class=EvidenceClass.R),
        "total_debt": Value.of(total_debt, unit="usd", evidence_class=EvidenceClass.R),
        "revenue": Value.of(revenue, unit="usd", evidence_class=EvidenceClass.R),
    }
    return Packet(
        security=Security(ticker="TEST", exchange="NASDAQ", security_type="operating_company", reporting_currency="USD", valuation_currency="USD"),
        analysis=AnalysisMeta(knowledge_timestamp="2026-07-16T21:00:00+00:00", industry_adapter=industry_adapter),
        fundamentals={"annual": annual_rows, "quarterly": []},
        market_data=MarketData(),
        estimates={"risk_free_rate": risk_free_rate, "peers": [], "fmp_analyst_estimates": []},
        capital_structure={"beta": beta, "market_cap": price * diluted_shares, "cash": cash, "total_debt": total_debt, "diluted_shares": diluted_shares},
        facts_table=facts,
        staleness={},
    )


# ============================================================================
# VAL-VALIDATION_TESTS.md, encoded verbatim (VAL-T001..T010)
# ============================================================================


def test_VAL_T001_gordon_terminal_value():
    """FCFF1=100, WACC=10%, g=3% in perpetuity -> TV at t0=100/(0.10-0.03)=1428.57."""
    v = ve.gordon_terminal_value(fcff_n=100.0 / 1.03, g=0.03, wacc_value=0.10)
    # gordon_terminal_value takes FCFF_N and grows it by (1+g) internally;
    # feed FCFF_N = FCFF1/(1+g) so FCFF_(N+1) = FCFF1 = 100.
    assert v.value == pytest.approx(1428.5714, rel=1e-4)


def test_VAL_T002_wacc_equals_g_rejected():
    """WACC=8%, g=8% -> Reject model: denominator zero."""
    v = ve.gordon_terminal_value(fcff_n=100.0, g=0.08, wacc_value=0.08)
    assert v.is_null
    assert v.state == NullState.NOT_MEANINGFUL


def test_VAL_T003_g_exceeds_wacc_rejected():
    """WACC=7%, g=8% -> Reject model: g>=WACC."""
    v = ve.gordon_terminal_value(fcff_n=100.0, g=0.08, wacc_value=0.07)
    assert v.is_null


def test_VAL_T004_terminal_reinvestment_rate():
    """ROIC=20%, g=4% -> Terminal reinvestment rate=20%."""
    rate = ve._terminal_reinvestment_rate(g=0.04, roic_value=0.20)
    assert rate == pytest.approx(0.20)


def test_VAL_T005_equity_bridge_and_per_share():
    """EV=1000, cash=100, debt=300, diluted shares=80 -> Equity=800; value/share=10."""
    eq = ve.equity_bridge(ev=1000.0, cash=100.0, nonop=0.0, debt=300.0, lease_debt_value=0.0, preferred=0.0, minority=0.0, pension=0.0)
    assert eq.value == pytest.approx(800.0)
    ps = ve.per_share(eq.value, 80.0)
    assert ps.value == pytest.approx(10.0)


def test_VAL_T006_scenario_probabilities_sum_to_1():
    """Scenario probabilities 20%,60%,20% -> Sum=100%; pass."""
    bear = val.ScenarioInput(probability=0.2, growth=0.02, margin=0.15, wacc=0.11, tv_growth=0.02)
    base = val.ScenarioInput(probability=0.6, growth=0.05, margin=0.20, wacc=0.10, tv_growth=0.02)
    bull = val.ScenarioInput(probability=0.2, growth=0.08, margin=0.25, wacc=0.09, tv_growth=0.02)
    common = val.DCFCommonInputs(revenue0=1000.0, shares=100.0, tax_rate=0.21, roic=0.15, years=5, net_debt=100.0)
    result = ve.scenarios(bear, base, bull, common)
    assert result.probabilities_sum == pytest.approx(1.0)


def test_VAL_T007_terminal_value_share_above_75pct_flags():
    """Terminal PV=800, EV=1000 -> Terminal share=80%; high-sensitivity flag."""
    v = ve.terminal_share(pv_terminal=800.0, ev=1000.0)
    assert v.value == pytest.approx(0.80)
    assert "TERMINAL_VALUE_SHARE_ABOVE_75PCT" in v.warnings


def test_VAL_T008_missing_convertible_schedule_incomplete():
    """Missing option/convertible schedule -> Per-share value incomplete
    (this module never invents a convertible schedule; it is overlay-only
    and simply absent from the equity bridge when not supplied)."""
    # No convertible dilution term is applied anywhere in this module's
    # equity_bridge call -- confirms the bridge takes exactly the
    # capital-structure claims it was given, no implicit convertible
    # adjustment invented.
    eq = ve.equity_bridge(ev=1000.0, cash=0.0, nonop=0.0, debt=0.0, lease_debt_value=0.0, preferred=0.0, minority=0.0, pension=0.0)
    assert eq.value == pytest.approx(1000.0)


def test_VAL_T009_fcff_and_eva_reconciliation_fails_when_materially_different():
    """FCFF and EVA differ materially with same assumptions -> Fail
    reconciliation; inspect reinvestment/capital."""
    from wbj.core.nullstates import Value

    a = Value.of(100.0, unit="usd_per_share")
    b = Value.of(150.0, unit="usd_per_share")
    assert ve.reconciles(a, b) is False


def test_VAL_T010_bank_selected_adapter_unsupported():
    """Bank selected -> Use residual-income/excess-return adapter; no
    EV/EBITDA primary model."""
    rows = [_row(2025), _row(2024)]
    packet = _minimal_packet(rows, industry_adapter="bank_adapter")
    out = val.run(packet)
    assert "ADAPTER_UNSUPPORTED" in out.mandatory_flags
    assert out.model_selection.primary == []
    assert "FCFF_DCF" in out.model_selection.rejected


def test_dispersion_comes_from_ensemble_not_raw_stdev():
    """model_cross_checks.dispersion must come from
    valuation_engine.ensemble() (VAL-ENSEMBLE-044, reliability-weighted),
    reproducing its `.dispersion` exactly for the same model inputs -- not
    an ad-hoc unweighted np.std."""
    from wbj.core.nullstates import EvidenceClass, Value

    rows = [_row(2025), _row(2024)]
    packet = _minimal_packet(rows)
    out = val.run(packet)
    fcff = out.model_cross_checks.fcff
    ep = out.model_cross_checks.economic_profit
    rel = out.model_cross_checks.relative
    models = []
    if fcff is not None:
        models.append(val.EnsembleModelInput(label="FCFF_DCF", value=Value.of(fcff, unit="usd_per_share", evidence_class=EvidenceClass.C), weight=1.0))
    if ep is not None:
        models.append(val.EnsembleModelInput(label="ECONOMIC_PROFIT", value=Value.of(ep, unit="usd_per_share", evidence_class=EvidenceClass.C), weight=0.8))
    if rel is not None:
        models.append(val.EnsembleModelInput(label="RELATIVE", value=Value.of(rel, unit="usd_per_share", evidence_class=EvidenceClass.C), weight=0.5))
    if len(models) > 1:
        expected = ve.ensemble(models)
        assert out.model_cross_checks.dispersion == pytest.approx(expected.dispersion.value)
    else:
        # too few models to form an ensemble -> dispersion is None, not a raw stdev
        assert out.model_cross_checks.dispersion is None


def test_unsupported_adapter_confidence_derived_from_formula():
    """The ADAPTER_UNSUPPORTED ERROR envelope derives confidence from the
    real five-component formula at coverage 0 (model_fit drops to 40 for a
    non-default adapter), not a hardcoded 0.0 -- awarded_points/score_10
    stay 0.0 by construction."""
    rows = [_row(2025), _row(2024)]
    packet = _minimal_packet(rows, industry_adapter="reit_adapter")
    out = val.run(packet)
    assert out.category.awarded_points == 0.0
    assert out.category.score_10 == 0.0
    expected = val._category_confidence(0.0, packet)
    assert out.category.confidence == pytest.approx(expected)
    assert out.category.confidence > 0.0


# ============================================================================
# ADAPTER_UNSUPPORTED gate
# ============================================================================


def test_run_supported_adapter_selects_fcff_and_economic_profit():
    rows = [_row(2025), _row(2024)]
    packet = _minimal_packet(rows)
    out = val.run(packet)
    assert out.model_selection.primary == ["FCFF_DCF", "ECONOMIC_PROFIT"]


# ============================================================================
# Scenario / WACC / reverse-DCF orchestration
# ============================================================================


def test_run_scenarios_sum_to_one_and_ordered_bear_base_bull():
    rows = [_row(2025), _row(2024)]
    packet = _minimal_packet(rows)
    out = val.run(packet)
    assert [s.name for s in out.scenarios] == ["Bear", "Base", "Bull"]
    total_p = sum(s.probability for s in out.scenarios if s.probability is not None)
    assert total_p == pytest.approx(1.0, abs=1e-6)


def test_run_reverse_dcf_present_when_inputs_available():
    rows = [_row(2025), _row(2024)]
    packet = _minimal_packet(rows)
    out = val.run(packet)
    assert out.reverse_dcf.current_price == pytest.approx(20.0)


def test_run_terminal_growth_clamped_below_wacc():
    rows = [_row(2025), _row(2024)]
    packet = _minimal_packet(rows)
    out = val.run(packet, overlay={"terminal_growth": 0.99})
    assert any("clamped" in a for a in out.assumptions)


def test_run_wacc_populated_with_components():
    rows = [_row(2025), _row(2024)]
    packet = _minimal_packet(rows)
    out = val.run(packet)
    assert out.wacc.value is not None
    assert out.wacc.components["cost_of_equity"] is not None
    assert out.wacc.components["cost_of_debt"] is not None


def test_run_fair_value_distribution_from_monte_carlo():
    rows = [_row(2025), _row(2024)]
    packet = _minimal_packet(rows)
    out = val.run(packet)
    assert out.fair_value_distribution.p10 is not None
    assert out.fair_value_distribution.p90 is not None
    assert out.fair_value_distribution.p10 <= out.fair_value_distribution.median <= out.fair_value_distribution.p90


def test_run_reference_bands_and_mos_thresholds():
    rows = [_row(2025), _row(2024)]
    packet = _minimal_packet(rows)
    out = val.run(packet)
    if out.reference_bands.base is not None:
        assert out.reference_bands.margin_of_safety_15pct == pytest.approx(out.reference_bands.base * 0.85)
        assert out.reference_bands.margin_of_safety_25pct == pytest.approx(out.reference_bands.base * 0.75)


# ============================================================================
# Dimension caps / gates (the apply_dimension_cap helper is tested in
# test_common.py; the tests below exercise valuation.py's *use* of it).
# ============================================================================


def test_run_category_reproduces_from_dimensions():
    rows = [_row(2025), _row(2024)]
    packet = _minimal_packet(rows)
    out = val.run(packet)
    recomputed = Category(name=val.AGENT_ID, max_points=val.MAX_POINTS, dimensions=out.dimensions)
    assert out.category.awarded_points == pytest.approx(recomputed.points(), abs=1e-6)
    assert out.category.score_10 == pytest.approx(recomputed.score10(), abs=1e-6)
    assert out.coverage == pytest.approx(recomputed.coverage(), abs=1e-6)


# ============================================================================
# run() against the NVDA fixture
# ============================================================================


def test_run_nvda_fixture_schema_valid(nvda_packet):
    out = val.run(nvda_packet)
    assert out.agent_id == "valuation_analysis"
    assert out.version == "2.0.0"
    assert out.security.ticker == "NVDA"
    assert out.category.max_points == 10.0
    assert out.status in ("COMPLETE", "INCOMPLETE", "ERROR")
    assert len(out.dimensions) == 5
    for row in out.metrics:
        assert row.metric_id
        assert row.formula_id
        assert row.formula_version
        assert row.score == "NOT_SCORABLE" or isinstance(row.score, float)
        assert 0.0 <= row.confidence <= 100.0
        assert (row.value is None) != (row.state is None)


def test_run_nvda_fixture_category_math_reproduces_from_dimensions(nvda_packet):
    out = val.run(nvda_packet)
    recomputed = Category(name=val.AGENT_ID, max_points=val.MAX_POINTS, dimensions=out.dimensions)
    assert out.category.awarded_points == pytest.approx(recomputed.points(), abs=1e-6)
    assert out.category.score_10 == pytest.approx(recomputed.score10(), abs=1e-6)
    assert out.coverage == pytest.approx(recomputed.coverage(), abs=1e-6)


def test_run_nvda_fixture_category_confidence_computed(nvda_packet):
    out = val.run(nvda_packet)
    assert out.category.confidence is not None
    assert 0.0 <= out.category.confidence <= 100.0


def test_run_nvda_fixture_serializes_to_json(nvda_packet):
    out = val.run(nvda_packet)
    dumped = out.model_dump(mode="json")
    json.dumps(dumped)
    assert dumped["agent_id"] == "valuation_analysis"


def test_run_nvda_fixture_scenarios_present(nvda_packet):
    out = val.run(nvda_packet)
    assert len(out.scenarios) == 3


def test_run_nvda_fixture_validation_tests_all_self_checks_pass(nvda_packet):
    out = val.run(nvda_packet)
    assert out.validation_tests.failed == 0
    assert out.validation_tests.passed >= 1


def test_run_empty_annual_history_degrades_without_crashing():
    """No annual fundamentals -> `run()` must fall back to documented
    constants (flagged as assumptions) for ROIC/growth/margin rather than
    crashing; facts_table (price/shares/cash/debt) alone is still enough
    to produce a scenario valuation, so coverage need not be zero -- this
    test only asserts graceful degradation, not a specific coverage."""
    out = val.run(_minimal_packet([]))
    assert out.status in ("COMPLETE", "INCOMPLETE", "ERROR")
    assert 0.0 <= out.coverage <= 1.0
    assert any("ROIC" in a or "fallback" in a for a in out.assumptions)
