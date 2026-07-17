"""Tests for `wbj.specialists.business` (Task 15): BUS-001..030, the five
weighted dimensions, mandatory flags/caps, and `run()` against the NVDA
golden fixture.

Sources of truth: `Cerebro/01_business_analysis/{FORMULAS,SCORING,
DECISION_RULES,VALIDATION_TESTS}.md`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wbj.core.nullstates import NullState
from wbj.core.scoring import Category
from wbj.schemas.packet import AnalysisMeta, MarketData, Packet, Security
import wbj.specialists.business as bus

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "packet" / "NVDA_packet.json"


@pytest.fixture(scope="module")
def nvda_packet() -> Packet:
    data = json.loads(_FIXTURE.read_text())
    return Packet.model_validate(data)


def _minimal_packet(
    annual_rows: list[dict],
    *,
    industry_adapter: str = "default_nonfinancial",
    security_type: str = "operating_company",
) -> Packet:
    return Packet(
        security=Security(
            ticker="TEST", exchange="NASDAQ", security_type=security_type,
            reporting_currency="USD", valuation_currency="USD",
        ),
        analysis=AnalysisMeta(
            knowledge_timestamp="2026-07-16T21:00:00+00:00", industry_adapter=industry_adapter,
        ),
        fundamentals={"annual": annual_rows, "quarterly": []},
        market_data=MarketData(),
        estimates={},
        capital_structure={},
        facts_table={},
        staleness={},
    )


def _row(year: int, **overrides) -> dict:
    base = dict(
        calendarYear=str(year),
        date=f"{year}-12-31",
        period="FY",
        revenue=1000.0,
        cogs=600.0,
        gross_profit=400.0,
        ebit=200.0,
        income_before_tax=190.0,
        income_tax_expense=40.0,
        net_income=150.0,
        operating_cash_flow=180.0,
        capex=-50.0,
        cash=300.0,
        inventory=80.0,
        net_receivables=90.0,
        total_current_assets=500.0,
        total_current_liabilities=300.0,
        total_debt=200.0,
        total_equity=600.0,
        total_assets=1200.0,
        total_liabilities=600.0,
        diluted_shares=100.0,
        stock_based_compensation=10.0,
        common_stock_repurchased=0.0,
        dividends_paid=0.0,
        debt_repayment=0.0,
        acquisitions_net=0.0,
    )
    base.update(overrides)
    return base


def _five_year_rows(**per_year_overrides) -> list[dict]:
    """Five newest-first, boring, growing annual rows (2021..2025) with a
    stable 20% operating margin, for tests that need enough history for
    BUS-STAB-009/010, ROIC history, and dilution CAGR."""
    years = [2025, 2024, 2023, 2022, 2021]
    rows = []
    for i, y in enumerate(years):
        rev = 1000.0 + (len(years) - 1 - i) * -100.0  # oldest is smallest -> growth
        row = _row(
            y,
            revenue=rev,
            cogs=rev * 0.6,
            gross_profit=rev * 0.4,
            ebit=rev * 0.2,
            income_before_tax=rev * 0.19,
            income_tax_expense=rev * 0.19 * 0.21,
            net_income=rev * 0.15,
            operating_cash_flow=rev * 0.18,
            capex=-rev * 0.05,
            total_debt=200.0,
            total_equity=600.0 + (len(years) - 1 - i) * -20.0,
            diluted_shares=100.0 + i,  # oldest (i=4) has fewest shares -> mild dilution
        )
        rows.append(row)
    return rows


# ============================================================================
# BUS-VALIDATION_TESTS.md, encoded verbatim (BUS-T001..T008)
# ============================================================================


def test_BUS_T001_nopat_and_roic():
    """EBIT=100, tax=25%, invested capital=500 -> NOPAT=75; ROIC=15%."""
    nopat_v = bus.nopat(100.0, 0.25)
    assert nopat_v.value == pytest.approx(75.0)
    roic_v = bus.roic(nopat_v.value, 500.0)
    assert roic_v.value == pytest.approx(0.15)


def test_BUS_T002_spread_and_eva():
    """ROIC=15%, WACC=10%, IC=500 -> Spread=5pts; economic value=25."""
    spread_v = bus.spread(0.15, 0.10)
    assert spread_v.value == pytest.approx(0.05)
    eva_v = bus.eva(75.0, 0.10, 500.0)
    assert eva_v.value == pytest.approx(25.0)


def test_BUS_T003_margin_range_and_stability():
    """Five margins 20%,21%,19%,22%,20% -> range=3pts; stable signal."""
    margins = [0.20, 0.21, 0.19, 0.22, 0.20]
    range_v = bus.margin_range(margins)
    assert range_v.value == pytest.approx(0.03)
    assert bus.margin_range_is_stable(range_v.value) is True


def test_BUS_T004_largest_customer_concentration_red_flag():
    """Largest customer share=35% -> concentration red flag."""
    v = bus.largest_customer_concentration(35.0, 100.0)
    assert v.value == pytest.approx(0.35)
    assert bus.is_concentration_red_flag(v.value) is True


def test_BUS_T005_cumulative_fcf_conversion():
    """Five-year FCF=500, net income=450 -> FCF conversion=1.111x."""
    v = bus.cumulative_fcf_conversion(500.0, 450.0)
    assert v.value == pytest.approx(500.0 / 450.0, rel=1e-4)


def test_BUS_T006_revenue_cagr_nonpositive_begin_not_meaningful():
    """Beginning revenue <=0 for CAGR -> NOT_MEANINGFUL."""
    v = bus.revenue_cagr(100.0, 0.0, 3.0)
    assert v.is_null
    assert v.state == NullState.NOT_MEANINGFUL


def test_BUS_T007_roic_below_wacc_no_excellent_or_wide_moat():
    """ROIC<WACC with an otherwise-high score -> no Excellent/wide-moat label."""
    assert bus.value_destruction_triggered(roic_value=0.09, wacc_value=0.11) is True
    capped = bus.capped_verdict(
        9.5,
        value_destruction=True,
        excellent_gate_passes=True,
    )
    uncapped = bus.capped_verdict(
        9.5,
        value_destruction=False,
        excellent_gate_passes=True,
    )
    assert capped != "Excellent business"
    assert uncapped == "Excellent business"


def test_BUS_T008_missing_nrr_non_subscription_no_penalty(nvda_packet):
    """Missing NRR for a non-subscription industrial -> use adapter, no
    penalty for N/A: the customer-economics dimension is NOT_SCORABLE
    (excluded from coverage denominator via valid_weight), not scored 0."""
    out = bus.run(nvda_packet)
    customer_dim = next(d for d in out.dimensions if d.name == bus.DIM_CUSTOMER)
    assert customer_dim.valid_weight() == 0.0
    # a NOT_SCORABLE dimension contributes 0 points but the category still
    # reproduces from dimensions -- it must not be silently defaulted.
    recomputed = Category(name=bus.AGENT_ID, max_points=bus.MAX_POINTS, dimensions=out.dimensions)
    assert out.category.awarded_points == pytest.approx(recomputed.points(), abs=1e-6)


# ============================================================================
# Individual formula behavior
# ============================================================================


def test_gross_and_operating_margin():
    assert bus.gross_margin(400.0, 1000.0).value == pytest.approx(0.40)
    assert bus.operating_margin(200.0, 1000.0).value == pytest.approx(0.20)


def test_margin_stability_stdev():
    v = bus.margin_stability([0.20, 0.21, 0.19, 0.22, 0.20])
    assert v.is_valid
    assert v.value >= 0.0


def test_customer_hhi_and_segment_hhi():
    hhi = bus.customer_hhi([0.5, 0.3, 0.2])
    assert hhi.value == pytest.approx(0.25 + 0.09 + 0.04)
    seg = bus.segment_hhi([1.0])
    assert seg.value == pytest.approx(1.0)


def test_diluted_share_cagr_reuses_core_cagr():
    v = bus.diluted_share_cagr(110.0, 100.0, 5.0)
    assert v.is_valid
    assert v.value == pytest.approx((110.0 / 100.0) ** (1 / 5) - 1)


def test_incremental_roic_and_allocation_spread():
    iroic = bus.incremental_roic(50.0, 400.0)
    assert iroic.value == pytest.approx(0.125)
    alloc = bus.capital_allocation_spread(iroic.value, 0.10)
    assert alloc.value == pytest.approx(0.025)


def test_guidance_accuracy_clipped_to_unit_interval():
    v = bus.guidance_accuracy(actual=1.0, guidance_midpoint=1.0, floor=0.01)
    assert v.value == pytest.approx(1.0)
    v2 = bus.guidance_accuracy(actual=-5.0, guidance_midpoint=1.0, floor=0.01)
    assert v2.value == pytest.approx(0.0)


def test_net_and_gross_revenue_retention():
    nrr = bus.net_revenue_retention(begin=1000.0, expansion=150.0, contraction=50.0, churn=50.0)
    assert nrr.value == pytest.approx((1000.0 + 150.0 - 50.0 - 50.0) / 1000.0)
    grr = bus.gross_revenue_retention(begin=1000.0, contraction=50.0, churn=50.0)
    assert grr.value == pytest.approx((1000.0 - 50.0 - 50.0) / 1000.0)


def test_ltv_cac_and_payback():
    ltv = bus.customer_ltv(arpu=120.0, gross_margin_pct=0.7, customer_life_years=4.0)
    assert ltv.value == pytest.approx(120.0 * 0.7 * 4.0)
    cac = bus.customer_acquisition_cost(spend=10000.0, new_customers=100.0)
    assert cac.value == pytest.approx(100.0)
    ratio = bus.ltv_to_cac(ltv.value, cac.value)
    assert ratio.value == pytest.approx(ltv.value / cac.value)
    payback = bus.cac_payback_months(cac=100.0, monthly_arpu=10.0, gross_margin_pct=0.5)
    assert payback.value == pytest.approx(100.0 / (10.0 * 0.5))


def test_reinvestment_rate_and_sustainable_growth():
    reinv = bus.reinvestment_rate(net_capex=40.0, dnwc=10.0, rd_adjustment=0.0, nopat_value=100.0)
    assert reinv.value == pytest.approx(0.5)
    sg = bus.fundamental_growth(reinv.value, 0.15)
    assert sg.value == pytest.approx(0.075)


# ============================================================================
# Dimension caps (SCORING.md "Gate / cap" column) -- the apply_dimension_cap
# helper itself is now tested once in test_common.py; the tests below
# exercise business.py's *use* of it (moat/durability caps).
# ============================================================================


def test_moat_capped_at_6_without_positive_spread(nvda_packet):
    """No positive ROIC-WACC spread (e.g. WACC not supplied, or spread<=0)
    -> moat dimension score capped at 6/10."""
    out = bus.run(nvda_packet, overlay={"wacc": 0.50})  # deliberately huge WACC -> spread<=0
    moat_dim = next(d for d in out.dimensions if d.name == bus.DIM_MOAT)
    assert moat_dim.score10() <= 6.0 + 1e-9


def test_durability_capped_at_6_with_concentration_red_flag():
    rows = _five_year_rows()
    packet = _minimal_packet(rows)
    # Supply recurring_revenue so all three durability members are valid
    # (>=70% usable coverage) and the dimension itself is scorable, letting
    # this test isolate the concentration cap rather than a coverage gate.
    out = bus.run(
        packet,
        overlay={"wacc": 0.09, "largest_customer_share": 0.35, "recurring_revenue": 500.0},
    )
    assert "CONCENTRATION_RED_FLAG" in out.mandatory_flags
    durability_dim = next(d for d in out.dimensions if d.name == bus.DIM_DURABILITY)
    assert durability_dim.score10() <= 6.0 + 1e-9


def test_wide_moat_gate_margin_condition_uses_5pp_not_3pp():
    """DECISION_RULES.md wide-moat gate condition 2 says the 5y operating-
    margin range must be 'no more than 5 percentage points' (<=0.05) -- a
    company with a 4pp range must PASS condition 2 (it fails the stricter
    <=0.03 BUS-RANGE-010 'positive moat signal', which is a different
    threshold for a different purpose)."""
    assert bus.wide_moat_margin_range_ok(0.04) is True   # 4pp passes the gate
    assert bus.wide_moat_margin_range_ok(0.05) is True    # exactly 5pp is "no more than 5"
    assert bus.wide_moat_margin_range_ok(0.0501) is False  # above 5pp fails
    # ...while BUS-RANGE-010's own 'positive moat signal' stays at <=0.03:
    assert bus.margin_range_is_stable(0.04) is False


def test_dilution_red_flag_when_diluted_cagr_above_5pct():
    rows = [
        _row(2025, diluted_shares=130.0),
        _row(2024, diluted_shares=100.0),
    ]
    out = bus.run(_minimal_packet(rows))
    assert "DILUTION_RED_FLAG" in out.mandatory_flags


def test_value_destruction_flag_when_roic_below_wacc():
    rows = _five_year_rows()
    packet = _minimal_packet(rows)
    out = bus.run(packet, overlay={"wacc": 0.50})
    assert "VALUE_DESTRUCTION" in out.mandatory_flags


# ============================================================================
# Judgment requests
# ============================================================================


def test_run_moat_classification_and_quantitative_effects_are_judgment_requests(nvda_packet):
    out = bus.run(nvda_packet)
    ids = {jr.metric_id for jr in out.judgment_requests}
    assert "moat_classification" in ids
    assert "moat_quantitative_effects_count" in ids
    assert "three_thesis_killers" in ids
    assert out.moat.classification == "NotScorable"


# ============================================================================
# run() against the NVDA fixture
# ============================================================================


def test_run_nvda_fixture_schema_valid(nvda_packet):
    out = bus.run(nvda_packet)
    assert out.agent_id == "business_analysis"
    assert out.version == "2.0.0"
    assert out.security.ticker == "NVDA"
    assert out.category.max_points == 20.0
    assert out.status in ("COMPLETE", "INCOMPLETE", "ERROR")
    assert len(out.dimensions) == 5
    assert len(out.metrics) == 30
    for row in out.metrics:
        assert row.metric_id
        assert row.formula_id
        assert row.formula_version
        assert row.score == "NOT_SCORABLE" or isinstance(row.score, float)
        assert 0.0 <= row.confidence <= 100.0
        assert (row.value is None) != (row.state is None)


def test_run_nvda_fixture_category_math_reproduces_from_dimensions(nvda_packet):
    out = bus.run(nvda_packet)
    recomputed = Category(name=bus.AGENT_ID, max_points=bus.MAX_POINTS, dimensions=out.dimensions)
    assert out.category.awarded_points == pytest.approx(recomputed.points(), abs=1e-6)
    assert out.category.score_10 == pytest.approx(recomputed.score10(), abs=1e-6)
    assert out.coverage == pytest.approx(recomputed.coverage(), abs=1e-6)


def test_run_nvda_fixture_category_confidence_computed(nvda_packet):
    out = bus.run(nvda_packet)
    assert out.category.confidence is not None
    assert 0.0 <= out.category.confidence <= 100.0


def test_run_nvda_fixture_serializes_to_json(nvda_packet):
    out = bus.run(nvda_packet)
    dumped = out.model_dump(mode="json")
    json.dumps(dumped)
    assert dumped["agent_id"] == "business_analysis"


def test_run_nvda_fixture_extension_fields_populated(nvda_packet):
    out = bus.run(nvda_packet)
    assert out.roic_history is not None
    assert out.roic_wacc_spread_history is not None
    assert out.margin_stability is not None
    assert out.three_thesis_killers == []  # populated later by Task 20's judgment overlay


def test_run_empty_annual_history_degrades_to_error_status_without_crashing():
    out = bus.run(_minimal_packet([]))
    assert out.status == "ERROR"
    assert out.coverage == 0.0
    assert out.category.awarded_points == 0.0
    assert len(out.metrics) == 30


def test_run_nvda_fixture_validation_tests_all_self_checks_pass(nvda_packet):
    out = bus.run(nvda_packet)
    assert out.validation_tests.failed == 0
    assert out.validation_tests.passed >= 1


def test_run_category_reproduces_from_dimensions_even_with_cap():
    rows = _five_year_rows()
    packet = _minimal_packet(rows)
    out = bus.run(packet, overlay={"wacc": 0.09, "largest_customer_share": 0.35})
    recomputed = Category(name=bus.AGENT_ID, max_points=bus.MAX_POINTS, dimensions=out.dimensions)
    assert out.category.awarded_points == pytest.approx(recomputed.points(), abs=1e-6)
    assert out.category.score_10 == pytest.approx(recomputed.score10(), abs=1e-6)
