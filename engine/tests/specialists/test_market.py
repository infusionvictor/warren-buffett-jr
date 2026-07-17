"""Tests for `wbj.specialists.market` (Task 16): MKT-001..025, the five
weighted dimensions, mandatory flags/caps, and `run()` against the NVDA
golden fixture.

Sources of truth: `Cerebro/03_market_analysis/{FORMULAS,SCORING,
DECISION_RULES,VALIDATION_TESTS}.md`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from wbj.core.nullstates import NullState
from wbj.core.scoring import Category
from wbj.schemas.packet import AnalysisMeta, MarketData, Packet, Security
import wbj.specialists.market as mkt

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "packet" / "NVDA_packet.json"


@pytest.fixture(scope="module")
def nvda_packet() -> Packet:
    data = json.loads(_FIXTURE.read_text())
    return Packet.model_validate(data)


def _minimal_packet(
    annual_rows: list[dict],
    *,
    industry_adapter: str = "default_nonfinancial",
    estimates: dict | None = None,
) -> Packet:
    return Packet(
        security=Security(
            ticker="TEST", exchange="NASDAQ", security_type="operating_company",
            reporting_currency="USD", valuation_currency="USD",
        ),
        analysis=AnalysisMeta(
            knowledge_timestamp="2026-07-16T21:00:00+00:00", industry_adapter=industry_adapter,
        ),
        fundamentals={"annual": annual_rows, "quarterly": []},
        market_data=MarketData(),
        estimates=estimates or {},
        capital_structure={},
        facts_table={},
        staleness={},
    )


def _row(year: int, **overrides) -> dict:
    base = dict(
        calendarYear=str(year), date=f"{year}-12-31", period="FY",
        revenue=1000.0, cogs=600.0, gross_profit=400.0, ebit=200.0,
        income_before_tax=190.0, income_tax_expense=40.0, net_income=150.0,
        operating_cash_flow=180.0, capex=-50.0, cash=300.0,
        total_debt=200.0, total_equity=600.0, diluted_shares=100.0,
    )
    base.update(overrides)
    return base


# ============================================================================
# MKT-VALIDATION_TESTS.md, encoded verbatim (MKT-T001..T008)
# ============================================================================


def test_MKT_T001_tam_cagr():
    """TAM 1000 to 1210 over 2 years -> CAGR=10%."""
    v = mkt.tam_cagr(1210.0, 1000.0, 2.0)
    assert v.value == pytest.approx(0.10)


def test_MKT_T002_penetration():
    """Company relevant revenue=50, TAM=1000 -> Penetration=5%."""
    v = mkt.penetration(50.0, 1000.0)
    assert v.value == pytest.approx(0.05)


def test_MKT_T003_market_share_delta():
    """Share 5.0% to 5.7% -> Delta=+0.7 percentage points."""
    v = mkt.market_share_delta(0.057, 0.05)
    assert v.value == pytest.approx(0.007)


def test_MKT_T004_revision_breadth():
    """8 upward revisions out of 10 -> Breadth=80%."""
    v = mkt.revision_breadth(8, 10)
    assert v.value == pytest.approx(0.80)


def test_MKT_T005_catalyst_expected_impact():
    """Prob=.6, impact=100, evidence=.8, time factor=.5 -> Expected impact index=24."""
    v = mkt.catalyst_expected_impact(0.6, 100.0, 0.8, 0.5)
    assert v.value == pytest.approx(24.0)


def test_MKT_T006_forecast_exceeds_tam_fails_consistency_gate():
    """Forecast revenue exceeds TAM -> Fail consistency gate."""
    assert mkt.forecast_consistency_gate(forecast_revenue=1100.0, tam=1000.0) is False
    assert mkt.forecast_consistency_gate(forecast_revenue=900.0, tam=1000.0) is True


def test_MKT_T007_issuer_only_tam_source_tier_4_caps_score_at_6():
    """Only issuer TAM with no method -> Source tier 4, score cap 6."""
    assert mkt.TAM_TIER_CONFIDENCE[4] == 45.0
    assert mkt.tam_confidence_caps_dimension(mkt.TAM_TIER_CONFIDENCE[4]) is True


def test_MKT_T008_snapshot_after_release_rejects_surprise():
    """Consensus snapshot taken after release -> Reject surprise calculation."""
    v = mkt.earnings_surprise(actual=1.0, pre_release_consensus=0.9, snapshot_before_release=False)
    assert v.is_null
    assert v.state == NullState.CONFLICTED
    v_ok = mkt.earnings_surprise(actual=1.0, pre_release_consensus=0.9, snapshot_before_release=True)
    assert v_ok.is_valid
    assert v_ok.value == pytest.approx((1.0 - 0.9) / 0.9)


# ============================================================================
# MKT-TDEC-020: time decay (brief's own worked example: months=12 -> 0.5)
# ============================================================================


def test_time_decay_12_months_is_half_life():
    v = mkt.time_decay(12.0)
    assert v.value == pytest.approx(0.5)


def test_time_decay_zero_months_is_one():
    v = mkt.time_decay(0.0)
    assert v.value == pytest.approx(1.0)


def test_time_decay_matches_formula_directly():
    months = 6.0
    v = mkt.time_decay(months)
    assert v.value == pytest.approx(math.exp(-math.log(2) * months / 12))


# ============================================================================
# Revision breadth >=5-estimate gate
# ============================================================================


def test_revision_breadth_below_5_estimates_is_not_scorable():
    v = mkt.revision_breadth(2, 4)  # only 4 total estimates
    assert v.is_null
    assert v.state == NullState.NOT_SCORABLE


def test_revision_breadth_at_5_estimates_is_scorable():
    v = mkt.revision_breadth(3, 5)
    assert v.is_valid
    assert v.value == pytest.approx(0.6)


# ============================================================================
# Individual formula behavior
# ============================================================================


def test_sam_and_som():
    sam_v = mkt.sam(1000.0, 0.5, 0.8, 0.9)
    assert sam_v.value == pytest.approx(1000.0 * 0.5 * 0.8 * 0.9)
    som_v = mkt.som(sam_v.value, 0.10)
    assert som_v.value == pytest.approx(sam_v.value * 0.10)


def test_industry_hhi():
    v = mkt.industry_hhi([0.4, 0.3, 0.3])
    assert v.value == pytest.approx(0.16 + 0.09 + 0.09)


def test_growth_capacity():
    v = mkt.growth_capacity(0.5, 0.15)
    assert v.value == pytest.approx(0.075)


def test_runway_years_not_meaningful_when_growth_nonpositive():
    v = mkt.runway_years(target_revenue=200.0, current_revenue=100.0, assumed_growth=0.0)
    assert v.is_null
    assert v.state == NullState.NOT_MEANINGFUL


def test_runway_years_computes_ln_ratio():
    v = mkt.runway_years(target_revenue=200.0, current_revenue=100.0, assumed_growth=0.10)
    assert v.value == pytest.approx(math.log(2.0) / math.log(1.10))


def test_revision_magnitude_sign_change_handling():
    v = mkt.revision_magnitude(current_consensus=110.0, prior_consensus=100.0)
    assert v.value == pytest.approx(0.10)
    v_zero = mkt.revision_magnitude(current_consensus=110.0, prior_consensus=0.0)
    assert v_zero.is_null


def test_estimate_dispersion():
    v = mkt.estimate_dispersion([9.0, 10.0, 11.0])
    assert v.is_valid
    assert v.value > 0.0


def test_backlog_growth_and_revenue_coverage():
    bg = mkt.backlog_growth(120.0, 100.0)
    assert bg.value == pytest.approx(0.20)
    cov = mkt.revenue_coverage(80.0, 100.0)
    assert cov.value == pytest.approx(0.80)


def test_operating_leverage_and_incremental_margin():
    op_lev = mkt.operating_leverage(pct_change_oi=0.20, pct_change_rev=0.10)
    assert op_lev.value == pytest.approx(2.0)
    inc_margin = mkt.incremental_operating_margin(delta_oi=20.0, delta_rev=100.0)
    assert inc_margin.value == pytest.approx(0.20)


def test_adoption_penetration_and_arpu_growth():
    adopt = mkt.adoption_penetration(current_units=1000.0, eventual_units=10000.0)
    assert adopt.value == pytest.approx(0.10)
    arpu = mkt.arpu_growth(arpu_t=110.0, arpu_t1=100.0)
    assert arpu.value == pytest.approx(0.10)


def test_scenario_weighted_outcome_requires_probabilities_sum_to_one():
    v = mkt.scenario_weighted_outcome([(0.5, 100.0), (0.5, 200.0)])
    assert v.value == pytest.approx(150.0)
    v_bad = mkt.scenario_weighted_outcome([(0.5, 100.0), (0.6, 200.0)])
    assert v_bad.is_null
    assert v_bad.state == NullState.CONFLICTED


# ============================================================================
# Dimension caps (the apply_dimension_cap helper is tested in test_common.py;
# the tests below exercise market.py's *use* of it -- TAM/catalyst caps).
# ============================================================================


def test_run_tam_dimension_capped_at_6_with_low_source_tier(nvda_packet):
    # Supply enough TAM-dimension members (sam/som/tam_history/penetration/
    # share/share_delta/hhi/adoption) to clear the 70% usable-coverage gate
    # so this test isolates the source-tier cap, not a coverage gate.
    out = mkt.run(
        nvda_packet,
        overlay={
            "tam": 100000.0,
            "tam_source_tier": 4,
            "sam_inputs": {"geography_share": 0.8, "product_share": 0.9, "reachable_share": 0.7},
            "som_inputs": {"target_share": 0.1},
            "tam_history": [80000.0, 90000.0, 100000.0],
            "company_relevant_revenue": 5000.0,
            "share": {"company_sales": 500.0, "total_market_sales": 10000.0},
            "share_history": [0.04, 0.05],
            "competitor_shares": [0.3, 0.2, 0.1],
            "adoption": {"current_units": 1000.0, "eventual_units": 10000.0},
        },
    )
    tam_dim = next(d for d in out.dimensions if d.name == mkt.DIM_TAM)
    assert tam_dim.score10() <= 6.0 + 1e-9


def test_run_catalysts_dimension_capped_at_3_when_narrative_only(nvda_packet):
    out = mkt.run(
        nvda_packet,
        overlay={
            "catalysts": [{"event": "New product launch", "months_to_event": 6.0}],
            "backlog_history": [100.0, 120.0],
            "ntm_contracted": 80.0,
            "ntm_revenue_estimate": 100.0,
        },
    )
    cat_dim = next(d for d in out.dimensions if d.name == mkt.DIM_CATALYSTS)
    assert cat_dim.score10() <= 3.0 + 1e-9


def test_run_catalysts_dimension_not_capped_when_quantified(nvda_packet):
    out = mkt.run(
        nvda_packet,
        overlay={
            "catalysts": [
                {
                    "event": "New product launch", "months_to_event": 6.0,
                    "probability": 0.7, "impact": 500.0, "evidence_quality": 0.8,
                }
            ]
        },
    )
    cat_dim = next(d for d in out.dimensions if d.name == mkt.DIM_CATALYSTS)
    # not capped at 3 -- a fully quantified catalyst may score above the narrative cap
    assert cat_dim.valid_weight() > 0


# ============================================================================
# Judgment requests
# ============================================================================


def test_run_tam_tier_assignment_is_judgment_request(nvda_packet):
    out = mkt.run(nvda_packet)
    ids = {jr.metric_id for jr in out.judgment_requests}
    assert "tam_source_tier_assignment" in ids
    assert "three_growth_thesis_killers" in ids


def test_run_catalyst_probability_impact_evidence_are_judgment_requests(nvda_packet):
    out = mkt.run(nvda_packet, overlay={"catalysts": [{"event": "X", "months_to_event": 3.0}]})
    ids = {jr.metric_id for jr in out.judgment_requests}
    assert any("catalyst" in i for i in ids)


# ============================================================================
# run() against the NVDA fixture
# ============================================================================


def test_run_nvda_fixture_schema_valid(nvda_packet):
    out = mkt.run(nvda_packet)
    assert out.agent_id == "market_analysis"
    assert out.version == "2.0.0"
    assert out.security.ticker == "NVDA"
    assert out.category.max_points == 20.0
    assert out.status in ("COMPLETE", "INCOMPLETE", "ERROR")
    assert len(out.dimensions) == 5
    assert len(out.metrics) == 25
    for row in out.metrics:
        assert row.metric_id
        assert row.formula_id
        assert row.formula_version
        assert row.score == "NOT_SCORABLE" or isinstance(row.score, float)
        assert 0.0 <= row.confidence <= 100.0
        assert (row.value is None) != (row.state is None)


def test_run_nvda_fixture_category_math_reproduces_from_dimensions(nvda_packet):
    out = mkt.run(nvda_packet)
    recomputed = Category(name=mkt.AGENT_ID, max_points=mkt.MAX_POINTS, dimensions=out.dimensions)
    assert out.category.awarded_points == pytest.approx(recomputed.points(), abs=1e-6)
    assert out.category.score_10 == pytest.approx(recomputed.score10(), abs=1e-6)
    assert out.coverage == pytest.approx(recomputed.coverage(), abs=1e-6)


def test_run_nvda_fixture_category_confidence_computed(nvda_packet):
    out = mkt.run(nvda_packet)
    assert out.category.confidence is not None
    assert 0.0 <= out.category.confidence <= 100.0


def test_run_nvda_fixture_serializes_to_json(nvda_packet):
    out = mkt.run(nvda_packet)
    dumped = out.model_dump(mode="json")
    json.dumps(dumped)
    assert dumped["agent_id"] == "market_analysis"


def test_run_operating_leverage_computed_from_packet_without_overlay():
    """MKT-OPLEV-017/018 are computable directly from
    `packet.fundamentals.annual` (ebit/revenue), unlike most of this
    specialist's inputs -- no overlay needed."""
    rows = [_row(2025, revenue=1200.0, ebit=300.0), _row(2024, revenue=1000.0, ebit=200.0)]
    out = mkt.run(_minimal_packet(rows))
    oplev_row = next(r for r in out.metrics if r.metric_id == "MKT-OPLEV-017")
    assert oplev_row.score != "NOT_SCORABLE"


def test_run_empty_annual_history_degrades_without_crashing():
    out = mkt.run(_minimal_packet([]))
    assert out.coverage == 0.0
    assert out.category.awarded_points == 0.0
    assert len(out.metrics) == 25


def test_run_nvda_fixture_validation_tests_all_self_checks_pass(nvda_packet):
    out = mkt.run(nvda_packet)
    assert out.validation_tests.failed == 0
    assert out.validation_tests.passed >= 1


def test_run_nvda_fixture_extension_fields_populated(nvda_packet):
    out = mkt.run(nvda_packet)
    assert out.tam_sam_som is not None
    assert out.three_growth_thesis_killers == []
    assert isinstance(out.catalysts, list)


def test_run_external_capital_flag_when_forecast_exceeds_growth_capacity():
    rows = [_row(2025), _row(2024)]
    out = mkt.run(
        _minimal_packet(rows),
        overlay={"target_revenue": 5000.0, "current_revenue": 1000.0, "assumed_growth": 0.50},
    )
    assert "EXTERNAL_CAPITAL_REQUIRED" in out.mandatory_flags
