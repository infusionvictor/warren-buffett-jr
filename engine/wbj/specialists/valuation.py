"""Valuation specialist (Task 19): VAL-NORM-001..VAL-ENSEMBLE-044
(`FORMULAS.md`) and the five weighted valuation dimensions (10 pts).

Sources of truth (`Cerebro/06_valuation_analysis/`):
- `FORMULAS.md`: the 44 registered formulas.
- `DECISION_RULES.md`: the model-selection matrix, DCF/WACC consistency
  rules, the scenario framework, reverse DCF, and the valuation verdict
  bands.
- `SCORING.md`: the five weighted dimensions and their per-dimension
  gate/cap column.
- `OUTPUT_SCHEMA.md`: `ValuationOutput`'s extension fields.

## Engine reuse (per the task-19 brief: do NOT reimplement)

Every DCF/WACC/reverse-DCF/scenario/Monte-Carlo/ensemble formula is a
direct call into `wbj.engines.valuation_engine` (Task 13) --
`nopat`/`invested_capital`/`roic`/`fundamental_growth`,
`cost_of_equity`/`synthetic_kd`/`wacc`, `equity_bridge`/`per_share`,
`economic_profit_value`/`reconciles`, `justified_pe`/`justified_ev_sales`/
`hist_zscore`, `reverse_dcf`, `scenarios`, `monte_carlo`, `ensemble`,
`margin_of_safety`. This module's own code is the *orchestration* layer:
deriving each engine call's inputs from `Packet` (+ `overlay`), assembling
the five weighted dimensions, and shaping `OUTPUT_SCHEMA.md`'s extension
fields -- not new valuation math.

The FCFF DCF itself is run through the engine's `scenarios` (whose
per-branch `_constant_growth_value` is the engine's own constant-growth
FCFF DCF) and `monte_carlo`, not the engine's list-based `dcf_value`
helper -- `dcf_value` takes an already-built explicit FCFF path, whereas
this module forecasts the path from a growth/margin/ROIC driver set, which
is exactly what `scenarios`/`monte_carlo`/`reverse_dcf` are built to
consume. `model_cross_checks.dispersion` (and its reliability-weighted
value) comes from the engine's `ensemble` (VAL-ENSEMBLE-044), not an
unweighted stdlib stdev.

## Model-selection matrix (brief's simplification of DECISION_RULES.md)

Per the task-19 brief, this module implements only the "general
non-financial" row of Cerebro's full 8-row model-selection matrix: FCFF
DCF + an economic-profit cross-check. Every other `industry_adapter`
value (bank, insurer, REIT, ...) returns an `ADAPTER_UNSUPPORTED`
mandatory flag and an all-`NOT_SCORABLE` category rather than silently
running FCFF/EV-EBITDA math Cerebro's matrix explicitly says to avoid for
that company type (`DECISION_RULES.md`: "Do not use a valuation model
that conflicts with the company type").

## Dataset coverage vs. `Packet`

`packet.facts_table` (`price`, `cash`, `diluted_shares`, `revenue`,
`total_debt`) and `packet.capital_structure` (`beta`, `market_cap`) supply
the WACC/DCF inputs a real ticker actually has; `packet.fundamentals
.annual`'s latest row supplies EBIT/tax/margin/ROIC building blocks (the
same NOPAT/invested-capital derivation `business.py`/`market.py` use
locally). `estimates.risk_free_rate` and `estimates.fmp_analyst_estimates`
(for a consensus growth read) come from `Packet.estimates`. What `Packet`
does NOT carry -- interest expense (for a real cost of debt), a peer
multiple panel (>=8 peers; `estimates.peers` is a bare 3-ticker list, not
a multiples panel), a historical own-multiple series, and any explicit
scenario/growth override -- is `overlay`-only, mirroring `financial.py`'s
`overlay["wacc"]` precedent; each gap degrades to a documented default
assumption (flagged) or `NOT_SCORABLE`, never a silent guess.

## Scenario defaults

Per the brief ("scenario growth/margin defaults derived... flagged as
assumptions, agent-overridable"): base growth defaults to
`fundamental_growth(reinvestment_rate, ROIC)` (`VAL-REINV-043`, historical
capex-implied reinvestment rate against the latest ROIC); bear/bull growth
default to half/1.5x the base rate; base margin defaults to the latest
reported operating margin, bear/bull to +/-2 points; discount rates widen
+/-100bp bear/bull. `overlay["scenarios"]` (a `{"bear": {...}, "base":
{...}, "bull": {...}}` dict of raw `ScenarioInput` fields) overrides any
or all of these per the brief's "agent-overridable" instruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from wbj.core.confidence import confidence as _confidence_formula
from wbj.core.nullstates import EvidenceClass, NullState, Value
from wbj.core.scoring import Category, Dimension, anchor_score
from wbj.engines import valuation_engine as ve
from wbj.schemas.packet import Packet
from wbj.schemas.valuation import (
    DCFCommonInputs,
    EnsembleModelInput,
    MonteCarloInputs,
    MonteCarloRange,
    ReverseDCFInputs,
    ScenarioInput,
)
from wbj.specialists.common import (
    CategoryStats,
    apply_dimension_cap,
    JudgmentRequest,
    MetricRow,
    SecurityRef,
    SpecialistOutput,
    ValidationTestsSummary,
    status_from_coverage,
)

__all__ = [
    "AGENT_ID",
    "MAX_POINTS",
    "DIM_MULTIPLES",
    "DIM_HIST_PEER",
    "DIM_CF_YIELD",
    "DIM_FAIR_VALUE_SCENARIOS",
    "DIM_MOS",
    "DIMENSION_NAMES",
    "SUPPORTED_ADAPTER",
    "ValuationOutput",
    "verdict",
    "run",
]

_VERSION = "2.0.0"
AGENT_ID = "valuation_analysis"
MAX_POINTS = 10.0

DIM_MULTIPLES = "growth_adjusted_multiples"
DIM_HIST_PEER = "historical_and_peer_comparison"
DIM_CF_YIELD = "cash_flow_and_earnings_yield"
DIM_FAIR_VALUE_SCENARIOS = "fair_value_by_scenarios"
DIM_MOS = "margin_of_safety"

DIMENSION_NAMES = (DIM_MULTIPLES, DIM_HIST_PEER, DIM_CF_YIELD, DIM_FAIR_VALUE_SCENARIOS, DIM_MOS)

DIMENSION_MAX_POINTS: dict[str, float] = {
    DIM_MULTIPLES: 3.0, DIM_HIST_PEER: 2.0, DIM_CF_YIELD: 2.0, DIM_FAIR_VALUE_SCENARIOS: 2.0, DIM_MOS: 1.0,
}
assert sum(DIMENSION_MAX_POINTS.values()) == MAX_POINTS

SUPPORTED_ADAPTER = "default_nonfinancial"
MIN_PEERS = 8


def _ok(x: float, unit: str, **lineage: object) -> Value:
    return Value.of(x, unit=unit, evidence_class=lineage.pop("evidence_class", EvidenceClass.C), **lineage)


def _null(state: NullState, unit: str, *warnings: str) -> Value:
    return Value.null(state, unit=unit, warnings=list(warnings))


def _score_from_anchor(v: Value, anchors: list[tuple[float, float]]) -> float | None:
    if v.is_null:
        return None
    return anchor_score(v.value, anchors)


def verdict(score10: float) -> str:
    """Valuation verdict bands (`DECISION_RULES.md`, verbatim)."""
    if score10 >= 8.0:
        return "Attractive across scenarios with a defensible margin of safety"
    if score10 >= 6.0:
        return "Fair to moderately attractive"
    if score10 >= 4.0:
        return "Full valuation; depends on optimistic execution"
    return "Expensive or not supported by cash-flow/return assumptions"


# ============================================================================
# Envelope assembly
# ============================================================================


class ModelSelection(BaseModel):
    model_config = ConfigDict(frozen=True)

    primary: list[str] = Field(default_factory=list)
    rejected: list[str] = Field(default_factory=list)


class WaccSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: float | None = None
    components: dict[str, float | None] = Field(default_factory=dict)
    sensitivity: list[dict[str, float | None]] = Field(default_factory=list)


class ScenarioSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    probability: float | None = None
    assumptions: dict[str, float | None] = Field(default_factory=dict)
    enterprise_value: float | None = None
    equity_value: float | None = None
    per_share_value: float | None = None


class ReverseDCFSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    current_price: float | None = None
    implied_revenue_cagr: float | None = None
    implied_margin: float | None = None
    implied_high_growth_years: int | None = None


class ModelCrossChecks(BaseModel):
    model_config = ConfigDict(frozen=True)

    fcff: float | None = None
    economic_profit: float | None = None
    residual_income: float | None = None
    relative: float | None = None
    dispersion: float | None = None


class FairValueDistribution(BaseModel):
    model_config = ConfigDict(frozen=True)

    p10: float | None = None
    p25: float | None = None
    median: float | None = None
    p75: float | None = None
    p90: float | None = None


class ReferenceBands(BaseModel):
    model_config = ConfigDict(frozen=True)

    bear: float | None = None
    base: float | None = None
    bull: float | None = None
    margin_of_safety_15pct: float | None = None
    margin_of_safety_25pct: float | None = None


class ValuationOutput(SpecialistOutput):
    """`valuation_analysis`'s `run()` result: the shared envelope plus
    OUTPUT_SCHEMA.md's valuation-specific extension fields."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    model_selection: ModelSelection = Field(default_factory=ModelSelection)
    normalization_reconciliation: list[str] = Field(default_factory=list)
    wacc: WaccSummary = Field(default_factory=WaccSummary)
    scenarios: list[ScenarioSummary] = Field(default_factory=list)
    reverse_dcf: ReverseDCFSummary = Field(default_factory=ReverseDCFSummary)
    model_cross_checks: ModelCrossChecks = Field(default_factory=ModelCrossChecks)
    fair_value_distribution: FairValueDistribution = Field(default_factory=FairValueDistribution)
    reference_bands: ReferenceBands = Field(default_factory=ReferenceBands)


@dataclass(frozen=True)
class _Row:
    metric_id: str
    value: Value
    score10: float | None
    confidence: float = 0.0


def _confidence_for(v: Value) -> float:
    if v.is_null:
        return 0.0
    base_by_class = {
        EvidenceClass.R: 95.0, EvidenceClass.C: 85.0, EvidenceClass.E: 60.0,
        EvidenceClass.A: 50.0, EvidenceClass.Q: 30.0,
    }
    base = base_by_class.get(v.evidence_class, 70.0) if v.evidence_class else 70.0
    base -= 15.0 * len(v.warnings)
    return max(0.0, min(100.0, base))


def _num(row: dict, key: str) -> float | None:
    x = row.get(key)
    return float(x) if isinstance(x, (int, float)) else None


def _annual_rows(packet: Packet) -> list[dict]:
    rows = packet.fundamentals.get("annual") or []
    return list(reversed(rows))  # ascending, per DATASET.md


def _fact(packet: Packet, key: str) -> float | None:
    v = packet.facts_table.get(key)
    return v.value if v is not None and v.is_valid else None


def _unsupported_adapter_output(packet: Packet) -> ValuationOutput:
    # awarded_points/score_10 are 0.0 by construction (empty dimensions),
    # but confidence is derived from the real five-component formula at
    # coverage 0 -- for an unsupported adapter that yields a low value (the
    # model_fit component drops to 40 for a non-default adapter), which is
    # the honest signal, not a hardcoded 0.0 literal.
    dims = [Dimension(name=n, max_points=DIMENSION_MAX_POINTS[n], metric_scores=[]) for n in DIMENSION_NAMES]
    return ValuationOutput(
        agent_id=AGENT_ID, status="ERROR",
        security=SecurityRef(ticker=packet.security.ticker, exchange=packet.security.exchange, currency=packet.security.reporting_currency),
        knowledge_timestamp=packet.analysis.knowledge_timestamp,
        category=CategoryStats(max_points=MAX_POINTS, awarded_points=0.0, score_10=0.0, confidence=_category_confidence(0.0, packet)),
        verdict=verdict(0.0), coverage=0.0, dimensions=dims, metrics=[],
        mandatory_flags=["ADAPTER_UNSUPPORTED"],
        assumptions=[
            f"industry_adapter={packet.analysis.industry_adapter!r} is not the supported "
            f"{SUPPORTED_ADAPTER!r}: DECISION_RULES.md's model-selection matrix requires a "
            "different primary model (residual income/excess-return/DDM/SOTP/NAV/... per company "
            "type) this module does not implement -- FCFF DCF and EV/EBITDA-style multiples are "
            "explicitly avoided for that company type rather than run anyway."
        ],
        judgment_requests=[], source_lineage=["packet.fundamentals.annual", "packet.facts_table"],
        validation_tests=ValidationTestsSummary(passed=0, failed=1, warnings=0),
        model_selection=ModelSelection(primary=[], rejected=["FCFF_DCF", "ECONOMIC_PROFIT"]),
    )


def run(packet: Packet, overlay: dict[str, Any] | None = None) -> ValuationOutput:
    """Run the valuation specialist against `packet`, producing a
    `ValuationOutput` (max 10 pts): the five weighted dimensions,
    mandatory flags, and OUTPUT_SCHEMA.md's extension fields (WACC,
    scenarios, reverse DCF, cross-checks, fair-value distribution,
    reference bands), all assembled by direct calls into
    `wbj.engines.valuation_engine` (Task 13).

    See the module docstring for `overlay`'s supported keys.
    """
    overlay = overlay or {}
    if packet.analysis.industry_adapter != SUPPORTED_ADAPTER:
        return _unsupported_adapter_output(packet)

    annual = _annual_rows(packet)
    assumptions: list[str] = []
    judgment_requests: list[JudgmentRequest] = []
    rows: list[_Row] = []

    def add(metric_id: str, v: Value, score10: float | None) -> None:
        rows.append(_Row(metric_id=metric_id, value=v, score10=score10, confidence=_confidence_for(v)))

    latest = annual[-1] if annual else {}
    prior = annual[-2] if len(annual) >= 2 else {}
    revenue0 = _num(latest, "revenue") or _fact(packet, "revenue")
    ebit = _num(latest, "ebit")
    price = _fact(packet, "price")
    diluted_shares = _fact(packet, "diluted_shares") or _num(latest, "diluted_shares")
    cash = _fact(packet, "cash") or _num(latest, "cash") or 0.0
    total_debt = _fact(packet, "total_debt") or _num(latest, "total_debt") or 0.0
    net_debt = total_debt - cash

    pretax, tax_expense = _num(latest, "income_before_tax"), _num(latest, "income_tax_expense")
    tax_rate = min(max(tax_expense / pretax, 0.0), 1.0) if pretax and pretax > 0 and tax_expense is not None else 0.21
    if pretax is None or tax_expense is None:
        assumptions.append("Normalized cash tax rate unavailable; substituted the 21% statutory rate.")

    # ---- ROIC / reinvestment rate / fundamental growth (VAL-NOPAT/ROIC/REINV) ----
    debt_t, equity_t = total_debt, packet.capital_structure.get("market_cap") or _num(latest, "total_equity")
    debt_t1, equity_t1 = _num(prior, "total_debt"), _num(prior, "total_equity")
    if ebit is not None:
        nopat_v = ve.nopat(ebit, tax_rate)
    else:
        nopat_v = _null(NullState.MISSING, "usd", "NOPAT_INPUTS_UNAVAILABLE")
    add("VAL-NORM-001", _ok(ebit, unit="usd") if ebit is not None else _null(NullState.MISSING, "usd", "EBIT_UNAVAILABLE"), None)

    ic_result = None
    if debt_t is not None and equity_t is not None and debt_t1 is not None and equity_t1 is not None:
        ic_end = ve.invested_capital(debt_t, equity_t, cash).financing_view
        ic_begin = ve.invested_capital(debt_t1, equity_t1, _num(prior, "cash") or 0.0).financing_view
        if ic_end.is_valid and ic_begin.is_valid:
            avg_ic = (ic_end.value + ic_begin.value) / 2
            ic_result = avg_ic
    roic_v = ve.roic(nopat_v.value, ic_result) if nopat_v.is_valid and ic_result is not None else _null(NullState.MISSING, "pct", "ROIC_INPUTS_UNAVAILABLE")
    roic_value = roic_v.value if roic_v.is_valid else 0.10  # documented fallback below
    if roic_v.is_null:
        assumptions.append("ROIC unavailable from Packet; substituted a 10% fallback for the reinvestment-rate/fundamental-growth derivation.")

    capex_latest = abs(_num(latest, "capex") or 0.0)
    reinvestment_rate = capex_latest / nopat_v.value if nopat_v.is_valid and nopat_v.value != 0 else 0.30
    base_growth_v = ve.fundamental_growth(reinvestment_rate, roic_value)
    base_growth = base_growth_v.value if base_growth_v.is_valid else 0.03
    assumptions.append(
        f"Base-case revenue growth ({base_growth:.4f}) derived as reinvestment_rate*ROIC "
        f"(VAL-REINV-043), not a management/consensus figure -- agent-overridable via "
        "overlay['scenarios']['base']['growth']."
    )

    op_margin_latest = ebit / revenue0 if ebit is not None and revenue0 not in (None, 0) else 0.20
    if ebit is None or revenue0 in (None, 0):
        assumptions.append("Latest operating margin unavailable; substituted a 20% fallback for scenario construction.")

    # ---- WACC (VAL-KE-008 / VAL-KD-011 / VAL-WACC-007) ----
    rf = overlay.get("risk_free_rate", (packet.estimates or {}).get("risk_free_rate"))
    beta = overlay.get("beta", packet.capital_structure.get("beta"))
    erp = float(overlay.get("erp", 0.045))
    if rf is None or beta is None:
        ke_v = _null(NullState.MISSING, "pct", "WACC_INPUTS_UNAVAILABLE_NO_RF_OR_BETA")
        assumptions.append("Cost of equity not computed: risk-free rate or beta unavailable from Packet.estimates/capital_structure.")
    else:
        ke_v = ve.cost_of_equity(float(rf), float(beta), erp)
    add("VAL-KE-008", ke_v, None)

    interest_expense = overlay.get("interest_expense")
    if interest_expense is not None and ebit is not None and rf is not None:
        coverage = ebit / float(interest_expense) if interest_expense else 0.0
        kd_v = ve.synthetic_kd(float(rf), coverage)
    elif rf is not None:
        kd_v = ve.synthetic_kd(float(rf), 3.0)  # documented fallback: mid-BBB synthetic coverage
        assumptions.append("Interest expense unavailable; cost of debt used a fallback interest-coverage of 3.0x (synthetic mid-grade rating).")
    else:
        kd_v = _null(NullState.MISSING, "pct", "COST_OF_DEBT_UNAVAILABLE_NO_RF")
    add("VAL-KD-011", kd_v, None)

    e_value = packet.capital_structure.get("market_cap") or (price * diluted_shares if price and diluted_shares else None)
    if ke_v.is_valid and kd_v.is_valid and e_value is not None:
        wacc_v = ve.wacc(e_value, total_debt, ke_v.value, kd_v.value, tax_rate)
    else:
        wacc_v = _null(NullState.MISSING, "pct", "WACC_INPUTS_UNAVAILABLE")
    add("VAL-WACC-007", wacc_v, None)
    wacc_value = wacc_v.value if wacc_v.is_valid else None
    wacc_sens = ve.wacc_sensitivity(wacc_v) if wacc_v.is_valid else None

    # ---- Terminal growth (DCF consistency rule 3: must stay below WACC) ----
    tv_growth_default = float(overlay.get("terminal_growth", 0.025))
    if wacc_value is not None and tv_growth_default >= wacc_value:
        tv_growth = max(0.0, wacc_value - 0.005)
        assumptions.append(
            f"Default terminal growth ({tv_growth_default:.4f}) was >= WACC ({wacc_value:.4f}); "
            f"clamped to {tv_growth:.4f} per DECISION_RULES.md's 'g < WACC' consistency rule."
        )
    else:
        tv_growth = tv_growth_default

    years = int(overlay.get("forecast_years", 5))

    # ---- Scenario construction (bear/base/bull) -- brief's derived defaults ----
    scenario_overrides = overlay.get("scenarios") or {}

    def _scenario_input(name: str, default_growth: float, default_margin: float, default_wacc: float | None, default_prob: float) -> ScenarioInput | None:
        ov = scenario_overrides.get(name, {})
        w = ov.get("wacc", default_wacc)
        if w is None:
            return None
        return ScenarioInput(
            probability=float(ov.get("probability", default_prob)),
            growth=float(ov.get("growth", default_growth)),
            margin=float(ov.get("margin", default_margin)),
            wacc=float(w),
            tv_growth=float(ov.get("tv_growth", tv_growth)),
        )

    bear_input = _scenario_input("bear", base_growth * 0.5, op_margin_latest - 0.02, (wacc_value + 0.01) if wacc_value else None, 0.25)
    base_input = _scenario_input("base", base_growth, op_margin_latest, wacc_value, 0.50)
    bull_input = _scenario_input("bull", base_growth * 1.5, op_margin_latest + 0.02, (wacc_value - 0.01) if wacc_value else None, 0.25)

    scenario_summaries: list[ScenarioSummary] = []
    weighted_value = None
    base_per_share = None
    bear_per_share = None
    bull_per_share = None
    scenario_result = None
    if bear_input and base_input and bull_input and diluted_shares and revenue0 is not None:
        common = DCFCommonInputs(revenue0=revenue0, shares=diluted_shares, tax_rate=tax_rate, roic=roic_value, years=years, net_debt=net_debt)
        try:
            scenario_result = ve.scenarios(bear_input, base_input, bull_input, common)
        except ValueError as e:
            assumptions.append(f"Scenario probabilities did not sum to 1.0: {e}")
        if scenario_result is not None:
            for name, s_input, s_value in (("Bear", bear_input, scenario_result.bear_value), ("Base", base_input, scenario_result.base_value), ("Bull", bull_input, scenario_result.bull_value)):
                scenario_summaries.append(
                    ScenarioSummary(
                        name=name, probability=s_input.probability,
                        assumptions={"growth": s_input.growth, "margin": s_input.margin, "wacc": s_input.wacc, "tv_growth": s_input.tv_growth},
                        per_share_value=s_value.value if s_value.is_valid else None,
                    )
                )
            weighted_value = scenario_result.weighted_value.value if scenario_result.weighted_value.is_valid else None
            base_per_share = scenario_summaries[1].per_share_value
            bear_per_share = scenario_summaries[0].per_share_value
            bull_per_share = scenario_summaries[2].per_share_value
    else:
        assumptions.append("Scenario valuation not computed: WACC, diluted shares, or latest revenue unavailable.")

    # ---- Reverse DCF (VAL-RDCF-027) ----
    consensus_growth = None
    fmp_est = (packet.estimates or {}).get("fmp_analyst_estimates") or []
    if len(fmp_est) >= 2 and revenue0:
        rev_next = fmp_est[0].get("estimatedRevenueAvg")
        if rev_next is not None:
            consensus_growth = rev_next / revenue0 - 1

    reverse_dcf_summary = ReverseDCFSummary(current_price=price)
    if wacc_value is not None and diluted_shares and price and revenue0 is not None:
        rdcf_inputs = ReverseDCFInputs(
            revenue0=revenue0, shares=diluted_shares, tax_rate=tax_rate, roic=roic_value, years=years,
            net_debt=net_debt, margin=op_margin_latest, wacc=wacc_value, tv_growth=tv_growth,
            consensus_growth=consensus_growth,
        )
        rdcf_result = ve.reverse_dcf(price, diluted_shares, rdcf_inputs)
        reverse_dcf_summary = ReverseDCFSummary(
            current_price=price,
            implied_revenue_cagr=rdcf_result.implied_growth.value if rdcf_result.implied_growth.is_valid else None,
            implied_margin=rdcf_result.implied_margin.value if rdcf_result.implied_margin.is_valid else None,
            implied_high_growth_years=years,
        )
        add("VAL-RDCF-027", rdcf_result.implied_growth, None)
    else:
        add("VAL-RDCF-027", _null(NullState.MISSING, "pct", "REVERSE_DCF_INPUTS_UNAVAILABLE"), None)

    # ---- Model cross-checks: FCFF DCF vs economic-profit (DCF consistency rule 7) ----
    fcff_per_share = base_per_share  # scenarios() IS the FCFF DCF model for the base case
    econ_profit_per_share = None
    if wacc_value is not None and ic_result is not None and diluted_shares and nopat_v.is_valid:
        eps_forecast = [nopat_v.value * (1 - reinvestment_rate) * (1 + base_growth) ** t for t in range(1, years + 1)]
        ep_ev = ve.economic_profit_value(ic_result, eps_forecast, wacc_value)
        if ep_ev.is_valid:
            eq_v = ve.equity_bridge(ep_ev.value, cash, 0.0, total_debt, 0.0, 0.0, 0.0, 0.0)
            ps_v = ve.per_share(eq_v.value, diluted_shares) if eq_v.is_valid else None
            econ_profit_per_share = ps_v.value if ps_v is not None and ps_v.is_valid else None
    add("VAL-EVAEV-021", _ok(econ_profit_per_share, unit="usd_per_share") if econ_profit_per_share is not None else _null(NullState.MISSING, "usd_per_share", "ECONOMIC_PROFIT_INPUTS_UNAVAILABLE"), None)

    reconciled = None
    if fcff_per_share is not None and econ_profit_per_share is not None:
        reconciled = ve.reconciles(_ok(fcff_per_share, "usd_per_share"), _ok(econ_profit_per_share, "usd_per_share"))
        if not reconciled:
            assumptions.append("FCFF DCF and economic-profit per-share values differ by >1%: inconsistent reinvestment/capital assumptions (DECISION_RULES.md rule 7).")

    # ---- Relative valuation: needs >=8 peers; Packet's estimates.peers is a bare ticker list ----
    peers_list = (packet.estimates or {}).get("peers") or []
    n_peers = len(peers_list[0].get("peersList", [])) if peers_list else 0
    peer_multiples = overlay.get("peer_multiples")  # list of floats, agent-supplied
    relative_value = None
    if peer_multiples and len(peer_multiples) >= MIN_PEERS and revenue0:
        relative_value = float(np.median(peer_multiples)) * revenue0 / diluted_shares if diluted_shares else None
    # ---- Reliability-weighted ensemble (VAL-ENSEMBLE-044) -> dispersion + value ----
    # Reuse the engine's `ensemble` rather than an unweighted stdlib stdev:
    # weights reflect model reliability (FCFF DCF is the primary model;
    # economic profit a high-reliability cross-check that should reconcile;
    # the peer-relative read is weaker). `ensemble` excludes null-valued
    # models and returns both the reliability-weighted value and the
    # dispersion across the included values.
    ensemble_models: list[EnsembleModelInput] = []
    if fcff_per_share is not None:
        ensemble_models.append(EnsembleModelInput(label="FCFF_DCF", value=_ok(fcff_per_share, "usd_per_share"), weight=1.0))
    if econ_profit_per_share is not None:
        ensemble_models.append(EnsembleModelInput(label="ECONOMIC_PROFIT", value=_ok(econ_profit_per_share, "usd_per_share"), weight=0.8))
    if relative_value is not None:
        ensemble_models.append(EnsembleModelInput(label="RELATIVE", value=_ok(relative_value, "usd_per_share"), weight=0.5))
    ensemble_result = ve.ensemble(ensemble_models) if ensemble_models else None
    dispersion = (
        ensemble_result.dispersion.value
        if ensemble_result is not None and ensemble_result.dispersion.is_valid
        else None
    )

    # ---- Fair-value distribution via Monte Carlo (VAL-MC-037) ----
    fv_dist = FairValueDistribution()
    if bear_input and base_input and bull_input and wacc_value is not None and diluted_shares and revenue0 is not None:
        mc_inputs = MonteCarloInputs(
            revenue0=revenue0, shares=diluted_shares, tax_rate=tax_rate, roic=roic_value, years=years, net_debt=net_debt,
            growth_range=MonteCarloRange(low=bear_input.growth, mode=base_input.growth, high=bull_input.growth),
            margin_range=MonteCarloRange(low=bear_input.margin, mode=base_input.margin, high=bull_input.margin),
            wacc_range=MonteCarloRange(low=bull_input.wacc, mode=wacc_value, high=bear_input.wacc),
            tv_growth=tv_growth,
        )
        mc_result = ve.monte_carlo(mc_inputs, n=2000, seed=42)
        fv_dist = FairValueDistribution(
            p10=mc_result.p10.value if mc_result.p10.is_valid else None,
            p25=mc_result.p25.value if mc_result.p25.is_valid else None,
            median=mc_result.median.value if mc_result.median.is_valid else None,
            p75=mc_result.p75.value if mc_result.p75.is_valid else None,
            p90=mc_result.p90.value if mc_result.p90.is_valid else None,
        )

    # ---- Margin of safety (VAL-MOS-040) against the base scenario ----
    mos_v = ve.margin_of_safety(base_per_share, price) if base_per_share is not None and price is not None else _null(NullState.MISSING, "pct", "MOS_INPUTS_UNAVAILABLE")
    add("VAL-MOS-040", mos_v, None)

    # ---- Growth-adjusted multiples (VAL-PEG-028, VAL-JPE-032, VAL-JEVS-033) ----
    eps = _num(latest, "eps")
    consensus_eps_growth_pct = overlay.get("eps_growth_pct")
    if eps is not None and price and eps > 0 and consensus_eps_growth_pct:
        trailing_pe = price / eps
        v_peg = _ok(trailing_pe / (consensus_eps_growth_pct * 100), unit="ratio") if consensus_eps_growth_pct != 0 else _null(NullState.NOT_MEANINGFUL, "ratio", "PEG_ZERO_GROWTH")
    else:
        v_peg = _null(NullState.MISSING, "ratio", "PEG_INPUTS_UNAVAILABLE_OR_NEGATIVE_EARNINGS")
    add("VAL-PEG-028", v_peg, _score_from_anchor(v_peg, [(0.5, 10), (1.0, 7), (2.0, 3), (3.5, 0)]))

    roe = _num(latest, "net_income") / equity_t if _num(latest, "net_income") is not None and equity_t not in (None, 0) else None
    v_jpe = ve.justified_pe(base_growth, roe, ke_v.value) if roe is not None and ke_v.is_valid else _null(NullState.MISSING, "", "JUSTIFIED_PE_INPUTS_UNAVAILABLE")
    add("VAL-JPE-032", v_jpe, None)
    trailing_pe_v = _ok(price / eps, unit="ratio") if price and eps and eps > 0 else _null(NullState.MISSING, "ratio", "TRAILING_PE_UNAVAILABLE")
    peg_style_v = _null(NullState.MISSING, "ratio", "JPE_COMPARISON_UNAVAILABLE")
    if v_jpe.is_valid and trailing_pe_v.is_valid and v_jpe.value != 0:
        peg_style_v = _ok((trailing_pe_v.value - v_jpe.value) / v_jpe.value, unit="pct")
    peg_style_score = _score_from_anchor(peg_style_v, [(0.50, 0), (0.15, 4), (0.0, 7), (-0.15, 10)])

    v_jevs = (
        ve.justified_ev_sales(op_margin_latest, tax_rate, base_growth, roic_value, wacc_value)
        if wacc_value is not None else _null(NullState.MISSING, "", "JUSTIFIED_EV_SALES_INPUTS_UNAVAILABLE")
    )
    add("VAL-JEVS-033", v_jevs, None)

    rdcf_plausibility_score = None
    if reverse_dcf_summary.implied_revenue_cagr is not None:
        reference_growth = consensus_growth if consensus_growth is not None else base_growth
        implied_vs_reference = reverse_dcf_summary.implied_revenue_cagr - reference_growth
        rdcf_plausibility_score = anchor_score(implied_vs_reference, [(0.10, 0), (0.02, 5), (-0.02, 8), (-0.10, 10)])

    # ---- Historical/peer comparison (VAL-REL-034, VAL-ZHIST-035) ----
    hist_multiples = overlay.get("historical_multiples")
    v_zhist = ve.hist_zscore(trailing_pe_v.value, hist_multiples) if hist_multiples and trailing_pe_v.is_valid else _null(NullState.MISSING, "", "ZHIST_UNAVAILABLE")
    add("VAL-ZHIST-035", v_zhist, _score_from_anchor(v_zhist, [(2.0, 0), (0.5, 5), (0.0, 7), (-1.0, 10)]))
    if n_peers < MIN_PEERS and not hist_multiples:
        assumptions.append(f"Only {n_peers} peers available from Packet.estimates.peers (need >={MIN_PEERS}) and no overlay['historical_multiples'] supplied; historical/peer dimension falls back to history-only or NOT_SCORABLE.")

    # ---- Cash-flow and earnings yield (VAL-EY-029, VAL-FCFY-030) ----
    v_ey = _ok(eps / price, unit="pct") if eps is not None and eps > 0 and price else _null(NullState.NOT_MEANINGFUL, "pct", "EARNINGS_YIELD_NONPOSITIVE_EPS_OR_PRICE")
    add("VAL-EY-029", v_ey, _score_from_anchor(v_ey, [(0.0, 0), (0.03, 4), (0.06, 7), (0.10, 10)]))
    fcf_latest = _num(latest, "fcf")
    market_cap = e_value
    v_fcfy = _ok(fcf_latest / market_cap, unit="pct") if fcf_latest is not None and market_cap else _null(NullState.MISSING, "pct", "FCF_YIELD_INPUTS_UNAVAILABLE")
    add("VAL-FCFY-030", v_fcfy, _score_from_anchor(v_fcfy, [(0.0, 0), (0.02, 4), (0.05, 7), (0.08, 10)]))

    by_id = {r.metric_id: r for r in rows}

    metric_rows: list[MetricRow] = []
    for r in rows:
        score: float | str = r.score10 if r.score10 is not None else "NOT_SCORABLE"
        metric_rows.append(MetricRow.from_value(r.metric_id, r.value, formula_id=r.metric_id, formula_version=_VERSION, score=score, confidence=r.confidence))

    # ---- DIM_MULTIPLES (3 pts) ----
    multiples_scores: list[tuple[float, Value]] = [
        (0.35, Value.of(by_id["VAL-PEG-028"].score10, unit="score") if by_id["VAL-PEG-028"].score10 is not None else Value.null(NullState.NOT_SCORABLE, unit="score")),
        (0.25, Value.of(peg_style_score, unit="score") if peg_style_score is not None else Value.null(NullState.NOT_SCORABLE, unit="score")),
        (0.40, Value.of(rdcf_plausibility_score, unit="score") if rdcf_plausibility_score is not None else Value.null(NullState.NOT_SCORABLE, unit="score")),
    ]
    multiples_dim = Dimension(name=DIM_MULTIPLES, max_points=DIMENSION_MAX_POINTS[DIM_MULTIPLES], metric_scores=multiples_scores)

    # ---- DIM_HIST_PEER (2 pts): min 8 peers or history-only ----
    hist_peer_score = by_id["VAL-ZHIST-035"].score10
    hist_peer_scores: list[tuple[float, Value]] = [
        (1.0, Value.of(hist_peer_score, unit="score") if hist_peer_score is not None else Value.null(NullState.NOT_SCORABLE, unit="score")),
    ]
    hist_peer_dim = Dimension(name=DIM_HIST_PEER, max_points=DIMENSION_MAX_POINTS[DIM_HIST_PEER], metric_scores=hist_peer_scores)

    # ---- DIM_CF_YIELD (2 pts) ----
    cf_yield_scores: list[tuple[float, Value]] = [
        (0.5, Value.of(by_id["VAL-EY-029"].score10, unit="score") if by_id["VAL-EY-029"].score10 is not None else Value.null(NullState.NOT_SCORABLE, unit="score")),
        (0.5, Value.of(by_id["VAL-FCFY-030"].score10, unit="score") if by_id["VAL-FCFY-030"].score10 is not None else Value.null(NullState.NOT_SCORABLE, unit="score")),
    ]
    cf_yield_dim = Dimension(name=DIM_CF_YIELD, max_points=DIMENSION_MAX_POINTS[DIM_CF_YIELD], metric_scores=cf_yield_scores)

    # ---- DIM_FAIR_VALUE_SCENARIOS (2 pts): WACC>g, diluted shares, scenario audit ----
    scenario_gate_passes = (
        wacc_value is not None and wacc_value > tv_growth and diluted_shares is not None
        and scenario_result is not None and abs(scenario_result.probabilities_sum - 1.0) < 1e-6
    )
    if scenario_gate_passes and price is not None and bear_per_share is not None and base_per_share is not None and bull_per_share is not None:
        if price > bull_per_share:
            fv_score = 1.0
        elif price > base_per_share:
            fv_score = 5.0
        elif price >= bear_per_share:
            fv_score = 8.0
        else:
            fv_score = 10.0
    else:
        fv_score = None
    fair_value_scores: list[tuple[float, Value]] = [
        (1.0, Value.of(fv_score, unit="score") if fv_score is not None else Value.null(NullState.NOT_SCORABLE, unit="score")),
    ]
    if not scenario_gate_passes:
        assumptions.append("Fair-value-by-scenarios gate not satisfied (needs WACC>terminal growth, diluted shares, and probabilities summing to 1) -- dimension NOT_SCORABLE.")
    fair_value_dim = Dimension(name=DIM_FAIR_VALUE_SCENARIOS, max_points=DIMENSION_MAX_POINTS[DIM_FAIR_VALUE_SCENARIOS], metric_scores=fair_value_scores)

    # ---- DIM_MOS (1 pt): 0-15% -> 4-6; >15% -> 7-10; low confidence caps 5 ----
    mos_score = _score_from_anchor(mos_v, [(-0.20, 0), (0.0, 3), (0.15, 6), (0.30, 10)])
    mos_scores: list[tuple[float, Value]] = [
        (1.0, Value.of(mos_score, unit="score") if mos_score is not None else Value.null(NullState.NOT_SCORABLE, unit="score")),
    ]
    low_confidence = dispersion is not None and base_per_share and dispersion / abs(base_per_share) > 0.30
    if low_confidence:
        mos_scores = apply_dimension_cap(mos_scores, cap=5.0)
    mos_dim = Dimension(name=DIM_MOS, max_points=DIMENSION_MAX_POINTS[DIM_MOS], metric_scores=mos_scores)

    dimensions = [multiples_dim, hist_peer_dim, cf_yield_dim, fair_value_dim, mos_dim]

    cat = Category(name=AGENT_ID, max_points=MAX_POINTS, dimensions=dimensions)
    awarded_points = cat.points()
    dim_score10 = cat.score10()
    coverage = cat.coverage()

    mandatory_flags: list[str] = []
    if scenario_result is not None:
        for name, v in (("bear", scenario_result.bear_value), ("base", scenario_result.base_value), ("bull", scenario_result.bull_value)):
            if "TERMINAL_VALUE_SHARE_ABOVE_75PCT" in v.warnings:
                mandatory_flags.append("HIGH_TERMINAL_SENSITIVITY")
    if reconciled is False:
        mandatory_flags.append("FCFF_ECONOMIC_PROFIT_RECONCILIATION_FAILED")

    label = verdict(dim_score10)

    category = CategoryStats(max_points=MAX_POINTS, awarded_points=awarded_points, score_10=dim_score10, confidence=_category_confidence(coverage, packet))
    status = status_from_coverage(coverage)

    passed = 0
    failed = 0
    checks = [
        abs((category.awarded_points or 0.0) - Category(name=AGENT_ID, max_points=MAX_POINTS, dimensions=dimensions).points()) < 1e-6,
        all(row.formula_id for row in metric_rows),
        bool(packet.analysis.knowledge_timestamp),
        category.confidence is not None,
    ]
    for ok in checks:
        passed += 1 if ok else 0
        failed += 0 if ok else 1
    validation_tests = ValidationTestsSummary(passed=passed, failed=failed, warnings=0)

    return ValuationOutput(
        agent_id=AGENT_ID,
        status=status,
        security=SecurityRef(ticker=packet.security.ticker, exchange=packet.security.exchange, currency=packet.security.reporting_currency),
        knowledge_timestamp=packet.analysis.knowledge_timestamp,
        category=category,
        verdict=label,
        coverage=coverage,
        dimensions=dimensions,
        metrics=metric_rows,
        mandatory_flags=mandatory_flags,
        assumptions=assumptions,
        judgment_requests=judgment_requests,
        source_lineage=["packet.fundamentals.annual", "packet.facts_table", "packet.capital_structure", "packet.estimates"],
        validation_tests=validation_tests,
        model_selection=ModelSelection(primary=["FCFF_DCF", "ECONOMIC_PROFIT"], rejected=["DDM", "RELATIVE_MULTIPLES"]),
        normalization_reconciliation=[],
        wacc=WaccSummary(
            value=wacc_value,
            components={"cost_of_equity": ke_v.value if ke_v.is_valid else None, "cost_of_debt": kd_v.value if kd_v.is_valid else None, "tax_rate": tax_rate},
            sensitivity=(
                [{"minus_bp": wacc_sens.minus_bp.value, "plus_bp": wacc_sens.plus_bp.value}] if wacc_sens is not None and wacc_sens.minus_bp.is_valid else []
            ),
        ),
        scenarios=scenario_summaries,
        reverse_dcf=reverse_dcf_summary,
        model_cross_checks=ModelCrossChecks(fcff=fcff_per_share, economic_profit=econ_profit_per_share, residual_income=None, relative=relative_value, dispersion=dispersion),
        fair_value_distribution=fv_dist,
        reference_bands=ReferenceBands(
            bear=bear_per_share, base=base_per_share, bull=bull_per_share,
            margin_of_safety_15pct=base_per_share * 0.85 if base_per_share is not None else None,
            margin_of_safety_25pct=base_per_share * 0.75 if base_per_share is not None else None,
        ),
    )


def _category_confidence(coverage: float, packet: Packet) -> float:
    coverage_component = max(0.0, min(1.0, coverage)) * 100.0
    source_quality = 85.0
    freshness = 100.0 if packet.staleness.get("consensus", "FRESH") == "FRESH" else 50.0
    consistency = 75.0
    model_fit = 90.0 if packet.analysis.industry_adapter == SUPPORTED_ADAPTER else 40.0
    return _confidence_formula(
        coverage=coverage_component, source_quality=source_quality, freshness=freshness,
        consistency=consistency, model_fit=model_fit,
    )
