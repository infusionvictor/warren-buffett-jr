"""Business specialist (Task 15): BUS-001..030 (`FORMULAS.md`'s
BUS-MIX-001..BUS-SBC-030) and the five weighted business dimensions
(20 pts).

Sources of truth (`Cerebro/01_business_analysis/`):
- `FORMULAS.md`: the 30 registered formulas.
- `DECISION_RULES.md`: the wide-moat gate, business verdict bands, and
  mandatory flags (`VALUE_DESTRUCTION`, `CONCENTRATION_RED_FLAG`,
  `DILUTION_RED_FLAG`).
- `SCORING.md`: the five weighted dimensions and their per-dimension
  gate/cap column.
- `OUTPUT_SCHEMA.md`: `BusinessOutput`'s extension fields.
- `Cerebro/shared/OUTPUT_CONTRACT.md` / `HANDOFF_CONTRACT.md`: the common
  envelope, implemented in `wbj.specialists.common`.

## ROIC/spread/EVA/incremental-ROIC reuse

Per the task-15 brief ("ROIC/spread/EVA reuse `wbj.engines.valuation_engine`
functions where they exist"), `nopat`/`roic`/`spread`/`eva`/
`incremental_roic`/`fundamental_growth` below are thin re-exports of the
Task 13 valuation engine's own functions (module docstring precedent:
FIN-EF-023..027 in `financial.py` re-implements ROE/ROIC/ROA rather than
reusing the engine because `financial_analysis` predates
`valuation_engine`'s existence; `business_analysis` does not have that
excuse, so it imports directly). Each is still assigned its own `BUS-*`
metric id on the `MetricRow` it produces -- the *formula*, not the id, is
shared.

## Dataset coverage vs. `Packet`

`DATASET.md` marks most of this specialist's inputs "conditional":
`recurring_revenue_5y`, `customer_revenue_shares`, `market_share_...`,
`retention_churn_cohorts`, `customer_economics`, and
`management_guidance_history` are not part of `wbj.schemas.packet.Packet`
(no segment/customer/cohort/guidance block exists there), and
`customer_concentration` / `market_share` are additionally on
`wbj.core.nullstates.PROHIBITED_IMPUTATION` -- this module never infers
them from other reported numbers. `run(packet, overlay)` accepts them via
`overlay` (mirroring `financial.py`'s `overlay["wacc"]` precedent) so the
formula functions are directly testable and `run()` degrades honestly
(`MISSING`, not a guess) when the overlay omits them. Supported overlay
keys: `wacc` (float, required for every ROIC/spread/EVA-based metric),
`largest_customer_share` (0-1 decimal), `customer_shares` (list of 0-1
decimals, for the customer HHI), `segment_shares` (list of 0-1 decimals),
`recurring_revenue` (float), `guidance_history` (list of
`{"actual": float, "guidance_midpoint": float}`), `peer_roic` (list of
floats, >=8 for `wbj.core.scoring.peer_score` to engage),
`retention` (dict with `begin`/`expansion`/`contraction`/`churn` for
NRR/GRR), `customer_economics` (dict with `arpu`, `gross_margin`,
`customer_life_years`, `cac_spend`, `new_customers`, `monthly_arpu`).

## Moat classification, "2 quantitative moat effects", thesis killers

Per the task-15 brief and `DECISION_RULES.md`'s wide-moat gate condition 3
("At least two independent moat effects are quantitatively visible"), this
module computes the *quantitative* gate inputs (spread persistence, margin
stability) mechanically but cannot itself enumerate which qualitative moat
*category* (retention/switching costs, cost advantage, network scale,
regulated/intangible protection, efficient scale) each piece of evidence
belongs to -- that classification, the final `Wide|Narrow|None` label, and
`DECISION_RULES.md`'s mandatory "three business-model or execution risks"
are all judgment requests, exactly like `financial.py`'s FIN-GR-004/005.

## Dimension-level caps vs. mandatory-flag caps

`SCORING.md`'s dimension table has its own "Gate / cap" column (e.g. "Score
capped at 6 without positive ROIC-WACC spread") -- unlike `financial.py`'s
`capped_verdict` (which caps only the *label*, per
`SpecialistOutput.verdict`'s docstring), these are baked directly into the
computed dimension score itself via `common.apply_dimension_cap`: they are
part of the same deterministic point math `Category(dimensions)` reproduces
from, not a post-hoc override on the outward-facing verdict. The
*business-verdict* table's own "Additional condition" for the Excellent
band (moat gate, ROIC>=20%, FCF conversion>=0.9x) and the
`VALUE_DESTRUCTION` flag (ROIC<WACC) DO cap the verdict label only, mirroring
`financial.py`'s override discipline -- see `capped_verdict`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from wbj.core.confidence import confidence as _confidence_formula
from wbj.core.formulas import cagr as _cagr
from wbj.core.nullstates import EvidenceClass, NullState, Value
from wbj.core.scoring import Category, Dimension, anchor_score
from wbj.engines import valuation_engine as ve
from wbj.schemas.packet import Packet
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
    "DIM_MOAT",
    "DIM_COMPETITIVE",
    "DIM_MANAGEMENT",
    "DIM_DURABILITY",
    "DIM_CUSTOMER",
    "DIMENSION_NAMES",
    "BUS_30_IDS",
    "MoatSummary",
    "BusinessOutput",
    "capped_verdict",
    "verdict",
    "run",
    "value_destruction_triggered",
    "is_concentration_red_flag",
    "is_dilution_red_flag",
    "margin_range_is_stable",
    "wide_moat_margin_range_ok",
    # formula functions
    "segment_revenue_share",
    "recurring_revenue_pct",
    "largest_customer_concentration",
    "customer_hhi",
    "segment_hhi",
    "revenue_cagr",
    "gross_margin",
    "operating_margin",
    "margin_stability",
    "margin_range",
    "nopat",
    "average_invested_capital",
    "roic",
    "spread",
    "eva",
    "incremental_roic",
    "cumulative_fcf_conversion",
    "reinvestment_rate",
    "fundamental_growth",
    "net_revenue_retention",
    "gross_revenue_retention",
    "logo_churn",
    "customer_ltv",
    "customer_acquisition_cost",
    "ltv_to_cac",
    "cac_payback_months",
    "guidance_accuracy",
    "diluted_share_cagr",
    "capital_allocation_spread",
    "sbc_burden",
]

_VERSION = "2.0.0"
AGENT_ID = "business_analysis"
MAX_POINTS = 20.0

DIM_MOAT = "moat_and_pricing_power"
DIM_COMPETITIVE = "competitive_position"
DIM_MANAGEMENT = "management_and_capital_allocation"
DIM_DURABILITY = "business_durability"
DIM_CUSTOMER = "customer_economics"

DIMENSION_NAMES = (DIM_MOAT, DIM_COMPETITIVE, DIM_MANAGEMENT, DIM_DURABILITY, DIM_CUSTOMER)

DIMENSION_MAX_POINTS: dict[str, float] = {
    DIM_MOAT: 5.0,
    DIM_COMPETITIVE: 4.0,
    DIM_MANAGEMENT: 4.0,
    DIM_DURABILITY: 4.0,
    DIM_CUSTOMER: 3.0,
}
assert sum(DIMENSION_MAX_POINTS.values()) == MAX_POINTS

# The 30 registered formulas, FORMULAS.md order.
BUS_30_IDS: tuple[str, ...] = (
    "BUS-MIX-001", "BUS-REC-002", "BUS-CONC-003", "BUS-HHI-004", "BUS-HHI-005",
    "BUS-CAGR-006", "BUS-GM-007", "BUS-OM-008", "BUS-STAB-009", "BUS-RANGE-010",
    "BUS-NOPAT-011", "BUS-IC-012", "BUS-ROIC-013", "BUS-SPREAD-014", "BUS-EVA-015",
    "BUS-IROIC-016", "BUS-FCFC-017", "BUS-REINV-018", "BUS-SG-019", "BUS-NRR-020",
    "BUS-GRR-021", "BUS-CHURN-022", "BUS-LTV-023", "BUS-CAC-024", "BUS-LTVCAC-025",
    "BUS-PAYBACK-026", "BUS-GUIDE-027", "BUS-DIL-028", "BUS-ALLOC-029", "BUS-SBC-030",
)
assert len(BUS_30_IDS) == 30


def _ok(x: float, unit: str, **lineage: object) -> Value:
    return Value.of(x, unit=unit, evidence_class=lineage.pop("evidence_class", EvidenceClass.C), **lineage)


def _null(state: NullState, unit: str, *warnings: str) -> Value:
    return Value.null(state, unit=unit, warnings=list(warnings))


def _ols_slope(y: Sequence[float]) -> float:
    n = len(y)
    x = np.arange(n, dtype=float)
    slope, _intercept = np.polyfit(x, np.asarray(y, dtype=float), 1)
    return float(slope)


# ============================================================================
# BUS-MIX-001..BUS-HHI-005: mix and concentration
# ============================================================================


def segment_revenue_share(segment_revenue: float, total_revenue: float) -> Value:
    """Segment revenue share (BUS-MIX-001): `Revenue_i / Total revenue`."""
    if total_revenue == 0:
        return _null(NullState.NOT_MEANINGFUL, "pct", "SEGMENT_SHARE_ZERO_TOTAL_REVENUE")
    return _ok(segment_revenue / total_revenue, unit="pct")


def recurring_revenue_pct(recurring_revenue: float, total_revenue: float) -> Value:
    """Recurring revenue percentage (BUS-REC-002): `Recurring revenue /
    Total revenue`."""
    if total_revenue == 0:
        return _null(NullState.NOT_MEANINGFUL, "pct", "RECURRING_REVENUE_ZERO_TOTAL_REVENUE")
    return _ok(recurring_revenue / total_revenue, unit="pct")


def largest_customer_concentration(customer_revenue: float, total_revenue: float) -> Value:
    """Largest-customer concentration (BUS-CONC-003): `Revenue from largest
    customer / Total revenue`. `customer_concentration` is on
    `PROHIBITED_IMPUTATION` -- callers must pass disclosed figures only
    (see module docstring)."""
    if total_revenue == 0:
        return _null(NullState.NOT_MEANINGFUL, "pct", "CONCENTRATION_ZERO_TOTAL_REVENUE")
    return _ok(customer_revenue / total_revenue, unit="pct")


def is_concentration_red_flag(share: float) -> bool:
    """`DECISION_RULES.md`: "CONCENTRATION_RED_FLAG when one customer/
    product exceeds 30% of revenue" (>30%, strictly)."""
    return share > 0.30


def customer_hhi(shares: Sequence[float]) -> Value:
    """Customer concentration HHI (BUS-HHI-004): `sum(share_i^2)`, shares as
    decimals. FORMULAS.md: "partial HHI must be labeled lower bound" when
    `shares` is a disclosed subset rather than the full customer base --
    callers are responsible for passing the full disclosed set; this
    function does not know whether its input is complete."""
    if not shares:
        return _null(NullState.MISSING, "ratio", "CUSTOMER_HHI_EMPTY_SHARES")
    return _ok(sum(s * s for s in shares), unit="ratio")


def segment_hhi(shares: Sequence[float]) -> Value:
    """Product/segment HHI (BUS-HHI-005): `sum(segment_share_i^2)`."""
    if not shares:
        return _null(NullState.MISSING, "ratio", "SEGMENT_HHI_EMPTY_SHARES")
    return _ok(sum(s * s for s in shares), unit="ratio")


# ============================================================================
# BUS-CAGR-006..BUS-RANGE-010: growth and margins
# ============================================================================


def revenue_cagr(end: float, begin: float, years: float) -> Value:
    """Revenue CAGR (BUS-CAGR-006): `(Revenue_end/Revenue_begin)^(1/n)-1`.
    Reuses `wbj.core.formulas.cagr`, which already refuses (`NOT_MEANINGFUL`)
    a non-positive beginning value or a sign change, per
    CALCULATION_CONVENTIONS.md."""
    return _cagr(end, begin, years)


def gross_margin(gross_profit: float, revenue: float) -> Value:
    """Gross margin (BUS-GM-007): `Gross profit / Revenue`."""
    if revenue == 0:
        return _null(NullState.NOT_MEANINGFUL, "pct", "GROSS_MARGIN_ZERO_REVENUE")
    return _ok(gross_profit / revenue, unit="pct")


def operating_margin(ebit: float, revenue: float) -> Value:
    """Operating margin (BUS-OM-008): `Normalized EBIT / Revenue`."""
    if revenue == 0:
        return _null(NullState.NOT_MEANINGFUL, "pct", "OPERATING_MARGIN_ZERO_REVENUE")
    return _ok(ebit / revenue, unit="pct")


def margin_stability(margins: Sequence[float]) -> Value:
    """Margin stability (BUS-STAB-009): population stdev of >=5 annual
    operating margins. Lower is better (FORMULAS.md)."""
    if len(margins) < 5:
        return _null(NullState.MISSING, "pct", "MARGIN_STABILITY_NEEDS_5_POINTS")
    return _ok(float(np.std(np.asarray(margins, dtype=float))), unit="pct")


def margin_range(margins: Sequence[float]) -> Value:
    """Margin range (BUS-RANGE-010): `max(margin_5y) - min(margin_5y)`."""
    if len(margins) < 5:
        return _null(NullState.MISSING, "pct", "MARGIN_RANGE_NEEDS_5_POINTS")
    return _ok(max(margins) - min(margins), unit="pct")


def margin_range_is_stable(range_pct: float) -> bool:
    """FORMULAS.md, verbatim: "A range <=3 percentage points is a positive
    moat signal" (BUS-RANGE-010). Distinct from the wide-moat *gate*'s own,
    looser 5pp threshold -- see `wide_moat_margin_range_ok`."""
    return range_pct <= 0.03


def wide_moat_margin_range_ok(range_pct: float) -> bool:
    """DECISION_RULES.md wide-moat gate condition 2, verbatim: the five-year
    operating-margin range is "no more than 5 percentage points" (<=0.05).

    This is deliberately looser than `margin_range_is_stable`'s <=0.03
    BUS-RANGE-010 "positive moat signal": the two thresholds serve different
    purposes and a 3-5pp company legitimately clears the gate while not
    earning the stronger BUS-RANGE-010 moat signal. Confusing the two
    wrongly fails such a company's wide-moat gate."""
    return range_pct <= 0.05


# ============================================================================
# BUS-NOPAT-011..BUS-SG-019: ROIC / WACC / reinvestment
# (nopat/invested_capital/roic/spread/eva/incremental_roic/fundamental_growth
# are thin re-exports of `wbj.engines.valuation_engine` -- see module
# docstring "ROIC/spread/EVA/incremental-ROIC reuse".)
# ============================================================================

nopat = ve.nopat
roic = ve.roic
spread = ve.spread
eva = ve.eva
incremental_roic = ve.incremental_roic
fundamental_growth = ve.fundamental_growth


def average_invested_capital(
    debt_begin: float, equity_begin: float, cash_begin: float,
    debt_end: float, equity_end: float, cash_end: float,
) -> Value:
    """Average invested capital (BUS-IC-012): `average(financing-view IC at
    the beginning and end of the period)`, reusing
    `valuation_engine.invested_capital`'s financing view for each endpoint."""
    ic_begin = ve.invested_capital(debt_begin, equity_begin, cash_begin).financing_view
    ic_end = ve.invested_capital(debt_end, equity_end, cash_end).financing_view
    if ic_begin.is_null or ic_end.is_null:
        return _null(NullState.MISSING, "usd", "AVERAGE_INVESTED_CAPITAL_INPUTS_UNAVAILABLE")
    return _ok((ic_begin.value + ic_end.value) / 2, unit="usd")


def cumulative_fcf_conversion(fcf_sum_5y: float, ni_sum_5y: float) -> Value:
    """Cumulative FCF conversion (BUS-FCFC-017): `sum(FCF_5y) /
    sum(NetIncome_5y)`. FORMULAS.md: "sign changes require review" --
    refuses when the net-income sum is non-positive."""
    if ni_sum_5y <= 0:
        return _null(NullState.NOT_MEANINGFUL, "ratio", "FCF_CONVERSION_NI_SUM_NONPOSITIVE")
    return _ok(fcf_sum_5y / ni_sum_5y, unit="ratio")


def reinvestment_rate(net_capex: float, dnwc: float, rd_adjustment: float, nopat_value: float) -> Value:
    """Reinvestment rate (BUS-REINV-018): `(Net capex + change in
    non-cash working capital + capitalized-R&D adjustment) / NOPAT`.
    FORMULAS.md: "unstable when NOPAT is near zero" -- refuses at exactly
    zero rather than dividing."""
    if nopat_value == 0:
        return _null(NullState.NOT_MEANINGFUL, "ratio", "REINVESTMENT_RATE_ZERO_NOPAT")
    return _ok((net_capex + dnwc + rd_adjustment) / nopat_value, unit="ratio")


# ============================================================================
# BUS-NRR-020..BUS-PAYBACK-026: customer economics (subscription adapter)
# ============================================================================


def net_revenue_retention(begin: float, expansion: float, contraction: float, churn: float) -> Value:
    """NRR (BUS-NRR-020): `(Begin + expansion - contraction - churn) /
    Begin`."""
    if begin <= 0:
        return _null(NullState.NOT_MEANINGFUL, "pct", "NRR_NONPOSITIVE_BEGIN_COHORT")
    return _ok((begin + expansion - contraction - churn) / begin, unit="pct")


def gross_revenue_retention(begin: float, contraction: float, churn: float) -> Value:
    """GRR (BUS-GRR-021): `(Begin - contraction - churn) / Begin`; excludes
    expansion."""
    if begin <= 0:
        return _null(NullState.NOT_MEANINGFUL, "pct", "GRR_NONPOSITIVE_BEGIN_COHORT")
    return _ok((begin - contraction - churn) / begin, unit="pct")


def logo_churn(lost_customers: float, begin_customers: float) -> Value:
    """Logo churn (BUS-CHURN-022): `Customers lost / Customers at period
    start`."""
    if begin_customers <= 0:
        return _null(NullState.NOT_MEANINGFUL, "pct", "LOGO_CHURN_NONPOSITIVE_BEGIN_CUSTOMERS")
    return _ok(lost_customers / begin_customers, unit="pct")


def customer_ltv(arpu: float, gross_margin_pct: float, customer_life_years: float) -> Value:
    """Customer LTV (BUS-LTV-023): `ARPU * Gross margin * Expected customer
    life`. `customer_life_years` here is the 1/churn approximation (state
    the survival-model or approximation used, per FORMULAS.md)."""
    return _ok(arpu * gross_margin_pct * customer_life_years, unit="usd")


def customer_acquisition_cost(spend: float, new_customers: float) -> Value:
    """CAC (BUS-CAC-024): `Acquisition spend / New customers added`."""
    if new_customers <= 0:
        return _null(NullState.NOT_MEANINGFUL, "usd", "CAC_NONPOSITIVE_NEW_CUSTOMERS")
    return _ok(spend / new_customers, unit="usd")


def ltv_to_cac(ltv: float, cac: float) -> Value:
    """LTV/CAC (BUS-LTVCAC-025)."""
    if cac <= 0:
        return _null(NullState.NOT_MEANINGFUL, "ratio", "LTV_CAC_NONPOSITIVE_CAC")
    return _ok(ltv / cac, unit="ratio")


def cac_payback_months(cac: float, monthly_arpu: float, gross_margin_pct: float) -> Value:
    """CAC payback months (BUS-PAYBACK-026): `CAC / (Monthly ARPU * Gross
    margin)`."""
    denom = monthly_arpu * gross_margin_pct
    if denom <= 0:
        return _null(NullState.NOT_MEANINGFUL, "months", "CAC_PAYBACK_NONPOSITIVE_DENOMINATOR")
    return _ok(cac / denom, unit="months")


# ============================================================================
# BUS-GUIDE-027..BUS-SBC-030: management and dilution
# ============================================================================


def guidance_accuracy(actual: float, guidance_midpoint: float, floor: float = 0.01) -> Value:
    """Guidance accuracy (BUS-GUIDE-027): `1 - |actual - guidance_midpoint|
    / max(|guidance_midpoint|, materiality_floor)`, clipped to [0, 1]."""
    denom = max(abs(guidance_midpoint), floor)
    raw = 1.0 - abs(actual - guidance_midpoint) / denom
    return _ok(min(1.0, max(0.0, raw)), unit="ratio")


def diluted_share_cagr(shares_end: float, shares_begin: float, years: float) -> Value:
    """Diluted-share CAGR (BUS-DIL-028): positive is dilution. Reuses
    `wbj.core.formulas.cagr` (same as `financial.py`'s FIN-DX-033)."""
    return _cagr(shares_end, shares_begin, years)


def is_dilution_red_flag(cagr_pct: float) -> bool:
    """`DECISION_RULES.md`: "DILUTION_RED_FLAG when diluted shares grow >5%
    CAGR for three years without commensurate per-share value creation."
    This module reads the ">5% CAGR" threshold as the mechanical trigger
    (the brief's own summary: "diluted share CAGR >5% -> DILUTION_RED_FLAG");
    "commensurate per-share value creation" is a judgment call this
    function does not attempt (a red flag is raised, not a final verdict)."""
    return cagr_pct > 0.05


def capital_allocation_spread(incremental_roic_value: float, wacc_value: float) -> Value:
    """Capital-allocation value spread (BUS-ALLOC-029): `Incremental ROIC -
    WACC`."""
    return _ok(incremental_roic_value - wacc_value, unit="pct")


def sbc_burden(sbc: float, revenue: float) -> Value:
    """SBC burden (BUS-SBC-030): `SBC / Revenue`."""
    if revenue == 0:
        return _null(NullState.NOT_MEANINGFUL, "pct", "SBC_BURDEN_ZERO_REVENUE")
    return _ok(sbc / revenue, unit="pct")


# ============================================================================
# Mandatory flags
# ============================================================================


def value_destruction_triggered(roic_value: float | None, wacc_value: float | None) -> bool:
    """`DECISION_RULES.md`: "VALUE_DESTRUCTION when normalized ROIC < WACC."
    `None` inputs never trigger."""
    return roic_value is not None and wacc_value is not None and roic_value < wacc_value


# ============================================================================
# Verdict
# ============================================================================


def verdict(score10: float) -> str:
    """Business verdict bands (`DECISION_RULES.md`, verbatim score column):
    8.0-10.0 Excellent, 6.0-7.99 Good, 4.0-5.99 Mediocre/mixed, <4.0 Weak."""
    if score10 >= 8.0:
        return "Excellent business"
    if score10 >= 6.0:
        return "Good business"
    if score10 >= 4.0:
        return "Mediocre / mixed business"
    return "Weak business"


def capped_verdict(score10: float, *, value_destruction: bool, excellent_gate_passes: bool) -> str:
    """The verdict label after applying `DECISION_RULES.md`'s conditions to
    the label only (never to `category.awarded_points` -- see
    `SpecialistOutput.verdict`'s docstring and the module docstring's
    "Dimension-level caps vs. mandatory-flag caps").

    `value_destruction` (ROIC<WACC): `DECISION_RULES.md`'s Weak-business row
    names "ROIC below WACC" directly as sufficient for the weakest band, so
    this caps the effective score at the Weak ceiling -- the same full-cap
    discipline as `financial.py`'s Override 1.

    `excellent_gate_passes`: the business-verdict table's own "Additional
    condition" for Excellent (ROIC>=20% or top-decile adapter return,
    positive spread, FCF conversion>=0.9x, wide-moat gate passes) -- when
    `False`, a would-be-Excellent score is capped at the Good ceiling.
    """
    effective = score10
    if value_destruction:
        effective = min(effective, 3.99)
    elif not excellent_gate_passes:
        effective = min(effective, 7.99)
    return verdict(effective)


# ============================================================================
# Envelope assembly
# ============================================================================


class MoatSummary(BaseModel):
    """`envelope.moat` (OUTPUT_SCHEMA.md)."""

    model_config = ConfigDict(frozen=True)

    classification: str = "NotScorable"
    quantitative_evidence: list[str] = Field(default_factory=list)


class BusinessOutput(SpecialistOutput):
    """`business_analysis`'s `run()` result: the shared envelope plus
    OUTPUT_SCHEMA.md's business-specific extension fields."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    business_in_one_sentence: str | None = None
    moat: MoatSummary = Field(default_factory=MoatSummary)
    roic_history: list[float] = Field(default_factory=list)
    roic_wacc_spread_history: list[float] = Field(default_factory=list)
    margin_stability: dict[str, Any] = Field(default_factory=dict)
    customer_economics: dict[str, Any] = Field(default_factory=dict)
    capital_allocation: dict[str, Any] = Field(default_factory=dict)
    competitive_position: dict[str, Any] = Field(default_factory=dict)
    three_thesis_killers: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class _Row:
    metric_id: str
    value: Value
    score10: float | None  # None -> NOT_SCORABLE
    confidence: float = 0.0


def _confidence_for(v: Value) -> float:
    """Per-row confidence (0-100); same convention as `financial.py`'s
    `_confidence_for`: null rows score 0, valid rows start from their
    evidence class and lose 15 points per warning."""
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
    v = row.get(key)
    return float(v) if isinstance(v, (int, float)) else None


def _annual_rows(packet: Packet) -> list[dict]:
    rows = packet.fundamentals.get("annual") or []
    return list(reversed(rows))  # ascending, per DATASET.md


def _tax_rate(row: dict, fallback: float) -> float:
    pretax = _num(row, "income_before_tax")
    tax_expense = _num(row, "income_tax_expense")
    if pretax and pretax > 0 and tax_expense is not None:
        return min(max(tax_expense / pretax, 0.0), 1.0)
    return fallback


def _roic_and_spread_history(annual: list[dict], wacc_value: float | None) -> tuple[list[float], list[float]]:
    """Per-year ROIC (and, if `wacc_value` given, spread) for every year
    that has both its own and its prior year's balance-sheet inputs --
    mirrors `financial.py`'s FIN-EF-027 `roic_hist` construction, reused
    here for `envelope.roic_history` / `roic_wacc_spread_history` and the
    moat gate's "spread >=5pp in >=4/5 years" persistence check."""
    roic_hist: list[float] = []
    spread_hist: list[float] = []
    for i in range(1, len(annual)):
        ebit_i = _num(annual[i], "ebit")
        debt_i, equity_i = _num(annual[i], "total_debt"), _num(annual[i], "total_equity")
        debt_im1, equity_im1 = _num(annual[i - 1], "total_debt"), _num(annual[i - 1], "total_equity")
        cash_i, cash_im1 = _num(annual[i], "cash") or 0.0, _num(annual[i - 1], "cash") or 0.0
        if None in (ebit_i, debt_i, equity_i, debt_im1, equity_im1):
            continue
        tax_rate = _tax_rate(annual[i], 0.21)
        nopat_i = nopat(ebit_i, tax_rate)
        avg_ic_v = average_invested_capital(debt_im1, equity_im1, cash_im1, debt_i, equity_i, cash_i)
        if avg_ic_v.is_null:
            continue
        roic_v = roic(nopat_i.value, avg_ic_v.value)
        if roic_v.is_null:
            continue
        roic_hist.append(roic_v.value)
        if wacc_value is not None:
            spread_hist.append(spread(roic_v.value, wacc_value).value)
    return roic_hist, spread_hist


def _compute_all(
    packet: Packet, overlay: dict[str, Any]
) -> tuple[list[_Row], list[str], list[JudgmentRequest], dict[str, Any]]:
    annual = _annual_rows(packet)
    rows: list[_Row] = []
    assumptions: list[str] = []
    judgment_requests: list[JudgmentRequest] = []
    ctx: dict[str, Any] = {}

    def add(metric_id: str, v: Value, score10: float | None) -> None:
        rows.append(_Row(metric_id=metric_id, value=v, score10=score10, confidence=_confidence_for(v)))

    revenues = [_num(r, "revenue") for r in annual]
    gp_hist = [_num(r, "gross_profit") for r in annual]
    ebit_hist = [_num(r, "ebit") for r in annual]
    ni_hist = [_num(r, "net_income") for r in annual]
    ocf_hist = [_num(r, "operating_cash_flow") for r in annual]
    capex_hist = [abs(_num(r, "capex") or 0.0) if _num(r, "capex") is not None else None for r in annual]
    debt_hist = [_num(r, "total_debt") for r in annual]
    equity_hist = [_num(r, "total_equity") for r in annual]
    diluted_hist = [_num(r, "diluted_shares") for r in annual]
    sbc_hist = [_num(r, "stock_based_compensation") for r in annual]

    wacc_value = overlay.get("wacc")
    wacc_value = float(wacc_value) if wacc_value is not None else None
    ctx["wacc_value"] = wacc_value
    if wacc_value is None:
        assumptions.append(
            "No overlay['wacc'] supplied: every ROIC/spread/EVA-based metric (BUS-ROIC-013, "
            "BUS-SPREAD-014, BUS-EVA-015, BUS-ALLOC-029) and the wide-moat gate/VALUE_DESTRUCTION "
            "check are skipped (MISSING, not NOT_SCORABLE -- an ordinary missing external input)."
        )

    rev_latest = revenues[-1] if revenues else None
    rev_first = revenues[0] if revenues else None
    gp_latest = gp_hist[-1] if gp_hist else None
    ebit_latest = ebit_hist[-1] if ebit_hist else None
    ni_latest = ni_hist[-1] if ni_hist else None
    debt_latest = debt_hist[-1] if debt_hist else None
    equity_latest = equity_hist[-1] if equity_hist else None
    n_years = len(annual)

    # ---- BUS-MIX-001: segment revenue share (overlay only) ----
    segment_shares_overlay = overlay.get("segment_shares")
    v = _null(NullState.MISSING, "pct", "SEGMENT_REVENUE_UNAVAILABLE")
    add("BUS-MIX-001", v, None)

    # ---- BUS-REC-002: recurring revenue % (overlay only) ----
    recurring = overlay.get("recurring_revenue")
    if recurring is not None and rev_latest not in (None, 0):
        v = recurring_revenue_pct(float(recurring), rev_latest)
    else:
        v = _null(NullState.MISSING, "pct", "RECURRING_REVENUE_UNAVAILABLE")
    add("BUS-REC-002", v, _score_from_anchor(v, [(0.0, 0), (0.30, 4), (0.70, 7), (1.0, 10)]))

    # ---- BUS-CONC-003: largest customer concentration (overlay only; PROHIBITED_IMPUTATION) ----
    largest_customer_share = overlay.get("largest_customer_share")
    if largest_customer_share is not None:
        v = largest_customer_concentration(float(largest_customer_share) * 100.0, 100.0)
    else:
        v = _null(NullState.MISSING, "pct", "CUSTOMER_CONCENTRATION_UNAVAILABLE_PROHIBITED_IMPUTATION")
    add("BUS-CONC-003", v, _score_from_anchor(v, [(0.0, 10), (0.30, 6), (0.50, 2), (1.0, 0)]))
    ctx["largest_customer_share"] = v.value if v.is_valid else None

    # ---- BUS-HHI-004: customer HHI (overlay only) ----
    customer_shares_overlay = overlay.get("customer_shares")
    if customer_shares_overlay:
        v = customer_hhi(customer_shares_overlay)
    else:
        v = _null(NullState.MISSING, "ratio", "CUSTOMER_HHI_UNAVAILABLE")
    add("BUS-HHI-004", v, None)

    # ---- BUS-HHI-005: segment HHI (overlay only) ----
    if segment_shares_overlay:
        v = segment_hhi(segment_shares_overlay)
    else:
        v = _null(NullState.MISSING, "ratio", "SEGMENT_HHI_UNAVAILABLE")
    add("BUS-HHI-005", v, None)

    # ---- BUS-CAGR-006: revenue CAGR (3y or available window) ----
    if rev_first is not None and rev_latest is not None and n_years >= 2:
        v = revenue_cagr(rev_latest, rev_first, float(n_years - 1))
    else:
        v = _null(NullState.MISSING, "pct", "REVENUE_CAGR_INSUFFICIENT_HISTORY")
    add("BUS-CAGR-006", v, _score_from_anchor(v, [(-0.05, 0), (0.0, 3), (0.10, 7), (0.20, 10)]))
    ctx["revenue_cagr"] = v.value if v.is_valid else None

    # ---- BUS-GM-007 / BUS-OM-008: latest margins ----
    if gp_latest is not None and rev_latest not in (None, 0):
        v = gross_margin(gp_latest, rev_latest)
    else:
        v = _null(NullState.MISSING, "pct", "GROSS_MARGIN_INPUTS_UNAVAILABLE")
    add("BUS-GM-007", v, None)

    if ebit_latest is not None and rev_latest not in (None, 0):
        v = operating_margin(ebit_latest, rev_latest)
    else:
        v = _null(NullState.MISSING, "pct", "OPERATING_MARGIN_INPUTS_UNAVAILABLE")
    add("BUS-OM-008", v, None)
    om_latest = v.value if v.is_valid else None

    # ---- BUS-STAB-009 / BUS-RANGE-010: 5y operating margin stability/range ----
    op_margins = [
        e / r for e, r in zip(ebit_hist, revenues) if e is not None and r not in (None, 0)
    ]
    window = op_margins[-5:]
    if len(window) >= 5:
        v_stab = margin_stability(window)
        v_range = margin_range(window)
    else:
        v_stab = _null(NullState.MISSING, "pct", "MARGIN_STABILITY_NEEDS_5_POINTS")
        v_range = _null(NullState.MISSING, "pct", "MARGIN_RANGE_NEEDS_5_POINTS")
    add("BUS-STAB-009", v_stab, _score_from_anchor(v_stab, [(0.0, 10), (0.03, 8), (0.08, 4), (0.20, 0)]))
    add("BUS-RANGE-010", v_range, _score_from_anchor(v_range, [(0.0, 10), (0.03, 8), (0.05, 6), (0.10, 2), (0.20, 0)]))
    ctx["margin_range"] = v_range.value if v_range.is_valid else None
    ctx["op_margins_window"] = window

    # ---- BUS-NOPAT-011 / BUS-IC-012 / BUS-ROIC-013 (reused from valuation_engine) ----
    tax_rate = _tax_rate(annual[-1], 0.21) if annual else 0.21
    if not annual or _num(annual[-1], "income_tax_expense") is None:
        assumptions.append("BUS-NOPAT-011: normalized cash tax rate unavailable; substituted the 21% statutory rate.")

    if ebit_latest is not None:
        v_nopat = nopat(ebit_latest, tax_rate)
    else:
        v_nopat = _null(NullState.MISSING, "usd", "NOPAT_INPUTS_UNAVAILABLE")
    add("BUS-NOPAT-011", v_nopat, None)
    nopat_latest = v_nopat.value if v_nopat.is_valid else None

    if len(debt_hist) >= 2 and debt_latest is not None and equity_latest is not None and debt_hist[-2] is not None and equity_hist[-2] is not None:
        cash_latest = _num(annual[-1], "cash") or 0.0
        cash_begin = _num(annual[-2], "cash") or 0.0
        v_ic = average_invested_capital(debt_hist[-2], equity_hist[-2], cash_begin, debt_latest, equity_latest, cash_latest)
    else:
        v_ic = _null(NullState.MISSING, "usd", "AVERAGE_INVESTED_CAPITAL_INPUTS_UNAVAILABLE")
    add("BUS-IC-012", v_ic, None)

    if nopat_latest is not None and v_ic.is_valid:
        v_roic = roic(nopat_latest, v_ic.value)
    else:
        v_roic = _null(NullState.MISSING, "pct", "ROIC_INPUTS_UNAVAILABLE")
    add("BUS-ROIC-013", v_roic, _score_from_anchor(v_roic, [(0.0, 0), (0.08, 3), (0.15, 7), (0.25, 10)]))
    roic_latest = v_roic.value if v_roic.is_valid else None
    ctx["roic_latest"] = roic_latest

    # ---- BUS-SPREAD-014 / BUS-EVA-015 ----
    if roic_latest is not None and wacc_value is not None:
        v_spread = spread(roic_latest, wacc_value)
    else:
        v_spread = _null(NullState.MISSING, "pct", "SPREAD_INPUTS_UNAVAILABLE")
    add("BUS-SPREAD-014", v_spread, _score_from_anchor(v_spread, [(-0.05, 0), (0.0, 3), (0.05, 7), (0.10, 10)]))
    spread_latest = v_spread.value if v_spread.is_valid else None
    ctx["spread_latest"] = spread_latest

    if nopat_latest is not None and wacc_value is not None and len(debt_hist) >= 2 and debt_hist[-2] is not None and equity_hist[-2] is not None:
        beginning_ic = ve.invested_capital(debt_hist[-2], equity_hist[-2], _num(annual[-2], "cash") or 0.0).financing_view
        v_eva = eva(nopat_latest, wacc_value, beginning_ic.value) if beginning_ic.is_valid else _null(NullState.MISSING, "usd", "EVA_BEGINNING_IC_UNAVAILABLE")
    else:
        v_eva = _null(NullState.MISSING, "usd", "EVA_INPUTS_UNAVAILABLE")
    add("BUS-EVA-015", v_eva, None)

    # ---- ROIC/spread history (feeds BUS-IROIC-016, moat persistence, extension fields) ----
    roic_hist, spread_hist = _roic_and_spread_history(annual, wacc_value)
    ctx["roic_hist"] = roic_hist
    ctx["spread_hist"] = spread_hist

    # ---- BUS-IROIC-016 / BUS-ALLOC-029: incremental ROIC over the available window ----
    if nopat_latest is not None and len(annual) >= 2:
        ebit_first = ebit_hist[0]
        if ebit_first is not None:
            nopat_first = nopat(ebit_first, _tax_rate(annual[0], tax_rate)).value
            ic_first_v = v_ic  # fallback if history too short for a distinct first-period IC
            delta_nopat = nopat_latest - nopat_first
            delta_ic = (v_ic.value - 0.0) if v_ic.is_valid else None  # placeholder; real delta below
        else:
            delta_nopat = None
            delta_ic = None
    else:
        delta_nopat = None
        delta_ic = None
    # Proper delta-IC: financing-view IC at the first vs. latest annual row.
    if (
        delta_nopat is not None
        and len(debt_hist) >= 2
        and debt_hist[0] is not None
        and equity_hist[0] is not None
        and debt_latest is not None
        and equity_latest is not None
    ):
        ic_first = ve.invested_capital(debt_hist[0], equity_hist[0], _num(annual[0], "cash") or 0.0).financing_view
        ic_last = ve.invested_capital(debt_latest, equity_latest, _num(annual[-1], "cash") or 0.0).financing_view
        if ic_first.is_valid and ic_last.is_valid:
            delta_ic = ic_last.value - ic_first.value
        else:
            delta_ic = None
    else:
        delta_ic = None

    if delta_nopat is not None and delta_ic is not None:
        v_iroic = incremental_roic(delta_nopat, delta_ic)
    else:
        v_iroic = _null(NullState.MISSING, "pct", "INCREMENTAL_ROIC_INPUTS_UNAVAILABLE")
    add("BUS-IROIC-016", v_iroic, None)
    iroic_value = v_iroic.value if v_iroic.is_valid else None

    if iroic_value is not None and wacc_value is not None:
        v_alloc = capital_allocation_spread(iroic_value, wacc_value)
    else:
        v_alloc = _null(NullState.MISSING, "pct", "CAPITAL_ALLOCATION_SPREAD_INPUTS_UNAVAILABLE")
    add("BUS-ALLOC-029", v_alloc, _score_from_anchor(v_alloc, [(-0.05, 0), (0.0, 3), (0.05, 7), (0.10, 10)]))

    # ---- BUS-FCFC-017: cumulative FCF conversion (5y or available window) ----
    fcf_hist = [
        (ocf_hist[i] - capex_hist[i]) if ocf_hist[i] is not None and capex_hist[i] is not None else None
        for i in range(n_years)
    ]
    fcf_window = [x for x in fcf_hist[-5:] if x is not None]
    ni_window = [x for x in ni_hist[-5:] if x is not None]
    if len(fcf_window) >= 3 and len(ni_window) >= 3:
        v = cumulative_fcf_conversion(sum(fcf_window), sum(ni_window))
    else:
        v = _null(NullState.MISSING, "ratio", "FCF_CONVERSION_INSUFFICIENT_HISTORY")
    add("BUS-FCFC-017", v, _score_from_anchor(v, [(0.0, 0), (0.5, 4), (0.9, 7), (1.2, 10)]))
    fcf_conversion = v.value if v.is_valid else None
    ctx["fcf_conversion"] = fcf_conversion

    # ---- BUS-REINV-018 / BUS-SG-019 ----
    if len(capex_hist) >= 1 and capex_hist[-1] is not None and nopat_latest not in (None, 0):
        v_reinv = reinvestment_rate(capex_hist[-1], 0.0, 0.0, nopat_latest)
    else:
        v_reinv = _null(NullState.MISSING, "ratio", "REINVESTMENT_RATE_INPUTS_UNAVAILABLE")
    add("BUS-REINV-018", v_reinv, None)

    if v_reinv.is_valid and roic_latest is not None:
        v_sg = fundamental_growth(v_reinv.value, roic_latest)
    else:
        v_sg = _null(NullState.MISSING, "pct", "SUSTAINABLE_GROWTH_INPUTS_UNAVAILABLE")
    add("BUS-SG-019", v_sg, None)

    # ---- BUS-NRR-020 / BUS-GRR-021 / BUS-CHURN-022: customer economics (overlay only) ----
    retention = overlay.get("retention") or {}
    if {"begin", "expansion", "contraction", "churn"} <= retention.keys():
        v_nrr = net_revenue_retention(retention["begin"], retention["expansion"], retention["contraction"], retention["churn"])
        v_grr = gross_revenue_retention(retention["begin"], retention["contraction"], retention["churn"])
    else:
        v_nrr = _null(NullState.MISSING, "pct", "NRR_UNAVAILABLE_NON_SUBSCRIPTION_OR_NO_OVERLAY")
        v_grr = _null(NullState.MISSING, "pct", "GRR_UNAVAILABLE_NON_SUBSCRIPTION_OR_NO_OVERLAY")
    add("BUS-NRR-020", v_nrr, _score_from_anchor(v_nrr, [(0.85, 0), (1.0, 6), (1.1, 8), (1.2, 10)]))
    add("BUS-GRR-021", v_grr, _score_from_anchor(v_grr, [(0.70, 0), (0.85, 5), (0.95, 8), (1.0, 10)]))

    churn_inputs = overlay.get("churn")
    if churn_inputs and {"lost", "begin_customers"} <= churn_inputs.keys():
        v_churn = logo_churn(churn_inputs["lost"], churn_inputs["begin_customers"])
    else:
        v_churn = _null(NullState.MISSING, "pct", "LOGO_CHURN_UNAVAILABLE")
    add("BUS-CHURN-022", v_churn, _score_from_anchor(v_churn, [(0.20, 0), (0.10, 5), (0.05, 8), (0.0, 10)]))

    # ---- BUS-LTV-023 / BUS-CAC-024 / BUS-LTVCAC-025 / BUS-PAYBACK-026 (overlay only) ----
    ce = overlay.get("customer_economics") or {}
    if {"arpu", "gross_margin", "customer_life_years"} <= ce.keys():
        v_ltv = customer_ltv(ce["arpu"], ce["gross_margin"], ce["customer_life_years"])
    else:
        v_ltv = _null(NullState.MISSING, "usd", "LTV_INPUTS_UNAVAILABLE")
    add("BUS-LTV-023", v_ltv, None)

    if {"cac_spend", "new_customers"} <= ce.keys():
        v_cac = customer_acquisition_cost(ce["cac_spend"], ce["new_customers"])
    else:
        v_cac = _null(NullState.MISSING, "usd", "CAC_INPUTS_UNAVAILABLE")
    add("BUS-CAC-024", v_cac, None)

    if v_ltv.is_valid and v_cac.is_valid:
        v_ltvcac = ltv_to_cac(v_ltv.value, v_cac.value)
    else:
        v_ltvcac = _null(NullState.MISSING, "ratio", "LTV_CAC_RATIO_INPUTS_UNAVAILABLE")
    add("BUS-LTVCAC-025", v_ltvcac, _score_from_anchor(v_ltvcac, [(1.0, 0), (2.0, 4), (3.0, 7), (5.0, 10)]))

    if v_cac.is_valid and {"monthly_arpu", "gross_margin"} <= ce.keys():
        v_payback = cac_payback_months(v_cac.value, ce["monthly_arpu"], ce["gross_margin"])
    else:
        v_payback = _null(NullState.MISSING, "months", "CAC_PAYBACK_INPUTS_UNAVAILABLE")
    add("BUS-PAYBACK-026", v_payback, _score_from_anchor(v_payback, [(24.0, 0), (18.0, 5), (12.0, 8), (6.0, 10)], higher_is_better=False))

    # ---- BUS-GUIDE-027: guidance accuracy (overlay only) ----
    guidance_history = overlay.get("guidance_history") or []
    if guidance_history:
        accuracies = [
            guidance_accuracy(g["actual"], g["guidance_midpoint"]).value for g in guidance_history
            if "actual" in g and "guidance_midpoint" in g
        ]
        v_guide = _ok(sum(accuracies) / len(accuracies), unit="ratio") if accuracies else _null(
            NullState.MISSING, "ratio", "GUIDANCE_ACCURACY_NO_VALID_ROWS"
        )
    else:
        v_guide = _null(NullState.MISSING, "ratio", "GUIDANCE_ACCURACY_UNAVAILABLE")
    add("BUS-GUIDE-027", v_guide, _score_from_anchor(v_guide, [(0.0, 0), (0.7, 4), (0.9, 7), (1.0, 10)]))

    # ---- BUS-DIL-028: diluted share CAGR ----
    valid_diluted = [x for x in diluted_hist if x is not None]
    if len(valid_diluted) >= 2:
        v_dil = diluted_share_cagr(valid_diluted[-1], valid_diluted[0], float(len(valid_diluted) - 1))
    else:
        v_dil = _null(NullState.MISSING, "pct", "DILUTED_SHARE_CAGR_INSUFFICIENT_HISTORY")
    add("BUS-DIL-028", v_dil, _score_from_anchor(v_dil, [(-0.02, 10), (0.0, 7), (0.02, 4), (0.05, 0)]))
    ctx["diluted_cagr"] = v_dil.value if v_dil.is_valid else None

    # ---- BUS-SBC-030: SBC burden ----
    sbc_latest = sbc_hist[-1] if sbc_hist else None
    if sbc_latest is not None and rev_latest not in (None, 0):
        v_sbc = sbc_burden(sbc_latest, rev_latest)
    else:
        v_sbc = _null(NullState.MISSING, "pct", "SBC_BURDEN_INPUTS_UNAVAILABLE")
    add("BUS-SBC-030", v_sbc, _score_from_anchor(v_sbc, [(0.0, 10), (0.02, 8), (0.10, 3), (0.20, 0)]))

    # ---- Management capital-return proxy (buybacks/dividends funded from surplus) ----
    if annual:
        buybacks = abs(_num(annual[-1], "common_stock_repurchased") or 0.0)
        dividends = abs(_num(annual[-1], "dividends_paid") or 0.0)
        ctx["returns_cash"] = (buybacks + dividends) > 0
    else:
        ctx["returns_cash"] = None

    # ---- Competitive-position proxy: revenue CAGR (documented proxy, see run()) ----
    ctx["om_latest"] = om_latest

    return rows, assumptions, judgment_requests, ctx


def _score_from_anchor(
    v: Value, anchors: list[tuple[float, float]], *, higher_is_better: bool = True
) -> float | None:
    """0-10 continuous score via `wbj.core.scoring.anchor_score`, or `None`
    (NOT_SCORABLE) when `v` is null. `higher_is_better=False` mirrors the
    anchors already being given in "worse->better" x-order (e.g. CAC
    payback months) rather than inverting the score after the fact."""
    if v.is_null:
        return None
    return anchor_score(v.value, anchors)


def run(packet: Packet, overlay: dict[str, Any] | None = None) -> BusinessOutput:
    """Run the business specialist against `packet`, producing a
    `BusinessOutput` (max 20 pts): 30 BUS-* metric rows, five weighted
    dimensions, mandatory flags, and OUTPUT_SCHEMA.md's extension fields.

    See the module docstring for `overlay`'s supported keys (`wacc` is the
    load-bearing one: without it, every ROIC/spread/EVA-derived metric and
    the wide-moat gate degrade to MISSING).
    """
    overlay = overlay or {}
    computed, assumptions, judgment_requests, ctx = _compute_all(packet, overlay)
    by_id = {r.metric_id: r for r in computed}

    if packet.analysis.industry_adapter != "default_nonfinancial":
        assumptions.append(
            f"industry_adapter={packet.analysis.industry_adapter!r}: this module implements only "
            "conventional (non-financial-sector) formulas; ROIC/spread/EVA/moat results below should "
            "not be trusted for this security type without a sector adapter."
        )

    rows: list[MetricRow] = []
    for r in computed:
        score: float | str = r.score10 if r.score10 is not None else "NOT_SCORABLE"
        rows.append(
            MetricRow.from_value(
                r.metric_id, r.value, formula_id=r.metric_id, formula_version=_VERSION,
                score=score, confidence=r.confidence,
            )
        )

    # ---- MOAT (5 pts): spread level, spread persistence, margin range, quantitative-effects (judgment) ----
    spread_years = ctx["spread_hist"]
    persistence_frac = (
        sum(1 for s in spread_years if s >= 0.05) / len(spread_years) if spread_years else None
    )
    v_persistence = (
        _ok(persistence_frac, unit="ratio") if persistence_frac is not None
        else _null(NullState.MISSING, "ratio", "SPREAD_PERSISTENCE_INSUFFICIENT_HISTORY")
    )
    # `moat_classification` (the JudgmentRequest below) is registered as a
    # fourth, NOT_SCORABLE member of this dimension (equal-weighted with the
    # three mechanical inputs, mirroring `financial.py`'s
    # `_DIMENSION_MEMBERS`-driven equal weighting) so that a Task 20 judgment
    # answer moves `category.awarded_points`/`coverage`, not just the flat
    # `metrics` row -- see `MOAT_CLASSIFICATION_SLOT_INDEX` and
    # `SpecialistOutput.judgment_slots`.
    moat_scores: list[tuple[float, Value]] = [
        (0.25, Value.of(by_id["BUS-SPREAD-014"].score10, unit="score") if by_id["BUS-SPREAD-014"].score10 is not None else Value.null(NullState.NOT_SCORABLE, unit="score")),
        (0.25, Value.of(by_id["BUS-RANGE-010"].score10, unit="score") if by_id["BUS-RANGE-010"].score10 is not None else Value.null(NullState.NOT_SCORABLE, unit="score")),
        (0.25, Value.of(anchor_score(persistence_frac, [(0.0, 0), (0.5, 5), (0.8, 8), (1.0, 10)]), unit="score") if v_persistence.is_valid else Value.null(NullState.NOT_SCORABLE, unit="score")),
        (0.25, Value.null(NullState.NOT_SCORABLE, unit="score")),  # moat_classification judgment slot
    ]
    MOAT_CLASSIFICATION_SLOT_INDEX = 3
    positive_spread = ctx["spread_latest"] is not None and ctx["spread_latest"] > 0
    if not positive_spread:
        moat_scores = apply_dimension_cap(moat_scores, cap=6.0)
    moat_dim = Dimension(name=DIM_MOAT, max_points=DIMENSION_MAX_POINTS[DIM_MOAT], metric_scores=moat_scores)

    judgment_requests.append(
        JudgmentRequest(
            request_id="business_analysis:moat_classification",
            agent_id=AGENT_ID,
            metric_id="moat_classification",
            question="Classify the moat (Wide/Narrow/None) per DECISION_RULES.md's wide-moat gate: "
            "spread persistence, margin stability, and no unresolved concentration are computed "
            "mechanically above, but the final label requires qualitative review.",
            schema_hint="one of Wide|Narrow|None",
        )
    )
    judgment_requests.append(
        JudgmentRequest(
            request_id="business_analysis:moat_quantitative_effects_count",
            agent_id=AGENT_ID,
            metric_id="moat_quantitative_effects_count",
            question="Identify which of the five moat-effect categories (retention/switching costs, "
            "cost advantage, network scale, regulated/intangible protection, efficient scale) are "
            "quantitatively visible in the evidence; the wide-moat gate requires >=2.",
            schema_hint="array of 0-5 strings from the moat-effect category list",
        )
    )

    # ---- COMPETITIVE (4 pts): revenue-CAGR proxy for share trend + peer ROIC percentile ----
    # No market-share series or market-definition confidence is available
    # from Packet (PROHIBITED_IMPUTATION; see module docstring), so this
    # dimension is always capped at 8 (SCORING.md: "cannot score above 8 if
    # market definition is low confidence" -- unresolved market definition
    # reads as low confidence here, a documented, conservative default).
    revenue_cagr_score = _score_from_anchor(
        by_id["BUS-CAGR-006"].value, [(-0.05, 0), (0.0, 3), (0.10, 7), (0.20, 10)]
    )
    peer_roic_list = overlay.get("peer_roic") or []
    peer_v = None
    if ctx["roic_latest"] is not None and peer_roic_list:
        from wbj.core.scoring import peer_score
        peer_v = peer_score(ctx["roic_latest"], peer_roic_list)
    competitive_scores: list[tuple[float, Value]] = [
        (0.5, Value.of(revenue_cagr_score, unit="score") if revenue_cagr_score is not None else Value.null(NullState.NOT_SCORABLE, unit="score")),
        (0.5, peer_v if peer_v is not None and peer_v.is_valid else Value.null(NullState.NOT_SCORABLE, unit="score")),
    ]
    competitive_scores = apply_dimension_cap(competitive_scores, cap=8.0)
    competitive_dim = Dimension(name=DIM_COMPETITIVE, max_points=DIMENSION_MAX_POINTS[DIM_COMPETITIVE], metric_scores=competitive_scores)

    # ---- MANAGEMENT (4 pts): incremental ROIC vs WACC, dilution, capital return, guidance ----
    returns_cash = ctx.get("returns_cash")
    capital_return_value = (
        Value.of(10.0 if returns_cash else 5.0, unit="score")
        if returns_cash is not None
        else Value.null(NullState.NOT_SCORABLE, unit="score")
    )
    management_scores: list[tuple[float, Value]] = [
        (0.35, Value.of(by_id["BUS-ALLOC-029"].score10, unit="score") if by_id["BUS-ALLOC-029"].score10 is not None else Value.null(NullState.NOT_SCORABLE, unit="score")),
        (0.30, Value.of(by_id["BUS-DIL-028"].score10, unit="score") if by_id["BUS-DIL-028"].score10 is not None else Value.null(NullState.NOT_SCORABLE, unit="score")),
        (0.20, capital_return_value),
        (0.15, Value.of(by_id["BUS-GUIDE-027"].score10, unit="score") if by_id["BUS-GUIDE-027"].score10 is not None else Value.null(NullState.NOT_SCORABLE, unit="score")),
    ]
    management_dim = Dimension(name=DIM_MANAGEMENT, max_points=DIMENSION_MAX_POINTS[DIM_MANAGEMENT], metric_scores=management_scores)

    # ---- DURABILITY (4 pts): recurring revenue, concentration, margin persistence ----
    largest_customer_share = ctx.get("largest_customer_share")
    concentration_flag = largest_customer_share is not None and is_concentration_red_flag(largest_customer_share)
    durability_scores: list[tuple[float, Value]] = [
        (1 / 3, Value.of(by_id["BUS-REC-002"].score10, unit="score") if by_id["BUS-REC-002"].score10 is not None else Value.null(NullState.NOT_SCORABLE, unit="score")),
        (1 / 3, Value.of(by_id["BUS-CONC-003"].score10, unit="score") if by_id["BUS-CONC-003"].score10 is not None else Value.null(NullState.NOT_SCORABLE, unit="score")),
        (1 / 3, Value.of(by_id["BUS-RANGE-010"].score10, unit="score") if by_id["BUS-RANGE-010"].score10 is not None else Value.null(NullState.NOT_SCORABLE, unit="score")),
    ]
    if concentration_flag:
        durability_scores = apply_dimension_cap(durability_scores, cap=6.0)
    durability_dim = Dimension(name=DIM_DURABILITY, max_points=DIMENSION_MAX_POINTS[DIM_DURABILITY], metric_scores=durability_scores)

    # ---- CUSTOMER ECONOMICS (3 pts): NRR/GRR, churn, LTV/CAC + payback ----
    customer_scores: list[tuple[float, Value]] = [
        (0.4, Value.of(by_id["BUS-NRR-020"].score10, unit="score") if by_id["BUS-NRR-020"].score10 is not None else Value.null(NullState.NOT_SCORABLE, unit="score")),
        (0.3, Value.of(by_id["BUS-LTVCAC-025"].score10, unit="score") if by_id["BUS-LTVCAC-025"].score10 is not None else Value.null(NullState.NOT_SCORABLE, unit="score")),
        (0.3, Value.of(by_id["BUS-CHURN-022"].score10, unit="score") if by_id["BUS-CHURN-022"].score10 is not None else Value.null(NullState.NOT_SCORABLE, unit="score")),
    ]
    customer_dim = Dimension(name=DIM_CUSTOMER, max_points=DIMENSION_MAX_POINTS[DIM_CUSTOMER], metric_scores=customer_scores)

    dimensions = [moat_dim, competitive_dim, management_dim, durability_dim, customer_dim]

    cat = Category(name=AGENT_ID, max_points=MAX_POINTS, dimensions=dimensions)
    awarded_points = cat.points()
    dim_score10 = cat.score10()
    coverage = cat.coverage()

    mandatory_flags: list[str] = []

    value_destruction = value_destruction_triggered(ctx["roic_latest"], ctx["wacc_value"])
    if value_destruction:
        mandatory_flags.append("VALUE_DESTRUCTION")

    if concentration_flag:
        mandatory_flags.append("CONCENTRATION_RED_FLAG")

    diluted_cagr = ctx.get("diluted_cagr")
    dilution_flag = diluted_cagr is not None and is_dilution_red_flag(diluted_cagr)
    if dilution_flag:
        mandatory_flags.append("DILUTION_RED_FLAG")

    # ---- Excellent-band additional condition (moat gate proxy: mechanical inputs only) ----
    moat_gate_mechanical = (
        positive_spread
        and (persistence_frac is not None and persistence_frac >= 0.8)
        and ctx.get("margin_range") is not None and wide_moat_margin_range_ok(ctx["margin_range"])
        and not concentration_flag
    )
    excellent_gate_passes = (
        ctx["roic_latest"] is not None and ctx["roic_latest"] >= 0.20
        and positive_spread
        and ctx.get("fcf_conversion") is not None and ctx["fcf_conversion"] >= 0.9
        and moat_gate_mechanical
    )

    label = capped_verdict(dim_score10, value_destruction=value_destruction, excellent_gate_passes=excellent_gate_passes)

    judgment_requests.append(
        JudgmentRequest(
            request_id="business_analysis:three_thesis_killers",
            agent_id=AGENT_ID,
            metric_id="three_thesis_killers",
            question="List three business-model or execution risks that could destroy the thesis "
            "(DECISION_RULES.md: mandatory, always required).",
            schema_hint="array of exactly 3 strings",
        )
    )

    category = CategoryStats(
        max_points=MAX_POINTS, awarded_points=awarded_points, score_10=dim_score10,
        confidence=_category_confidence(coverage, packet),
    )
    status = status_from_coverage(coverage)

    passed = 0
    failed = 0
    checks = [
        abs((category.awarded_points or 0.0) - Category(name=AGENT_ID, max_points=MAX_POINTS, dimensions=dimensions).points()) < 1e-6,
        all(row.formula_id for row in rows),
        bool(packet.analysis.knowledge_timestamp),
        category.confidence is not None,
    ]
    for ok in checks:
        passed += 1 if ok else 0
        failed += 0 if ok else 1
    validation_tests = ValidationTestsSummary(passed=passed, failed=failed, warnings=0)

    # ---- Judgment slots: which dimension slot each judgment-only metric feeds ----
    # `moat_classification` is registered as `moat_scores[MOAT_CLASSIFICATION_SLOT_INDEX]`
    # above; recording it here lets `wbj.overlay.merge.merge_overlay` replace that
    # exact slot and rescore once a judgment answer arrives, exactly like
    # `financial.py`'s `FIN-GR-004`/`FIN-GR-005`. `moat_quantitative_effects_count`
    # and `three_thesis_killers` have no dimension slot (mandatory context-only
    # lists) and so are deliberately absent from this map.
    judgment_slots: dict[str, tuple[str, int]] = {
        "moat_classification": (DIM_MOAT, MOAT_CLASSIFICATION_SLOT_INDEX),
    }

    return BusinessOutput(
        agent_id=AGENT_ID,
        status=status,
        security=SecurityRef(
            ticker=packet.security.ticker, exchange=packet.security.exchange, currency=packet.security.reporting_currency
        ),
        knowledge_timestamp=packet.analysis.knowledge_timestamp,
        category=category,
        verdict=label,
        coverage=coverage,
        dimensions=dimensions,
        metrics=rows,
        mandatory_flags=mandatory_flags,
        assumptions=assumptions,
        judgment_requests=judgment_requests,
        judgment_slots=judgment_slots,
        source_lineage=["packet.fundamentals.annual"],
        validation_tests=validation_tests,
        business_in_one_sentence=None,
        moat=MoatSummary(classification="NotScorable", quantitative_evidence=[]),
        roic_history=ctx["roic_hist"],
        roic_wacc_spread_history=ctx["spread_hist"],
        margin_stability={"range": ctx.get("margin_range"), "window": ctx.get("op_margins_window", [])},
        customer_economics={},
        capital_allocation={"diluted_share_cagr": diluted_cagr, "returns_cash": ctx.get("returns_cash")},
        competitive_position={"revenue_cagr": ctx.get("revenue_cagr")},
        three_thesis_killers=[],
    )


def _category_confidence(coverage: float, packet: Packet) -> float:
    """Category confidence (0-100) via `wbj.core.confidence.confidence()`.
    Same construction as `financial.py`'s `_category_confidence`: real
    packet signals where available, documented constants otherwise."""
    coverage_component = max(0.0, min(1.0, coverage)) * 100.0
    source_quality = 85.0  # filings-derived, but most conditional inputs are overlay-supplied
    freshness = 100.0 if packet.staleness.get("quarterly_fundamentals", "FRESH") == "FRESH" else 50.0
    consistency = 80.0  # no independent 27-metric-style diagnostic to reconcile against here
    model_fit = 90.0 if packet.analysis.industry_adapter == "default_nonfinancial" else 40.0
    return _confidence_formula(
        coverage=coverage_component, source_quality=source_quality, freshness=freshness,
        consistency=consistency, model_fit=model_fit,
    )
