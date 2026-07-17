"""Market and Growth specialist (Task 16): MKT-001..025 (`FORMULAS.md`'s
MKT-TAM-001..MKT-SCEN-025) and the five weighted market/growth dimensions
(20 pts).

Sources of truth (`Cerebro/03_market_analysis/`):
- `FORMULAS.md`: the 25 registered formulas.
- `DECISION_RULES.md`: TAM source-quality tiers, the forecast-consistency
  gate, and the market verdict bands.
- `SCORING.md`: the five weighted dimensions and their per-dimension
  gate/cap column. Dimension "Primary inputs" columns reference formulas
  by their *numeric suffix only* (e.g. "MKT-TAM-001..008,021" means
  formulas #1-8 and #21 regardless of prefix letter) -- several formulas
  therefore legitimately feed two dimensions (e.g. MKT-PEN-005 feeds both
  TAM and Runway&Share), the same precedent as `financial.py`'s FIN-CF-015.
- `OUTPUT_SCHEMA.md`: `MarketOutput`'s extension fields.
- `Cerebro/shared/OUTPUT_CONTRACT.md` / `HANDOFF_CONTRACT.md`: the common
  envelope, implemented in `wbj.specialists.common`.

## Dataset coverage vs. `Packet`

Almost nothing this specialist needs (TAM/SAM/SOM, peer market shares,
consensus-revision history with a *frozen pre-release snapshot*, backlog/
RPO, catalyst registry, unit economics) is part of
`wbj.schemas.packet.Packet` -- `DATASET.md` marks most of it "conditional"
and sources it from primary market research / issuer KPI disclosures the
packet builder does not carry. `market_share` is additionally on
`wbj.core.nullstates.PROHIBITED_IMPUTATION`. Exactly two things ARE
computable straight from `Packet` without an overlay:

1. `MKT-OPLEV-017`/`MKT-INCM-018` (operating leverage / incremental
   margin), from `packet.fundamentals.annual`'s `ebit`/`revenue`.
2. `MKT-RSG-024` (sector relative strength), from
   `packet.market_data.sector`/`.benchmark` -- reusing
   `wbj.engines.indicators.relative_strength` directly (per the general
   "reuse the engine" discipline) when both series are non-empty. Per the
   task-17 brief's own note (shared here since it's the same packet
   field), `Packet.market_data.benchmark`/`.sector` may be empty (a Task
   10 limitation) -- this degrades to `NOT_SCORABLE` honestly rather than
   crashing.

Everything else is supplied, if at all, via an optional `overlay` dict
(mirroring `financial.py`'s `overlay["wacc"]` precedent): `tam`,
`tam_source_tier` (1-5, `DECISION_RULES.md`'s source-quality tiers),
`sam_inputs`, `som_inputs`, `company_relevant_revenue`, `company_sales`/
`total_market_sales` (share), `competitor_shares` (HHI), `target_revenue`/
`current_revenue`/`assumed_growth` (runway), `estimates` (a
`{"upward": int, "total": int, "current_consensus": float,
"prior_consensus": float, "individual_estimates": [float, ...],
"actual": float, "pre_release_consensus": float,
"snapshot_before_release": bool}` dict), `backlog_history`,
`ntm_contracted`/`ntm_revenue_estimate`, `catalysts` (a list of
`{"event": str, "months_to_event": float, "probability": float?,
"impact": float?, "evidence_quality": float?}` dicts -- probability/
impact/evidence_quality are judgment inputs, see below),
`adoption` (`current_units`/`eventual_units`), `arpu_t`/`arpu_t1`,
`scenarios` (list of `(probability, outcome)` pairs).

## TAM tier assignment, catalyst P/Impact/EvidenceQuality, thesis killers

Per the task-16 brief, TAM source-tier *assignment* and each catalyst's
probability/impact/evidence-quality are judgment requests: this module
computes `MKT-TDEC-020` (time decay) mechanically from `months_to_event`
but cannot itself assign a probability, dollar impact, or evidence-quality
score to a described event -- exactly like `financial.py`'s FIN-GR-004/005
and `business.py`'s moat classification. `overlay["catalysts"]` entries
that already carry `probability`/`impact`/`evidence_quality` (e.g. answers
merged back in by Task 20) are scored in full; entries missing any of the
three are "narrative-only" and trigger both a per-catalyst judgment
request and `SCORING.md`'s "narrative-only catalyst score capped at 3"
dimension cap. `DECISION_RULES.md`'s mandatory "three growth-thesis
killers" is likewise a judgment request.

## Dimension-level caps vs. mandatory flags

Same discipline as `business.py`: `SCORING.md`'s dimension-level "Gate /
cap" column (TAM confidence <60 caps TAM at 6; narrative-only catalysts
cap Catalysts at 3) is baked directly into the dimension's own point math
via the shared `common.apply_dimension_cap` -- part of the deterministic
math `Category(dimensions)` reproduces from. `DECISION_RULES.md`'s
forecast-consistency gate ("forecast growth > growth capacity by >5pts
requires an external-capital explanation") is instead recorded as an
`EXTERNAL_CAPITAL_REQUIRED` mandatory flag -- informational, not a point
cap (no verdict-label cap is specified for it in `DECISION_RULES.md`,
unlike `financial.py`'s overrides).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from wbj.core.confidence import confidence as _confidence_formula
from wbj.core.formulas import cagr as _cagr
from wbj.core.nullstates import EvidenceClass, NullState, Value
from wbj.core.scoring import Category, Dimension, anchor_score
from wbj.engines import indicators as ind
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
    "DIM_TAM",
    "DIM_REVISIONS",
    "DIM_CATALYSTS",
    "DIM_RUNWAY_SHARE",
    "DIM_OPLEV",
    "DIMENSION_NAMES",
    "MKT_25_IDS",
    "TAM_TIER_CONFIDENCE",
    "TamSamSom",
    "MarketOutput",
    "verdict",
    "run",
    "tam_confidence_caps_dimension",
    "forecast_consistency_gate",
    # formula functions
    "sam",
    "som",
    "tam_cagr",
    "penetration",
    "market_share",
    "market_share_delta",
    "industry_hhi",
    "growth_capacity",
    "runway_years",
    "revision_breadth",
    "revision_magnitude",
    "estimate_dispersion",
    "earnings_surprise",
    "backlog_growth",
    "revenue_coverage",
    "operating_leverage",
    "incremental_operating_margin",
    "catalyst_expected_impact",
    "time_decay",
    "adoption_penetration",
    "arpu_growth",
    "sector_breadth",
    "sector_relative_strength",
    "scenario_weighted_outcome",
]

_VERSION = "2.0.0"
AGENT_ID = "market_analysis"
MAX_POINTS = 20.0

DIM_TAM = "tam_and_industry_tailwind"
DIM_REVISIONS = "earnings_and_revenue_revisions"
DIM_CATALYSTS = "product_and_business_catalysts"
DIM_RUNWAY_SHARE = "growth_runway_and_share_capture"
DIM_OPLEV = "operating_leverage_and_market_confirmation"

DIMENSION_NAMES = (DIM_TAM, DIM_REVISIONS, DIM_CATALYSTS, DIM_RUNWAY_SHARE, DIM_OPLEV)

DIMENSION_MAX_POINTS: dict[str, float] = {
    DIM_TAM: 5.0, DIM_REVISIONS: 4.0, DIM_CATALYSTS: 4.0, DIM_RUNWAY_SHARE: 4.0, DIM_OPLEV: 3.0,
}
assert sum(DIMENSION_MAX_POINTS.values()) == MAX_POINTS

# The 25 registered formulas, FORMULAS.md order.
MKT_25_IDS: tuple[str, ...] = (
    "MKT-TAM-001", "MKT-SAM-002", "MKT-SOM-003", "MKT-CAGR-004", "MKT-PEN-005",
    "MKT-SHARE-006", "MKT-SHDELTA-007", "MKT-HHI-008", "MKT-GCAP-009", "MKT-RUN-010",
    "MKT-REVBR-011", "MKT-REVMAG-012", "MKT-DISP-013", "MKT-SURP-014", "MKT-BACK-015",
    "MKT-COVER-016", "MKT-OPLEV-017", "MKT-INCM-018", "MKT-CAT-019", "MKT-TDEC-020",
    "MKT-ADOPT-021", "MKT-ARPU-022", "MKT-SECB-023", "MKT-RSG-024", "MKT-SCEN-025",
)
assert len(MKT_25_IDS) == 25

# DECISION_RULES.md: TAM source-quality tiers -> confidence component.
TAM_TIER_CONFIDENCE: dict[int, float] = {1: 100.0, 2: 85.0, 3: 70.0, 4: 45.0, 5: 0.0}


def _ok(x: float, unit: str, **lineage: object) -> Value:
    return Value.of(x, unit=unit, evidence_class=lineage.pop("evidence_class", EvidenceClass.C), **lineage)


def _null(state: NullState, unit: str, *warnings: str) -> Value:
    return Value.null(state, unit=unit, warnings=list(warnings))


# ============================================================================
# MKT-TAM-001..MKT-HHI-008: TAM/SAM/SOM, penetration, share
# ============================================================================


def sam(tam: float, geography_share: float, product_share: float, reachable_share: float) -> Value:
    """SAM (MKT-SAM-002): `TAM * eligible geography share * eligible
    product share * reachable customer share`."""
    return _ok(tam * geography_share * product_share * reachable_share, unit="usd", evidence_class=EvidenceClass.A)


def som(sam_value: float, target_share: float) -> Value:
    """SOM (MKT-SOM-003): `SAM * defensible target share` -- a
    scenario-specific assumption (FORMULAS.md)."""
    return _ok(sam_value * target_share, unit="usd", evidence_class=EvidenceClass.A)


def tam_cagr(end: float, begin: float, years: float) -> Value:
    """TAM CAGR (MKT-CAGR-004): `(TAM_end/TAM_begin)^(1/n)-1`. Reuses
    `wbj.core.formulas.cagr`, which refuses a non-positive/sign-changing
    base per CALCULATION_CONVENTIONS.md."""
    return _cagr(end, begin, years)


def penetration(company_relevant_revenue: float, tam: float) -> Value:
    """Company penetration (MKT-PEN-005): `Company relevant revenue / TAM`
    under the same definition. FORMULAS.md: "Do not divide total company
    revenue by a narrow TAM" -- callers are responsible for passing
    TAM-mapped revenue, not total company revenue."""
    if tam <= 0:
        return _null(NullState.NOT_MEANINGFUL, "pct", "PENETRATION_NONPOSITIVE_TAM")
    return _ok(company_relevant_revenue / tam, unit="pct")


def market_share(company_sales: float, total_market_sales: float) -> Value:
    """Market share (MKT-SHARE-006): `Company relevant sales / Total market
    sales`. `market_share` is on `PROHIBITED_IMPUTATION` -- callers must
    pass disclosed figures only (see module docstring)."""
    if total_market_sales <= 0:
        return _null(NullState.NOT_MEANINGFUL, "pct", "MARKET_SHARE_NONPOSITIVE_TOTAL")
    return _ok(company_sales / total_market_sales, unit="pct")


def market_share_delta(share_t: float, share_t1: float) -> Value:
    """Market-share change (MKT-SHDELTA-007): `Share_t - Share_t-1`, in
    percentage points (FORMULAS.md, verbatim: "not percent change")."""
    return _ok(share_t - share_t1, unit="pp")


def industry_hhi(shares: Sequence[float]) -> Value:
    """Industry HHI (MKT-HHI-008): `sum(competitor_share_i^2)`, decimals.
    A partial competitor set is a lower bound (FORMULAS.md) -- callers'
    responsibility to disclose completeness."""
    if not shares:
        return _null(NullState.MISSING, "ratio", "INDUSTRY_HHI_EMPTY_SHARES")
    return _ok(sum(s * s for s in shares), unit="ratio")


# ============================================================================
# MKT-GCAP-009 / MKT-RUN-010: growth capacity and runway
# ============================================================================


def growth_capacity(reinvestment_rate: float, roic_value: float) -> Value:
    """Fundamental growth capacity (MKT-GCAP-009): `Reinvestment rate *
    ROIC` -- a consistency ceiling for internally financed growth, not a
    forecast guarantee (FORMULAS.md)."""
    return _ok(reinvestment_rate * roic_value, unit="pct")


def runway_years(target_revenue: float, current_revenue: float, assumed_growth: float) -> Value:
    """Runway years to target share (MKT-RUN-010): `ln(Target revenue /
    Current revenue) / ln(1 + Assumed growth)`. Not meaningful when
    assumed growth <=0, or either revenue figure is non-positive
    (FORMULAS.md)."""
    if assumed_growth <= 0 or target_revenue <= 0 or current_revenue <= 0:
        return _null(NullState.NOT_MEANINGFUL, "years", "RUNWAY_NONPOSITIVE_GROWTH_OR_REVENUE")
    return _ok(math.log(target_revenue / current_revenue) / math.log(1 + assumed_growth), unit="years")


def forecast_consistency_gate(forecast_revenue: float, tam: float) -> bool:
    """`DECISION_RULES.md`'s forecast-consistency gate: "Company revenue <=
    TAM in same definition." Returns `True` (passes) iff
    `forecast_revenue <= tam`."""
    return forecast_revenue <= tam


# ============================================================================
# MKT-REVBR-011..MKT-SURP-014: estimate revisions
# ============================================================================


def revision_breadth(upward_count: int, total_count: int, *, min_estimates: int = 5) -> Value:
    """Positive revision breadth (MKT-REVBR-011): `Upward revisions / Total
    revisions`. `SCORING.md`: "Requires... >=5 estimates" -- below that,
    `NOT_SCORABLE` rather than a low-confidence guess (`earnings_revisions`
    is on `PROHIBITED_IMPUTATION`)."""
    if total_count < min_estimates:
        return _null(
            NullState.NOT_SCORABLE, "pct",
            f"REVISION_BREADTH_NEEDS_{min_estimates}_ESTIMATES: got {total_count}",
        )
    return _ok(upward_count / total_count, unit="pct")


def revision_magnitude(current_consensus: float, prior_consensus: float) -> Value:
    """Net revision magnitude (MKT-REVMAG-012): `(Current consensus - Prior
    consensus) / abs(Prior consensus)`."""
    if prior_consensus == 0:
        return _null(NullState.NOT_MEANINGFUL, "pct", "REVISION_MAGNITUDE_ZERO_PRIOR_CONSENSUS")
    return _ok((current_consensus - prior_consensus) / abs(prior_consensus), unit="pct")


def estimate_dispersion(estimates: Sequence[float]) -> Value:
    """Estimate dispersion (MKT-DISP-013): `stdev(analyst estimates) /
    abs(consensus mean)`. Higher dispersion lowers forecast confidence
    (FORMULAS.md)."""
    if len(estimates) < 2:
        return _null(NullState.MISSING, "ratio", "DISPERSION_NEEDS_2_ESTIMATES")
    mean = sum(estimates) / len(estimates)
    if mean == 0:
        return _null(NullState.NOT_MEANINGFUL, "ratio", "DISPERSION_ZERO_MEAN")
    stdev = float(np.std(np.asarray(estimates, dtype=float)))
    return _ok(stdev / abs(mean), unit="ratio")


def earnings_surprise(actual: float, pre_release_consensus: float, *, snapshot_before_release: bool) -> Value:
    """Earnings/revenue surprise (MKT-SURP-014): `(Actual - Pre-release
    consensus) / abs(Pre-release consensus)`. `AGENT.md`: "Do not use
    post-release consensus to calculate earnings surprise" --
    `snapshot_before_release=False` rejects the calculation outright
    (`CONFLICTED`, per MKT-T008) rather than silently computing a
    contaminated number."""
    if not snapshot_before_release:
        return _null(
            NullState.CONFLICTED, "pct",
            "SNAPSHOT_AFTER_RELEASE_REJECTED: consensus must be frozen before the release date",
        )
    if pre_release_consensus == 0:
        return _null(NullState.NOT_MEANINGFUL, "pct", "SURPRISE_ZERO_PRE_RELEASE_CONSENSUS")
    return _ok((actual - pre_release_consensus) / abs(pre_release_consensus), unit="pct")


# ============================================================================
# MKT-BACK-015 / MKT-COVER-016: backlog
# ============================================================================


def backlog_growth(backlog_t: float, backlog_t1: float) -> Value:
    """Backlog/RPO growth (MKT-BACK-015): `(Backlog_t - Backlog_t-1) /
    Backlog_t-1`."""
    if backlog_t1 <= 0:
        return _null(NullState.NOT_MEANINGFUL, "pct", "BACKLOG_GROWTH_NONPOSITIVE_PRIOR")
    return _ok((backlog_t - backlog_t1) / backlog_t1, unit="pct")


def revenue_coverage(ntm_contracted: float, ntm_revenue_estimate: float) -> Value:
    """Revenue coverage (MKT-COVER-016): `NTM contracted backlog/RPO / NTM
    revenue estimate`. Only the convertible/recognized portion should be
    passed (FORMULAS.md)."""
    if ntm_revenue_estimate <= 0:
        return _null(NullState.NOT_MEANINGFUL, "ratio", "REVENUE_COVERAGE_NONPOSITIVE_ESTIMATE")
    return _ok(ntm_contracted / ntm_revenue_estimate, unit="ratio")


# ============================================================================
# MKT-OPLEV-017 / MKT-INCM-018: operating leverage
# ============================================================================


def operating_leverage(pct_change_oi: float, pct_change_rev: float) -> Value:
    """Operating leverage (MKT-OPLEV-017): `%change(operating income) /
    %change(revenue)`. Not meaningful across a loss sign change or a
    zero-revenue-change base (FORMULAS.md: "use incremental margin
    instead" in the sign-change case)."""
    if pct_change_rev == 0:
        return _null(NullState.NOT_MEANINGFUL, "ratio", "OPERATING_LEVERAGE_ZERO_REVENUE_CHANGE")
    return _ok(pct_change_oi / pct_change_rev, unit="ratio")


def incremental_operating_margin(delta_oi: float, delta_rev: float) -> Value:
    """Incremental operating margin (MKT-INCM-018): `change(operating
    income) / change(revenue)`."""
    if delta_rev == 0:
        return _null(NullState.NOT_MEANINGFUL, "pct", "INCREMENTAL_MARGIN_ZERO_REVENUE_CHANGE")
    return _ok(delta_oi / delta_rev, unit="pct")


# ============================================================================
# MKT-CAT-019 / MKT-TDEC-020: catalysts
# ============================================================================


def time_decay(months_to_event: float) -> Value:
    """Catalyst time-decay factor (MKT-TDEC-020): `exp(-ln(2) *
    months_to_event / 12)` -- a 12-month half-life (FORMULAS.md, verbatim;
    brief's worked example: `months=12 -> 0.5`)."""
    return _ok(math.exp(-math.log(2) * months_to_event / 12), unit="ratio")


def catalyst_expected_impact(
    probability: float, impact: float, evidence_quality: float, time_decay_factor: float
) -> Value:
    """Catalyst expected financial impact (MKT-CAT-019): `Probability *
    Estimated annual financial impact * Evidence quality * Time-decay
    factor`. Probability and impact are explicit scenario assumptions,
    never disguised as reported facts (FORMULAS.md)."""
    return _ok(
        probability * impact * evidence_quality * time_decay_factor,
        unit="usd", evidence_class=EvidenceClass.A,
    )


# ============================================================================
# MKT-ADOPT-021 / MKT-ARPU-022: adoption curve, ARPU
# ============================================================================


def adoption_penetration(current_units: float, eventual_units: float) -> Value:
    """Adoption-curve penetration (MKT-ADOPT-021): `Current units /
    Estimated eventual addressable units`. Eventual market is an
    assumption (FORMULAS.md)."""
    if eventual_units <= 0:
        return _null(NullState.NOT_MEANINGFUL, "pct", "ADOPTION_NONPOSITIVE_EVENTUAL_UNITS")
    return _ok(current_units / eventual_units, unit="pct", evidence_class=EvidenceClass.A)


def arpu_growth(arpu_t: float, arpu_t1: float) -> Value:
    """ARPU growth (MKT-ARPU-022): `(ARPU_t - ARPU_t-1) / ARPU_t-1`."""
    if arpu_t1 <= 0:
        return _null(NullState.NOT_MEANINGFUL, "pct", "ARPU_GROWTH_NONPOSITIVE_PRIOR")
    return _ok((arpu_t - arpu_t1) / arpu_t1, unit="pct")


# ============================================================================
# MKT-SECB-023 / MKT-RSG-024: sector breadth and relative strength
# ============================================================================


def sector_breadth(above_50dma_count: int, valid_members: int) -> Value:
    """Sector breadth (MKT-SECB-023): `Members above 50DMA / Valid sector
    members`. `Packet` carries no per-constituent panel (only a sector
    index), so `run()` can only populate this from `overlay`."""
    if valid_members <= 0:
        return _null(NullState.NOT_MEANINGFUL, "pct", "SECTOR_BREADTH_ZERO_VALID_MEMBERS")
    return _ok(above_50dma_count / valid_members, unit="pct")


def sector_relative_strength(sector_close: pd.Series, benchmark_close: pd.Series, n: int) -> Value:
    """Sector relative strength (MKT-RSG-024): sector N-day total return
    minus broad-benchmark N-day total return. Reuses
    `wbj.engines.indicators.relative_strength` directly (Task 11's engine)
    rather than reimplementing the return math -- context, not a company
    technical score (FORMULAS.md)."""
    if len(sector_close) <= n or len(benchmark_close) <= n:
        return _null(NullState.MISSING, "pp", "SECTOR_RELATIVE_STRENGTH_INSUFFICIENT_HISTORY")
    series = ind.relative_strength(sector_close, benchmark_close, n)
    last = series.iloc[-1]
    if pd.isna(last):
        return _null(NullState.MISSING, "pp", "SECTOR_RELATIVE_STRENGTH_NAN")
    return _ok(float(last), unit="pp")


# ============================================================================
# MKT-SCEN-025: scenario-weighted outcome (diagnostic, not a dimension member)
# ============================================================================


def scenario_weighted_outcome(scenarios: Sequence[tuple[float, float]], *, tolerance: float = 1e-6) -> Value:
    """Scenario-weighted market outcome (MKT-SCEN-025): `sum(probability_i
    * outcome_i)`. Probabilities must sum to 1 (FORMULAS.md) -- `CONFLICTED`
    when they don't, rather than silently renormalizing."""
    if not scenarios:
        return _null(NullState.MISSING, "usd", "SCENARIO_WEIGHTED_OUTCOME_EMPTY")
    total_p = sum(p for p, _ in scenarios)
    if abs(total_p - 1.0) > tolerance:
        return _null(
            NullState.CONFLICTED, "usd",
            f"SCENARIO_PROBABILITIES_DO_NOT_SUM_TO_1: sum={total_p:.6f}",
        )
    return _ok(sum(p * o for p, o in scenarios), unit="usd", evidence_class=EvidenceClass.A)


def tam_confidence_caps_dimension(confidence_component: float) -> bool:
    """`SCORING.md`: "TAM confidence <60 caps score at 6"."""
    return confidence_component < 60.0


# ============================================================================
# Scoring helper. (SCORING.md's "Gate / cap" column is applied via the shared
# wbj.specialists.common.apply_dimension_cap -- imported above, not local.)
# ============================================================================


def _score_from_anchor(v: Value, anchors: list[tuple[float, float]]) -> float | None:
    if v.is_null:
        return None
    return anchor_score(v.value, anchors)


# ============================================================================
# Verdict
# ============================================================================


def verdict(score10: float) -> str:
    """Market verdict bands (`DECISION_RULES.md`, verbatim)."""
    if score10 >= 8.0:
        return "Large expanding opportunity"
    if score10 >= 6.0:
        return "Attractive with one material uncertainty"
    if score10 >= 4.0:
        return "Mixed or mature opportunity"
    return "Shrinking, saturated, or poorly evidenced"


# ============================================================================
# Envelope assembly
# ============================================================================


class TamSamSom(BaseModel):
    """`envelope.tam_sam_som` (OUTPUT_SCHEMA.md)."""

    model_config = ConfigDict(frozen=True)

    tam: float | None = None
    sam: float | None = None
    som_scenarios: list[float] = Field(default_factory=list)
    source_tier: int | None = None


class MarketOutput(SpecialistOutput):
    """`market_analysis`'s `run()` result: the shared envelope plus
    OUTPUT_SCHEMA.md's market-specific extension fields."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    market_definition: str | None = None
    tam_sam_som: TamSamSom = Field(default_factory=TamSamSom)
    penetration_and_share: dict[str, Any] = Field(default_factory=dict)
    revision_dashboard: dict[str, Any] = Field(default_factory=dict)
    catalysts: list[dict[str, Any]] = Field(default_factory=list)
    growth_capacity_check: dict[str, Any] = Field(default_factory=dict)
    three_growth_thesis_killers: list[str] = Field(default_factory=list)


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
    v = row.get(key)
    return float(v) if isinstance(v, (int, float)) else None


def _annual_rows(packet: Packet) -> list[dict]:
    rows = packet.fundamentals.get("annual") or []
    return list(reversed(rows))  # ascending, per DATASET.md


def _reinvestment_rate_and_roic(annual: list[dict]) -> tuple[float | None, float | None]:
    """Locally derived reinvestment rate and ROIC from
    `packet.fundamentals.annual` -- for `MKT-GCAP-009` only. Deliberately
    self-contained (not imported from `business.py`/`valuation_engine`)
    per the module docstring's "reuse the engine, not another specialist"
    line: `market_analysis` runs independently of `business_analysis`
    (CLAUDE.md: "Ningun agente ve ni altera el score de otro"), and this
    is a two-line formula, not worth a cross-specialist coupling."""
    if len(annual) < 2:
        return None, None
    ebit_latest = _num(annual[-1], "ebit")
    capex_latest = abs(_num(annual[-1], "capex") or 0.0)
    debt_latest, equity_latest = _num(annual[-1], "total_debt"), _num(annual[-1], "total_equity")
    debt_begin, equity_begin = _num(annual[-2], "total_debt"), _num(annual[-2], "total_equity")
    if None in (ebit_latest, debt_latest, equity_latest, debt_begin, equity_begin):
        return None, None
    pretax = _num(annual[-1], "income_before_tax")
    tax_expense = _num(annual[-1], "income_tax_expense")
    tax_rate = min(max(tax_expense / pretax, 0.0), 1.0) if pretax and pretax > 0 and tax_expense is not None else 0.21
    cash_latest = _num(annual[-1], "cash") or 0.0
    cash_begin = _num(annual[-2], "cash") or 0.0
    nopat_value = ebit_latest * (1 - tax_rate)
    ic_end = debt_latest + equity_latest - cash_latest
    ic_begin = debt_begin + equity_begin - cash_begin
    avg_ic = (ic_end + ic_begin) / 2
    if avg_ic <= 0 or nopat_value == 0:
        return None, None
    reinvestment_rate = capex_latest / nopat_value
    roic_value = nopat_value / avg_ic
    return reinvestment_rate, roic_value


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

    # ---- MKT-TAM-001..MKT-CAGR-004: TAM/SAM/SOM (overlay only) ----
    tam_value = overlay.get("tam")
    v_tam = _ok(float(tam_value), unit="usd", evidence_class=EvidenceClass.E) if tam_value is not None else _null(
        NullState.MISSING, "usd", "TAM_UNAVAILABLE"
    )
    add("MKT-TAM-001", v_tam, None)
    ctx["tam"] = v_tam.value if v_tam.is_valid else None

    sam_inputs = overlay.get("sam_inputs")
    if sam_inputs and v_tam.is_valid:
        v_sam = sam(v_tam.value, sam_inputs["geography_share"], sam_inputs["product_share"], sam_inputs["reachable_share"])
    else:
        v_sam = _null(NullState.MISSING, "usd", "SAM_UNAVAILABLE")
    add("MKT-SAM-002", v_sam, None)

    som_inputs = overlay.get("som_inputs")
    if som_inputs and v_sam.is_valid:
        v_som = som(v_sam.value, som_inputs["target_share"])
    else:
        v_som = _null(NullState.MISSING, "usd", "SOM_UNAVAILABLE")
    add("MKT-SOM-003", v_som, None)

    tam_history = overlay.get("tam_history")
    if tam_history and len(tam_history) >= 2:
        v_tam_cagr = tam_cagr(tam_history[-1], tam_history[0], float(len(tam_history) - 1))
    else:
        v_tam_cagr = _null(NullState.MISSING, "pct", "TAM_CAGR_UNAVAILABLE")
    add("MKT-CAGR-004", v_tam_cagr, _score_from_anchor(v_tam_cagr, [(-0.05, 0), (0.0, 3), (0.10, 7), (0.20, 10)]))

    # ---- MKT-PEN-005: penetration ----
    company_relevant_revenue = overlay.get("company_relevant_revenue")
    if company_relevant_revenue is not None and v_tam.is_valid:
        v_pen = penetration(float(company_relevant_revenue), v_tam.value)
    else:
        v_pen = _null(NullState.MISSING, "pct", "PENETRATION_UNAVAILABLE")
    add("MKT-PEN-005", v_pen, _score_from_anchor(v_pen, [(0.50, 0), (0.20, 4), (0.05, 8), (0.0, 10)]))

    # ---- MKT-SHARE-006 / MKT-SHDELTA-007: market share (overlay only; PROHIBITED_IMPUTATION) ----
    share_inputs = overlay.get("share")
    if share_inputs and {"company_sales", "total_market_sales"} <= share_inputs.keys():
        v_share = market_share(share_inputs["company_sales"], share_inputs["total_market_sales"])
    else:
        v_share = _null(NullState.MISSING, "pct", "MARKET_SHARE_UNAVAILABLE_PROHIBITED_IMPUTATION")
    add("MKT-SHARE-006", v_share, _score_from_anchor(v_share, [(0.0, 2), (0.10, 5), (0.25, 8), (0.50, 10)]))

    share_history = overlay.get("share_history")
    if share_history and len(share_history) >= 2:
        v_shdelta = market_share_delta(share_history[-1], share_history[-2])
    else:
        v_shdelta = _null(NullState.MISSING, "pp", "MARKET_SHARE_DELTA_UNAVAILABLE")
    add("MKT-SHDELTA-007", v_shdelta, _score_from_anchor(v_shdelta, [(-0.02, 0), (0.0, 5), (0.01, 8), (0.03, 10)]))

    # ---- MKT-HHI-008: industry HHI (overlay only) ----
    competitor_shares = overlay.get("competitor_shares")
    v_hhi = industry_hhi(competitor_shares) if competitor_shares else _null(NullState.MISSING, "ratio", "INDUSTRY_HHI_UNAVAILABLE")
    add("MKT-HHI-008", v_hhi, None)

    # ---- MKT-GCAP-009: fundamental growth capacity (derived from packet) ----
    reinvestment_rate, roic_value = _reinvestment_rate_and_roic(annual)
    if reinvestment_rate is not None and roic_value is not None:
        v_gcap = growth_capacity(reinvestment_rate, roic_value)
    else:
        v_gcap = _null(NullState.MISSING, "pct", "GROWTH_CAPACITY_INPUTS_UNAVAILABLE")
    add("MKT-GCAP-009", v_gcap, _score_from_anchor(v_gcap, [(0.0, 0), (0.05, 4), (0.10, 7), (0.20, 10)]))
    ctx["growth_capacity"] = v_gcap.value if v_gcap.is_valid else None

    # ---- MKT-RUN-010: runway years (overlay only) ----
    target_revenue = overlay.get("target_revenue")
    current_revenue = overlay.get("current_revenue")
    assumed_growth = overlay.get("assumed_growth")
    if None not in (target_revenue, current_revenue, assumed_growth):
        v_run = runway_years(float(target_revenue), float(current_revenue), float(assumed_growth))
    else:
        v_run = _null(NullState.MISSING, "years", "RUNWAY_INPUTS_UNAVAILABLE")
    add("MKT-RUN-010", v_run, _score_from_anchor(v_run, [(15.0, 0), (8.0, 5), (4.0, 8), (1.0, 10)]))
    ctx["assumed_growth"] = float(assumed_growth) if assumed_growth is not None else None

    # ---- MKT-REVBR-011..MKT-SURP-014: estimate revisions (overlay only) ----
    est = overlay.get("estimates") or {}
    upward, total = est.get("upward"), est.get("total")
    if upward is not None and total is not None:
        v_revbr = revision_breadth(int(upward), int(total))
    else:
        v_revbr = _null(NullState.NOT_SCORABLE, "pct", "REVISION_BREADTH_UNAVAILABLE")
    add("MKT-REVBR-011", v_revbr, _score_from_anchor(v_revbr, [(0.30, 0), (0.50, 5), (0.70, 8), (0.90, 10)]))

    cur_c, prior_c = est.get("current_consensus"), est.get("prior_consensus")
    if cur_c is not None and prior_c is not None:
        v_revmag = revision_magnitude(cur_c, prior_c)
    else:
        v_revmag = _null(NullState.MISSING, "pct", "REVISION_MAGNITUDE_UNAVAILABLE")
    add("MKT-REVMAG-012", v_revmag, _score_from_anchor(v_revmag, [(-0.10, 0), (0.0, 5), (0.05, 8), (0.15, 10)]))

    individual = est.get("individual_estimates")
    v_disp = estimate_dispersion(individual) if individual else _null(NullState.MISSING, "ratio", "DISPERSION_UNAVAILABLE")
    add("MKT-DISP-013", v_disp, _score_from_anchor(v_disp, [(0.0, 10), (0.05, 8), (0.15, 4), (0.30, 0)]))

    actual, pre_release = est.get("actual"), est.get("pre_release_consensus")
    snapshot_before = est.get("snapshot_before_release", True)
    if actual is not None and pre_release is not None:
        v_surp = earnings_surprise(actual, pre_release, snapshot_before_release=bool(snapshot_before))
    else:
        v_surp = _null(NullState.MISSING, "pct", "SURPRISE_UNAVAILABLE")
    add("MKT-SURP-014", v_surp, _score_from_anchor(v_surp, [(-0.10, 0), (0.0, 5), (0.03, 8), (0.10, 10)]))

    # ---- MKT-BACK-015 / MKT-COVER-016: backlog (overlay only) ----
    backlog_history = overlay.get("backlog_history")
    if backlog_history and len(backlog_history) >= 2:
        v_back = backlog_growth(backlog_history[-1], backlog_history[-2])
    else:
        v_back = _null(NullState.MISSING, "pct", "BACKLOG_GROWTH_UNAVAILABLE")
    add("MKT-BACK-015", v_back, _score_from_anchor(v_back, [(-0.10, 0), (0.0, 5), (0.10, 8), (0.25, 10)]))

    ntm_contracted, ntm_estimate = overlay.get("ntm_contracted"), overlay.get("ntm_revenue_estimate")
    if ntm_contracted is not None and ntm_estimate is not None:
        v_cover = revenue_coverage(ntm_contracted, ntm_estimate)
    else:
        v_cover = _null(NullState.MISSING, "ratio", "REVENUE_COVERAGE_UNAVAILABLE")
    add("MKT-COVER-016", v_cover, _score_from_anchor(v_cover, [(0.0, 0), (0.30, 5), (0.60, 8), (0.90, 10)]))

    # ---- MKT-OPLEV-017 / MKT-INCM-018: operating leverage (from packet) ----
    ebit_hist = [_num(r, "ebit") for r in annual]
    rev_hist = [_num(r, "revenue") for r in annual]
    if len(ebit_hist) >= 2 and None not in (ebit_hist[-1], ebit_hist[-2], rev_hist[-1], rev_hist[-2]) and ebit_hist[-2] != 0 and rev_hist[-2] != 0:
        pct_oi = (ebit_hist[-1] - ebit_hist[-2]) / abs(ebit_hist[-2])
        pct_rev = (rev_hist[-1] - rev_hist[-2]) / abs(rev_hist[-2])
        v_oplev = operating_leverage(pct_oi, pct_rev)
        v_incm = incremental_operating_margin(ebit_hist[-1] - ebit_hist[-2], rev_hist[-1] - rev_hist[-2])
    else:
        v_oplev = _null(NullState.MISSING, "ratio", "OPERATING_LEVERAGE_INPUTS_UNAVAILABLE")
        v_incm = _null(NullState.MISSING, "pct", "INCREMENTAL_MARGIN_INPUTS_UNAVAILABLE")
    add("MKT-OPLEV-017", v_oplev, _score_from_anchor(v_oplev, [(0.0, 0), (1.0, 5), (1.5, 8), (2.5, 10)]))
    add("MKT-INCM-018", v_incm, _score_from_anchor(v_incm, [(0.0, 0), (0.20, 5), (0.35, 8), (0.50, 10)]))

    # ---- MKT-CAT-019 / MKT-TDEC-020: catalysts (overlay only) ----
    catalysts_overlay = overlay.get("catalysts") or []
    catalyst_rows: list[dict[str, Any]] = []
    catalyst_impacts: list[float] = []
    any_quantified = False
    for i, c in enumerate(catalysts_overlay):
        months = c.get("months_to_event")
        v_td = time_decay(float(months)) if months is not None else _null(NullState.MISSING, "ratio", "CATALYST_MISSING_TIMING")
        has_pie = {"probability", "impact", "evidence_quality"} <= c.keys()
        if has_pie and v_td.is_valid:
            v_cat = catalyst_expected_impact(c["probability"], c["impact"], c["evidence_quality"], v_td.value)
            any_quantified = True
            catalyst_impacts.append(v_cat.value)
        else:
            v_cat = _null(NullState.NOT_SCORABLE, "usd", "CATALYST_NARRATIVE_ONLY_MISSING_P_IMPACT_EVIDENCE")
            judgment_requests.append(
                JudgmentRequest(
                    request_id=f"market_analysis:catalyst_{i}",
                    agent_id=AGENT_ID,
                    metric_id=f"catalyst_{i}_probability_impact_evidence",
                    question=f"Assess probability, estimated annual financial impact, and evidence "
                    f"quality for catalyst {c.get('event', f'#{i}')!r}.",
                    schema_hint="{probability: 0-1, impact: usd, evidence_quality: 0-1}",
                )
            )
        catalyst_rows.append(
            {
                "event": c.get("event"), "date_window": c.get("months_to_event"),
                "evidence_class": "A" if has_pie else "Q",
                "probability_assumption": c.get("probability"),
                "financial_impact_assumption": c.get("impact"),
            }
        )
    if catalysts_overlay:
        v_cat_agg = _ok(sum(catalyst_impacts), unit="usd", evidence_class=EvidenceClass.A) if catalyst_impacts else _null(
            NullState.NOT_SCORABLE, "usd", "CATALYST_NARRATIVE_ONLY_MISSING_P_IMPACT_EVIDENCE"
        )
        v_td_agg = time_decay(float(catalysts_overlay[0].get("months_to_event", 0.0))) if catalysts_overlay[0].get("months_to_event") is not None else _null(NullState.MISSING, "ratio", "CATALYST_MISSING_TIMING")
    else:
        v_cat_agg = _null(NullState.MISSING, "usd", "CATALYST_REGISTRY_UNAVAILABLE")
        v_td_agg = _null(NullState.MISSING, "ratio", "CATALYST_REGISTRY_UNAVAILABLE")
    add("MKT-CAT-019", v_cat_agg, _score_from_anchor(v_cat_agg, [(0.0, 3), (50.0, 6), (200.0, 10)]) if v_cat_agg.is_valid else None)
    # A near-term catalyst (high time-decay factor) is more actionable for
    # this scoring cycle than a distant one -- documented reading (no
    # explicit 0-10 band in FORMULAS.md for MKT-TDEC-020 itself).
    add("MKT-TDEC-020", v_td_agg, _score_from_anchor(v_td_agg, [(0.0, 0), (0.25, 3), (0.5, 6), (1.0, 10)]))
    ctx["catalyst_rows"] = catalyst_rows
    ctx["any_catalyst_quantified"] = any_quantified
    ctx["has_catalysts"] = bool(catalysts_overlay)

    # ---- MKT-ADOPT-021 / MKT-ARPU-022 (overlay only) ----
    adoption = overlay.get("adoption")
    if adoption and {"current_units", "eventual_units"} <= adoption.keys():
        v_adopt = adoption_penetration(adoption["current_units"], adoption["eventual_units"])
    else:
        v_adopt = _null(NullState.MISSING, "pct", "ADOPTION_UNAVAILABLE")
    add("MKT-ADOPT-021", v_adopt, _score_from_anchor(v_adopt, [(0.50, 0), (0.20, 4), (0.05, 8), (0.0, 10)]))

    arpu_t, arpu_t1 = overlay.get("arpu_t"), overlay.get("arpu_t1")
    if arpu_t is not None and arpu_t1 is not None:
        v_arpu = arpu_growth(arpu_t, arpu_t1)
    else:
        v_arpu = _null(NullState.MISSING, "pct", "ARPU_GROWTH_UNAVAILABLE")
    add("MKT-ARPU-022", v_arpu, _score_from_anchor(v_arpu, [(-0.10, 0), (0.0, 5), (0.05, 8), (0.15, 10)]))

    # ---- MKT-SECB-023 / MKT-RSG-024: sector breadth / relative strength ----
    breadth_overlay = overlay.get("sector_breadth")
    if breadth_overlay and {"above_50dma_count", "valid_members"} <= breadth_overlay.keys():
        v_secb = sector_breadth(breadth_overlay["above_50dma_count"], breadth_overlay["valid_members"])
    else:
        v_secb = _null(NullState.MISSING, "pct", "SECTOR_BREADTH_UNAVAILABLE_NO_CONSTITUENT_PANEL")
    add("MKT-SECB-023", v_secb, _score_from_anchor(v_secb, [(0.20, 0), (0.50, 5), (0.70, 8), (0.90, 10)]))

    sector_rows, benchmark_rows = packet.market_data.sector, packet.market_data.benchmark
    if sector_rows and benchmark_rows:
        sector_close = pd.Series([r.close for r in sector_rows])
        benchmark_close = pd.Series([r.close for r in benchmark_rows])
        v_rsg = sector_relative_strength(sector_close, benchmark_close, 63)
    else:
        v_rsg = _null(NullState.MISSING, "pp", "SECTOR_RELATIVE_STRENGTH_UNAVAILABLE_EMPTY_MARKET_DATA")
    add("MKT-RSG-024", v_rsg, _score_from_anchor(v_rsg, [(-0.10, 0), (0.0, 5), (0.05, 8), (0.15, 10)]))

    # ---- MKT-SCEN-025: scenario-weighted outcome (diagnostic, not a dimension member) ----
    scenarios = overlay.get("scenarios")
    v_scen = scenario_weighted_outcome(scenarios) if scenarios else _null(NullState.MISSING, "usd", "SCENARIOS_UNAVAILABLE")
    add("MKT-SCEN-025", v_scen, None)

    return rows, assumptions, judgment_requests, ctx


def run(packet: Packet, overlay: dict[str, Any] | None = None) -> MarketOutput:
    """Run the market specialist against `packet`, producing a
    `MarketOutput` (max 20 pts): 25 MKT-* metric rows, five weighted
    dimensions, mandatory flags, and OUTPUT_SCHEMA.md's extension fields.

    See the module docstring for `overlay`'s supported keys. Only
    `MKT-OPLEV-017`/`018` (operating leverage) and `MKT-RSG-024` (sector
    relative strength, when `packet.market_data.sector`/`.benchmark` are
    non-empty) are computable from `packet` alone.
    """
    overlay = overlay or {}
    computed, assumptions, judgment_requests, ctx = _compute_all(packet, overlay)
    by_id = {r.metric_id: r for r in computed}

    if packet.analysis.industry_adapter != "default_nonfinancial":
        assumptions.append(
            f"industry_adapter={packet.analysis.industry_adapter!r}: growth-capacity/ROIC results below "
            "should not be trusted for this security type without a sector adapter."
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

    # ---- TAM dimension (5 pts) ----
    # SCORING.md's "primary inputs" column lists formulas #1-8 (TAM, SAM,
    # SOM, CAGR, PEN, SHARE, SHDELTA, HHI) + #21 (ADOPT), but FORMULAS.md
    # gives no numeric BAD/GOOD/EXCELLENT band for a raw TAM/SAM/SOM dollar
    # figure or a bare HHI value (its scoring direction is genuinely
    # ambiguous: a low HHI is both "more room to consolidate" and "more
    # competitive" -- FORMULAS.md only says "higher means more
    # concentration", not which is better for THIS company's thesis).
    # Those four (MKT-TAM-001/MKT-SAM-002/MKT-SOM-003/MKT-HHI-008) are
    # computed and reported as metric rows (feeding `tam_sam_som`) but are
    # diagnostic, not scored dimension members -- the same
    # core27=False precedent as `financial.py`'s FIN-DX rows. The five
    # members with a defensible directional anchor (CAGR, penetration,
    # share level, share delta, adoption) carry the dimension's weight.
    tam_source_tier = overlay.get("tam_source_tier")
    tam_confidence = TAM_TIER_CONFIDENCE.get(int(tam_source_tier)) if tam_source_tier is not None else None
    # When the caller hasn't supplied `tam_source_tier`, `tam_source_tier_assignment`
    # becomes a JudgmentRequest below -- registered as a sixth, NOT_SCORABLE
    # member of this dimension (equal-weighted with the five mechanical
    # inputs) so a Task 20 judgment answer moves `category.awarded_points`/
    # `coverage`, not just the flat `metrics` row, mirroring `financial.py`'s
    # `_DIMENSION_MEMBERS`-driven equal weighting. When a tier *is* supplied,
    # there is no judgment to score and the dimension keeps its original
    # five equally-weighted members (the tier instead feeds the confidence
    # cap below, as before).
    tam_needs_judgment = tam_confidence is None
    tam_members = ("MKT-CAGR-004", "MKT-PEN-005", "MKT-SHARE-006", "MKT-SHDELTA-007", "MKT-ADOPT-021")
    tam_weight = 1 / (len(tam_members) + 1) if tam_needs_judgment else 1 / len(tam_members)
    tam_scores: list[tuple[float, Value]] = []
    for mid in tam_members:
        s = by_id[mid].score10
        tam_scores.append((tam_weight, Value.of(s, unit="score") if s is not None else Value.null(NullState.NOT_SCORABLE, unit="score")))
    tam_source_tier_slot_index: int | None = None
    if tam_needs_judgment:
        tam_source_tier_slot_index = len(tam_scores)
        tam_scores.append((tam_weight, Value.null(NullState.NOT_SCORABLE, unit="score")))  # tam_source_tier_assignment judgment slot
        judgment_requests.append(
            JudgmentRequest(
                request_id="market_analysis:tam_source_tier_assignment",
                agent_id=AGENT_ID,
                metric_id="tam_source_tier_assignment",
                question="Assign the TAM source-quality tier (1-5, DECISION_RULES.md) based on the "
                "market-size source's methodology and independence.",
                schema_hint="integer 1-5",
            )
        )
    elif tam_confidence_caps_dimension(tam_confidence):
        tam_scores = apply_dimension_cap(tam_scores, cap=6.0)
    tam_dim = Dimension(name=DIM_TAM, max_points=DIMENSION_MAX_POINTS[DIM_TAM], metric_scores=tam_scores)

    # ---- Revisions dimension (4 pts) ----
    revisions_scores: list[tuple[float, Value]] = []
    for mid in ("MKT-REVBR-011", "MKT-REVMAG-012", "MKT-DISP-013", "MKT-SURP-014"):
        s = by_id[mid].score10
        revisions_scores.append((0.25, Value.of(s, unit="score") if s is not None else Value.null(NullState.NOT_SCORABLE, unit="score")))
    revisions_dim = Dimension(name=DIM_REVISIONS, max_points=DIMENSION_MAX_POINTS[DIM_REVISIONS], metric_scores=revisions_scores)

    # ---- Catalysts dimension (4 pts) ----
    catalyst_scores: list[tuple[float, Value]] = []
    for mid in ("MKT-BACK-015", "MKT-COVER-016", "MKT-CAT-019", "MKT-TDEC-020"):
        s = by_id[mid].score10
        catalyst_scores.append((0.25, Value.of(s, unit="score") if s is not None else Value.null(NullState.NOT_SCORABLE, unit="score")))
    narrative_only = ctx.get("has_catalysts") and not ctx.get("any_catalyst_quantified")
    if narrative_only:
        catalyst_scores = apply_dimension_cap(catalyst_scores, cap=3.0)
    catalysts_dim = Dimension(name=DIM_CATALYSTS, max_points=DIMENSION_MAX_POINTS[DIM_CATALYSTS], metric_scores=catalyst_scores)

    # ---- Runway & share dimension (4 pts) ----
    # MKT-HHI-008 excluded from the scored weight for the same reason as
    # the TAM dimension above (no defensible directional band); still
    # computed/reported as a metric row.
    runway_scores: list[tuple[float, Value]] = []
    for mid in (
        "MKT-PEN-005", "MKT-SHARE-006", "MKT-SHDELTA-007",
        "MKT-GCAP-009", "MKT-RUN-010", "MKT-BACK-015", "MKT-ADOPT-021", "MKT-ARPU-022",
    ):
        s = by_id[mid].score10
        runway_scores.append((1 / 8, Value.of(s, unit="score") if s is not None else Value.null(NullState.NOT_SCORABLE, unit="score")))
    runway_dim = Dimension(name=DIM_RUNWAY_SHARE, max_points=DIMENSION_MAX_POINTS[DIM_RUNWAY_SHARE], metric_scores=runway_scores)

    # ---- Operating leverage dimension (3 pts) ----
    oplev_scores: list[tuple[float, Value]] = []
    for mid in ("MKT-OPLEV-017", "MKT-INCM-018", "MKT-SECB-023", "MKT-RSG-024"):
        s = by_id[mid].score10
        oplev_scores.append((0.25, Value.of(s, unit="score") if s is not None else Value.null(NullState.NOT_SCORABLE, unit="score")))
    oplev_dim = Dimension(name=DIM_OPLEV, max_points=DIMENSION_MAX_POINTS[DIM_OPLEV], metric_scores=oplev_scores)

    dimensions = [tam_dim, revisions_dim, catalysts_dim, runway_dim, oplev_dim]

    cat = Category(name=AGENT_ID, max_points=MAX_POINTS, dimensions=dimensions)
    awarded_points = cat.points()
    dim_score10 = cat.score10()
    coverage = cat.coverage()

    mandatory_flags: list[str] = []

    # ---- Forecast-consistency gate: forecast growth vs growth capacity ----
    growth_cap = ctx.get("growth_capacity")
    assumed_growth = ctx.get("assumed_growth")
    if growth_cap is not None and assumed_growth is not None and (assumed_growth - growth_cap) > 0.05:
        mandatory_flags.append("EXTERNAL_CAPITAL_REQUIRED")

    label = verdict(dim_score10)

    judgment_requests.append(
        JudgmentRequest(
            request_id="market_analysis:three_growth_thesis_killers",
            agent_id=AGENT_ID,
            metric_id="three_growth_thesis_killers",
            question="List three events that would invalidate the growth thesis "
            "(DECISION_RULES.md: mandatory, always required).",
            schema_hint="array of exactly 3 strings",
        )
    )

    category = CategoryStats(
        max_points=MAX_POINTS, awarded_points=awarded_points, score_10=dim_score10,
        confidence=_category_confidence(coverage, packet, tam_confidence),
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
    # `tam_source_tier_assignment` is registered as `tam_scores[tam_source_tier_slot_index]`
    # above (only when the judgment was actually requested); recording it here
    # lets `wbj.overlay.merge.merge_overlay` replace that exact slot and
    # rescore once a judgment answer arrives, exactly like `financial.py`'s
    # `FIN-GR-004`/`FIN-GR-005`. The catalyst-assessment and thesis-killer
    # requests have no dimension slot (their answers are dict/array-shaped
    # and never reduce to a single 0-10 score under `merge.py`'s documented
    # answer-scoring rules, so there is no slot for them to move) and are
    # deliberately absent from this map.
    judgment_slots: dict[str, tuple[str, int]] = {}
    if tam_source_tier_slot_index is not None:
        judgment_slots["tam_source_tier_assignment"] = (DIM_TAM, tam_source_tier_slot_index)

    return MarketOutput(
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
        source_lineage=["packet.fundamentals.annual", "packet.market_data.sector", "packet.market_data.benchmark"],
        validation_tests=validation_tests,
        market_definition=None,
        tam_sam_som=TamSamSom(
            tam=ctx.get("tam"), sam=by_id["MKT-SAM-002"].value.value if by_id["MKT-SAM-002"].value.is_valid else None,
            som_scenarios=[], source_tier=tam_source_tier,
        ),
        penetration_and_share={},
        revision_dashboard={},
        catalysts=ctx.get("catalyst_rows", []),
        growth_capacity_check={"growth_capacity": growth_cap, "assumed_growth": assumed_growth},
        three_growth_thesis_killers=[],
    )


def _category_confidence(coverage: float, packet: Packet, tam_confidence: float | None) -> float:
    """Category confidence (0-100) via `wbj.core.confidence.confidence()`.
    Same construction as `financial.py`/`business.py`'s
    `_category_confidence`: real packet signals where available, documented
    constants otherwise. `source_quality` folds in the TAM source tier
    (DECISION_RULES.md's own confidence-component table) when known."""
    coverage_component = max(0.0, min(1.0, coverage)) * 100.0
    source_quality = tam_confidence if tam_confidence is not None else 60.0
    freshness = 100.0 if packet.staleness.get("consensus", "FRESH") == "FRESH" else 50.0
    consistency = 75.0
    model_fit = 90.0 if packet.analysis.industry_adapter == "default_nonfinancial" else 40.0
    return _confidence_formula(
        coverage=coverage_component, source_quality=source_quality, freshness=freshness,
        consistency=consistency, model_fit=model_fit,
    )
