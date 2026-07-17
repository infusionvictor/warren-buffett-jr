"""Financial specialist (Task 14): FIN-001..033, the core-27 diagnostic, and
the five weighted financial dimensions (15 pts).

Sources of truth (`Cerebro/02_financial_analysis/`):
- `FORMULAS.md`: FIN-GR-001..005, FIN-PR-006..011, FIN-CF-012..016,
  FIN-BS-017..022, FIN-EF-023..027 (the 27 "core" formulas) plus
  FIN-DX-028..033 (diagnostics, not part of the core-27 count).
- `DECISION_RULES.md`: the core-27 point math, mandatory overrides, and
  financial verdict bands.
- `SCORING.md`: the five weighted dimensions (3 pts each) and their
  primary-input formula lists.
- `OUTPUT_SCHEMA.md`: `FinancialOutput`'s extension fields.
- `Cerebro/shared/OUTPUT_CONTRACT.md` / `HANDOFF_CONTRACT.md`: the common
  envelope, implemented in `wbj.specialists.common`.

## Band-edge convention

`FORMULAS.md` phrases every explicit band as e.g. "BAD <0%; GOOD 0-10%;
EXCELLENT >10%": the GOOD band is closed (`[lo, hi]`) and EXCELLENT is the
open ray strictly beyond `hi` — `x == hi` scores GOOD, not EXCELLENT (the
task brief's own worked example: `yoy=0.10 -> GOOD not EXCELLENT`).
`band_score` implements this once; every metric below with an explicit
numeric band in `FORMULAS.md` calls it.

## Metrics without an explicit numeric band

About half of the 27 core formulas (`FORMULAS.md`'s "Rules / caveats"
column) describe a *qualitative* three-way classification (e.g. FIN-GR-002
"Negative=decelerating, near zero=stable, positive=accelerating;
materiality threshold 1 percentage point/year") rather than an explicit
`BAD <x; GOOD x-y; EXCELLENT >y` band. Per the task-14 brief ("Register
FIN-001..033 with exact BAD/GOOD/EXCELLENT bands from FORMULAS.md" — for
the ones that don't spell out numbers, a defensible reading is required),
each such metric's docstring below states the specific threshold this
module uses and, where `FORMULAS.md` already anchors a materiality
threshold elsewhere in the same document (the recurring "1 percentage
point/year" trend threshold, and GR-003's "+/-2 percentage-point band" for
a peer/diff comparison), reuses that anchor rather than inventing a new
number. These are this module's own dated (2.0.0) interpretation, not a
value taken directly from a `FORMULAS.md` table cell — flagged individually
below and in the Task 14 commit message.

## Judgment-only metrics

FIN-GR-004 (organic growth quality) and FIN-GR-005 (market-share trend)
require an `organic_growth_bridge` / `market_share_series` dataset
(`DATASET.md`: both "conditional") that Task 10's `Packet` schema does not
carry, and both `organic_growth` and `market_share` are on
`wbj.core.nullstates.PROHIBITED_IMPUTATION` — Cerebro explicitly forbids
inferring them from other reported numbers. `run()` therefore always
returns `NOT_SCORABLE` plus a `JudgmentRequest` for these two, regardless
of `overlay` (there is no mechanical path to a number from this packet
shape). The formula functions themselves remain directly callable and
tested in isolation for the day a packet does carry the bridge/series data.

## WACC

FIN-EF-026 (ROIC vs. cost of capital) needs a WACC, which `DATASET.md`
sources from "the valuation agent input packet" (Task 13's engine) — not
part of `wbj.schemas.packet.Packet`. `run(packet, overlay)` accepts an
optional `overlay["wacc"]` (a plain float, e.g. supplied by the Task 21
aggregator once the valuation specialist has run); without it, FIN-EF-026
and DECISION_RULES.md's mandatory Override 2 (`ROIC < WACC` -> category
cannot be `Excellent`) are both skipped (`MISSING`, not `NOT_SCORABLE` —
this is an ordinary missing external input, not a judgment call), and an
assumption note records the omission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from wbj.core.confidence import confidence as _confidence_formula
from wbj.core.formulas import cagr, register_formula, yoy
from wbj.core.nullstates import EvidenceClass, NullState, Value
from wbj.core.scoring import Category, Dimension
from wbj.schemas.packet import Packet
from wbj.specialists.common import (
    CategoryStats,
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
    "DIMENSION_NAMES",
    "CORE_27_IDS",
    "Core27Summary",
    "FinancialOutput",
    "band_score",
    "core27_diagnostic",
    "capped_verdict",
    "verdict",
    "run",
    # formula functions (Value-returning, direct-callable, VALIDATION_TESTS-facing)
    "yoy_revenue_growth",
    "revenue_growth_trend",
    "growth_vs_peers",
    "organic_growth_quality",
    "market_share_trend",
    "net_profit_status",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "margin_trend",
    "profit_vs_revenue_growth",
    "free_cash_flow",
    "fcf_growth",
    "fcf_margin",
    "cash_vs_earnings",
    "capital_dependence",
    "current_ratio",
    "quick_ratio",
    "debt_to_equity",
    "interest_coverage",
    "debt_vs_revenue_trend",
    "liquidity_trend",
    "roe",
    "roic",
    "roa",
    "roic_vs_wacc",
    "return_trend",
    "net_debt_to_ebitda",
    "debt_to_fcf",
    "accrual_ratio",
    "cash_conversion_cycle",
    "sbc_to_revenue",
    "diluted_share_cagr",
]

_VERSION = "2.0.0"
AGENT_ID = "financial_analysis"
MAX_POINTS = 15.0

DIM_REVENUE = "revenue_quality_and_growth"
DIM_EPS_FCF = "eps_and_free_cash_flow"
DIM_MARGINS = "margins"
DIM_BALANCE = "balance_and_liquidity"
DIM_RETURNS = "cash_conversion_and_capital_efficiency"

DIMENSION_NAMES = (DIM_REVENUE, DIM_EPS_FCF, DIM_MARGINS, DIM_BALANCE, DIM_RETURNS)

# The 27 core formulas, in FORMULAS.md order. FIN-DX-028..033 are
# diagnostics and deliberately excluded (DECISION_RULES.md: "Each of the
# 27 core formulas...").
CORE_27_IDS: tuple[str, ...] = (
    "FIN-GR-001", "FIN-GR-002", "FIN-GR-003", "FIN-GR-004", "FIN-GR-005",
    "FIN-PR-006", "FIN-PR-007", "FIN-PR-008", "FIN-PR-009", "FIN-PR-010", "FIN-PR-011",
    "FIN-CF-012", "FIN-CF-013", "FIN-CF-014", "FIN-CF-015", "FIN-CF-016",
    "FIN-BS-017", "FIN-BS-018", "FIN-BS-019", "FIN-BS-020", "FIN-BS-021", "FIN-BS-022",
    "FIN-EF-023", "FIN-EF-024", "FIN-EF-025", "FIN-EF-026", "FIN-EF-027",
)
assert len(CORE_27_IDS) == 27

# Dimension membership (SCORING.md "Primary inputs"). FIN-CF-015 legitimately
# appears in both EPS_FCF and RETURNS (SCORING.md lists it under both
# dimensions' primary inputs) -- the core-27 diagnostic still counts it once.
#
# SCORING.md lists two dimensions with a trailing "+ <extra>" input beyond
# their numbered core metrics:
#   - RETURNS: "FIN-CF-015 and FIN-EF-023..027 + dilution" -> the dilution
#     diagnostics FIN-DX-032 (SBC/revenue) and FIN-DX-033 (diluted-share
#     CAGR) are included here and scored (see `band_sbc_to_revenue` /
#     `band_diluted_share_cagr`). SCORING.md's own dimension rule names
#     "heavy dilution" as a 0-3 (BAD) signal, so they must carry weight.
#   - BALANCE: "FIN-BS-017..022 + diagnostics" -> the balance diagnostics
#     FIN-DX-028 (net-debt/EBITDA) and FIN-DX-029 (debt/FCF) are
#     deliberately NOT added as scored members. FORMULAS.md itself labels
#     FIN-DX-028 "Diagnostic and risk-agent input" (it primarily feeds the
#     risk specialist, Task 18) and both are frequently NOT_MEANINGFUL /
#     unavailable from this packet's field set (no D&A field -> no EBITDA);
#     folding an always-MISSING metric into the weighted score would only
#     depress the balance dimension's coverage without adding signal. They
#     remain computed and reported (in `balance_and_maturities`), just not
#     as weighted dimension inputs -- documented here rather than silently.
# Diagnostics stay `core27=False`: including a dilution metric in a
# dimension does NOT add it to the 27-metric core diagnostic count.
_DIMENSION_MEMBERS: dict[str, tuple[str, ...]] = {
    DIM_REVENUE: ("FIN-GR-001", "FIN-GR-002", "FIN-GR-003", "FIN-GR-004", "FIN-GR-005"),
    DIM_EPS_FCF: (
        "FIN-PR-006", "FIN-PR-011",
        "FIN-CF-012", "FIN-CF-013", "FIN-CF-014", "FIN-CF-015", "FIN-CF-016",
    ),
    DIM_MARGINS: ("FIN-PR-007", "FIN-PR-008", "FIN-PR-009", "FIN-PR-010"),
    DIM_BALANCE: ("FIN-BS-017", "FIN-BS-018", "FIN-BS-019", "FIN-BS-020", "FIN-BS-021", "FIN-BS-022"),
    DIM_RETURNS: (
        "FIN-CF-015", "FIN-EF-023", "FIN-EF-024", "FIN-EF-025", "FIN-EF-026", "FIN-EF-027",
        "FIN-DX-032", "FIN-DX-033",
    ),
}


def _ok(x: float, unit: str, **lineage: object) -> Value:
    return Value.of(x, unit=unit, evidence_class=lineage.pop("evidence_class", EvidenceClass.C), **lineage)


def _null(state: NullState, unit: str, *warnings: str) -> Value:
    return Value.null(state, unit=unit, warnings=list(warnings))


# ============================================================================
# band_score: the shared BAD/GOOD/EXCELLENT -> 0/1/2 mapping
# ============================================================================


def band_score(x: float, lo: float, hi: float, *, higher_is_better: bool = True) -> int:
    """0 (BAD) / 1 (GOOD) / 2 (EXCELLENT) from a closed-GOOD band `[lo, hi]`.

    `higher_is_better=True` (e.g. FIN-GR-001 "BAD <0%; GOOD 0-10%; EXCELLENT
    >10%"): `x < lo` -> BAD, `lo <= x <= hi` -> GOOD, `x > hi` -> EXCELLENT.

    `higher_is_better=False` (e.g. FIN-BS-019 D/E "BAD >2.0; GOOD 1.0-2.0;
    EXCELLENT <1.0"): `x > hi` -> BAD, `lo <= x <= hi` -> GOOD, `x < lo` ->
    EXCELLENT.

    Either way the GOOD band is closed at both ends and the worse-than-GOOD
    tier is the open ray excluding its boundary -- `x == hi` (or `x == lo`)
    always lands in GOOD, never in the adjacent tier. See the module
    docstring's "Band-edge convention" for the FORMULAS.md phrasing this
    encodes.
    """
    if higher_is_better:
        if x < lo:
            return 0
        if x <= hi:
            return 1
        return 2
    else:
        if x > hi:
            return 0
        if x >= lo:
            return 1
        return 2


def _ols_slope(y: Sequence[float]) -> float:
    """OLS slope of `y` against its own index (0, 1, 2, ...) -- "per period"
    slope for an equally-spaced annual series, per FORMULAS.md's repeated
    "OLS slope ... over 3 years" phrasing."""
    n = len(y)
    x = np.arange(n, dtype=float)
    slope, _intercept = np.polyfit(x, np.asarray(y, dtype=float), 1)
    return float(slope)


# ============================================================================
# FIN-GR-001..005: Revenue quality and growth
# ============================================================================


@register_formula(id="FIN-GR-001", version=_VERSION, unit="pct", inputs=["revenue_cur", "revenue_prior"])
def yoy_revenue_growth(revenue_cur: float, revenue_prior: float) -> Value:
    """YoY revenue growth (FIN-GR-001): `(Revenue_t - Revenue_t-1) /
    Revenue_t-1`. Band: BAD <0%; GOOD 0-10%; EXCELLENT >10%."""
    return yoy(revenue_cur, revenue_prior)


def band_yoy_revenue_growth(pct: float) -> int:
    return band_score(pct, 0.0, 0.10)


@register_formula(id="FIN-GR-002", version=_VERSION, unit="pct_per_period", inputs=["growth_rates"])
def revenue_growth_trend(growth_rates: Sequence[float]) -> Value:
    """Revenue-growth trend (FIN-GR-002): OLS slope of >=3 annual growth
    rates. Requires >=3 points per FORMULAS.md's "three annual growth
    rates"."""
    if len(growth_rates) < 3:
        return _null(NullState.MISSING, "pct_per_period", "GROWTH_TREND_NEEDS_3_POINTS")
    return _ok(_ols_slope(growth_rates), unit="pct_per_period")


def band_revenue_growth_trend(slope: float) -> int:
    """Materiality threshold 1 percentage point/year (FORMULAS.md, verbatim)."""
    return band_score(slope, -0.01, 0.01)


@register_formula(
    id="FIN-GR-003", version=_VERSION, unit="pct", inputs=["company_growth", "peer_median_growth"],
)
def growth_vs_peers(company_growth: float, peer_median_growth: float) -> Value:
    """Growth vs. competitors (FIN-GR-003): `company growth - peer median
    growth`. Band: +/-2 percentage-point band (FORMULAS.md, verbatim)."""
    return _ok(company_growth - peer_median_growth, unit="pct")


def band_growth_vs_peers(diff: float) -> int:
    return band_score(diff, -0.02, 0.02)


@register_formula(
    id="FIN-GR-004", version=_VERSION, unit="ratio", inputs=["organic_growth", "total_growth"],
)
def organic_growth_quality(organic_growth: float, total_growth: float) -> Value:
    """Organic growth quality (FIN-GR-004): `organic growth / total
    reported growth`. Per FORMULAS.md: "If total growth <=0, classify from
    bridge rather than ratio" -- this module refuses (`NOT_MEANINGFUL`)
    rather than dividing by a non-positive/immaterial total growth, since
    "classify from bridge" needs judgment this formula alone can't supply.
    Judgment-only for `run()` regardless (see module docstring): `Packet`
    never carries `organic_growth_bridge`."""
    if total_growth <= 0:
        return _null(NullState.NOT_MEANINGFUL, "ratio", "ORGANIC_GROWTH_TOTAL_GROWTH_NONPOSITIVE")
    return _ok(organic_growth / total_growth, unit="ratio")


@register_formula(id="FIN-GR-005", version=_VERSION, unit="pct_per_period", inputs=["shares"])
def market_share_trend(shares: Sequence[float]) -> Value:
    """Market-share trend (FIN-GR-005): OLS slope of >=3 annual share
    observations. Judgment-only for `run()` (see module docstring):
    `Packet` never carries `market_share_series`."""
    if len(shares) < 3:
        return _null(NullState.MISSING, "pct_per_period", "MARKET_SHARE_TREND_NEEDS_3_POINTS")
    return _ok(_ols_slope(shares), unit="pct_per_period")


# ============================================================================
# FIN-PR-006..011: Profitability
# ============================================================================


@register_formula(
    id="FIN-PR-006", version=_VERSION, unit="usd", inputs=["net_income_history"],
)
def net_profit_status(net_income_history: Sequence[float]) -> Value:
    """Net profit status (FIN-PR-006): the latest normalized net income,
    classified Loss / small-volatile / strong-consistent by
    `net_profit_status_band` (FORMULAS.md gives no numeric band for this
    row -- see that function for the documented classification rule)."""
    if not net_income_history:
        return _null(NullState.MISSING, "usd", "NET_INCOME_HISTORY_EMPTY")
    return _ok(net_income_history[-1], unit="usd")


def net_profit_status_band(net_income_history: Sequence[float]) -> int:
    """FORMULAS.md gives FIN-PR-006 no numeric band ("Loss, small/volatile
    profit, or strong consistent profit" only). This module's reading:
    BAD if the latest net income is <=0 (a loss); EXCELLENT if net income
    was positive in every one of the last 3 years *and* its
    coefficient of variation over that window is <15% (consistent, not
    volatile); GOOD otherwise (positive but volatile, or too little
    history to judge consistency)."""
    if not net_income_history or net_income_history[-1] <= 0:
        return 0
    window = list(net_income_history[-3:])
    if len(window) < 3 or any(v <= 0 for v in window):
        return 1
    mean = sum(window) / len(window)
    if mean == 0:
        return 1
    cv = (sum((v - mean) ** 2 for v in window) / len(window)) ** 0.5 / abs(mean)
    return 2 if cv < 0.15 else 1


@register_formula(id="FIN-PR-007", version=_VERSION, unit="pct", inputs=["revenue", "cogs"])
def gross_margin(revenue: float, cogs: float) -> Value:
    """Gross margin (FIN-PR-007): `(Revenue - COGS) / Revenue`. Band: BAD
    <20%; GOOD 20-40%; EXCELLENT >40%."""
    if revenue == 0:
        return _null(NullState.NOT_MEANINGFUL, "pct", "GROSS_MARGIN_ZERO_REVENUE")
    return _ok((revenue - cogs) / revenue, unit="pct")


def band_gross_margin(pct: float) -> int:
    return band_score(pct, 0.20, 0.40)


@register_formula(id="FIN-PR-008", version=_VERSION, unit="pct", inputs=["ebit", "revenue"])
def operating_margin(ebit: float, revenue: float) -> Value:
    """Operating margin (FIN-PR-008): `Normalized EBIT / Revenue`. Band:
    BAD <10%; GOOD 10-20%; EXCELLENT >20%."""
    if revenue == 0:
        return _null(NullState.NOT_MEANINGFUL, "pct", "OPERATING_MARGIN_ZERO_REVENUE")
    return _ok(ebit / revenue, unit="pct")


def band_operating_margin(pct: float) -> int:
    return band_score(pct, 0.10, 0.20)


@register_formula(id="FIN-PR-009", version=_VERSION, unit="pct", inputs=["net_income", "revenue"])
def net_margin(net_income: float, revenue: float) -> Value:
    """Net margin (FIN-PR-009): `Normalized net income / Revenue`. Band:
    BAD <5%; GOOD 5-10%; EXCELLENT >10%."""
    if revenue == 0:
        return _null(NullState.NOT_MEANINGFUL, "pct", "NET_MARGIN_ZERO_REVENUE")
    return _ok(net_income / revenue, unit="pct")


def band_net_margin(pct: float) -> int:
    return band_score(pct, 0.05, 0.10)


@register_formula(id="FIN-PR-010", version=_VERSION, unit="pct_per_period", inputs=["margins"])
def margin_trend(margins: Sequence[float]) -> Value:
    """Margin trend (FIN-PR-010): OLS slope of >=3 annual margins. This
    module tracks *net* margin's trend (the dimension already tracks
    gross/operating/net margin *levels* via FIN-PR-007..009; FORMULAS.md
    doesn't say which margin FIN-PR-010 tracks, and net margin is the
    bottom-line summary figure)."""
    if len(margins) < 3:
        return _null(NullState.MISSING, "pct_per_period", "MARGIN_TREND_NEEDS_3_POINTS")
    return _ok(_ols_slope(margins), unit="pct_per_period")


def band_margin_trend(slope: float) -> int:
    """0.5 percentage-point/year materiality threshold (FORMULAS.md, verbatim)."""
    return band_score(slope, -0.005, 0.005)


@register_formula(
    id="FIN-PR-011", version=_VERSION, unit="pct", inputs=["ni_growth", "revenue_growth"],
)
def profit_vs_revenue_growth(ni_growth: float, revenue_growth: float) -> Value:
    """Profit vs. revenue growth (FIN-PR-011): `net-income growth - revenue
    growth`. Per FORMULAS.md: "not meaningful across a loss-to-profit sign
    change" -- callers must not compute `ni_growth` across a sign flip
    (`wbj.core.formulas.yoy`/`cagr` already refuse those cases upstream).
    Band: reuses GR-003's +/-2 percentage-point convention (FORMULAS.md
    gives no separate number for this row)."""
    return _ok(ni_growth - revenue_growth, unit="pct")


def band_profit_vs_revenue_growth(diff: float) -> int:
    return band_score(diff, -0.02, 0.02)


# ============================================================================
# FIN-CF-012..016: Cash flow
# ============================================================================


@register_formula(id="FIN-CF-012", version=_VERSION, unit="usd", inputs=["ocf", "capex"])
def free_cash_flow(ocf: float, capex: float) -> Value:
    """Free cash flow (FIN-CF-012): `Operating cash flow - Capital
    expenditures` (`capex` is the positive spend amount)."""
    return _ok(ocf - capex, unit="usd")


def free_cash_flow_band(fcf: float, fcf_margin_pct: float | None) -> int:
    """FORMULAS.md gives FIN-CF-012 no independent numeric band (only
    FIN-CF-014's *margin* has one). This module scores the raw dollar
    figure by sign, then reuses FIN-CF-014's margin band to decide
    GOOD-vs-EXCELLENT when a margin is available: BAD if `fcf <= 0`;
    EXCELLENT if `fcf > 0` and the FCF margin already scores EXCELLENT
    (>10%); GOOD otherwise."""
    if fcf <= 0:
        return 0
    if fcf_margin_pct is not None and fcf_margin_pct > 0.10:
        return 2
    return 1


@register_formula(id="FIN-CF-013", version=_VERSION, unit="pct", inputs=["fcf_cur", "fcf_prior"])
def fcf_growth(fcf_cur: float, fcf_prior: float) -> Value:
    """FCF growth (FIN-CF-013): `(FCF_t - FCF_t-1) / abs(FCF_t-1)`. Per
    FORMULAS.md: "If sign changes, report transition rather than
    percentage" -- refuses when `fcf_prior <= 0` (a sign change or
    already-negative base makes the percentage meaningless)."""
    if fcf_prior <= 0:
        return _null(NullState.NOT_MEANINGFUL, "pct", "FCF_GROWTH_PRIOR_NONPOSITIVE_TRANSITION")
    return _ok((fcf_cur - fcf_prior) / abs(fcf_prior), unit="pct")


def band_fcf_growth(pct: float) -> int:
    """Reuses FIN-GR-001's growth band convention (FORMULAS.md gives no
    separate number for this row)."""
    return band_score(pct, 0.0, 0.10)


@register_formula(id="FIN-CF-014", version=_VERSION, unit="pct", inputs=["fcf", "revenue"])
def fcf_margin(fcf: float, revenue: float) -> Value:
    """FCF margin (FIN-CF-014): `FCF / Revenue`. Band: BAD <0%; GOOD
    0-10%; EXCELLENT >10%."""
    if revenue == 0:
        return _null(NullState.NOT_MEANINGFUL, "pct", "FCF_MARGIN_ZERO_REVENUE")
    return _ok(fcf / revenue, unit="pct")


def band_fcf_margin(pct: float) -> int:
    return band_score(pct, 0.0, 0.10)


@register_formula(id="FIN-CF-015", version=_VERSION, unit="ratio", inputs=["ocf_cum", "ni_cum"])
def cash_vs_earnings(ocf_cum: float, ni_cum: float) -> Value:
    """Cash vs. earnings (FIN-CF-015): `Operating cash flow / Net income`,
    cumulative ratio preferred (FORMULAS.md). `ni_cum` <= 0 makes the
    ratio's sign meaningless."""
    if ni_cum <= 0:
        return _null(NullState.NOT_MEANINGFUL, "ratio", "CASH_VS_EARNINGS_NI_NONPOSITIVE")
    return _ok(ocf_cum / ni_cum, unit="ratio")


def band_cash_vs_earnings(ratio: float) -> int:
    """FORMULAS.md gives no exact numbers for "OCF<earnings / approx equal
    / OCF>earnings" -- this module reads "approx equal" as within 10%."""
    return band_score(ratio, 0.9, 1.1)


@register_formula(
    id="FIN-CF-016", version=_VERSION, unit="usd",
    inputs=["cash_uses", "ocf", "cash_begin"],
)
def capital_dependence(cash_uses: float, ocf: float, cash_begin: float) -> Value:
    """Capital dependence (FIN-CF-016): `External financing need = max(0,
    Cash uses - OCF - Beginning excess cash)` (FORMULAS.md, verbatim)."""
    return _ok(max(0.0, cash_uses - ocf - cash_begin), unit="usd")


def capital_dependence_band(need: float, returns_cash: bool) -> int:
    """FORMULAS.md classifies FIN-CF-016 qualitatively ("external
    dependence, self-funded, or funds growth plus returns cash"), no
    numbers. This module's reading: BAD if `need > 0` (external financing
    need); EXCELLENT if `need <= 0` and the company also returned cash
    (`returns_cash`, e.g. dividends/buybacks funded from surplus OCF); GOOD
    if `need <= 0` but no cash was returned (self-funded, not yet
    returning capital)."""
    if need > 0:
        return 0
    return 2 if returns_cash else 1


def is_externally_dependent(need: float, debt_issuance: float, equity_issuance: float) -> bool:
    """External-dependence classification used by the mandatory Override 1
    check (DECISION_RULES.md: "loss + negative FCF + external financing
    need triggers override"). True when either the mechanical financing
    need (`need > 0`) or an actual net capital raise (`debt_issuance > 0`
    or `equity_issuance > 0`) is observed -- FIN-T006 exercises the latter
    ("NI=-10, FCF=-20, equity issuance positive")."""
    return need > 0 or debt_issuance > 0 or equity_issuance > 0


def override_1_triggered(net_income: float | None, fcf: float | None, externally_dependent: bool) -> bool:
    """Mandatory Override 1 (DECISION_RULES.md #1): "If normalized net
    income is negative, FCF is negative, and the company needs external
    financing, category verdict is capped at Bad/Avoid." `None` inputs
    (metric not scorable) never trigger the override."""
    return net_income is not None and fcf is not None and net_income < 0 and fcf < 0 and externally_dependent


def override_2_triggered(roic_value: float | None, wacc_value: float | None) -> bool:
    """Mandatory Override 2 (DECISION_RULES.md #2): "If ROIC < WACC,
    category cannot be Excellent." `None` inputs (ROIC or WACC not
    available) never trigger the override."""
    return roic_value is not None and wacc_value is not None and roic_value < wacc_value


def reconciliation_check(dim_score10: float, core27_score10: float, *, tolerance: float = 1.5) -> str | None:
    """DECISION_RULES.md: "The core diagnostic must reconcile
    directionally; a difference greater than 1.5 points between the
    weighted category score and the 27-metric score requires an
    explanation." Returns the explanatory flag string when the two 0-10
    scores diverge by more than `tolerance`, else `None`."""
    diff = abs(dim_score10 - core27_score10)
    if diff > tolerance:
        return (
            "CORE27_RECONCILIATION_WARNING: weighted dimension score_10="
            f"{dim_score10:.2f} vs core-27 score_10={core27_score10:.2f} "
            f"(diff={diff:.2f} > {tolerance})"
        )
    return None


def verdict(score10: float) -> str:
    """Financial verdict bands (DECISION_RULES.md, verbatim):
    8.0-10.0 Excellent, 6.0-7.99 Good, 4.0-5.99 Mixed/watch, <4.0 Weak."""
    if score10 >= 8.0:
        return "Excellent financial health"
    if score10 >= 6.0:
        return "Good with limited weaknesses"
    if score10 >= 4.0:
        return "Mixed / watch"
    return "Weak / high financial risk"


def capped_verdict(score10: float, *, override_1: bool, override_2: bool) -> str:
    """The verdict label after applying DECISION_RULES.md's mandatory
    overrides *to the label only* (never to `category.awarded_points` --
    see the `SpecialistOutput.verdict` docstring and HANDOFF_CONTRACT.md).

    Override 1 (loss + negative FCF + external dependence) caps the verdict
    at Bad/Avoid (the weakest band); Override 2 (ROIC < WACC) caps it below
    Excellent. Both are expressed as a ceiling on the score fed to
    `verdict()`, so the label is degraded without moving any points.
    """
    effective = score10
    if override_1:
        effective = min(effective, 3.99)  # Bad/Avoid
    if override_2:
        effective = min(effective, 7.99)  # cannot be Excellent
    return verdict(effective)


def core27_diagnostic(bands: Sequence[int | None]) -> tuple[int, float, float, float, float]:
    """The core-27 diagnostic (DECISION_RULES.md), as a pure function of the
    27 core metrics' band points so it is directly testable and is the
    single implementation `run()` calls.

    `bands[i]` is that core metric's 0/1/2 band (BAD/GOOD/EXCELLENT), or
    `None` when the metric is NOT_SCORABLE. Returns `(valid_count, points,
    maximum_valid_points, percent, score_10)` where
    `percent = points / (2 * valid_count) * 100` and
    `score_10 = percent / 10` (both `0.0` when no metric is valid).
    """
    valid = [b for b in bands if b is not None]
    points = float(sum(valid))
    maximum_valid_points = float(2 * len(valid))
    percent = (points / maximum_valid_points * 100.0) if valid else 0.0
    return len(valid), points, maximum_valid_points, percent, percent / 10.0


# ============================================================================
# FIN-BS-017..022: Balance sheet and liquidity
# ============================================================================


@register_formula(
    id="FIN-BS-017", version=_VERSION, unit="ratio", inputs=["current_assets", "current_liabilities"],
)
def current_ratio(current_assets: float, current_liabilities: float) -> Value:
    """Current ratio (FIN-BS-017): `Current assets / Current liabilities`.
    Band: BAD <1.0; GOOD 1.0-1.5; EXCELLENT 1.5-3.0 (>3 carries an
    `IDLE_CAPITAL` caveat, not a different score tier, per FORMULAS.md:
    ">3 may indicate idle capital")."""
    if current_liabilities == 0:
        return _null(NullState.NOT_MEANINGFUL, "ratio", "CURRENT_RATIO_ZERO_LIABILITIES")
    ratio = current_assets / current_liabilities
    warnings = ["CURRENT_RATIO_ABOVE_3_POSSIBLE_IDLE_CAPITAL"] if ratio > 3.0 else []
    return Value.of(ratio, unit="ratio", evidence_class=EvidenceClass.C, warnings=warnings)


def band_current_ratio(ratio: float) -> int:
    return band_score(ratio, 1.0, 1.5)


@register_formula(
    id="FIN-BS-018", version=_VERSION, unit="ratio",
    inputs=["current_assets", "inventory", "current_liabilities"],
)
def quick_ratio(current_assets: float, inventory: float, current_liabilities: float) -> Value:
    """Quick ratio (FIN-BS-018): `(Current assets - Inventory) / Current
    liabilities`. Band: BAD <0.7; GOOD 0.7-1.0; EXCELLENT >1.0."""
    if current_liabilities == 0:
        return _null(NullState.NOT_MEANINGFUL, "ratio", "QUICK_RATIO_ZERO_LIABILITIES")
    return _ok((current_assets - inventory) / current_liabilities, unit="ratio")


def band_quick_ratio(ratio: float) -> int:
    return band_score(ratio, 0.7, 1.0)


@register_formula(id="FIN-BS-019", version=_VERSION, unit="ratio", inputs=["debt", "equity"])
def debt_to_equity(debt: float, equity: float) -> Value:
    """Debt to equity (FIN-BS-019): `Total interest-bearing debt /
    Shareholders equity`. Band: BAD >2.0; GOOD 1.0-2.0; EXCELLENT <1.0.
    Per DECISION_RULES.md mandatory override 4: `NOT_MEANINGFUL` when
    equity is negative."""
    if equity < 0:
        return _null(NullState.NOT_MEANINGFUL, "ratio", "NEGATIVE_EQUITY_DEBT_TO_EQUITY_NOT_MEANINGFUL")
    if equity == 0:
        return _null(NullState.NOT_MEANINGFUL, "ratio", "DEBT_TO_EQUITY_ZERO_EQUITY")
    return _ok(debt / equity, unit="ratio")


def band_debt_to_equity(ratio: float) -> int:
    return band_score(ratio, 1.0, 2.0, higher_is_better=False)


@register_formula(id="FIN-BS-020", version=_VERSION, unit="ratio", inputs=["ebit", "interest_expense"])
def interest_coverage(ebit: float, interest_expense: float) -> Value:
    """Interest coverage (FIN-BS-020): `Normalized EBIT / Cash interest
    expense`. Band: BAD <1.5x; GOOD 1.5-3x; EXCELLENT >3x. Mandatory
    warning (`SOLVENCY_WARNING`) below 1.5x, per DECISION_RULES.md
    override 3."""
    if interest_expense == 0:
        return _null(NullState.NOT_MEANINGFUL, "ratio", "INTEREST_COVERAGE_ZERO_INTEREST_EXPENSE")
    coverage = ebit / interest_expense
    warnings = ["SOLVENCY_WARNING"] if coverage < 1.5 else []
    return Value.of(coverage, unit="ratio", evidence_class=EvidenceClass.C, warnings=warnings)


def band_interest_coverage(coverage: float) -> int:
    return band_score(coverage, 1.5, 3.0)


@register_formula(
    id="FIN-BS-021", version=_VERSION, unit="pct", inputs=["debt_growth", "revenue_growth"],
)
def debt_vs_revenue_trend(debt_growth: float, revenue_growth: float) -> Value:
    """Debt vs. revenue trend (FIN-BS-021): `debt growth - revenue
    growth`. Band: reuses GR-003's +/-2 percentage-point convention
    (FORMULAS.md gives no separate number); lower is better (debt growing
    faster than revenue is worse)."""
    return _ok(debt_growth - revenue_growth, unit="pct")


def band_debt_vs_revenue_trend(diff: float) -> int:
    return band_score(diff, -0.02, 0.02, higher_is_better=False)


@register_formula(id="FIN-BS-022", version=_VERSION, unit="ratio_per_period", inputs=["ratios"])
def liquidity_trend(ratios: Sequence[float]) -> Value:
    """Liquidity trend (FIN-BS-022): OLS slope of >=3 annual current/quick
    ratios."""
    if len(ratios) < 3:
        return _null(NullState.MISSING, "ratio_per_period", "LIQUIDITY_TREND_NEEDS_3_POINTS")
    return _ok(_ols_slope(ratios), unit="ratio_per_period")


def band_liquidity_trend(slope: float) -> int:
    """FORMULAS.md gives no numeric threshold; this module uses a
    0.05-ratio-point/year materiality band, mirroring the "small trend
    slope = stable" idiom used everywhere else in FORMULAS.md."""
    return band_score(slope, -0.05, 0.05)


# ============================================================================
# FIN-EF-023..027: Efficiency / return on capital
# ============================================================================


@register_formula(id="FIN-EF-023", version=_VERSION, unit="pct", inputs=["net_income", "avg_equity"])
def roe(net_income: float, avg_equity: float) -> Value:
    """ROE (FIN-EF-023): `Net income / Average shareholders equity`. Band:
    BAD <8%; GOOD 8-15%; EXCELLENT >15%."""
    if avg_equity <= 0:
        return _null(NullState.NOT_MEANINGFUL, "pct", "ROE_NONPOSITIVE_AVG_EQUITY")
    return _ok(net_income / avg_equity, unit="pct")


def band_roe(pct: float) -> int:
    return band_score(pct, 0.08, 0.15)


@register_formula(id="FIN-EF-024", version=_VERSION, unit="pct", inputs=["nopat", "avg_invested_capital"])
def roic(nopat: float, avg_invested_capital: float) -> Value:
    """ROIC (FIN-EF-024): `NOPAT / Average invested capital`. Band: BAD
    <8%; GOOD 8-15%; EXCELLENT >15%."""
    if avg_invested_capital <= 0:
        return _null(NullState.NOT_MEANINGFUL, "pct", "ROIC_NONPOSITIVE_INVESTED_CAPITAL")
    return _ok(nopat / avg_invested_capital, unit="pct")


def band_roic(pct: float) -> int:
    return band_score(pct, 0.08, 0.15)


@register_formula(id="FIN-EF-025", version=_VERSION, unit="pct", inputs=["net_income", "avg_assets"])
def roa(net_income: float, avg_assets: float) -> Value:
    """ROA (FIN-EF-025): `Net income / Average total assets`. Band: BAD
    <3%; GOOD 3-8%; EXCELLENT >8%."""
    if avg_assets <= 0:
        return _null(NullState.NOT_MEANINGFUL, "pct", "ROA_NONPOSITIVE_AVG_ASSETS")
    return _ok(net_income / avg_assets, unit="pct")


def band_roa(pct: float) -> int:
    return band_score(pct, 0.03, 0.08)


@register_formula(id="FIN-EF-026", version=_VERSION, unit="pct", inputs=["roic_value", "wacc_value"])
def roic_vs_wacc(roic_value: float, wacc_value: float) -> Value:
    """ROIC vs. cost of capital (FIN-EF-026): `ROIC - WACC`. Band: BAD
    below WACC; GOOD approximately WACC (+/-1 pt); EXCELLENT above WACC --
    read numerically as BAD spread < -1pp, GOOD -1..+1pp, EXCELLENT >
    +1pp (FORMULAS.md, verbatim +/-1pt band)."""
    return _ok(roic_value - wacc_value, unit="pct")


def band_roic_vs_wacc(spread: float) -> int:
    return band_score(spread, -0.01, 0.01)


@register_formula(id="FIN-EF-027", version=_VERSION, unit="pct_per_period", inputs=["returns"])
def return_trend(returns: Sequence[float]) -> Value:
    """Return trend (FIN-EF-027): OLS slope of >=3 annual ROIC (or adapter
    return) observations. Band: 1 percentage-point/year threshold
    (FORMULAS.md, verbatim)."""
    if len(returns) < 3:
        return _null(NullState.MISSING, "pct_per_period", "RETURN_TREND_NEEDS_3_POINTS")
    return _ok(_ols_slope(returns), unit="pct_per_period")


def band_return_trend(slope: float) -> int:
    return band_score(slope, -0.01, 0.01)


# ============================================================================
# FIN-DX-028..033: Diagnostics (not part of the core-27 count)
# ============================================================================


@register_formula(id="FIN-DX-028", version=_VERSION, unit="ratio", inputs=["net_debt", "ebitda"])
def net_debt_to_ebitda(net_debt: float, ebitda: float) -> Value:
    """Net debt to EBITDA (FIN-DX-028). Not meaningful for negative EBITDA
    (FORMULAS.md, verbatim)."""
    if ebitda <= 0:
        return _null(NullState.NOT_MEANINGFUL, "ratio", "NET_DEBT_TO_EBITDA_NONPOSITIVE_EBITDA")
    return _ok(net_debt / ebitda, unit="ratio")


@register_formula(id="FIN-DX-029", version=_VERSION, unit="ratio", inputs=["net_debt", "fcf"])
def debt_to_fcf(net_debt: float, fcf: float) -> Value:
    """Debt to FCF (FIN-DX-029). Not meaningful when FCF<=0 (FORMULAS.md,
    verbatim)."""
    if fcf <= 0:
        return _null(NullState.NOT_MEANINGFUL, "ratio", "DEBT_TO_FCF_NONPOSITIVE_FCF")
    return _ok(net_debt / fcf, unit="ratio")


@register_formula(
    id="FIN-DX-030", version=_VERSION, unit="ratio", inputs=["net_income", "ocf", "avg_assets"],
)
def accrual_ratio(net_income: float, ocf: float, avg_assets: float) -> Value:
    """Accrual ratio (FIN-DX-030): `(Net income - Operating cash flow) /
    Average total assets`. Higher positive accruals indicate lower cash
    quality (FORMULAS.md)."""
    if avg_assets <= 0:
        return _null(NullState.NOT_MEANINGFUL, "ratio", "ACCRUAL_RATIO_NONPOSITIVE_AVG_ASSETS")
    return _ok((net_income - ocf) / avg_assets, unit="ratio")


@register_formula(
    id="FIN-DX-031", version=_VERSION, unit="days", inputs=["dso", "dio", "dpo"],
)
def cash_conversion_cycle(dso: float, dio: float, dpo: float) -> Value:
    """Cash-conversion cycle (FIN-DX-031): `DSO + DIO - DPO`."""
    return _ok(dso + dio - dpo, unit="days")


@register_formula(id="FIN-DX-032", version=_VERSION, unit="pct", inputs=["sbc", "revenue"])
def sbc_to_revenue(sbc: float, revenue: float) -> Value:
    """SBC to revenue (FIN-DX-032): `Stock-based compensation / Revenue`."""
    if revenue == 0:
        return _null(NullState.NOT_MEANINGFUL, "pct", "SBC_TO_REVENUE_ZERO_REVENUE")
    return _ok(sbc / revenue, unit="pct")


def band_sbc_to_revenue(pct: float) -> int:
    """FORMULAS.md gives FIN-DX-032 no numeric band (only "Report with
    diluted-share CAGR"); dilution is a scored input of the RETURNS
    dimension (SCORING.md, "+ dilution", "heavy dilution" -> BAD). This
    module's reading (lower SBC drag is better): EXCELLENT <2% of revenue;
    GOOD 2-10%; BAD >10%."""
    return band_score(pct, 0.02, 0.10, higher_is_better=False)


@register_formula(
    id="FIN-DX-033", version=_VERSION, unit="pct", inputs=["shares_end", "shares_begin", "years"],
)
def diluted_share_cagr(shares_end: float, shares_begin: float, years: float) -> Value:
    """Diluted-share CAGR (FIN-DX-033): positive is dilution (FORMULAS.md)."""
    return cagr(shares_end, shares_begin, years)


def band_diluted_share_cagr(cagr_pct: float) -> int:
    """FORMULAS.md gives FIN-DX-033 no numeric band ("Positive is dilution"
    only). This module's reading (share-count shrinkage is best): EXCELLENT
    <0%/year (net buybacks return capital); GOOD 0-2%/year (modest
    dilution); BAD >2%/year (material dilution -- Task 15's business brief
    independently anchors a >5% "DILUTION_RED_FLAG", so >2% BAD is the
    stricter, still-defensible line for a *scored* input)."""
    return band_score(cagr_pct, 0.0, 0.02, higher_is_better=False)


# ============================================================================
# Envelope assembly: dimensions, core-27 diagnostic, extension fields
# ============================================================================


class Core27Summary(BaseModel):
    """`envelope.core_27_metrics` (OUTPUT_SCHEMA.md)."""

    model_config = ConfigDict(frozen=True)

    valid_count: int
    points: float
    maximum_valid_points: float
    percent: float
    score_10: float
    rows: list[MetricRow] = Field(default_factory=list)


class FinancialOutput(SpecialistOutput):
    """`financial_analysis`'s `run()` result: the shared envelope plus
    OUTPUT_SCHEMA.md's financial-specific extension fields."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    core_27_metrics: Core27Summary
    profitability_and_cash: dict[str, MetricRow] = Field(default_factory=dict)
    balance_and_maturities: dict[str, MetricRow] = Field(default_factory=dict)
    return_on_capital: dict[str, MetricRow] = Field(default_factory=dict)
    dilution_and_sbc: dict[str, MetricRow] = Field(default_factory=dict)
    mandatory_overrides: list[str] = Field(default_factory=list)
    strongest_metric: str | None = None
    weakest_metric: str | None = None


@dataclass(frozen=True)
class _MetricResult:
    """One core-27 (or diagnostic) metric's computed `Value` plus the
    core-27 band points it earns (`None` when not part of the 27, or not
    scorable).

    `dimensions` documents which dimension(s) this metric feeds at its call
    site in `_compute_all` (readable proof the call site agrees with
    `_DIMENSION_MEMBERS`, the table `run()` actually assembles dimensions
    from) -- it is not itself consumed downstream.
    """

    metric_id: str
    value: Value
    band: int | None
    dimensions: tuple[str, ...] = field(default_factory=tuple)
    confidence: float = 0.0
    core27: bool = True


def _confidence_for(v: Value) -> float:
    """Per-metric-row confidence (0-100). `OUTPUT_CONTRACT.md` requires a
    `confidence` per row but doesn't pin its formula down to this
    granularity (`wbj.core.confidence.confidence` is the *category*-level
    aggregate, from coverage/source_quality/freshness/consistency/
    model_fit). This module's reading: null values score 0; valid values
    start from their evidence class (R=95, C=85, E=60, A=50, Q=30,
    unset=70) and lose 15 points per attached warning, floored at 0."""
    if v.is_null:
        return 0.0
    base_by_class = {
        EvidenceClass.R: 95.0,
        EvidenceClass.C: 85.0,
        EvidenceClass.E: 60.0,
        EvidenceClass.A: 50.0,
        EvidenceClass.Q: 30.0,
    }
    base = base_by_class.get(v.evidence_class, 70.0) if v.evidence_class else 70.0
    base -= 15.0 * len(v.warnings)
    return max(0.0, min(100.0, base))


def _attach(v: Value, *, period: str | None, source_locator: str | None) -> Value:
    updates: dict[str, Any] = {}
    if period is not None and v.period is None:
        updates["period"] = period
    if source_locator is not None and v.source_locator is None:
        updates["source_locator"] = source_locator
    return v.model_copy(update=updates) if updates else v


def _annual_rows(packet: Packet) -> list[dict]:
    """`packet.fundamentals["annual"]` oldest -> newest (the packet stores
    newest-first; `DATASET.md` requires ascending order)."""
    rows = packet.fundamentals.get("annual") or []
    return list(reversed(rows))


def _quarterly_rows(packet: Packet) -> list[dict]:
    rows = packet.fundamentals.get("quarterly") or []
    return list(reversed(rows))


def _num(row: dict, key: str) -> float | None:
    v = row.get(key)
    return float(v) if isinstance(v, (int, float)) else None


def _latest_balance_row(packet: Packet) -> dict | None:
    """Most recent balance-sheet snapshot: latest quarterly row if present
    (FORMULAS.md marks current/quick ratio "quarterly" frequency), else
    the latest annual row."""
    q = _quarterly_rows(packet)
    if q:
        return q[-1]
    a = _annual_rows(packet)
    return a[-1] if a else None


def _compute_all(
    packet: Packet, overlay: dict[str, Any]
) -> tuple[list[_MetricResult], list[str], list[JudgmentRequest], dict[str, Any]]:
    """Compute every core-27 + diagnostic metric from `packet` (+
    `overlay`), returning `(results, assumptions, judgment_requests, ctx)`.

    `ctx` carries the handful of raw values `run()` needs for the
    mandatory-override checks (`override_1_triggered`/`override_2_triggered`)
    without re-deriving them from the assembled `MetricRow`s: `net_income`,
    `fcf`, `roic_value`, `wacc_value`, `capital_dependence_need`,
    `debt_issuance`, `equity_issuance`, `returns_cash`.
    """
    annual = _annual_rows(packet)
    results: list[_MetricResult] = []
    assumptions: list[str] = []
    judgment_requests: list[JudgmentRequest] = []
    ctx: dict[str, Any] = {}

    latest_year = annual[-1].get("calendarYear") if annual else None
    period_label = f"FY{latest_year}" if latest_year else None

    def add(metric_id: str, v: Value, band: int | None, dims: tuple[str, ...], *, core27: bool = True) -> None:
        v = _attach(v, period=period_label, source_locator="packet.fundamentals.annual")
        results.append(
            _MetricResult(metric_id=metric_id, value=v, band=band, dimensions=dims, confidence=_confidence_for(v), core27=core27)
        )

    # Known simplification: `revenues` drops missing years rather than
    # preserving their position, so a *gap in the middle* of the reported
    # history (present in FMP data would be unusual -- a company either
    # reports a year or its series is simply shorter, not holed) would
    # silently treat two non-adjacent years as if they were consecutive
    # for GR-001/GR-002. `debt_hist`/`equity_hist`/`assets_hist` avoid this
    # by staying index-aligned to `annual` and indexing from the end
    # instead of filtering.
    revenues = [_num(r, "revenue") for r in annual]
    revenues = [x for x in revenues if x is not None]
    cogs_hist = [_num(r, "cogs") for r in annual]
    ebit_hist = [_num(r, "ebit") for r in annual]
    ni_hist = [_num(r, "net_income") for r in annual]
    ocf_hist = [_num(r, "operating_cash_flow") for r in annual]
    capex_hist = [abs(_num(r, "capex") or 0.0) for r in annual]
    debt_hist = [_num(r, "total_debt") for r in annual]
    equity_hist = [_num(r, "total_equity") for r in annual]
    assets_hist = [_num(r, "total_assets") for r in annual]

    # ---- FIN-GR-001: YoY revenue growth ----
    if len(revenues) >= 2:
        v = yoy_revenue_growth(revenues[-1], revenues[-2])
    else:
        v = _null(NullState.MISSING, "pct", "INSUFFICIENT_REVENUE_HISTORY")
    add("FIN-GR-001", v, _band_or_none(v, band_yoy_revenue_growth), (DIM_REVENUE,))

    # ---- FIN-GR-002: revenue growth trend ----
    growth_rates = [
        (revenues[i] - revenues[i - 1]) / revenues[i - 1]
        for i in range(1, len(revenues))
        if revenues[i - 1] not in (None, 0)
    ]
    v = revenue_growth_trend(growth_rates) if len(growth_rates) >= 3 else _null(
        NullState.MISSING, "pct_per_period", "GROWTH_TREND_NEEDS_3_POINTS"
    )
    add("FIN-GR-002", v, _band_or_none(v, band_revenue_growth_trend), (DIM_REVENUE,))

    # ---- FIN-GR-003: growth vs peers (no peer growth dataset in Packet) ----
    v = _null(NullState.MISSING, "pct", "PEER_GROWTH_DATA_UNAVAILABLE")
    add("FIN-GR-003", v, None, (DIM_REVENUE,))

    # ---- FIN-GR-004 / FIN-GR-005: judgment-only (see module docstring) ----
    v_organic = _null(NullState.NOT_SCORABLE, "ratio", "ORGANIC_GROWTH_BRIDGE_UNAVAILABLE_JUDGMENT_REQUIRED")
    add("FIN-GR-004", v_organic, None, (DIM_REVENUE,))
    judgment_requests.append(
        JudgmentRequest(
            request_id="financial_analysis:FIN-GR-004",
            agent_id=AGENT_ID,
            metric_id="FIN-GR-004",
            question="Classify organic growth quality (BAD/GOOD/EXCELLENT) from the "
            "organic/acquired/FX/divestiture revenue bridge; the packet does not carry it.",
            schema_hint="one of BAD|GOOD|EXCELLENT",
        )
    )
    v_share = _null(NullState.NOT_SCORABLE, "pct_per_period", "MARKET_SHARE_SERIES_UNAVAILABLE_JUDGMENT_REQUIRED")
    add("FIN-GR-005", v_share, None, (DIM_REVENUE,))
    judgment_requests.append(
        JudgmentRequest(
            request_id="financial_analysis:FIN-GR-005",
            agent_id=AGENT_ID,
            metric_id="FIN-GR-005",
            question="Classify market-share trend (losing/stable/gaining) from a 3-year "
            "consistent-market-definition share series; the packet does not carry it.",
            schema_hint="one of BAD|GOOD|EXCELLENT",
        )
    )

    # ---- FIN-PR-006: net profit status ----
    valid_ni = [x for x in ni_hist if x is not None]
    v = net_profit_status(valid_ni) if valid_ni else _null(NullState.MISSING, "usd", "NET_INCOME_UNAVAILABLE")
    add("FIN-PR-006", v, net_profit_status_band(valid_ni) if valid_ni else None, (DIM_EPS_FCF,))
    ctx["net_income"] = valid_ni[-1] if valid_ni else None

    # ---- FIN-PR-007..009: margins (latest annual) ----
    rev_latest = revenues[-1] if revenues else None
    cogs_latest = cogs_hist[-1] if cogs_hist else None
    ebit_latest = ebit_hist[-1] if ebit_hist else None
    ni_latest = ni_hist[-1] if ni_hist else None

    if rev_latest is not None and cogs_latest is not None:
        v = gross_margin(rev_latest, cogs_latest)
    else:
        v = _null(NullState.MISSING, "pct", "GROSS_MARGIN_INPUTS_UNAVAILABLE")
    add("FIN-PR-007", v, _band_or_none(v, band_gross_margin), (DIM_MARGINS,))

    if rev_latest is not None and ebit_latest is not None:
        v = operating_margin(ebit_latest, rev_latest)
    else:
        v = _null(NullState.MISSING, "pct", "OPERATING_MARGIN_INPUTS_UNAVAILABLE")
    add("FIN-PR-008", v, _band_or_none(v, band_operating_margin), (DIM_MARGINS,))

    if rev_latest is not None and ni_latest is not None:
        v = net_margin(ni_latest, rev_latest)
    else:
        v = _null(NullState.MISSING, "pct", "NET_MARGIN_INPUTS_UNAVAILABLE")
    add("FIN-PR-009", v, _band_or_none(v, band_net_margin), (DIM_MARGINS,))

    # ---- FIN-PR-010: margin trend (net margin) ----
    # Iterate `annual` directly (not the pre-filtered `revenues`/`ni_hist`
    # lists, which can differ in length when a year is missing one field but
    # not the other) so a missing revenue in one year never misaligns net
    # income from a different year.
    net_margins = [
        ni / rev
        for row in annual
        for ni in (_num(row, "net_income"),)
        for rev in (_num(row, "revenue"),)
        if ni is not None and rev not in (None, 0)
    ]
    v = margin_trend(net_margins) if len(net_margins) >= 3 else _null(
        NullState.MISSING, "pct_per_period", "MARGIN_TREND_NEEDS_3_POINTS"
    )
    add("FIN-PR-010", v, _band_or_none(v, band_margin_trend), (DIM_MARGINS,))

    # ---- FIN-PR-011: profit vs revenue growth ----
    if len(revenues) >= 2 and len(valid_ni) >= 2 and valid_ni[-2] > 0 and revenues[-2] != 0:
        rev_growth = (revenues[-1] - revenues[-2]) / revenues[-2]
        ni_growth = (valid_ni[-1] - valid_ni[-2]) / valid_ni[-2]
        v = profit_vs_revenue_growth(ni_growth, rev_growth)
    else:
        v = _null(NullState.NOT_MEANINGFUL, "pct", "PROFIT_VS_REVENUE_GROWTH_SIGN_CHANGE_OR_MISSING")
    add("FIN-PR-011", v, _band_or_none(v, band_profit_vs_revenue_growth), (DIM_EPS_FCF,))

    # ---- FIN-CF-012: free cash flow ----
    ocf_latest = ocf_hist[-1] if ocf_hist else None
    capex_latest = capex_hist[-1] if capex_hist else None
    if ocf_latest is not None and capex_latest is not None:
        v = free_cash_flow(ocf_latest, capex_latest)
    else:
        v = _null(NullState.MISSING, "usd", "FCF_INPUTS_UNAVAILABLE")
    fcf_latest = v.value if v.is_valid else None
    ctx["fcf"] = fcf_latest

    # ---- FIN-CF-014: FCF margin (computed before FCF's own band, which needs it) ----
    if fcf_latest is not None and rev_latest not in (None, 0):
        v_margin = fcf_margin(fcf_latest, rev_latest)
    else:
        v_margin = _null(NullState.MISSING, "pct", "FCF_MARGIN_INPUTS_UNAVAILABLE")

    add(
        "FIN-CF-012", v, free_cash_flow_band(fcf_latest, v_margin.value if v_margin.is_valid else None) if fcf_latest is not None else None,
        (DIM_EPS_FCF,),
    )
    add("FIN-CF-014", v_margin, _band_or_none(v_margin, band_fcf_margin), (DIM_EPS_FCF,))

    # ---- FIN-CF-013: FCF growth ----
    fcf_hist = [
        (ocf_hist[i] - capex_hist[i]) if ocf_hist[i] is not None else None
        for i in range(len(annual))
    ]
    valid_fcf = [x for x in fcf_hist if x is not None]
    if len(valid_fcf) >= 2:
        v = fcf_growth(valid_fcf[-1], valid_fcf[-2])
    else:
        v = _null(NullState.MISSING, "pct", "INSUFFICIENT_FCF_HISTORY")
    add("FIN-CF-013", v, _band_or_none(v, band_fcf_growth), (DIM_EPS_FCF,))

    # ---- FIN-CF-015: cash vs earnings (3y cumulative) ----
    ocf_window = [x for x in ocf_hist[-3:] if x is not None]
    ni_window = [x for x in ni_hist[-3:] if x is not None]
    if len(ocf_window) == 3 and len(ni_window) == 3:
        v = cash_vs_earnings(sum(ocf_window), sum(ni_window))
    else:
        v = _null(NullState.MISSING, "ratio", "CASH_VS_EARNINGS_INSUFFICIENT_HISTORY")
    add("FIN-CF-015", v, _band_or_none(v, band_cash_vs_earnings), (DIM_EPS_FCF, DIM_RETURNS))

    # ---- FIN-CF-016: capital dependence ----
    latest = annual[-1] if annual else {}
    cash_uses = capex_latest or 0.0
    buybacks = abs(_num(latest, "common_stock_repurchased") or 0.0)
    dividends = abs(_num(latest, "dividends_paid") or 0.0)
    cash_uses_total = cash_uses + buybacks + dividends
    cash_begin_row = annual[-2] if len(annual) >= 2 else {}
    cash_begin = _num(cash_begin_row, "cash") or 0.0
    # `debt_repayment` follows FMP's net convention: negative = net debt
    # repaid (cash outflow), positive = net new debt raised. Only the
    # positive (issuance) side counts as "external dependence" here; a
    # repayment is the opposite of raising capital.
    debt_issuance = max(0.0, _num(latest, "debt_repayment") or 0.0)
    equity_issuance = float(overlay.get("equity_issuance", 0.0))
    if ocf_latest is not None:
        v = capital_dependence(cash_uses_total, ocf_latest, cash_begin)
        need = v.value
        returns_cash = (buybacks + dividends) > 0
        band = capital_dependence_band(need, returns_cash)
    else:
        v = _null(NullState.MISSING, "usd", "CAPITAL_DEPENDENCE_INPUTS_UNAVAILABLE")
        need = 0.0
        returns_cash = False
        band = None
    add("FIN-CF-016", v, band, (DIM_EPS_FCF,))
    ctx["capital_dependence_need"] = need
    ctx["debt_issuance"] = debt_issuance
    ctx["equity_issuance"] = equity_issuance
    ctx["returns_cash"] = returns_cash
    ctx["externally_dependent"] = is_externally_dependent(need, debt_issuance, equity_issuance)

    # ---- FIN-BS-017/018: current & quick ratio (latest balance-sheet row) ----
    bal = _latest_balance_row(packet) or {}
    ca = _num(bal, "total_current_assets")
    cl = _num(bal, "total_current_liabilities")
    inv = _num(bal, "inventory")
    bal_period = f"{bal.get('period', '')}{bal.get('calendarYear', '')}" or None

    if ca is not None and cl is not None:
        v = current_ratio(ca, cl)
    else:
        v = _null(NullState.MISSING, "ratio", "CURRENT_RATIO_INPUTS_UNAVAILABLE")
    v = _attach(v, period=bal_period, source_locator="packet.fundamentals")
    results.append(_MetricResult("FIN-BS-017", v, _band_or_none(v, band_current_ratio), (DIM_BALANCE,), _confidence_for(v)))

    if ca is not None and cl is not None and inv is not None:
        v = quick_ratio(ca, inv, cl)
    else:
        v = _null(NullState.MISSING, "ratio", "QUICK_RATIO_INPUTS_UNAVAILABLE")
    v = _attach(v, period=bal_period, source_locator="packet.fundamentals")
    results.append(_MetricResult("FIN-BS-018", v, _band_or_none(v, band_quick_ratio), (DIM_BALANCE,), _confidence_for(v)))

    # ---- FIN-BS-019: debt to equity ----
    debt_latest = debt_hist[-1] if debt_hist else None
    equity_latest = equity_hist[-1] if equity_hist else None
    if debt_latest is not None and equity_latest is not None:
        v = debt_to_equity(debt_latest, equity_latest)
    else:
        v = _null(NullState.MISSING, "ratio", "DEBT_TO_EQUITY_INPUTS_UNAVAILABLE")
    add("FIN-BS-019", v, _band_or_none(v, band_debt_to_equity), (DIM_BALANCE,))

    # ---- FIN-BS-020: interest coverage (Packet carries no interest_expense field) ----
    interest_expense = overlay.get("interest_expense")
    if interest_expense is not None and ebit_latest is not None:
        v = interest_coverage(ebit_latest, float(interest_expense))
    else:
        v = _null(NullState.MISSING, "ratio", "INTEREST_EXPENSE_UNAVAILABLE")
        assumptions.append(
            "FIN-BS-020 (interest coverage) not computed: `interest_expense` is not part of "
            "Packet.fundamentals and was not supplied via overlay['interest_expense']."
        )
    add("FIN-BS-020", v, _band_or_none(v, band_interest_coverage), (DIM_BALANCE,))

    # ---- FIN-BS-021: debt vs revenue trend ----
    if len(debt_hist) >= 2 and debt_hist[-2] not in (None, 0) and len(revenues) >= 2 and revenues[-2] != 0:
        debt_growth = (debt_hist[-1] - debt_hist[-2]) / debt_hist[-2]
        rev_growth = (revenues[-1] - revenues[-2]) / revenues[-2]
        v = debt_vs_revenue_trend(debt_growth, rev_growth)
    else:
        v = _null(NullState.MISSING, "pct", "DEBT_VS_REVENUE_TREND_INPUTS_UNAVAILABLE")
    add("FIN-BS-021", v, _band_or_none(v, band_debt_vs_revenue_trend), (DIM_BALANCE,))

    # ---- FIN-BS-022: liquidity trend (current ratio, 3y) ----
    liq_series = [
        (ca_i / cl_i)
        for ca_i, cl_i in zip(
            [_num(r, "total_current_assets") for r in annual[-3:]],
            [_num(r, "total_current_liabilities") for r in annual[-3:]],
        )
        if ca_i is not None and cl_i not in (None, 0)
    ]
    v = liquidity_trend(liq_series) if len(liq_series) >= 3 else _null(
        NullState.MISSING, "ratio_per_period", "LIQUIDITY_TREND_NEEDS_3_POINTS"
    )
    add("FIN-BS-022", v, _band_or_none(v, band_liquidity_trend), (DIM_BALANCE,))

    # ---- FIN-EF-023: ROE ----
    if len(equity_hist) >= 2 and equity_hist[-1] is not None and equity_hist[-2] is not None and ni_latest is not None:
        avg_equity = (equity_hist[-1] + equity_hist[-2]) / 2
        v = roe(ni_latest, avg_equity)
    elif equity_latest is not None and ni_latest is not None:
        v = roe(ni_latest, equity_latest)
        v = (
            Value.of(
                v.value, unit=v.unit, evidence_class=v.evidence_class, warnings=[*v.warnings, "END_BALANCE_PROXY"]
            )
            if v.is_valid
            else v
        )
    else:
        v = _null(NullState.MISSING, "pct", "ROE_INPUTS_UNAVAILABLE")
    add("FIN-EF-023", v, _band_or_none(v, band_roe), (DIM_RETURNS,))

    # ---- FIN-EF-024: ROIC ----
    tax_rate = None
    if annual:
        pretax = _num(annual[-1], "income_before_tax")
        tax_expense = _num(annual[-1], "income_tax_expense")
        if pretax and pretax > 0 and tax_expense is not None:
            tax_rate = min(max(tax_expense / pretax, 0.0), 1.0)
    if tax_rate is None:
        tax_rate = 0.21
        assumptions.append("FIN-EF-024 (ROIC): normalized cash tax rate unavailable; substituted the 21% statutory rate.")

    if ebit_latest is not None and debt_latest is not None and equity_latest is not None:
        nopat_value = ebit_latest * (1 - tax_rate)
        cash_latest = _num(annual[-1], "cash") or 0.0
        ic_end = debt_latest + equity_latest - cash_latest
        if len(debt_hist) >= 2 and debt_hist[-2] is not None and equity_hist[-2] is not None:
            cash_begin_ic = _num(annual[-2], "cash") or 0.0
            ic_begin = debt_hist[-2] + equity_hist[-2] - cash_begin_ic
            avg_ic = (ic_begin + ic_end) / 2
        else:
            avg_ic = ic_end
        v = roic(nopat_value, avg_ic)
    else:
        v = _null(NullState.MISSING, "pct", "ROIC_INPUTS_UNAVAILABLE")
    add("FIN-EF-024", v, _band_or_none(v, band_roic), (DIM_RETURNS,))
    roic_value = v.value if v.is_valid else None
    ctx["roic_value"] = roic_value

    # ---- FIN-EF-025: ROA ----
    if len(assets_hist) >= 2 and assets_hist[-1] is not None and assets_hist[-2] is not None and ni_latest is not None:
        avg_assets = (assets_hist[-1] + assets_hist[-2]) / 2
        v = roa(ni_latest, avg_assets)
    elif assets_hist and assets_hist[-1] is not None and ni_latest is not None:
        v = roa(ni_latest, assets_hist[-1])
    else:
        v = _null(NullState.MISSING, "pct", "ROA_INPUTS_UNAVAILABLE")
    add("FIN-EF-025", v, _band_or_none(v, band_roa), (DIM_RETURNS,))

    # ---- FIN-EF-026: ROIC vs WACC (needs overlay["wacc"]) ----
    wacc_value = overlay.get("wacc")
    ctx["wacc_value"] = float(wacc_value) if wacc_value is not None else None
    if roic_value is not None and wacc_value is not None:
        v = roic_vs_wacc(roic_value, float(wacc_value))
    else:
        v = _null(NullState.MISSING, "pct", "WACC_NOT_SUPPLIED")
        if wacc_value is None:
            assumptions.append(
                "FIN-EF-026 (ROIC vs WACC) and mandatory Override 2 not evaluated: no "
                "overlay['wacc'] was supplied (Packet does not carry a WACC)."
            )
    add("FIN-EF-026", v, _band_or_none(v, band_roic_vs_wacc), (DIM_RETURNS,))

    # ---- FIN-EF-027: return trend (ROIC history) ----
    # ROIC_t = NOPAT_t / avg(InvestedCapital_t-1, InvestedCapital_t), for
    # every year t that has both its own and its prior year's balance-sheet
    # inputs -- one fewer observation than years of annual history.
    roic_hist: list[float] = []
    for i in range(1, len(annual)):
        ebit_i = ebit_hist[i]
        debt_i, equity_i, debt_im1, equity_im1 = debt_hist[i], equity_hist[i], debt_hist[i - 1], equity_hist[i - 1]
        cash_i, cash_im1 = _num(annual[i], "cash"), _num(annual[i - 1], "cash")
        if None in (ebit_i, debt_i, equity_i, debt_im1, equity_im1, cash_i, cash_im1):
            continue
        pretax_i, tax_i = _num(annual[i], "income_before_tax"), _num(annual[i], "income_tax_expense")
        tax_rate_i = min(max(tax_i / pretax_i, 0.0), 1.0) if pretax_i and pretax_i > 0 and tax_i is not None else tax_rate
        nopat_i = ebit_i * (1 - tax_rate_i)
        avg_ic_i = ((debt_i + equity_i - cash_i) + (debt_im1 + equity_im1 - cash_im1)) / 2
        if avg_ic_i <= 0:
            continue
        roic_hist.append(nopat_i / avg_ic_i)
    v = return_trend(roic_hist[-3:]) if len(roic_hist) >= 3 else _null(
        NullState.MISSING, "pct_per_period", "RETURN_TREND_NEEDS_3_POINTS"
    )
    add("FIN-EF-027", v, _band_or_none(v, band_return_trend), (DIM_RETURNS,))

    # ---- Diagnostics (not part of core-27) ----
    net_debt_latest = (debt_latest or 0.0) - (_num(annual[-1], "cash") or 0.0) if annual else None
    v = _null(NullState.MISSING, "ratio", "EBITDA_UNAVAILABLE_NO_DA_FIELD")
    add("FIN-DX-028", v, None, (), core27=False)

    if net_debt_latest is not None and fcf_latest is not None:
        v = debt_to_fcf(net_debt_latest, fcf_latest)
    else:
        v = _null(NullState.MISSING, "ratio", "DEBT_TO_FCF_INPUTS_UNAVAILABLE")
    add("FIN-DX-029", v, None, (), core27=False)

    if ni_latest is not None and ocf_latest is not None and len(assets_hist) >= 2 and assets_hist[-1] and assets_hist[-2]:
        avg_assets = (assets_hist[-1] + assets_hist[-2]) / 2
        v = accrual_ratio(ni_latest, ocf_latest, avg_assets)
    else:
        v = _null(NullState.MISSING, "ratio", "ACCRUAL_RATIO_INPUTS_UNAVAILABLE")
    add("FIN-DX-030", v, None, (), core27=False)

    v = _null(NullState.MISSING, "days", "PAYABLES_FIELD_UNAVAILABLE")
    add("FIN-DX-031", v, None, (), core27=False)

    # FIN-DX-032/033 (dilution) ARE scored -- SCORING.md names them a
    # primary input of the RETURNS dimension ("+ dilution"). They stay
    # core27=False (not part of the 27-metric diagnostic count) but carry a
    # band so they contribute to the weighted dimension score.
    sbc_latest = _num(annual[-1], "stock_based_compensation") if annual else None
    if sbc_latest is not None and rev_latest not in (None, 0):
        v = sbc_to_revenue(sbc_latest, rev_latest)
    else:
        v = _null(NullState.MISSING, "pct", "SBC_TO_REVENUE_INPUTS_UNAVAILABLE")
    add("FIN-DX-032", v, _band_or_none(v, band_sbc_to_revenue), (DIM_RETURNS,), core27=False)

    diluted_hist = [_num(r, "diluted_shares") for r in annual]
    valid_diluted = [x for x in diluted_hist if x is not None]
    if len(valid_diluted) >= 2:
        v = diluted_share_cagr(valid_diluted[-1], valid_diluted[0], len(valid_diluted) - 1)
    else:
        v = _null(NullState.MISSING, "pct", "DILUTED_SHARE_CAGR_INSUFFICIENT_HISTORY")
    add("FIN-DX-033", v, _band_or_none(v, band_diluted_share_cagr), (DIM_RETURNS,), core27=False)

    return results, assumptions, judgment_requests, ctx


def _band_or_none(v: Value, band_fn) -> int | None:
    if v.is_null:
        return None
    return band_fn(v.value)


def run(packet: Packet, overlay: dict[str, Any] | None = None) -> FinancialOutput:
    """Run the financial specialist against `packet`, producing a
    `FinancialOutput` (max 15 pts): the core-27 diagnostic, five weighted
    dimensions, mandatory flags/overrides, and OUTPUT_SCHEMA.md's
    extension fields.

    `overlay` may supply `wacc` (float, FIN-EF-026 / mandatory Override 2)
    and `equity_issuance` (float, Override 1's external-dependence check);
    see the module docstring for why these two come from `overlay` rather
    than `packet` itself.
    """
    overlay = overlay or {}
    results, assumptions, judgment_requests, ctx = _compute_all(packet, overlay)
    by_id = {r.metric_id: r for r in results}

    if packet.analysis.industry_adapter != "default_nonfinancial":
        assumptions.append(
            f"industry_adapter={packet.analysis.industry_adapter!r}: this module implements only "
            "conventional (non-financial-sector) formulas -- Cerebro's mandatory override 5 "
            "(banks/insurers/REITs must use a sector adapter) is not yet implemented, so FCF/ROIC "
            "results below should not be trusted for this security type."
        )

    # ---- MetricRow assembly ----
    rows: list[MetricRow] = []
    for r in results:
        score: float | str = r.band * 5.0 if r.band is not None else "NOT_SCORABLE"
        rows.append(
            MetricRow.from_value(
                r.metric_id, r.value, formula_id=r.metric_id, formula_version=_VERSION,
                score=score, confidence=r.confidence,
            )
        )

    # ---- Dimensions: each metric contributes a 0-10 score (band*5) at equal weight ----
    dimensions: list[Dimension] = []
    for dim_name in DIMENSION_NAMES:
        member_ids = _DIMENSION_MEMBERS[dim_name]
        weight = 1.0 / len(member_ids)
        metric_scores: list[tuple[float, Value]] = []
        for mid in member_ids:
            r = by_id[mid]
            if r.band is not None:
                metric_scores.append((weight, Value.of(r.band * 5.0, unit="score")))
            else:
                metric_scores.append((weight, Value.null(NullState.NOT_SCORABLE, unit="score")))
        dimensions.append(Dimension(name=dim_name, max_points=3.0, metric_scores=metric_scores))

    # ---- Category points/score_10 (ALWAYS reproducible from dimensions) ----
    cat = Category(name=AGENT_ID, max_points=MAX_POINTS, dimensions=dimensions)
    awarded_points = cat.points()
    dim_score10 = cat.score10()
    coverage = cat.coverage()

    # ---- Core-27 diagnostic (single implementation, shared with tests) ----
    core27_results = [r for r in results if r.core27]
    assert len(core27_results) == 27, f"expected 27 core metrics, got {len(core27_results)}"
    valid = [r for r in core27_results if r.band is not None]
    valid_count, points, maximum_valid_points, percent, core27_score10 = core27_diagnostic(
        [r.band for r in core27_results]
    )
    core27_rows = [row for row in rows if row.metric_id in {r.metric_id for r in core27_results}]
    core27 = Core27Summary(
        valid_count=valid_count, points=points, maximum_valid_points=maximum_valid_points,
        percent=percent, score_10=core27_score10, rows=core27_rows,
    )

    mandatory_flags: list[str] = []
    mandatory_overrides: list[str] = []

    # ---- Reconciliation (DECISION_RULES.md: >1.5-point gap requires an explanation) ----
    recon_warning = reconciliation_check(dim_score10, core27_score10) if valid else None
    if recon_warning is not None:
        mandatory_flags.append(recon_warning)

    # ---- SOLVENCY_WARNING (DECISION_RULES.md override 3: coverage <1.5x) ----
    if "SOLVENCY_WARNING" in by_id["FIN-BS-020"].value.warnings:
        mandatory_flags.append("SOLVENCY_WARNING")

    # ---- Negative equity -> D/E NOT_MEANINGFUL (DECISION_RULES.md override 4) ----
    if "NEGATIVE_EQUITY_DEBT_TO_EQUITY_NOT_MEANINGFUL" in by_id["FIN-BS-019"].value.warnings:
        mandatory_flags.append("NEGATIVE_EQUITY_DEBT_TO_EQUITY_NOT_MEANINGFUL")

    # ---- Override 1: loss + negative FCF + external dependence -> Bad/Avoid ----
    override_1 = override_1_triggered(ctx["net_income"], ctx["fcf"], ctx["externally_dependent"])
    if override_1:
        mandatory_flags.append("OVERRIDE_1_LOSS_NEGATIVE_FCF_EXTERNAL_DEPENDENCE")
        mandatory_overrides.append("OVERRIDE_1_LOSS_NEGATIVE_FCF_EXTERNAL_DEPENDENCE_CAPS_BAD_AVOID")

    # ---- Override 2: ROIC < WACC -> cannot be Excellent ----
    override_2 = override_2_triggered(ctx["roic_value"], ctx["wacc_value"])
    if override_2:
        mandatory_flags.append("OVERRIDE_2_ROIC_BELOW_WACC")
        mandatory_overrides.append("OVERRIDE_2_ROIC_BELOW_WACC_CAPS_EXCELLENT")

    # ---- Overrides cap the VERDICT LABEL only; category points stay reproducible ----
    # (HANDOFF_CONTRACT.md rejects a packet whose category points don't
    # reproduce from dimension scores, with no exception for overrides --
    # so the cap lives on `verdict`, not on `awarded_points`/`score_10`.)
    label = capped_verdict(dim_score10, override_1=override_1, override_2=override_2)

    # ---- Category confidence (Task 5 five-component formula) ----
    category = CategoryStats(
        max_points=MAX_POINTS,
        awarded_points=awarded_points,
        score_10=dim_score10,
        confidence=_category_confidence(coverage, packet, reconciled=recon_warning is None),
    )

    status = status_from_coverage(coverage)

    # ---- Extension field groupings ----
    def _subset(ids: Sequence[str]) -> dict[str, MetricRow]:
        row_by_id = {row.metric_id: row for row in rows}
        return {i: row_by_id[i] for i in ids if i in row_by_id}

    profitability_and_cash = _subset(["FIN-PR-007", "FIN-PR-008", "FIN-PR-009", "FIN-CF-012", "FIN-CF-014", "FIN-CF-015"])
    balance_and_maturities = _subset(
        ["FIN-BS-017", "FIN-BS-018", "FIN-BS-019", "FIN-BS-020", "FIN-BS-021", "FIN-BS-022", "FIN-DX-028", "FIN-DX-029"]
    )
    return_on_capital = _subset(["FIN-EF-023", "FIN-EF-024", "FIN-EF-025", "FIN-EF-026", "FIN-EF-027"])
    dilution_and_sbc = _subset(["FIN-DX-032", "FIN-DX-033"])

    numeric_core27 = [(r.metric_id, r.band) for r in valid]
    strongest_metric = max(numeric_core27, key=lambda t: t[1])[0] if numeric_core27 else None
    weakest_metric = min(numeric_core27, key=lambda t: t[1])[0] if numeric_core27 else None

    # ---- Internal self-checks -> validation_tests summary ----
    # Mirrors HANDOFF_CONTRACT.md's handoff-rejection checklist at a small
    # scale: does category math reproduce from dimensions (always -- the
    # override cap is on the verdict label, never on the points, so this
    # check has no override exception); does every row carry a formula id;
    # is the knowledge timestamp present; is confidence a real number.
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
    warnings_count = 1 if recon_warning is not None else 0

    validation_tests = ValidationTestsSummary(passed=passed, failed=failed, warnings=warnings_count)

    # ---- Judgment slots: which dimension slot each judgment-only metric feeds ----
    # FIN-GR-004/FIN-GR-005 are registered as `revenue_quality_and_growth`
    # members (`_DIMENSION_MEMBERS`) but left NOT_SCORABLE pending a Task 20
    # judgment answer. Recording (dimension_name, slot_index) here lets
    # `wbj.overlay.merge.merge_overlay` replace the exact slot and rescore,
    # so answering the request moves `awarded_points`/`coverage`, not just
    # the flat `metrics` row. Derived mechanically from `_DIMENSION_MEMBERS`
    # (the same ordered table `dimensions` was assembled from above), so a
    # judgment metric only maps to a slot if it genuinely is a dimension
    # member; the mandatory context-only requests have no entry.
    judgment_slots: dict[str, tuple[str, int]] = {}
    for req in judgment_requests:
        for dim_name in DIMENSION_NAMES:
            members = _DIMENSION_MEMBERS[dim_name]
            if req.metric_id in members:
                judgment_slots[req.metric_id] = (dim_name, members.index(req.metric_id))
                break

    return FinancialOutput(
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
        source_lineage=["packet.fundamentals.annual", "packet.fundamentals.quarterly"],
        validation_tests=validation_tests,
        core_27_metrics=core27,
        profitability_and_cash=profitability_and_cash,
        balance_and_maturities=balance_and_maturities,
        return_on_capital=return_on_capital,
        dilution_and_sbc=dilution_and_sbc,
        mandatory_overrides=mandatory_overrides,
        strongest_metric=strongest_metric,
        weakest_metric=weakest_metric,
    )


def _category_confidence(coverage: float, packet: Packet, *, reconciled: bool) -> float:
    """Category confidence (0-100) via `wbj.core.confidence.confidence()` —
    the Task 5 five-component formula (`0.30*coverage + 0.25*source_quality
    + 0.20*freshness + 0.15*consistency + 0.10*model_fit`).

    HANDOFF_CONTRACT.md rejects a packet whose category confidence is
    absent, so every `FinancialOutput` must carry a real number here. The
    five components are derived from real packet signals; where the packet
    doesn't supply a component directly, this module uses a documented
    constant (flagged below). This mapping is this module's own dated
    (2.0.0) construction, shared with Tasks 15-19 in spirit.

    - coverage: the category's own `Category.coverage()`, rescaled 0-100.
    - source_quality: 90 — `packet.fundamentals` are reported financial
      statements from regulatory filings (EDGAR/FMP), the top of
      `DATA_POLICY.md`'s source hierarchy. A constant (the packet doesn't
      carry a per-source-quality score at this layer).
    - freshness: 100 if `packet.staleness["quarterly_fundamentals"]` is
      FRESH, else 50 — the financial specialist's inputs are the quarterly/
      annual fundamentals, whose staleness the packet tracks directly
      (`DATA_POLICY.md`: stale data lowers confidence, never the value).
    - consistency: 90 if the core-27 diagnostic reconciled with the
      weighted dimensions (within DECISION_RULES.md's 1.5-pt tolerance),
      else 60 — a real, computed agreement signal.
    - model_fit: 90 for `default_nonfinancial` (conventional formulas fit
      an operating company), else 40 — a bank/insurer/REIT scored with
      conventional (non-adapter) FCF/ROIC is a poor model fit (mandatory
      override 5, not yet implemented; see module docstring).
    """
    coverage_component = max(0.0, min(1.0, coverage)) * 100.0
    source_quality = 90.0
    freshness = 100.0 if packet.staleness.get("quarterly_fundamentals", "FRESH") == "FRESH" else 50.0
    consistency = 90.0 if reconciled else 60.0
    model_fit = 90.0 if packet.analysis.industry_adapter == "default_nonfinancial" else 40.0
    return _confidence_formula(
        coverage=coverage_component,
        source_quality=source_quality,
        freshness=freshness,
        consistency=consistency,
        model_fit=model_fit,
    )
