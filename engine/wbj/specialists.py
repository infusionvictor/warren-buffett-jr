"""Technical, Valuation and Market & Growth category scorers.

These build on the data already flowing through the working app — Yahoo
price/volume (via `wbj.marketdata`), the live price, `wbj.targets` scenarios,
and EDGAR fundamentals — to score the three categories that the quick
scorecard previously left NOT_SCORABLE.

Faithful to the Cerebro methodology's dimension architecture
(`Cerebro/0{3,4,6}_*/SCORING.md`) AND to its non-negotiable rule "missing
evidence is never neutral": each specialist only scores the dimensions it has
real evidence for and returns the rest as NOT_SCORABLE, so `Category.coverage`
honestly reflects how much of the methodology the engine could actually apply.

Dimensions the compute engine cannot derive from these free sources are left
NOT_SCORABLE by design and belong to the orchestrator's qualitative research
(Market TAM/catalysts, consensus revisions) or paid feeds (peer/history
multiples, earnings-gap events):

    Technical  : trend, relative strength, volume, volatility/liquidity scored;
                 earnings-gap and breakout/base NOT_SCORABLE (need event dates
                 and pattern touch-records).
    Valuation  : growth-adjusted multiples, cash-flow/earnings yield, fair
                 value by scenarios, margin of safety scored; peer/history
                 comparison NOT_SCORABLE (needs >=8 peers or a P/E history).
    Market     : growth runway and operating leverage scored from fundamentals;
                 TAM, consensus revisions and catalysts NOT_SCORABLE.
"""

from __future__ import annotations

from wbj import indicators as ind
from wbj.core.nullstates import EvidenceClass, NullState, Value
from wbj.core.scoring import Category, Dimension, anchor_score

# --------------------------------------------------------------------------
# Small helpers (mirror wbj.quick so scoring stays consistent across modules)
# --------------------------------------------------------------------------


def _val(x: float | None, name: str, unit: str = "ratio") -> Value:
    if x is None:
        return Value.null(NullState.MISSING, unit=unit, warnings=[f"MISSING: {name}"])
    return Value.of(x, unit=unit, evidence_class=EvidenceClass.C)


def _scored(v: Value, anchors: list[tuple[float, float]]) -> Value:
    if v.is_null:
        return v
    return Value.of(anchor_score(v.value, anchors), unit="score", evidence_class=EvidenceClass.C)


def _missing(name: str, reason: str) -> Value:
    return Value.null(NullState.NOT_SCORABLE, unit="score", warnings=[f"{name}: {reason}"])


def _dim(name: str, max_points: float, weighted: list[tuple[float, Value]]) -> Dimension:
    return Dimension(name=name, max_points=max_points, metric_scores=weighted)


def _even(name: str, max_points: float, scores: list[Value]) -> Dimension:
    w = 1.0 / len(scores)
    return _dim(name, max_points, [(w, s) for s in scores])


def _latest(series: list[dict]) -> float | None:
    return series[-1]["val"] if series else None


# ==========================================================================
# TECHNICAL & MOMENTUM  (20 pts)  — Cerebro/04_technical_momentum/SCORING.md
# ==========================================================================

# close vs SMA200 (fractional distance); slightly above trend is healthiest.
_A_CLOSE_SMA200 = [(-0.25, 0.0), (-0.05, 3.0), (0.0, 5.0), (0.10, 7.5), (0.25, 10.0)]
# SMA50 vs SMA200 (golden/death-cross spread as a fraction).
_A_SMA_STACK = [(-0.10, 0.0), (-0.02, 3.0), (0.0, 5.0), (0.05, 8.0), (0.12, 10.0)]
# annualized slope of the SMA200 (trend direction).
_A_SMA200_SLOPE = [(-0.30, 0.0), (-0.05, 3.0), (0.0, 5.0), (0.20, 9.0), (0.40, 10.0)]
# relative return vs SPY over each window (excess return).
_A_RS_63 = [(-0.15, 0.0), (-0.03, 3.0), (0.0, 5.0), (0.08, 8.0), (0.18, 10.0)]
_A_RS_126 = [(-0.20, 0.0), (-0.05, 3.0), (0.0, 5.0), (0.12, 8.0), (0.25, 10.0)]
_A_RS_252 = [(-0.30, 0.0), (-0.08, 3.0), (0.0, 5.0), (0.18, 8.0), (0.40, 10.0)]
# accumulation: up/down volume ratio and normalized OBV slope.
_A_UD_VOL = [(0.6, 0.0), (0.9, 3.0), (1.0, 5.0), (1.2, 7.5), (1.6, 10.0)]
_A_OBV_SLOPE = [(-0.6, 0.0), (-0.1, 3.0), (0.0, 5.0), (0.2, 8.0), (0.6, 10.0)]
# volatility quality (annualized; lower is steadier) and liquidity ($/day).
_A_VOL_QUALITY = [(0.15, 10.0), (0.30, 8.0), (0.45, 5.0), (0.65, 3.0), (0.90, 0.0)]
_A_LIQUIDITY = [(1e6, 0.0), (1e7, 5.0), (5e7, 8.0), (2e8, 10.0)]


def technical_category(history: list[dict], benchmark: list[dict]) -> Category:
    """Score Technical & Momentum from ~1y of daily close+volume and SPY."""
    closes = [r["close"] for r in history]
    volumes = [r["volume"] for r in history]
    n = len(closes)
    reason_short = f"solo {n} sesiones (se requieren 200+)"

    # -- Primary price trend (4) : capped without a valid SMA200 (>=200 sess).
    close = closes[-1] if closes else None
    s50, s200 = ind.sma(closes, 50), ind.sma(closes, 200)
    close_sma200 = _val((close / s200 - 1) if (close and s200) else None, "close_vs_sma200")
    sma_stack = _val((s50 / s200 - 1) if (s50 and s200) else None, "sma50_vs_sma200")
    # window=40 so the SMA200 slope is available from ~1 year of daily data
    # (200 + 40 = 240 sessions) rather than needing 263.
    sma200_slope = _val(ind.sma_slope_annualized(closes, 200, window=40), "sma200_slope")
    trend = _dim("Primary price trend", 4.0, [
        (0.4, _scored(close_sma200, _A_CLOSE_SMA200)),
        (0.35, _scored(sma_stack, _A_SMA_STACK)),
        (0.25, _scored(sma200_slope, _A_SMA200_SLOPE)),
    ])

    # -- Relative strength vs the broad market (4).
    def _rs(window: int, name: str) -> Value:
        rt, rb = ind.roc(closes, window), ind.roc([b["close"] for b in benchmark], window)
        return _val((rt - rb) if (rt is not None and rb is not None) else None, name)

    # Long window is 240 (not 252) so 12-month RS is available from ~251
    # trading days of Yahoo "1y" data.
    rs = _dim("Relative strength", 4.0, [
        (0.30, _scored(_rs(63, "rs_3m"), _A_RS_63)),
        (0.35, _scored(_rs(126, "rs_6m"), _A_RS_126)),
        (0.35, _scored(_rs(240, "rs_12m"), _A_RS_252)),
    ])

    # -- Volume and institutional demand (3).
    ud = _val(ind.up_down_volume_ratio(closes, volumes), "up_down_volume")
    obv = _val(ind.obv_slope(closes, volumes), "obv_slope")
    volume = _even("Volume and institutional demand", 3.0,
                   [_scored(ud, _A_UD_VOL), _scored(obv, _A_OBV_SLOPE)])

    # -- Sector breadth and volatility quality (3): breadth needs a
    #    point-in-time sector universe (missing) -> scored on vol+liquidity.
    vq = _val(ind.realized_vol(closes), "volatility_quality", unit="ann_vol")
    liq = _val(ind.avg_dollar_volume(closes, volumes), "liquidity", unit="usd")
    breadth = _missing("sector_breadth", "requiere universo sectorial point-in-time")
    volq = _dim("Sector breadth and volatility quality", 3.0, [
        (0.4, _scored(vq, _A_VOL_QUALITY)),
        (0.3, _scored(liq, _A_LIQUIDITY)),
        (0.3, breadth),
    ])

    # -- Dimensions the engine cannot evidence from price/volume alone.
    gap = _dim("Earnings-gap behavior", 3.0,
               [(1.0, _missing("earnings_gap", "requiere fechas de earnings y holds de gap"))])
    base = _dim("Breakout and base quality", 3.0,
                [(1.0, _missing("breakout_base", "requiere registros de toques de niveles/pivotes"))])

    return Category(name="technical", max_points=20.0,
                    dimensions=[trend, rs, volume, volq, gap, base])


# ==========================================================================
# VALUATION  (10 pts)  — Cerebro/06_valuation_analysis/SCORING.md
# ==========================================================================

# PEG (P/E to growth): lower is cheaper for the growth.
_A_PEG = [(0.5, 10.0), (1.0, 7.5), (1.5, 6.0), (2.5, 3.0), (4.0, 0.0)]
# earnings yield in excess of the risk-free rate.
_A_EY_EXCESS = [(-0.03, 0.0), (0.0, 4.0), (0.02, 6.0), (0.05, 9.0), (0.08, 10.0)]
# free-cash-flow yield (FCF / market cap).
_A_FCF_YIELD = [(-0.02, 0.0), (0.0, 4.0), (0.03, 7.0), (0.06, 9.0), (0.10, 10.0)]
# base-scenario upside (price vs 12m base target).
_A_BASE_UPSIDE = [(-0.20, 0.0), (0.0, 5.0), (0.15, 8.0), (0.30, 10.0)]
# reward/risk skew: distance to bull vs distance to bear.
_A_REWARD_RISK = [(0.5, 0.0), (1.0, 4.0), (2.0, 7.0), (3.0, 10.0)]
# margin of safety (same as base upside but its own 1-pt dimension).
_A_MOS = [(-0.05, 0.0), (0.0, 3.0), (0.10, 5.0), (0.15, 7.0), (0.25, 9.0), (0.40, 10.0)]


def valuation_category(packet: dict, market: dict) -> Category:
    """Score Valuation from live price, targets, and EDGAR cash flows."""
    targets = market.get("targets", {})
    price = market.get("price")
    rf = market.get("risk_free", 0.043)

    ok = targets.get("status") == "ok"
    peg_reason = "requiere precio, EPS>0 y crecimiento" if not ok else ""

    # -- Growth-adjusted multiples (3): PEG = P/E / (growth% ).
    if ok and targets["growth_base"] > 0:
        peg = targets["pe_now"] / (targets["growth_base"] * 100.0)
        peg_v = _scored(_val(peg, "peg"), _A_PEG)
    else:
        peg_v = _missing("peg", peg_reason or "crecimiento no positivo")
    multiples = _dim("Growth-adjusted multiples", 3.0, [(1.0, peg_v)])

    # -- Historical and peer comparison (2): needs >=8 peers or a P/E history.
    peer = _dim("Historical and peer comparison", 2.0,
                [(1.0, _missing("peer_history", "requiere >=8 pares o historial de multiplos (FMP)"))])

    # -- Cash-flow and earnings yield (2).
    a = packet["annual"]
    ni_l = _latest(a["net_income"])
    sh_l = _latest(a.get("diluted_shares", []))
    ocf_l, capex_l = _latest(a["operating_cash_flow"]), _latest(a["capex"])
    fcf = (ocf_l - capex_l) if (ocf_l is not None and capex_l is not None) else None
    ey_excess = _val((1.0 / targets["pe_now"] - rf) if ok else None, "earnings_yield_excess")
    mktcap = (price * sh_l) if (price and sh_l) else None
    fcf_yield = _val((fcf / mktcap) if (fcf is not None and mktcap) else None, "fcf_yield")
    yields = _even("Cash-flow and earnings yield", 2.0,
                   [_scored(ey_excess, _A_EY_EXCESS), _scored(fcf_yield, _A_FCF_YIELD)])

    # -- Fair value by scenarios (2): price vs bull/base/bear.
    if ok:
        by = {s["key"]: s for s in targets["scenarios"]}
        base_up = _scored(_val(by["base"]["upside"], "base_upside"), _A_BASE_UPSIDE)
        bull_t, bear_t, px = by["bull"]["target"], by["bear"]["target"], targets["price"]
        rr = (bull_t - px) / (px - bear_t) if px > bear_t else None
        reward_risk = _scored(_val(rr, "reward_risk"), _A_REWARD_RISK)
    else:
        base_up = _missing("base_upside", "sin targets")
        reward_risk = _missing("reward_risk", "sin targets")
    fair_value = _even("Fair value by scenarios", 2.0, [base_up, reward_risk])

    # -- Margin of safety (1).
    mos_v = _scored(_val(by["base"]["upside"], "margin_of_safety"), _A_MOS) if ok \
        else _missing("margin_of_safety", "sin targets")
    mos = _dim("Margin of safety", 1.0, [(1.0, mos_v)])

    return Category(name="valuation", max_points=10.0,
                    dimensions=[multiples, peer, yields, fair_value, mos])


# ==========================================================================
# MARKET & GROWTH  (20 pts)  — Cerebro/03_market_analysis/SCORING.md
# Only the fundamentals-derivable dimensions are scorable by the engine; TAM,
# consensus revisions and catalysts belong to the orchestrator / paid feeds.
# ==========================================================================

_A_REV_CAGR = [(-0.05, 0.0), (0.0, 3.0), (0.08, 6.0), (0.15, 8.0), (0.30, 10.0)]
_A_ROE_CAP = [(0.0, 0.0), (0.08, 4.0), (0.15, 7.0), (0.30, 10.0)]
_A_ACCEL = [(-0.15, 0.0), (-0.03, 3.0), (0.0, 5.0), (0.05, 8.0), (0.15, 10.0)]
_A_INCR_MARGIN = [(-0.10, 0.0), (0.0, 3.0), (0.15, 5.0), (0.30, 7.5), (0.50, 10.0)]


def _cagr(series: list[dict], years: int) -> float | None:
    rows = series[-years:]
    if len(rows) < 3:
        return None
    begin, end = rows[0]["val"], rows[-1]["val"]
    if begin <= 0 or end <= 0:
        return None
    return (end / begin) ** (1 / (len(rows) - 1)) - 1


def market_category(packet: dict) -> Category:
    """Score the fundamentals-derivable slice of Market & Growth.

    TAM, revisions and catalysts stay NOT_SCORABLE (need consensus estimates
    and qualitative research) so the category reports low coverage honestly.
    """
    a = packet["annual"]
    rev, ni, eq, op = a["revenue"], a["net_income"], a["equity"], a.get("operating_income", [])

    # -- Growth runway and share capture (4): trajectory + reinvestment capacity.
    cagr5 = _cagr(rev, 5)
    cagr2 = _cagr(rev, 3)  # 3 points -> ~2y CAGR, the recent leg
    accel = (cagr2 - cagr5) if (cagr2 is not None and cagr5 is not None) else None
    ni_l, eq_l = _latest(ni), _latest(eq)
    eq_avg = (eq[-1]["val"] + eq[-2]["val"]) / 2 if len(eq) >= 2 else eq_l
    roe = (ni_l / eq_avg) if (ni_l is not None and eq_avg) else None
    runway = _dim("Growth runway and share capture", 4.0, [
        (0.45, _scored(_val(cagr5, "revenue_cagr_5y"), _A_REV_CAGR)),
        (0.30, _scored(_val(roe, "reinvestment_capacity_roe"), _A_ROE_CAP)),
        (0.25, _scored(_val(accel, "growth_acceleration"), _A_ACCEL)),
    ])

    # -- Operating leverage and market confirmation (3): incremental margin.
    #    Sector-breadth "confirmation" needs a universe -> left out of the
    #    dimension weight; scored on incremental economics alone.
    incr = None
    if len(rev) >= 2 and len(op) >= 2:
        d_rev = rev[-1]["val"] - rev[-2]["val"]
        d_op = op[-1]["val"] - op[-2]["val"]
        if d_rev > 0:
            incr = d_op / d_rev
    op_lev = _dim("Operating leverage", 3.0,
                  [(1.0, _scored(_val(incr, "incremental_margin"), _A_INCR_MARGIN))])

    # -- Dimensions requiring external market data / research.
    tam = _dim("TAM and industry tailwind", 5.0,
               [(1.0, _missing("tam", "requiere dimensionamiento de mercado externo"))])
    revisions = _dim("Earnings and revenue revisions", 4.0,
                     [(1.0, _missing("revisions", "requiere consenso congelado con timestamp (FMP/FinnHub)"))])
    catalysts = _dim("Product and business catalysts", 4.0,
                     [(1.0, _missing("catalysts", "requiere investigacion cualitativa del orquestador"))])

    return Category(name="market", max_points=20.0,
                    dimensions=[tam, revisions, catalysts, runway, op_lev])
