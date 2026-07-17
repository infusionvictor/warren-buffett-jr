"""Quick 6-agent scorecard (1-10 per category) from the MVP EDGAR packet.

Honest-scoring rules (Cerebro: "sin evidencia, no hay número"):
- Financial, Business, Risk: scored from real EDGAR fundamentals (partial
  metric sets — a *quick* read, not the full specialist methodology).
- Market, Technical, Valuation: NOT_SCORABLE until their engines land
  (need consensus estimates / OHLCV / market price — never imputed).
- Overall 1-10 weights only the covered categories and reports how many
  of the 100 evidence points that covers.
"""

from __future__ import annotations

from statistics import pstdev

from wbj.core.formulas import yoy
from wbj.core.nullstates import EvidenceClass, NullState, Value
from wbj.core.scoring import (
    CATEGORY_WEIGHTS,
    COVERAGE_USABLE,
    Category,
    Dimension,
    anchor_score,
)
from wbj.specialists import market_category, technical_category, valuation_category

# --- Anchors aligned with Cerebro FIN/BUS band cutoffs (quick defaults) ---
_A_REV_GROWTH = [(-0.10, 0.0), (0.0, 3.0), (0.10, 6.0), (0.25, 9.0), (0.40, 10.0)]
_A_NET_MARGIN = [(-0.10, 0.0), (0.0, 3.0), (0.10, 6.0), (0.20, 8.5), (0.30, 10.0)]
_A_FCF_MARGIN = [(-0.10, 0.0), (0.0, 3.0), (0.10, 6.5), (0.25, 10.0)]
_A_DEBT_EQUITY = [(0.0, 10.0), (0.5, 8.0), (1.0, 6.0), (2.0, 3.0), (4.0, 0.0)]
_A_OP_MARGIN = [(0.0, 0.0), (0.10, 4.0), (0.20, 7.0), (0.35, 10.0)]
_A_GROSS_MARGIN = [(0.0, 0.0), (0.20, 4.0), (0.40, 7.0), (0.60, 10.0)]
_A_ROE = [(0.0, 0.0), (0.08, 4.0), (0.15, 7.0), (0.30, 10.0)]
# margin stability: 5y net-margin std-dev (lower = steadier = moat signal)
_A_MARGIN_STDEV = [(0.01, 10.0), (0.03, 7.0), (0.06, 4.0), (0.12, 0.0)]
# interest coverage: <1.5x solvency warning band, >5x comfortable
_A_INT_COVERAGE = [(0.0, 0.0), (1.5, 2.0), (3.0, 5.0), (5.0, 7.0), (10.0, 10.0)]

_QUICK_LABEL = {
    "business": "Business (quick)",
    "financial": "Financial (quick)",
    "market": "Market & Growth",
    "technical": "Technical & Momentum",
    "risk": "Risk & Resilience (quick)",
    "valuation": "Valuation",
}
_NS_REASON = {
    "market": "necesita datos de mercado/consenso",
    "technical": "necesita historial de precios",
    "valuation": "necesita precio de mercado",
}


def _unscored_dim_names(cat: Category) -> list[str]:
    """Names of dimensions that came back NOT_SCORABLE (for partial reasons)."""
    return [d.name for d in cat.dimensions if d.score10_value().is_null]


def _val(x: float | None, name: str, unit: str = "ratio") -> Value:
    if x is None:
        return Value.null(NullState.MISSING, unit=unit, warnings=[f"MISSING: {name}"])
    return Value.of(x, unit=unit, evidence_class=EvidenceClass.C)


def _scored(v: Value, anchors: list[tuple[float, float]]) -> Value:
    if v.is_null:
        return v
    return Value.of(anchor_score(v.value, anchors), unit="score", evidence_class=EvidenceClass.C)


def _latest(series: list[dict]) -> float | None:
    return series[-1]["val"] if series else None


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or not den:
        return None
    return num / den


def _margin_history(ni: list[dict], rev: list[dict], years: int = 5) -> list[float]:
    by_end = {r["end"]: r["val"] for r in rev if r.get("val")}
    pairs = [(r["val"] / by_end[r["end"]]) for r in ni if r["end"] in by_end]
    return pairs[-years:]


def _dim(name: str, max_points: float, scores: list[Value]) -> Dimension:
    w = 1.0 / len(scores)
    return Dimension(name=name, max_points=max_points, metric_scores=[(w, s) for s in scores])


def quick_scorecard(packet: dict, market: dict | None = None) -> dict:
    """Compute the 6-category scorecard from an MVP packet.

    Business/Financial/Risk score from EDGAR fundamentals. When a `market`
    bundle (`wbj.marketdata.bundle`) is supplied, Technical, Valuation and
    the fundamentals-derivable slice of Market & Growth are scored too;
    without it those three stay NOT_SCORABLE (preserving the offline/CLI
    behaviour). Categories below the 70% coverage threshold are reported as
    ``partial`` and excluded from the headline overall to keep the evidence
    count honest.
    """
    a = packet["annual"]
    rev, ni = a["revenue"], a["net_income"]
    ocf, capex = a["operating_cash_flow"], a["capex"]
    debt, eq = a["long_term_debt"], a["equity"]
    op = a.get("operating_income", [])
    gp = a.get("gross_profit", [])
    interest = a.get("interest_expense", [])

    rev_l, ni_l = _latest(rev), _latest(ni)
    ocf_l, capex_l = _latest(ocf), _latest(capex)
    debt_l, eq_l = _latest(debt), _latest(eq)
    op_l, gp_l, int_l = _latest(op), _latest(gp), _latest(interest)

    growth = (
        yoy(rev[-1]["val"], rev[-2]["val"])
        if len(rev) >= 2
        else Value.null(NullState.MISSING, unit="ratio", warnings=["MISSING: revenue history"])
    )
    fcf = (ocf_l - capex_l) if ocf_l is not None and capex_l is not None else None
    margins = _margin_history(ni, rev)
    margin_stdev = _val(pstdev(margins) if len(margins) >= 3 else None, "margin_stability")
    # ROE on average equity when two years exist (Cerebro convention).
    eq_avg = (eq[-1]["val"] + eq[-2]["val"]) / 2 if len(eq) >= 2 else eq_l
    roe = _val(_ratio(ni_l, eq_avg), "roe")

    net_margin = _val(_ratio(ni_l, rev_l), "net_margin")
    fcf_margin = _val(_ratio(fcf, rev_l), "fcf_margin")
    d_e = _val(_ratio(debt_l, eq_l), "debt_to_equity")
    op_margin = _val(_ratio(op_l, rev_l), "operating_margin")
    gross_margin = _val(_ratio(gp_l, rev_l), "gross_margin")
    # EBIT proxy / interest expense; interest of 0/missing -> NOT_MEANINGFUL.
    coverage = (
        _val(_ratio(op_l, int_l), "interest_coverage")
        if int_l
        else Value.null(NullState.NOT_MEANINGFUL, unit="x", warnings=["no interest expense reported"])
    )
    fcf_positive = _val(None if fcf is None else (1.0 if fcf > 0 else 0.0), "fcf_positive")

    categories: dict[str, Category] = {
        "financial": Category(name="financial", max_points=15.0, dimensions=[
            _dim("Profitability", 7.5, [_scored(net_margin, _A_NET_MARGIN),
                                        _scored(fcf_margin, _A_FCF_MARGIN)]),
            _dim("Growth & Balance Sheet", 7.5, [_scored(growth, _A_REV_GROWTH),
                                                 _scored(d_e, _A_DEBT_EQUITY)]),
        ]),
        "business": Category(name="business", max_points=20.0, dimensions=[
            _dim("Margin power", 10.0, [_scored(op_margin, _A_OP_MARGIN),
                                        _scored(gross_margin, _A_GROSS_MARGIN)]),
            _dim("Durability & returns", 10.0, [_scored(margin_stdev, _A_MARGIN_STDEV),
                                                _scored(roe, _A_ROE)]),
        ]),
        "risk": Category(name="risk", max_points=15.0, dimensions=[
            _dim("Solvency", 7.5, [_scored(d_e, _A_DEBT_EQUITY),
                                   _scored(coverage, _A_INT_COVERAGE)]),
            _dim("Cash generation", 7.5, [_scored(fcf_positive, [(0.0, 2.0), (1.0, 10.0)]),
                                          _scored(fcf_margin, _A_FCF_MARGIN)]),
        ]),
    }

    # Market-data-driven specialists (only when a market bundle is supplied).
    if market is not None:
        categories["technical"] = technical_category(
            market.get("history", []), market.get("benchmark", []))
        categories["valuation"] = valuation_category(packet, market)
        categories["market"] = market_category(packet)

    rows: list[dict] = []
    covered_pts = 0.0
    weighted = 0.0
    partial_keys: list[str] = []
    for key in ("business", "financial", "market", "technical", "risk", "valuation"):
        max_pts = float(CATEGORY_WEIGHTS[key])
        cat = categories.get(key)
        cov = cat.coverage() if cat is not None else 0.0
        score10 = round(cat.score10(), 1) if (cat is not None and cov > 0) else None

        if score10 is None:
            rows.append({
                "key": key, "label": _QUICK_LABEL[key], "max_points": max_pts,
                "score10": None, "points": None, "coverage": round(cov, 2),
                "status": "not_scorable",
                "reason": _NS_REASON.get(key, "sin datos suficientes"),
            })
            continue

        row = {
            "key": key, "label": _QUICK_LABEL[key], "max_points": max_pts,
            "score10": score10, "points": round(cat.points(), 2),
            "coverage": round(cov, 2),
        }
        if cov >= COVERAGE_USABLE:
            covered_pts += max_pts
            weighted += max_pts * score10
            row["status"] = "scored"
        else:
            # Below 70% coverage: show the partial read, flag what's missing,
            # but do NOT let it inflate the headline evidence count.
            partial_keys.append(key)
            row["status"] = "partial"
            missing = _unscored_dim_names(cat)
            row["reason"] = (
                f"cobertura {cov:.0%} — pendiente: {', '.join(missing)}"
                if missing else f"cobertura {cov:.0%}"
            )
        rows.append(row)

    overall = round(weighted / covered_pts, 1) if covered_pts else None
    note = (
        "Scorecard con fundamentales SEC EDGAR"
        + ("" if market is None else " + precio/volumen (Yahoo) y targets")
        + f"; {int(covered_pts)}/100 puntos de evidencia con cobertura suficiente."
    )
    if partial_keys:
        note += (
            " Categorías parciales (no cuentan al overall): "
            + ", ".join(_QUICK_LABEL[k] for k in partial_keys) + "."
        )
    return {
        "categories": rows,
        "overall_10": overall,
        "evidence_points_covered": int(covered_pts),
        "evidence_points_total": 100,
        "disclaimer": note,
    }
