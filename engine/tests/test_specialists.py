"""Tests for the Technical, Valuation and Market & Growth specialists and
their integration into the quick scorecard via a `market` bundle."""

import math

from wbj.quick import quick_scorecard
from wbj.specialists import market_category, technical_category, valuation_category


def _series(vals):
    return [{"end": e, "val": v, "form": "10-K", "fp": "FY"} for e, v in vals]


def _packet():
    yrs = ["2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31"]
    return {
        "ticker": "TEST", "entity": "Test Corp",
        "annual": {
            "revenue": _series(zip(yrs, [80e9, 90e9, 100e9, 110e9, 125e9])),
            "net_income": _series(zip(yrs, [20e9, 23e9, 26e9, 28e9, 32e9])),
            "operating_cash_flow": _series(zip(yrs, [25e9, 28e9, 31e9, 34e9, 38e9])),
            "capex": _series(zip(yrs, [5e9, 6e9, 6e9, 7e9, 8e9])),
            "long_term_debt": _series([(yrs[-1], 40e9)]),
            "equity": _series(zip(yrs[-2:], [70e9, 80e9])),
            "operating_income": _series(zip(yrs[-2:], [30e9, 35e9])),
            "gross_profit": _series([(yrs[-1], 60e9)]),
            "interest_expense": _series([(yrs[-1], 2e9)]),
            "diluted_shares": _series(zip(yrs, [7.0e9, 6.9e9, 6.8e9, 6.7e9, 6.4e9])),
        },
    }


def _history(n=260):
    """Uptrending daily close+volume with genuine down days (sin wobble)."""
    out = []
    base = 20200101
    for i in range(n):
        close = 50 + 0.2 * i + 3 * math.sin(i / 4.0)
        out.append({"time": f"day{i:03d}", "close": round(close, 4), "volume": 1_000_000.0})
    return out


def _benchmark(n=260):
    return [{"time": f"day{i:03d}", "close": 400 + 0.1 * i} for i in range(n)]


def _targets():
    return {
        "status": "ok", "price": 100.0, "eps": 5.0, "pe_now": 20.0,
        "growth_base": 0.12, "horizon": "12 meses",
        "scenarios": [
            {"key": "bear", "label": "Bear", "target": 85.0, "upside": -0.15},
            {"key": "base", "label": "Medio", "target": 118.0, "upside": 0.18},
            {"key": "bull", "label": "Bull", "target": 140.0, "upside": 0.40},
        ],
    }


def _market():
    return {
        "price": 100.0, "history": _history(), "benchmark": _benchmark(),
        "risk_free": 0.043, "targets": _targets(),
    }


# --- Direct specialist behaviour ------------------------------------------

def test_technical_scores_trend_rs_volume_but_flags_gaps():
    cat = technical_category(_history(), _benchmark())
    assert cat.name == "technical" and len(cat.dimensions) == 6
    # Trend / RS / volume / volatility are evidenced; gap + breakout are not.
    cov = cat.coverage()
    assert 0.5 < cov < 0.70  # substantive but incomplete (no gap/breakout/breadth)
    assert 0 <= cat.score10() <= 10


def test_valuation_scores_from_targets_and_cashflow():
    cat = valuation_category(_packet(), _market())
    assert cat.name == "valuation"
    assert cat.coverage() >= 0.70  # multiples + yields + fair value + MOS
    assert 0 <= cat.score10() <= 10


def test_valuation_not_scorable_without_price():
    market = _market()
    market["targets"] = {"status": "not_scorable", "reason": "sin precio"}
    market["price"] = None
    cat = valuation_category(_packet(), market)
    # Only the cash-flow yield partially survives; multiples/fair-value/MOS gone.
    assert cat.coverage() < 0.70


def test_market_is_low_coverage_partial():
    cat = market_category(_packet())
    assert cat.name == "market"
    # Only growth-runway + operating-leverage (7 of 20 pts) are evidenced.
    assert 0.2 < cat.coverage() < 0.5


# --- Integration through quick_scorecard ----------------------------------

def test_scorecard_with_market_scores_valuation_and_flags_partials():
    sc = quick_scorecard(_packet(), _market())
    by = {r["key"]: r for r in sc["categories"]}
    assert by["valuation"]["status"] == "scored"
    assert by["technical"]["status"] == "partial"
    assert by["market"]["status"] == "partial"
    # Partial categories carry a score and a reason, but never inflate evidence.
    for k in ("technical", "market"):
        assert by[k]["score10"] is not None and by[k]["reason"]
    # Counted: business 20 + financial 15 + risk 15 + valuation 10 = 60.
    assert sc["evidence_points_covered"] == 60


def test_scorecard_without_market_keeps_offline_behaviour():
    sc = quick_scorecard(_packet())
    by = {r["key"]: r for r in sc["categories"]}
    assert by["technical"]["status"] == "not_scorable"
    assert by["valuation"]["status"] == "not_scorable"
    assert by["market"]["status"] == "not_scorable"
    assert sc["evidence_points_covered"] == 50
