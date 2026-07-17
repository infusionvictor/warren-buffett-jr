"""Tests for analyst-consensus normalization and the Market revisions lift."""

import json
from pathlib import Path

from wbj.marketdata import consensus_from_finnhub, consensus_from_fmp
from wbj.quick import quick_scorecard
from wbj.specialists import market_category

FIX = Path(__file__).parent / "fixtures"


def _load(*parts):
    return json.loads((FIX.joinpath(*parts)).read_text(encoding="utf-8"))


def _series(vals):
    return [{"end": e, "val": v, "form": "10-K", "fp": "FY"} for e, v in vals]


def _packet():
    yrs = ["2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31"]
    return {
        "ticker": "NVDA", "entity": "Test",
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
            "diluted_shares": _series(zip(yrs, [7e9, 6.9e9, 6.8e9, 6.7e9, 6.4e9])),
        },
    }


# last actual: revenue 125e9, eps = 32e9 / 6.4e9 = 5.0
LAST_REV, LAST_EPS = 125e9, 5.0


def test_consensus_from_fmp_normalizes():
    est = _load("fmp", "analyst_estimates.json")
    cal = _load("fmp", "earnings_calendar.json")
    c = consensus_from_fmp(est, cal, LAST_REV, LAST_EPS)
    assert c["source"] == "fmp"
    assert c["num_analysts"] == 42
    assert abs(c["fwd_rev_growth"] - (140e9 / 125e9 - 1)) < 1e-6   # 0.12
    assert abs(c["rev_dispersion"] - (16e9 / 140e9)) < 1e-6         # ~0.114
    assert c["beat_rate"] == 1.0   # both calendar rows beat estimate
    assert c["n_surprises"] == 2


def test_consensus_from_finnhub_normalizes():
    eps = _load("finnhub", "eps_estimate.json")
    rev = _load("finnhub", "revenue_estimate.json")
    c = consensus_from_finnhub(eps, rev, LAST_REV, LAST_EPS)
    assert c["source"] == "finnhub"
    assert c["num_analysts"] == 37       # forward revenue row analyst count
    assert c["beat_rate"] == 1.0         # the one actual (0.81) beat 0.75
    assert c["n_surprises"] == 1


def test_consensus_none_on_empty():
    assert consensus_from_fmp([], [], LAST_REV, LAST_EPS) is None
    assert consensus_from_finnhub({}, {}, LAST_REV, LAST_EPS) is None


def test_revisions_gated_below_five_analysts():
    cat_no = market_category(_packet(), None)               # no consensus
    cat_few = market_category(_packet(), {"num_analysts": 3})  # too few
    for cat in (cat_no, cat_few):
        rev_dim = next(d for d in cat.dimensions if d.name.startswith("Earnings"))
        assert rev_dim.score10_value().is_null


def test_market_coverage_lifts_with_consensus():
    base = market_category(_packet(), None).coverage()
    c = consensus_from_fmp(_load("fmp", "analyst_estimates.json"),
                           _load("fmp", "earnings_calendar.json"), LAST_REV, LAST_EPS)
    lifted = market_category(_packet(), c).coverage()
    assert base < 0.45                     # ~0.35 fundamentals only
    assert 0.5 < lifted < 0.70             # ~0.55 with revisions, still partial


def test_scorecard_market_partial_with_consensus():
    c = consensus_from_fmp(_load("fmp", "analyst_estimates.json"),
                           _load("fmp", "earnings_calendar.json"), LAST_REV, LAST_EPS)
    market = {"history": [], "benchmark": [], "risk_free": 0.043,
              "targets": {"status": "not_scorable", "reason": "x"}, "consensus": c}
    sc = quick_scorecard(_packet(), market)
    m = next(r for r in sc["categories"] if r["key"] == "market")
    assert m["status"] == "partial" and m["coverage"] >= 0.5
