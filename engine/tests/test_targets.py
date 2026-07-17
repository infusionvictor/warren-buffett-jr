"""Tests for Bull/Base/Bear targets and Spanish narrative."""

import httpx

from wbj.targets import live_price, narrative, price_targets


def _series(vals):
    return [{"end": e, "val": v, "form": "10-K", "fp": "FY"} for e, v in vals]


def _packet():
    years = ["2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31"]
    return {
        "ticker": "TEST",
        "annual": {
            "revenue": _series(list(zip(years, [80e9, 90e9, 100e9, 110e9, 125e9]))),
            "net_income": _series(list(zip(years, [20e9, 23e9, 26e9, 28e9, 32e9]))),
            "operating_cash_flow": _series(list(zip(years, [25e9, 28e9, 31e9, 34e9, 38e9]))),
            "capex": _series(list(zip(years, [5e9, 6e9, 6e9, 7e9, 8e9]))),
            "long_term_debt": _series([(years[-1], 40e9)]),
            "equity": _series(list(zip(years[-2:], [70e9, 80e9]))),
            "operating_income": _series([(years[-1], 35e9)]),
            "gross_profit": _series([(years[-1], 60e9)]),
            "interest_expense": _series([(years[-1], 2e9)]),
            "diluted_shares": _series([(years[-1], 4e9)]),
        },
    }


def test_targets_three_scenarios_ordered():
    t = price_targets(_packet(), price=200.0)
    assert t["status"] == "ok"
    assert t["eps"] == 8.0 and t["pe_now"] == 25.0
    by = {r["key"]: r for r in t["scenarios"]}
    assert set(by) == {"bear", "base", "bull"}
    assert by["bear"]["target"] < by["base"]["target"] < by["bull"]["target"]
    # base: eps * (1+g) * pe_now, g = NI 5y CAGR = (32/20)^(1/4)-1
    g = (32 / 20) ** 0.25 - 1
    assert abs(by["base"]["target"] - round(8 * (1 + g) * 25, 2)) < 0.02
    assert "crecimiento" in by["base"]["assumptions"]


def test_targets_not_scorable_without_price_or_eps():
    assert price_targets(_packet(), price=None)["status"] == "not_scorable"
    p = _packet()
    p["annual"]["net_income"] = _series([("2025-12-31", -1e9)])
    assert price_targets(p, price=200.0)["status"] == "not_scorable"


def test_narrative_covers_key_facts_in_spanish():
    p = _packet()
    t = price_targets(p, price=200.0)
    from wbj.quick import quick_scorecard
    sents = narrative(p, quick_scorecard(p), t)
    joined = " ".join(sents)
    assert "Vendio" in joined and "ganancia neta" in joined
    assert "flujo de caja libre" in joined
    assert "12 meses" in joined
    assert 4 <= len(sents) <= 7


def test_live_price_fmp_primary():
    def handler(request):
        assert "financialmodelingprep" in str(request.url)
        assert request.url.params["symbol"] == "AAPL"
        return httpx.Response(200, json=[{"symbol": "AAPL", "price": 212.5}])
    c = httpx.Client(transport=httpx.MockTransport(handler))
    assert live_price("AAPL", fmp_api_key="k", client=c) == 212.5


def test_live_price_falls_back_to_yahoo_then_none():
    calls = []
    def handler(request):
        calls.append(request.url.host)
        if "financialmodelingprep" in request.url.host:
            return httpx.Response(403)
        return httpx.Response(200, json={"chart": {"result": [{"meta": {"regularMarketPrice": 74.35}}]}})
    c = httpx.Client(transport=httpx.MockTransport(handler))
    assert live_price("NFLX", fmp_api_key="k", client=c) == 74.35
    assert len(calls) == 2  # FMP tried first, then Yahoo

    dead = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    assert live_price("NFLX", fmp_api_key=None, client=dead) is None
