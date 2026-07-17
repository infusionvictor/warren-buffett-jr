"""Tests for wbj.providers.finnhub.FinnhubProvider and wbj.providers.fred.FredProvider."""

import json
from pathlib import Path

import httpx

from wbj.config import Settings
from wbj.core.nullstates import NullState, Value
from wbj.providers.cache import Cache
from wbj.providers.finnhub import FinnhubProvider
from wbj.providers.fred import FredProvider

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _load_fixture(provider: str, name: str):
    return json.loads((FIXTURES_DIR / provider / f"{name}.json").read_text())


def _capturing_handler(provider, fixture_name, captured):
    def handler(request):
        captured["request"] = request
        return httpx.Response(200, json=_load_fixture(provider, fixture_name))

    return handler


# =============================================================================
# FinnhubProvider
# =============================================================================


def _make_finnhub(tmp_path, handler, finnhub_api_key="testkey"):
    settings = Settings(finnhub_api_key=finnhub_api_key)
    cache = Cache(tmp_path)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return FinnhubProvider(settings, cache, client=client)


def test_finnhub_available_true_when_api_key_set(tmp_path):
    p = _make_finnhub(tmp_path, lambda request: httpx.Response(200, json={}))
    assert p.available is True


def test_finnhub_available_false_when_api_key_missing(tmp_path):
    p = _make_finnhub(
        tmp_path, lambda request: httpx.Response(200, json={}), finnhub_api_key=None
    )
    assert p.available is False


def test_finnhub_all_methods_return_none_and_skip_network_when_unavailable(tmp_path):
    def handler(request):
        raise AssertionError("transport should not be called when unavailable")

    p = _make_finnhub(tmp_path, handler, finnhub_api_key=None)

    assert p.estimates("NVDA") is None
    assert p.revenue_estimates("NVDA") is None
    assert p.earnings_calendar("NVDA") is None
    assert p.quote("NVDA") is None


def test_finnhub_estimates_url_and_params(tmp_path):
    captured = {}
    p = _make_finnhub(
        tmp_path, _capturing_handler("finnhub", "eps_estimate", captured)
    )

    result = p.estimates("NVDA")

    req = captured["request"]
    assert req.url.path == "/api/v1/stock/eps-estimate"
    assert req.url.params.get("symbol") == "NVDA"
    assert req.url.params.get("token") == "testkey"
    assert result == _load_fixture("finnhub", "eps_estimate")


def test_finnhub_revenue_estimates_url_and_params(tmp_path):
    captured = {}
    p = _make_finnhub(
        tmp_path, _capturing_handler("finnhub", "revenue_estimate", captured)
    )

    result = p.revenue_estimates("NVDA")

    req = captured["request"]
    assert req.url.path == "/api/v1/stock/revenue-estimate"
    assert req.url.params.get("symbol") == "NVDA"
    assert result == _load_fixture("finnhub", "revenue_estimate")


def test_finnhub_earnings_calendar_url_and_params(tmp_path):
    captured = {}
    p = _make_finnhub(
        tmp_path, _capturing_handler("finnhub", "earnings_calendar", captured)
    )

    result = p.earnings_calendar("NVDA")

    req = captured["request"]
    assert req.url.path == "/api/v1/calendar/earnings"
    assert req.url.params.get("symbol") == "NVDA"
    assert result == _load_fixture("finnhub", "earnings_calendar")


def test_finnhub_quote_url_and_params(tmp_path):
    captured = {}
    p = _make_finnhub(tmp_path, _capturing_handler("finnhub", "quote", captured))

    result = p.quote("NVDA")

    req = captured["request"]
    assert req.url.path == "/api/v1/quote"
    assert req.url.params.get("symbol") == "NVDA"
    assert result == _load_fixture("finnhub", "quote")


def test_finnhub_methods_use_distinct_cache_keys(tmp_path):
    cache = Cache(tmp_path)

    def handler(request):
        path = request.url.path
        if "eps-estimate" in path:
            return httpx.Response(200, json=_load_fixture("finnhub", "eps_estimate"))
        if "quote" in path:
            return httpx.Response(200, json=_load_fixture("finnhub", "quote"))
        return httpx.Response(200, json={})

    settings = Settings(finnhub_api_key="testkey")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    p = FinnhubProvider(settings, cache, client=client)

    p.estimates("NVDA")
    p.quote("NVDA")

    assert cache.get("NVDA", "estimates") == _load_fixture("finnhub", "eps_estimate")
    assert cache.get("NVDA", "quote") == _load_fixture("finnhub", "quote")


def test_finnhub_get_json_serves_from_cache_without_hitting_transport(tmp_path):
    cache = Cache(tmp_path)
    cache.put("NVDA", "quote", _load_fixture("finnhub", "quote"))

    def handler(request):
        raise AssertionError("transport should not be called on cache hit")

    settings = Settings(finnhub_api_key="testkey")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    p = FinnhubProvider(settings, cache, client=client)

    result = p.quote("NVDA")
    assert result == _load_fixture("finnhub", "quote")


# =============================================================================
# FredProvider
# =============================================================================


def _make_fred(tmp_path, handler, fred_api_key="testkey"):
    settings = Settings(fred_api_key=fred_api_key)
    cache = Cache(tmp_path)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return FredProvider(settings, cache, client=client)


def test_fred_available_true_when_api_key_set(tmp_path):
    p = _make_fred(tmp_path, lambda request: httpx.Response(200, json={}))
    assert p.available is True


def test_fred_available_false_when_api_key_missing(tmp_path):
    p = _make_fred(
        tmp_path, lambda request: httpx.Response(200, json={}), fred_api_key=None
    )
    assert p.available is False


def test_fred_series_returns_none_and_skips_network_when_unavailable(tmp_path):
    def handler(request):
        raise AssertionError("transport should not be called when unavailable")

    p = _make_fred(tmp_path, handler, fred_api_key=None)

    assert p.series("DGS10") is None


def test_fred_series_url_and_params(tmp_path):
    captured = {}
    p = _make_fred(tmp_path, _capturing_handler("fred", "dgs10", captured))

    result = p.series("DGS10")

    req = captured["request"]
    assert req.url.path == "/fred/series/observations"
    assert req.url.params.get("series_id") == "DGS10"
    assert req.url.params.get("sort_order") == "desc"
    assert req.url.params.get("limit") == "120"
    assert req.url.params.get("api_key") == "testkey"
    assert req.url.params.get("file_type") == "json"
    assert result == _load_fixture("fred", "dgs10")


def test_fred_series_custom_limit(tmp_path):
    captured = {}
    p = _make_fred(tmp_path, _capturing_handler("fred", "dgs10", captured))

    p.series("DGS10", limit=30)

    assert captured["request"].url.params.get("limit") == "30"


def test_fred_series_cache_key_is_namespaced_by_series_id(tmp_path):
    cache = Cache(tmp_path)

    def handler(request):
        return httpx.Response(200, json=_load_fixture("fred", "dgs10"))

    settings = Settings(fred_api_key="testkey")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    p = FredProvider(settings, cache, client=client)

    p.series("DGS10")

    assert cache.get("_macro", "fred_DGS10") == _load_fixture("fred", "dgs10")


def test_fred_series_serves_from_cache_without_hitting_transport(tmp_path):
    cache = Cache(tmp_path)
    cache.put("_macro", "fred_DGS10", _load_fixture("fred", "dgs10"))

    def handler(request):
        raise AssertionError("transport should not be called on cache hit")

    settings = Settings(fred_api_key="testkey")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    p = FredProvider(settings, cache, client=client)

    result = p.series("DGS10")
    assert result == _load_fixture("fred", "dgs10")


# --- risk_free_rate ----------------------------------------------------------


def test_fred_risk_free_rate_divides_latest_observation_by_100(tmp_path):
    def handler(request):
        return httpx.Response(200, json=_load_fixture("fred", "dgs10"))

    p = _make_fred(tmp_path, handler)

    result = p.risk_free_rate()

    assert isinstance(result, Value)
    assert result.is_valid
    assert result.value == 0.0425
    assert result.unit == "ratio"
    assert result.source_name == "FRED"
    assert result.source_locator == "DGS10"


def test_fred_risk_free_rate_skips_dot_placeholder_values(tmp_path):
    def handler(request):
        return httpx.Response(200, json=_load_fixture("fred", "dgs10_with_dot"))

    p = _make_fred(tmp_path, handler)

    result = p.risk_free_rate()

    assert result.is_valid
    assert result.value == 0.0418


def test_fred_risk_free_rate_all_missing_returns_null(tmp_path):
    def handler(request):
        return httpx.Response(200, json=_load_fixture("fred", "dgs10_all_missing"))

    p = _make_fred(tmp_path, handler)

    result = p.risk_free_rate()

    assert result.is_null
    assert result.state == NullState.MISSING


def test_fred_risk_free_rate_unavailable_returns_null_without_network(tmp_path):
    def handler(request):
        raise AssertionError("transport should not be called when unavailable")

    p = _make_fred(tmp_path, handler, fred_api_key=None)

    result = p.risk_free_rate()

    assert result.is_null
    assert result.state == NullState.MISSING


def test_fred_risk_free_rate_empty_observations_returns_null(tmp_path):
    def handler(request):
        return httpx.Response(200, json={"observations": []})

    p = _make_fred(tmp_path, handler)

    result = p.risk_free_rate()

    assert result.is_null
    assert result.state == NullState.MISSING
