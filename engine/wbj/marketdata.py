"""Market-data bundle for the Technical and Valuation specialists.

Extends the keyless Yahoo access already used by `wbj.targets` to also pull
daily VOLUME and a broad-market benchmark (SPY), plus the risk-free rate
(10-year Treasury via Yahoo's ^TNX). Everything degrades to None/[] on
failure so specialists mark the affected dimensions MISSING rather than
inventing numbers.

The `market` dict returned by `bundle()` is the contract consumed by
`wbj.quick.quick_scorecard(packet, market=...)`:

    {
      "price":      float | None,          # latest trade price
      "history":    [{"time","close","volume"}...],  # ticker, ~1y daily
      "benchmark":  [{"time","close"}...],           # SPY, ~1y daily
      "risk_free":  float,                 # 10y yield as a decimal (e.g. 0.043)
      "targets":    {...},                 # wbj.targets.price_targets output
    }
"""

from __future__ import annotations

import httpx

from wbj.targets import live_price, price_targets

_YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{t}"
_YAHOO_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh) warren-buffett-jr"}
_BENCHMARK = "SPY"
_RISK_FREE_DEFAULT = 0.043  # ~10y UST fallback when ^TNX is unreachable


def ohlcv(ticker: str, client: httpx.Client | None = None) -> list[dict]:
    """Daily close + volume for the last year (Yahoo, keyless).

    Returns [{"time","close","volume"}...] oldest first; [] on failure.
    Rows with a missing close are skipped; a missing volume becomes 0.0.
    """
    own = client is None
    client = client or httpx.Client(timeout=8.0)
    try:
        r = client.get(
            _YAHOO_URL.format(t=ticker.upper()),
            params={"range": "1y", "interval": "1d"},
            headers=_YAHOO_HEADERS,
        )
        if r.status_code != 200:
            return []
        result = r.json()["chart"]["result"][0]
        stamps = result.get("timestamp") or []
        quote = result["indicators"]["quote"][0]
        closes = quote.get("close") or []
        volumes = quote.get("volume") or []
        from datetime import datetime, timezone

        out = []
        for i, (ts, c) in enumerate(zip(stamps, closes)):
            if not isinstance(c, (int, float)):
                continue
            v = volumes[i] if i < len(volumes) else None
            day = datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()
            out.append({
                "time": day,
                "close": round(float(c), 4),
                "volume": float(v) if isinstance(v, (int, float)) else 0.0,
            })
        return out
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
        return []
    finally:
        if own:
            client.close()


def _closes(ticker: str, client: httpx.Client) -> list[dict]:
    """Benchmark close series [{"time","close"}...]; [] on failure."""
    return [{"time": r["time"], "close": r["close"]} for r in ohlcv(ticker, client)]


def risk_free_rate(client: httpx.Client) -> float:
    """Latest 10-year Treasury yield as a decimal, via Yahoo ^TNX.

    ^TNX quotes the yield * 10 (e.g. 43.0 => 4.30%). Falls back to a fixed
    default if the request fails, so valuation yields always have a hurdle.
    """
    try:
        r = client.get(
            _YAHOO_URL.format(t="%5ETNX"),  # ^TNX url-encoded
            params={"range": "5d", "interval": "1d"},
            headers=_YAHOO_HEADERS,
        )
        if r.status_code == 200:
            meta = r.json()["chart"]["result"][0]["meta"]
            px = meta.get("regularMarketPrice")
            if isinstance(px, (int, float)) and px > 0:
                return round(float(px) / 100.0, 4)
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
        pass
    return _RISK_FREE_DEFAULT


def bundle(ticker: str, packet: dict, fmp_api_key: str | None = None) -> dict:
    """Assemble the full market bundle used by the Technical/Valuation scorers."""
    client = httpx.Client(timeout=8.0)
    try:
        price = live_price(ticker, fmp_api_key=fmp_api_key, client=client)
        history = ohlcv(ticker, client)
        benchmark = _closes(_BENCHMARK, client)
        rf = risk_free_rate(client)
        targets = price_targets(packet, price)
        return {
            "price": price,
            "history": history,
            "benchmark": benchmark,
            "risk_free": rf,
            "targets": targets,
        }
    finally:
        client.close()
