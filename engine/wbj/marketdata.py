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
      "consensus":  {...} | None,          # normalized analyst consensus (keys only)
    }

`consensus` is populated only when an FMP or FinnHub key is configured; it
feeds the Market & Growth "Earnings and revenue revisions" dimension. Without
keys it is None and that dimension stays NOT_SCORABLE (this lift is the reason
to add keys — TAM and catalysts still need qualitative research, so Market
rises to partial ~55% coverage, not full).
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


# --------------------------------------------------------------------------
# Analyst consensus (FMP / FinnHub) — feeds Market "revisions" dimension
# --------------------------------------------------------------------------

_MIN_ANALYSTS = 5  # Cerebro MKT-REVBR gate: needs >=5 estimates


def _last_actual_eps(packet: dict) -> float | None:
    a = packet.get("annual", {})
    ni = a.get("net_income") or []
    sh = a.get("diluted_shares") or []
    if ni and sh and sh[-1]["val"]:
        return ni[-1]["val"] / sh[-1]["val"]
    return None


def _last_actual_revenue(packet: dict) -> float | None:
    rev = packet.get("annual", {}).get("revenue") or []
    return rev[-1]["val"] if rev else None


def _surprise_stats(pairs: list[tuple[float, float]]) -> tuple[float | None, float | None, int]:
    """(beat_rate, mean_relative_surprise, n) over (actual, estimate) pairs."""
    clean = [(a, e) for a, e in pairs
             if isinstance(a, (int, float)) and isinstance(e, (int, float)) and e != 0]
    if not clean:
        return None, None, 0
    beats = sum(1 for a, e in clean if a >= e)
    mags = [(a - e) / abs(e) for a, e in clean]
    return beats / len(clean), sum(mags) / len(mags), len(clean)


def consensus_from_fmp(estimates, calendar, last_rev, last_eps, recent: int = 8) -> dict | None:
    """Normalize FMP analyst-estimates + earnings-calendar into a consensus dict."""
    if not isinstance(estimates, list) or not estimates:
        return None
    fwd = estimates[0]  # nearest forward fiscal year
    rev_avg = fwd.get("estimatedRevenueAvg")
    rev_lo, rev_hi = fwd.get("estimatedRevenueLow"), fwd.get("estimatedRevenueHigh")
    eps_avg = fwd.get("estimatedEpsAvg")
    n = fwd.get("numberAnalystEstimatedRevenue")

    pairs = []
    if isinstance(calendar, list):
        for row in calendar[:recent]:
            pairs.append((row.get("eps"), row.get("epsEstimated")))
    beat_rate, surprise_mag, n_surp = _surprise_stats(pairs)

    return _assemble_consensus(
        source="fmp", num_analysts=n, rev_avg=rev_avg, rev_lo=rev_lo, rev_hi=rev_hi,
        eps_avg=eps_avg, last_rev=last_rev, last_eps=last_eps,
        beat_rate=beat_rate, surprise_mag=surprise_mag, n_surp=n_surp,
    )


def consensus_from_finnhub(eps_est, rev_est, last_rev, last_eps) -> dict | None:
    """Normalize FinnHub eps/revenue estimate payloads into a consensus dict."""
    eps_rows = eps_est.get("data") if isinstance(eps_est, dict) else None
    rev_rows = rev_est.get("data") if isinstance(rev_est, dict) else None
    if not eps_rows and not rev_rows:
        return None

    def _fwd(rows, key):
        # first row whose actual is null == nearest forward estimate
        for r in rows or []:
            if r.get(f"{key}Actual") is None:
                return r
        return (rows or [None])[0]

    fwd_rev = _fwd(rev_rows, "revenue") or {}
    fwd_eps = _fwd(eps_rows, "eps") or {}
    n = fwd_rev.get("numberAnalysts") or fwd_eps.get("numberAnalysts")

    pairs = [(r.get("epsActual"), r.get("epsAvg")) for r in (eps_rows or [])
             if r.get("epsActual") is not None]
    beat_rate, surprise_mag, n_surp = _surprise_stats(pairs)

    return _assemble_consensus(
        source="finnhub", num_analysts=n,
        rev_avg=fwd_rev.get("revenueAvg"), rev_lo=fwd_rev.get("revenueLow"),
        rev_hi=fwd_rev.get("revenueHigh"), eps_avg=fwd_eps.get("epsAvg"),
        last_rev=last_rev, last_eps=last_eps,
        beat_rate=beat_rate, surprise_mag=surprise_mag, n_surp=n_surp,
    )


def _assemble_consensus(*, source, num_analysts, rev_avg, rev_lo, rev_hi, eps_avg,
                        last_rev, last_eps, beat_rate, surprise_mag, n_surp) -> dict:
    fwd_rev_growth = (rev_avg / last_rev - 1) if (rev_avg and last_rev) else None
    fwd_eps_growth = (eps_avg / last_eps - 1) if (eps_avg and last_eps) else None
    rev_dispersion = ((rev_hi - rev_lo) / rev_avg
                      if (rev_hi and rev_lo and rev_avg) else None)
    return {
        "source": source,
        "num_analysts": num_analysts,
        "fwd_rev_growth": fwd_rev_growth,
        "fwd_eps_growth": fwd_eps_growth,
        "rev_dispersion": rev_dispersion,
        "beat_rate": beat_rate,
        "surprise_mag": surprise_mag,
        "n_surprises": n_surp,
    }


def consensus(ticker: str, settings, packet: dict) -> dict | None:
    """Fetch and normalize analyst consensus via FMP (preferred) or FinnHub.

    Returns None when no key is configured or nothing usable comes back, so
    the Market "revisions" dimension stays honestly NOT_SCORABLE.
    """
    if settings is None:
        return None
    from wbj.providers.cache import Cache
    from wbj.providers.finnhub import FinnhubProvider
    from wbj.providers.fmp import FMPProvider

    last_rev = _last_actual_revenue(packet)
    last_eps = _last_actual_eps(packet)
    cache = Cache(settings.cache_dir)

    fmp = FMPProvider(settings, cache)
    if fmp.available:
        c = consensus_from_fmp(
            fmp.analyst_estimates(ticker), fmp.earnings_calendar(ticker),
            last_rev, last_eps)
        if c:
            return c

    fh = FinnhubProvider(settings, cache)
    if fh.available:
        c = consensus_from_finnhub(
            fh.estimates(ticker), fh.revenue_estimates(ticker), last_rev, last_eps)
        if c:
            return c
    return None


def bundle(ticker: str, packet: dict, settings=None) -> dict:
    """Assemble the full market bundle used by the Technical/Valuation/Market scorers.

    `settings` (a `wbj.config.Settings`) enables the paid providers: its FMP
    key sharpens `live_price`, and FMP/FinnHub keys populate `consensus`.
    Passing None keeps everything on the keyless Yahoo path.
    """
    fmp_key = getattr(settings, "fmp_api_key", None)
    client = httpx.Client(timeout=8.0)
    try:
        price = live_price(ticker, fmp_api_key=fmp_key, client=client)
        history = ohlcv(ticker, client)
        benchmark = _closes(_BENCHMARK, client)
        rf = risk_free_rate(client)
        targets = price_targets(packet, price)
    finally:
        client.close()
    return {
        "price": price,
        "history": history,
        "benchmark": benchmark,
        "risk_free": rf,
        "targets": targets,
        "consensus": consensus(ticker, settings, packet),
    }
