"""Insider trading (SEC Form 4) and 13F institutional ownership.

Report content required by the project vision (point 2) and CLAUDE.md
(sections 4-5):

- **SEC insider buying/selling** — every relevant Form 4 buy/sell, where an
  insider only counts as *material* when their total buys OR total sells over
  the trailing window exceed **$1,000,000** (the vision's threshold).
- **13F institutional holders** — the recognized funds/investors holding the
  company, ranked, with quarter-over-quarter share changes.

This is informational context, NOT a scored category — it never feeds the
100-point scorecard. The "does management have a track record at other
successful companies" question is qualitative research the compute engine
can't derive; it stays the orchestrator's job and is surfaced as a prompt,
not fabricated here.

Data comes from FMP (`insider_trades`, `institutional_holders`). Without an
FMP key `bundle()` leaves `ownership` as None and the report shows the section
as pending a key — never invented.
"""

from __future__ import annotations

from datetime import date

MATERIAL_USD = 1_000_000  # vision: insider activity matters above $1M total


def _direction(transaction_type: str | None) -> str:
    """Classify an FMP transactionType into buy / sell / other.

    Open-market purchases (P-*) are the bullish signal; sales (S-*) the
    bearish one. Awards, option exercises, gifts and tax withholding are
    "other" — they carry little discretionary signal, so they don't count
    toward the $1M buy/sell materiality.
    """
    t = (transaction_type or "").strip().upper()
    if t.startswith("P") or "PURCHASE" in t:
        return "buy"
    if t.startswith("S") or "SALE" in t:
        return "sell"
    return "other"


def _within(d: str | None, today: date, months: int) -> bool:
    if not d:
        return False
    try:
        td = date.fromisoformat(d[:10])
    except ValueError:
        return False
    return (today - td).days <= months * 31


def insider_summary(
    trades, today: date | None = None, months: int = 12, min_value: float = MATERIAL_USD
) -> dict | None:
    """Aggregate Form 4 trades into material insider buyers/sellers.

    Returns None when there is nothing usable. Otherwise a dict with the
    material buyers and sellers (each insider's trailing-window total in that
    direction exceeds `min_value`), aggregate values, and the largest
    individual transactions.
    """
    if not isinstance(trades, list) or not trades:
        return None
    today = today or date.today()

    by_person: dict[tuple[str, str], dict] = {}
    all_tx: list[dict] = []
    for row in trades:
        if not _within(row.get("transactionDate"), today, months):
            continue
        direction = _direction(row.get("transactionType"))
        if direction == "other":
            continue
        shares = row.get("securitiesTransacted") or 0
        price = row.get("price") or 0
        value = float(shares) * float(price)
        if value <= 0:
            continue
        name = row.get("reportingName") or "—"
        key = (name, direction)
        agg = by_person.setdefault(key, {"insider": name, "direction": direction,
                                          "value": 0.0, "shares": 0.0, "n": 0})
        agg["value"] += value
        agg["shares"] += float(shares)
        agg["n"] += 1
        all_tx.append({"insider": name, "direction": direction, "value": round(value),
                       "shares": float(shares), "price": float(price),
                       "date": (row.get("transactionDate") or "")[:10]})

    if not all_tx:
        return None

    material = [
        {**a, "value": round(a["value"])}
        for a in by_person.values() if a["value"] >= min_value
    ]
    buys = sorted([m for m in material if m["direction"] == "buy"],
                  key=lambda m: m["value"], reverse=True)
    sells = sorted([m for m in material if m["direction"] == "sell"],
                   key=lambda m: m["value"], reverse=True)
    total_buy = sum(m["value"] for m in buys)
    total_sell = sum(m["value"] for m in sells)

    return {
        "window_months": months,
        "material_threshold": min_value,
        "material_buys": buys,
        "material_sells": sells,
        "total_material_buy_usd": total_buy,
        "total_material_sell_usd": total_sell,
        "net_material_usd": total_buy - total_sell,
        "n_transactions": len(all_tx),
        "top_transactions": sorted(all_tx, key=lambda t: t["value"], reverse=True)[:8],
    }


def institutional_summary(holders, price: float | None = None, top: int = 10) -> dict | None:
    """Rank 13F institutional holders and summarize net share changes."""
    if not isinstance(holders, list) or not holders:
        return None
    rows = []
    for h in holders:
        shares = h.get("shares")
        if not isinstance(shares, (int, float)):
            continue
        change = h.get("change") if isinstance(h.get("change"), (int, float)) else None
        rows.append({
            "holder": h.get("holder") or "—",
            "shares": float(shares),
            "change": float(change) if change is not None else None,
            "value_usd": round(float(shares) * price) if price else None,
            "as_of": (h.get("dateReported") or "")[:10],
        })
    if not rows:
        return None
    rows.sort(key=lambda r: r["shares"], reverse=True)
    changes = [r["change"] for r in rows if r["change"] is not None]
    return {
        "n_holders_reported": len(rows),
        "total_shares": sum(r["shares"] for r in rows),
        "net_change_shares": sum(changes) if changes else None,
        "n_increasing": sum(1 for c in changes if c > 0),
        "n_decreasing": sum(1 for c in changes if c < 0),
        "as_of": rows[0]["as_of"],
        "top_holders": rows[:top],
    }


def ownership(ticker: str, settings, price: float | None = None,
              today: date | None = None) -> dict | None:
    """Fetch insider (Form 4) + 13F ownership via FMP. None without a key."""
    if settings is None:
        return None
    from wbj.providers.cache import Cache
    from wbj.providers.fmp import FMPProvider

    fmp = FMPProvider(settings, Cache(settings.cache_dir))
    if not fmp.available:
        return None

    insiders = insider_summary(fmp.insider_trades(ticker), today=today)
    institutions = institutional_summary(fmp.institutional_holders(ticker), price=price)
    if insiders is None and institutions is None:
        return None
    return {
        "source": "fmp",
        "insiders": insiders,
        "institutions": institutions,
        # Qualitative, not derivable from filings — orchestrator research.
        "management_track_record": None,
    }
