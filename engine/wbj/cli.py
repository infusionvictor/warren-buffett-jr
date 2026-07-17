"""CLI for wbj compute engine.

MVP pipeline: EDGAR companyfacts (no API key needed) -> mini packet ->
formulas -> anchor scores -> Financial category points. FMP enriches
the packet when an API key is configured, but is not required.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import typer

from wbj.config import load_settings
from wbj.core.nullstates import EvidenceClass, NullState, Value
from wbj.core.scoring import Category, Dimension, anchor_score
from wbj.core.formulas import yoy
from wbj.providers.cache import Cache
from wbj.providers.edgar import EdgarProvider
from wbj.providers.fmp import FMPProvider
from wbj.quick import quick_scorecard

app = typer.Typer()

# Concept tags to try, in preference order, per us-gaap taxonomy drift.
_REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
]
_NET_INCOME_TAGS = ["NetIncomeLoss"]
_OCF_TAGS = ["NetCashProvidedByUsedInOperatingActivities"]
_CAPEX_TAGS = ["PaymentsToAcquirePropertyPlantAndEquipment"]
_DEBT_TAGS = ["LongTermDebtNoncurrent", "LongTermDebt"]
_EQUITY_TAGS = ["StockholdersEquity"]
_OP_INCOME_TAGS = ["OperatingIncomeLoss"]
_GROSS_PROFIT_TAGS = ["GrossProfit"]
_INTEREST_TAGS = ["InterestExpense", "InterestExpenseNonoperating"]
_SHARES_TAGS = [
    "WeightedAverageNumberOfDilutedSharesOutstanding",
    "WeightedAverageNumberOfSharesOutstandingBasic",
]

# Scoring anchors (0-10) — MVP defaults, aligned with Cerebro-style bands.
_ANCHORS_REV_GROWTH = [(-0.10, 0.0), (0.0, 3.0), (0.10, 6.0), (0.25, 9.0), (0.40, 10.0)]
_ANCHORS_NET_MARGIN = [(-0.10, 0.0), (0.0, 3.0), (0.10, 6.0), (0.20, 8.5), (0.30, 10.0)]
_ANCHORS_FCF_MARGIN = [(-0.10, 0.0), (0.0, 3.0), (0.10, 6.5), (0.25, 10.0)]
_ANCHORS_DEBT_EQUITY = [(0.0, 10.0), (0.5, 8.0), (1.0, 6.0), (2.0, 3.0), (4.0, 0.0)]


def _providers():
    settings = load_settings()
    cache = Cache(settings.cache_dir)
    return settings, EdgarProvider(settings, cache), FMPProvider(settings, cache)


def _annual_series(facts: dict, tags: list[str]) -> list[dict]:
    """Extract annual (10-K FY) datapoints for the first tag that has data."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in tags:
        units = gaap.get(tag, {}).get("units", {})
        rows = units.get("USD") or units.get("shares") or []
        annual = [r for r in rows if r.get("form") == "10-K" and r.get("fp") == "FY"]
        if annual:
            # Deduplicate restatements: keep the latest filing per fiscal year end.
            by_end: dict[str, dict] = {}
            for r in sorted(annual, key=lambda r: r.get("filed", "")):
                by_end[r["end"]] = r
            return sorted(by_end.values(), key=lambda r: r["end"])
    return []


def _latest(series: list[dict]) -> float | None:
    return series[-1]["val"] if series else None


def _build_packet(ticker: str) -> dict:
    settings, edgar, fmp = _providers()
    cik = edgar.cik_for(ticker)
    if cik is None:
        raise typer.BadParameter(f"Ticker {ticker!r} not found in SEC EDGAR.")
    facts = edgar.companyfacts(cik)
    if facts is None:
        raise typer.Exit(code=1)

    packet: dict = {
        "ticker": ticker.upper(),
        "cik": cik,
        "as_of": date.today().isoformat(),
        "entity": facts.get("entityName"),
        "sources": {"edgar": "companyfacts", "fmp": fmp.available},
        "annual": {
            "revenue": _annual_series(facts, _REVENUE_TAGS),
            "net_income": _annual_series(facts, _NET_INCOME_TAGS),
            "operating_cash_flow": _annual_series(facts, _OCF_TAGS),
            "capex": _annual_series(facts, _CAPEX_TAGS),
            "long_term_debt": _annual_series(facts, _DEBT_TAGS),
            "equity": _annual_series(facts, _EQUITY_TAGS),
            "operating_income": _annual_series(facts, _OP_INCOME_TAGS),
            "gross_profit": _annual_series(facts, _GROSS_PROFIT_TAGS),
            "interest_expense": _annual_series(facts, _INTEREST_TAGS),
            "diluted_shares": _annual_series(facts, _SHARES_TAGS),
        },
    }
    if fmp.available:
        packet["fmp_profile"] = fmp.profile(ticker)
    return packet


def _metric(name: str, raw: float | None, unit: str = "ratio") -> Value:
    if raw is None:
        return Value.null(NullState.MISSING, unit=unit, warnings=[f"MISSING: {name}"])
    return Value.of(raw, unit=unit, evidence_class=EvidenceClass.C)


def _score(v: Value, anchors: list[tuple[float, float]]) -> Value:
    if v.is_null:
        return v
    return Value.of(anchor_score(v.value, anchors), unit="score", evidence_class=EvidenceClass.C)


def _compute(packet: dict) -> dict:
    a = packet["annual"]
    rev, ni = a["revenue"], a["net_income"]
    ocf, capex = a["operating_cash_flow"], a["capex"]
    debt, eq = a["long_term_debt"], a["equity"]

    # Metrics from the two most recent fiscal years.
    growth = (
        yoy(rev[-1]["val"], rev[-2]["val"])
        if len(rev) >= 2
        else Value.null(NullState.MISSING, unit="ratio", warnings=["MISSING: revenue history"])
    )
    rev_latest, ni_latest = _latest(rev), _latest(ni)
    ocf_latest, capex_latest = _latest(ocf), _latest(capex)
    debt_latest, eq_latest = _latest(debt), _latest(eq)

    net_margin = _metric(
        "net_margin",
        (ni_latest / rev_latest) if rev_latest and ni_latest is not None else None,
    )
    fcf = (
        (ocf_latest - capex_latest)
        if ocf_latest is not None and capex_latest is not None
        else None
    )
    fcf_margin = _metric("fcf_margin", (fcf / rev_latest) if rev_latest and fcf is not None else None)
    debt_equity = _metric(
        "debt_to_equity",
        (debt_latest / eq_latest) if eq_latest and debt_latest is not None else None,
    )

    profitability = Dimension(
        name="Profitability",
        max_points=7.5,
        metric_scores=[
            (0.5, _score(net_margin, _ANCHORS_NET_MARGIN)),
            (0.5, _score(fcf_margin, _ANCHORS_FCF_MARGIN)),
        ],
    )
    growth_health = Dimension(
        name="Growth & Balance Sheet",
        max_points=7.5,
        metric_scores=[
            (0.5, _score(growth, _ANCHORS_REV_GROWTH)),
            (0.5, _score(debt_equity, _ANCHORS_DEBT_EQUITY)),
        ],
    )
    financial = Category(name="Financial", max_points=15.0, dimensions=[profitability, growth_health])

    def fmt(v: Value) -> float | str:
        return round(v.value, 4) if v.is_valid else str(v.state)

    return {
        "ticker": packet["ticker"],
        "entity": packet["entity"],
        "as_of": packet["as_of"],
        "fiscal_year_end": rev[-1]["end"] if rev else None,
        "metrics": {
            "revenue_usd": rev_latest,
            "revenue_yoy": fmt(growth),
            "net_margin": fmt(net_margin),
            "fcf_margin": fmt(fcf_margin),
            "debt_to_equity": fmt(debt_equity),
        },
        "scores": {
            "dimensions": {
                d.name: fmt(d.score10_value()) for d in financial.dimensions
            },
            "category": {
                "name": financial.name,
                "points": round(financial.points(), 2),
                "max_points": financial.max_points,
                "score10": round(financial.score10(), 2),
                "coverage": round(financial.coverage(), 2),
            },
        },
        "scorecard": quick_scorecard(packet),
    }


def _out_dir(settings, ticker: str) -> Path:
    d = settings.reports_dir / ticker.upper() / date.today().isoformat()
    d.mkdir(parents=True, exist_ok=True)
    return d


@app.command()
def fetch(ticker: str) -> None:
    """Fetch raw EDGAR data for a ticker (cache-first)."""
    _, edgar, _ = _providers()
    cik = edgar.cik_for(ticker)
    if cik is None:
        typer.echo(f"{ticker}: not found in EDGAR")
        raise typer.Exit(1)
    facts = edgar.companyfacts(cik)
    n = len(facts.get("facts", {}).get("us-gaap", {})) if facts else 0
    typer.echo(f"{ticker.upper()}: CIK {cik}, {n} us-gaap concepts fetched")


@app.command()
def packet(ticker: str) -> None:
    """Build and save the data packet for a ticker."""
    settings, _, _ = _providers()
    p = _build_packet(ticker)
    path = _out_dir(settings, ticker) / "packet.json"
    path.write_text(json.dumps(p, indent=2))
    typer.echo(f"packet -> {path}")


@app.command()
def compute(ticker: str) -> None:
    """Compute metrics and scores for a ticker."""
    typer.echo(json.dumps(_compute(_build_packet(ticker)), indent=2))


@app.command()
def analyze(ticker: str) -> None:
    """Run the full MVP pipeline: fetch -> packet -> compute -> save report."""
    settings, _, _ = _providers()
    p = _build_packet(ticker)
    result = _compute(p)

    out = _out_dir(settings, ticker)
    (out / "packet.json").write_text(json.dumps(p, indent=2))
    (out / "scores.json").write_text(json.dumps(result, indent=2))

    m, cat = result["metrics"], result["scores"]["category"]
    typer.echo(f"\n=== {result['entity']} ({result['ticker']}) — FY end {result['fiscal_year_end']} ===")
    typer.echo(f"Revenue:        ${m['revenue_usd']:,.0f}" if m["revenue_usd"] else "Revenue: MISSING")
    typer.echo(f"Revenue YoY:    {m['revenue_yoy']}")
    typer.echo(f"Net margin:     {m['net_margin']}")
    typer.echo(f"FCF margin:     {m['fcf_margin']}")
    typer.echo(f"Debt/Equity:    {m['debt_to_equity']}")
    typer.echo("--- Scores (0-10) ---")
    for name, s in result["scores"]["dimensions"].items():
        typer.echo(f"{name}: {s}")
    typer.echo(
        f"Category {cat['name']}: {cat['points']}/{cat['max_points']} pts "
        f"(score {cat['score10']}/10, coverage {cat['coverage']:.0%})"
    )
    typer.echo(f"\nSaved: {out}/packet.json, scores.json")


@app.command()
def scorecard(ticker: str) -> None:
    """Quick 1-10 scorecard across the 6 agent categories."""
    settings, _, _ = _providers()
    p = _build_packet(ticker)
    sc = quick_scorecard(p)
    out = _out_dir(settings, ticker)
    (out / "scorecard.json").write_text(json.dumps(sc, indent=2))

    typer.echo(f"\n=== Quick Scorecard — {p['entity']} ({p['ticker']}) ===")
    for row in sc["categories"]:
        if row["status"] == "scored":
            bar = "█" * int(round(row["score10"])) + "░" * (10 - int(round(row["score10"])))
            typer.echo(f"{row['label']:<28} {bar}  {row['score10']}/10  ({row['points']}/{row['max_points']:.0f} pts)")
        else:
            typer.echo(f"{row['label']:<28} {'·' * 10}  N/S — {row['reason']}")
    typer.echo(f"\nOverall (quick): {sc['overall_10']}/10 on {sc['evidence_points_covered']}/100 evidence pts")
    typer.echo(f"Saved: {out}/scorecard.json")


@app.command()
def aggregate(ticker: str) -> None:
    """Aggregate specialist outputs for a ticker."""
    typer.echo(f"aggregate {ticker}: not implemented")
    raise typer.Exit(1)


@app.command()
def report(ticker: str) -> None:
    """Generate report for a ticker."""
    typer.echo(f"report {ticker}: not implemented")
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
