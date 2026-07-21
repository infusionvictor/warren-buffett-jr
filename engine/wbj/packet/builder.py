"""Packet builder: assembles the pydantic `Packet` handed to the six
Cerebro specialists from raw provider payloads.

Implements the first three phases of `Cerebro/00_main_agent/ORCHESTRATION.md`:

- Phase 0 (freeze the analysis clock): `knowledge_timestamp` is always the
  caller-supplied `now` — this module never calls `datetime.now()` itself,
  so packet construction is deterministic and testable.
- Phase 1 (validate common data + common facts table): hard-rejects packets
  missing timestamps, currency, any diluted-share source, or a full
  252-session daily OHLCV history; reconciles FMP vs EDGAR for
  revenue/diluted_shares/cash/total_debt via `wbj.packet.reconcile`.
- Phase 3 (freeze packets): `packet_hash` is a sha256 of the canonical
  (sorted-key, compact-separator) JSON of the packet, excluding the hash
  field itself, so a rebuild from identical inputs reproduces the same
  hash and any single changed input changes it.

`Providers` is a plain container of the four provider objects
(`wbj.providers.{fmp,edgar,finnhub,fred}` or their fakes in tests) that
`build_packet` reads from — it doesn't otherwise depend on their concrete
classes, only on the method surface documented on each fake in
`engine/tests/fixtures/packet/make_packet_fixture.py`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

from wbj.core.nullstates import EvidenceClass, NullState, Value
from wbj.packet.reconcile import reconcile
from wbj.packet.staleness import staleness_state
from wbj.schemas.packet import AnalysisMeta, MarketData, OHLCVRow, Packet, Security

# Minimum daily OHLCV sessions required to accept a packet (task-10 brief).
_MIN_DAILY_SESSIONS = 252

# Benchmark/sector index ticker used for relative-strength and beta/
# correlation metrics (technical.py's TECH-RS-011/RSS-012, risk.py's
# RSK-BETA-003/DBETA-004/CORR-005). SPY (S&P 500 ETF) is used for both the
# broad-market benchmark and, absent a per-sector ETF mapping, as the
# sector proxy too -- a per-GICS-sector ETF map (XLK, XLF, XLE, ...) is a
# reasonable future enhancement but not required for these metrics to
# score: technical.py/risk.py only need *a* liquid, long-history index
# series aligned to the stock's trading calendar, not a sector-precise one.
_BENCHMARK_TICKER = "SPY"


class PacketRejected(Exception):
    """Raised when a packet fails one of the Phase 0/1 hard-reject checks:
    missing timestamps, missing currency, no diluted share count from any
    source, or fewer than 252 daily sessions."""


@dataclass
class Providers:
    """The four data sources `build_packet` reads from.

    Duck-typed against `wbj.providers.{fmp,edgar,finnhub,fred}`'s public
    method surface (see the Fake* classes in
    `tests/fixtures/packet/make_packet_fixture.py` for the exact contract
    this module relies on).
    """

    fmp: Any
    edgar: Any
    finnhub: Any
    fred: Any


# --- canonical field mapping -------------------------------------------
#
# Raw FMP statement keys -> Cerebro/shared/DATA_DICTIONARY.md canonical
# names. Unmapped keys (date, symbol, period, calendarYear, acceptedDate,
# eps, ...) pass through unchanged since the data dictionary doesn't fix a
# canonical name for them. Mapped keys are *replaced*, not duplicated, so
# no raw FMP key survives into `packet.fundamentals`.
CANONICAL_FIELD_MAP: dict[str, str] = {
    "revenue": "revenue",
    "costOfRevenue": "cogs",
    "grossProfit": "gross_profit",
    "operatingIncome": "ebit",
    "incomeBeforeTax": "income_before_tax",
    "incomeTaxExpense": "income_tax_expense",
    "netIncome": "net_income",
    "weightedAverageShsOutDil": "diluted_shares",
    "weightedAverageShsOut": "basic_shares",
    "cashAndCashEquivalents": "cash",
    "netReceivables": "net_receivables",
    "inventory": "inventory",
    "totalCurrentAssets": "total_current_assets",
    "totalCurrentLiabilities": "total_current_liabilities",
    "shortTermDebt": "short_term_debt",
    "longTermDebt": "long_term_debt",
    "totalDebt": "total_debt",
    "totalAssets": "total_assets",
    "totalLiabilities": "total_liabilities",
    "totalStockholdersEquity": "total_equity",
    "netCashProvidedByOperatingActivities": "operating_cash_flow",
    "capitalExpenditure": "capex",
    "freeCashFlow": "fcf",
    "acquisitionsNet": "acquisitions_net",
    "debtRepayment": "debt_repayment",
    "commonStockRepurchased": "common_stock_repurchased",
    "dividendsPaid": "dividends_paid",
    "stockBasedCompensation": "stock_based_compensation",
    # --- additional statement fields (task: fuller FMP statement mapping) --
    # These are genuine FMP statement data (not judgment/qualitative
    # inputs) that several specialist formulas are documented to want from
    # `Packet.fundamentals` (see e.g. risk.py's RSK-ICOV-011/AQI-023/
    # DEPI-025/SGAI-026/ALT-030 and financial.py's FIN-BS-020/DX-028/
    # DX-031 module comments: "not part of Packet.fundamentals"). As of
    # this change the current specialist code for those specific metrics
    # reads its inputs from an `overlay` dict (a separate judgment-layer
    # input, Task 20) rather than `Packet.fundamentals`, or is hardcoded
    # to MISSING pending a future code change -- see the packet-builder
    # commit message / task report for the exact list. Mapping the raw
    # data in here regardless: (a) is what those docstrings ask for, (b)
    # is immediately useful to any other row that already reads
    # `packet.fundamentals.annual/quarterly` directly, and (c) is the
    # natural source for a future overlay-from-packet default.
    "interestExpense": "interest_expense",
    "ebitda": "ebitda",
    "depreciationAndAmortization": "depreciation_and_amortization",
    "sellingGeneralAndAdministrativeExpenses": "sga",
    "researchAndDevelopmentExpenses": "rnd_expense",
    "accountPayables": "accounts_payable",
    "retainedEarnings": "retained_earnings",
    # FMP's /stable/ balance-sheet statement only exposes *net* PP&E, not
    # gross -- classic Beneish AQI/DEPI use gross PP&E, so this is a
    # documented, honest proxy rather than a true substitute; named
    # `ppe_net` (not `ppe`) so it's never silently mistaken for the gross
    # figure those formulas actually call for.
    "propertyPlantEquipmentNet": "ppe_net",
}


def _map_statement_row(row: dict) -> dict:
    """Apply `CANONICAL_FIELD_MAP` to one raw FMP statement row."""
    return {CANONICAL_FIELD_MAP.get(k, k): v for k, v in row.items()}


def _merge_statement_period(income_row: dict, balance_row: dict, cashflow_row: dict) -> dict:
    """Merge one fiscal period's income/balance/cashflow rows into a single
    canonical-name record."""
    merged_raw = {**income_row, **balance_row, **cashflow_row}
    return _map_statement_row(merged_raw)


def _merge_statements(income: list[dict], balance: list[dict], cashflow: list[dict]) -> list[dict]:
    return [
        _merge_statement_period(inc, bal, cf)
        for inc, bal, cf in zip(income or [], balance or [], cashflow or [])
    ]


# --- FMP-side Value extraction -------------------------------------------


def _fmp_value(row: dict | None, key: str, unit: str) -> Value:
    if not row or row.get(key) is None:
        return Value.null(NullState.MISSING, unit=unit, source_name="FMP")
    return Value.of(
        row[key],
        unit=unit,
        source_name="FMP",
        period=row.get("date"),
        as_of=row.get("acceptedDate"),
        evidence_class=EvidenceClass.R,
    )


def _fmp_diluted_shares(income_annual: list[dict], income_quarterly: list[dict]) -> Value:
    for row in (income_annual[:1] or []) + (income_quarterly[:1] or []):
        val = _fmp_value(row, "weightedAverageShsOutDil", "shares")
        if val.is_valid:
            return val
    return Value.null(NullState.MISSING, unit="shares", source_name="FMP")


# --- EDGAR-side Value extraction -----------------------------------------


def _edgar_entries(companyfacts: dict, taxonomy: str, tag: str, unit: str) -> list[dict]:
    return (
        (companyfacts or {})
        .get("facts", {})
        .get(taxonomy, {})
        .get(tag, {})
        .get("units", {})
        .get(unit, [])
    )


def _edgar_value_at(
    companyfacts: dict, taxonomy: str, tag: str, unit: str, target_date: str | None
) -> Value:
    """Latest EDGAR XBRL fact for `tag`, preferring the entry whose `end`
    matches `target_date` (the FMP statement period it's being reconciled
    against) and falling back to the most recent entry otherwise."""
    entries = _edgar_entries(companyfacts, taxonomy, tag, unit)
    if not entries:
        return Value.null(NullState.MISSING, unit=unit, source_name="EDGAR")
    entry = next((e for e in entries if e.get("end") == target_date), None)
    if entry is None:
        entry = max(entries, key=lambda e: e.get("end", ""))
    return Value.of(
        entry["val"],
        unit=unit,
        source_name="EDGAR",
        source_locator=entry.get("accn"),
        period=entry.get("end"),
        as_of=entry.get("filed"),
        evidence_class=EvidenceClass.R,
    )


def _edgar_total_debt(companyfacts: dict, target_date: str | None) -> Value:
    """EDGAR has no single "total debt" tag; sum the noncurrent + current
    debt tags for the target period, plus ASC-842 operating-lease
    liabilities. If either debt half is unavailable, the sum isn't safe
    to report — return MISSING rather than silently treating a missing
    half as zero.

    FMP's "totalDebt" aggregate includes operating-lease liabilities;
    without adding EDGAR's matching tags, a lease-heavy filer's EDGAR sum
    undershoots FMP's figure enough to trip the >5% CONFLICTED threshold
    on an otherwise-agreeing pair of sources (observed live on NVDA:
    FMP's totalDebt was exactly LongTermDebtNoncurrent + DebtCurrent +
    OperatingLeaseLiabilityNoncurrent + OperatingLeaseLiabilityCurrent).
    Post-2019 GAAP filers book the lease liability's present value
    directly on the balance sheet, so no separate PV-of-payments
    calculation is needed here. The lease tags themselves default to 0
    when absent (pre-842 filers or non-lessees) rather than forcing the
    whole figure to MISSING — their absence is a legitimate zero, not a
    data gap the way a missing debt half is.
    """
    long_term = _edgar_value_at(companyfacts, "us-gaap", "LongTermDebtNoncurrent", "USD", target_date)
    current = _edgar_value_at(companyfacts, "us-gaap", "DebtCurrent", "USD", target_date)
    if long_term.is_null or current.is_null:
        return Value.null(NullState.MISSING, unit="USD", source_name="EDGAR")
    lease_noncurrent = _edgar_value_at(companyfacts, "us-gaap", "OperatingLeaseLiabilityNoncurrent", "USD", target_date)
    lease_current = _edgar_value_at(companyfacts, "us-gaap", "OperatingLeaseLiabilityCurrent", "USD", target_date)
    lease_total = (lease_noncurrent.value if lease_noncurrent.is_valid else 0) + (
        lease_current.value if lease_current.is_valid else 0
    )
    return Value.of(
        long_term.value + current.value + lease_total,
        unit="USD",
        source_name="EDGAR",
        period=target_date,
        evidence_class=EvidenceClass.C,
    )


def _edgar_diluted_shares(companyfacts: dict, target_date: str | None) -> Value:
    """Weighted-average diluted shares tag first; falls back to EDGAR's
    basic `dei:EntityCommonStockSharesOutstanding` tag (most recent filing)
    when the weighted-diluted tag isn't reported."""
    weighted = _edgar_value_at(
        companyfacts, "us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding", "shares", target_date
    )
    if weighted.is_valid:
        return weighted
    basic = _edgar_value_at(companyfacts, "dei", "EntityCommonStockSharesOutstanding", "shares", None)
    if basic.is_valid:
        return basic
    return Value.null(NullState.MISSING, unit="shares", source_name="EDGAR")


# --- staleness age derivation ---------------------------------------------


def _age_days(now: datetime, date_str: str | None) -> float:
    if not date_str:
        return float("inf")
    return (now.date() - date.fromisoformat(date_str[:10])).days


def _max_date(dates: list[str | None]) -> str | None:
    clean = [d for d in dates if d]
    return max(clean) if clean else None


def _ohlcv_row(bar: dict) -> OHLCVRow:
    """One raw FMP `/historical-price-eod` bar -> `OHLCVRow`, shared by the
    stock and benchmark/sector series so both use the same adjusted-close
    convention (`adj_close` falls back to `close` when FMP omits it)."""
    return OHLCVRow(
        date=bar["date"],
        open=bar["open"],
        high=bar["high"],
        low=bar["low"],
        close=bar["close"],
        adj_close=bar.get("adjClose", bar["close"]),
        volume=bar["volume"],
    )


# --- build_packet -----------------------------------------------------------


def build_packet(ticker: str, providers: Providers, now: datetime) -> Packet:
    """Build the analysis `Packet` for `ticker` from `providers`, framed at
    the analysis clock `now` (never the wall clock — callers own `now`).

    Raises `PacketRejected` for the Phase 0/1 hard-reject conditions: no
    currency, no derivable market timestamp, fewer than 252 daily
    sessions, or no diluted share count from any source.
    """
    fmp, edgar, finnhub = providers.fmp, providers.edgar, providers.finnhub

    profile_raw = fmp.profile(ticker)
    profile: dict = (profile_raw[0] if isinstance(profile_raw, list) and profile_raw else profile_raw) or {}

    income_annual = fmp.income_annual(ticker) or []
    income_quarterly = fmp.income_quarterly(ticker) or []
    balance_annual = fmp.balance_annual(ticker) or []
    balance_quarterly = fmp.balance_quarterly(ticker) or []
    cashflow_annual = fmp.cashflow_annual(ticker) or []
    cashflow_quarterly = fmp.cashflow_quarterly(ticker) or []
    ohlcv_raw = fmp.ohlcv_daily(ticker) or []
    peers = fmp.peers(ticker) or []
    analyst_estimates = fmp.analyst_estimates(ticker) or []
    insider_trades = fmp.insider_trades(ticker) or []
    institutional_holders = fmp.institutional_holders(ticker) or []
    earnings_calendar = fmp.earnings_calendar(ticker) or []

    quote = finnhub.quote(ticker)
    finnhub_eps_estimate = finnhub.estimates(ticker)
    finnhub_revenue_estimate = finnhub.revenue_estimates(ticker)

    cik = edgar.cik_for(ticker)
    companyfacts = edgar.companyfacts(cik) if cik is not None else {}

    risk_free_rate = providers.fred.risk_free_rate()

    # --- Phase 0: freeze the analysis clock ---------------------------

    currency = profile.get("currency")
    if not currency:
        raise PacketRejected(f"packet rejected for {ticker}: missing currency")

    market_timestamp: str | None = None
    if ohlcv_raw:
        market_timestamp = f"{ohlcv_raw[0]['date']}T21:00:00+00:00"
    elif quote and quote.get("t") is not None:
        market_timestamp = datetime.fromtimestamp(quote["t"], tz=timezone.utc).isoformat()

    if market_timestamp is None:
        raise PacketRejected(
            f"packet rejected for {ticker}: missing timestamps (no OHLCV or quote data available)"
        )

    if len(ohlcv_raw) < _MIN_DAILY_SESSIONS:
        raise PacketRejected(
            f"packet rejected for {ticker}: fewer than {_MIN_DAILY_SESSIONS} daily sessions "
            f"({len(ohlcv_raw)} available)"
        )

    # --- Phase 1: common facts table -----------------------------------

    latest_annual_date = income_annual[0]["date"] if income_annual else None

    fmp_revenue = _fmp_value(income_annual[0] if income_annual else None, "revenue", "USD")
    edgar_revenue = _edgar_value_at(companyfacts, "us-gaap", "Revenues", "USD", latest_annual_date)

    fmp_cash = _fmp_value(balance_annual[0] if balance_annual else None, "cashAndCashEquivalents", "USD")
    edgar_cash = _edgar_value_at(
        companyfacts, "us-gaap", "CashAndCashEquivalentsAtCarryingValue", "USD", latest_annual_date
    )

    fmp_total_debt = _fmp_value(balance_annual[0] if balance_annual else None, "totalDebt", "USD")
    edgar_total_debt = _edgar_total_debt(companyfacts, latest_annual_date)

    fmp_diluted_shares = _fmp_diluted_shares(income_annual, income_quarterly)
    edgar_diluted_shares = _edgar_diluted_shares(companyfacts, latest_annual_date)

    if fmp_diluted_shares.is_null and edgar_diluted_shares.is_null:
        raise PacketRejected(
            f"packet rejected for {ticker}: no diluted share count available from any source"
        )

    price_value = Value.null(NullState.MISSING, unit="usd_per_share", source_name="FMP")
    if profile.get("price") is not None:
        price_value = Value.of(
            profile["price"],
            unit="usd_per_share",
            source_name="FMP",
            as_of=market_timestamp,
            evidence_class=EvidenceClass.R,
        )

    facts_table: dict[str, Value] = {
        "revenue": reconcile("revenue", fmp_revenue, edgar_revenue),
        "diluted_shares": reconcile("diluted_shares", fmp_diluted_shares, edgar_diluted_shares),
        "cash": reconcile("cash", fmp_cash, edgar_cash),
        "total_debt": reconcile("total_debt", fmp_total_debt, edgar_total_debt),
        "price": price_value,
    }

    # --- fundamentals: canonical-name annual + quarterly records --------

    fundamentals = {
        "annual": _merge_statements(income_annual, balance_annual, cashflow_annual),
        "quarterly": _merge_statements(income_quarterly, balance_quarterly, cashflow_quarterly),
    }

    # --- market data ------------------------------------------------------

    daily_rows = [_ohlcv_row(bar) for bar in ohlcv_raw]

    # Benchmark (and, absent a per-sector ETF map, sector-proxy) series:
    # SPY aligned to the stock's own trading dates via an inner join, so
    # `close.tail(n)` position-alignment in technical.py/risk.py lines up
    # with the same calendar dates on both sides. `today=now.date()` keeps
    # this deterministic given `now` (never the wall clock).
    benchmark_raw = fmp.ohlcv_daily(_BENCHMARK_TICKER, today=now.date()) or []
    stock_dates = {bar["date"] for bar in ohlcv_raw}
    benchmark_aligned_raw = [bar for bar in benchmark_raw if bar["date"] in stock_dates]
    benchmark_rows = [_ohlcv_row(bar) for bar in benchmark_aligned_raw]

    market_data = MarketData(daily=daily_rows, benchmark=benchmark_rows, sector=benchmark_rows, adjusted=True)

    # --- staleness ----------------------------------------------------------

    consensus_dates = [row.get("date") for row in earnings_calendar if row.get("eps") is not None]
    if not consensus_dates:
        consensus_dates = [row.get("date") for row in earnings_calendar]
    latest_consensus_date = _max_date(consensus_dates)

    latest_quarterly_date = income_quarterly[0]["date"] if income_quarterly else None
    latest_holder_date = _max_date([row.get("dateReported") for row in institutional_holders])

    staleness = {
        "daily_market": staleness_state("daily_market", _age_days(now, ohlcv_raw[0]["date"])),
        "quarterly_fundamentals": staleness_state(
            "quarterly_fundamentals", _age_days(now, latest_quarterly_date)
        ),
        "consensus": staleness_state("consensus", _age_days(now, latest_consensus_date)),
        "peer_set": staleness_state("peer_set", _age_days(now, latest_holder_date)),
    }

    # --- remaining packet blocks --------------------------------------------

    estimates = {
        "fmp_analyst_estimates": analyst_estimates,
        "finnhub_eps_estimate": finnhub_eps_estimate,
        "finnhub_revenue_estimate": finnhub_revenue_estimate,
        "peers": peers,
        "risk_free_rate": risk_free_rate.value,
    }

    capital_structure = {
        "diluted_shares": facts_table["diluted_shares"].value,
        "total_debt": facts_table["total_debt"].value,
        "cash": facts_table["cash"].value,
        "market_cap": profile.get("mktCap"),
        "beta": profile.get("beta"),
    }

    security = Security(
        ticker=ticker,
        exchange=profile.get("exchangeShortName", ""),
        security_type="operating_company",
        reporting_currency=currency,
        valuation_currency=currency,
    )
    analysis = AnalysisMeta(
        knowledge_timestamp=now.isoformat(),
        market_timestamp=market_timestamp,
        industry_adapter="default_nonfinancial",
    )

    packet = Packet(
        security=security,
        analysis=analysis,
        fundamentals=fundamentals,
        market_data=market_data,
        estimates=estimates,
        capital_structure=capital_structure,
        insiders=insider_trades,
        institutional_holders=institutional_holders,
        facts_table=facts_table,
        staleness=staleness,
        packet_hash="",
    )

    return packet.model_copy(update={"packet_hash": _hash_packet(packet)})


def _hash_packet(packet: Packet) -> str:
    """sha256 of the canonical (sorted-key, compact-separator) JSON of
    `packet`, excluding `packet_hash` itself. Deterministic across rebuilds
    from identical inputs; changes if any packet content changes."""
    payload = packet.model_dump(mode="json")
    payload.pop("packet_hash", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
