"""Tests for wbj.packet.builder.build_packet.

Uses fake provider objects (see tests/fixtures/packet/make_packet_fixture.py)
reading the fmp/edgar/finnhub/fred fixture JSON — no MockTransport needed
at this layer per the task-10 design decision.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone

import pytest

from wbj.packet.builder import PacketRejected, Providers, build_packet
from wbj.schemas.packet import Packet

from .conftest import FakeEdgarProvider, FakeFinnhubProvider, FakeFMPProvider, FakeFredProvider, generate_ohlcv_sessions


# --- canonical field mapping -------------------------------------------------


def test_annual_fundamentals_use_canonical_field_names(fake_providers, fixed_now):
    packet = build_packet("NVDA", fake_providers, fixed_now)

    latest = packet.fundamentals["annual"][0]
    assert latest["net_income"] == 72880000000
    assert latest["operating_cash_flow"] == 64089000000
    assert latest["capex"] == -3236000000
    assert latest["diluted_shares"] == 24700000000
    assert latest["cash"] == 8589000000
    assert latest["total_debt"] == 9710000000
    assert latest["revenue"] == 130497000000
    assert latest["ebit"] == 81453000000
    # raw FMP keys must not leak through
    assert "netIncome" not in latest
    assert "weightedAverageShsOutDil" not in latest


def test_annual_fundamentals_has_five_fiscal_years(fake_providers, fixed_now):
    packet = build_packet("NVDA", fake_providers, fixed_now)

    assert len(packet.fundamentals["annual"]) == 5


def test_quarterly_fundamentals_use_canonical_field_names(fake_providers, fixed_now):
    packet = build_packet("NVDA", fake_providers, fixed_now)

    latest_q = packet.fundamentals["quarterly"][0]
    assert latest_q["net_income"] == 18775000000
    assert latest_q["operating_cash_flow"] == 27414000000


def test_annual_fundamentals_maps_additional_statement_fields(fake_providers, fixed_now):
    """Task: fuller FMP statement mapping -- interest expense, D&A, SG&A,
    accounts payable, retained earnings, and net PP&E should all appear
    under their canonical names once FMP's raw statement rows carry them,
    even though today's specialist formulas for RSK-ICOV-011/AQI-023/etc.
    read these from `overlay` rather than `Packet.fundamentals` (see
    CANONICAL_FIELD_MAP's docstring)."""
    income = copy.deepcopy(FakeFMPProvider()._get("income_annual"))
    income[0]["interestExpense"] = 257000000
    income[0]["sellingGeneralAndAdministrativeExpenses"] = 3200000000
    balance = copy.deepcopy(FakeFMPProvider()._get("balance_annual"))
    balance[0]["accountPayables"] = 6300000000
    balance[0]["retainedEarnings"] = 55000000000
    balance[0]["propertyPlantEquipmentNet"] = 6200000000
    fmp = FakeFMPProvider(
        ohlcv=fake_providers.fmp._ohlcv,
        benchmark_ohlcv=fake_providers.fmp._benchmark_ohlcv,
        overrides={"income_annual": income, "balance_annual": balance},
    )
    providers = Providers(fmp=fmp, edgar=fake_providers.edgar, finnhub=fake_providers.finnhub, fred=fake_providers.fred)

    packet = build_packet("NVDA", providers, fixed_now)

    latest = packet.fundamentals["annual"][0]
    assert latest["interest_expense"] == 257000000
    assert latest["sga"] == 3200000000
    assert latest["accounts_payable"] == 6300000000
    assert latest["retained_earnings"] == 55000000000
    assert latest["ppe_net"] == 6200000000
    # raw FMP keys must not leak through
    assert "interestExpense" not in latest
    assert "accountPayables" not in latest
    assert "propertyPlantEquipmentNet" not in latest


# --- >=252 daily sessions enforced ------------------------------------------


def test_full_ohlcv_history_accepted(fake_providers, fixed_now):
    packet = build_packet("NVDA", fake_providers, fixed_now)

    assert len(packet.market_data.daily) >= 252


def test_fewer_than_252_sessions_rejects(fixed_now):
    short_ohlcv = generate_ohlcv_sessions(end=(fixed_now - timedelta(days=1)).date(), sessions=100)
    providers = Providers(
        fmp=FakeFMPProvider(ohlcv=short_ohlcv),
        edgar=FakeEdgarProvider(),
        finnhub=FakeFinnhubProvider(),
        fred=FakeFredProvider(),
    )

    with pytest.raises(PacketRejected, match="daily sessions"):
        build_packet("NVDA", providers, fixed_now)


def test_exactly_252_sessions_accepted(fixed_now):
    ohlcv = generate_ohlcv_sessions(end=(fixed_now - timedelta(days=1)).date(), sessions=252)
    providers = Providers(
        fmp=FakeFMPProvider(ohlcv=ohlcv),
        edgar=FakeEdgarProvider(),
        finnhub=FakeFinnhubProvider(),
        fred=FakeFredProvider(),
    )

    packet = build_packet("NVDA", providers, fixed_now)

    assert len(packet.market_data.daily) == 252


# --- benchmark/sector series --------------------------------------------


def test_benchmark_and_sector_populated_from_spy(fake_providers, fixed_now):
    packet = build_packet("NVDA", fake_providers, fixed_now)

    assert len(packet.market_data.benchmark) > 0
    assert packet.market_data.benchmark == packet.market_data.sector
    # Genuinely different data from the stock's own series (fixture uses a
    # different synthetic start price for the SPY-like benchmark).
    assert packet.market_data.benchmark[0].close != packet.market_data.daily[0].close


def test_benchmark_aligned_to_stock_trading_dates(fixed_now):
    """Inner-join alignment: a benchmark bar on a date the stock didn't
    trade must be dropped; a stock date missing from the raw benchmark
    series must not appear in the aligned output either."""
    stock_ohlcv = generate_ohlcv_sessions(end=(fixed_now - timedelta(days=1)).date(), sessions=260)
    stock_dates = {bar["date"] for bar in stock_ohlcv}
    benchmark_ohlcv = generate_ohlcv_sessions(
        end=(fixed_now - timedelta(days=1)).date(), sessions=260, start_price=400.0
    )
    # Inject one extra benchmark bar on a date the stock never traded.
    benchmark_ohlcv = [{**benchmark_ohlcv[0], "date": "1999-01-01"}, *benchmark_ohlcv]

    providers = Providers(
        fmp=FakeFMPProvider(ohlcv=stock_ohlcv, benchmark_ohlcv=benchmark_ohlcv),
        edgar=FakeEdgarProvider(),
        finnhub=FakeFinnhubProvider(),
        fred=FakeFredProvider(),
    )

    packet = build_packet("NVDA", providers, fixed_now)

    benchmark_dates = {row.date for row in packet.market_data.benchmark}
    assert "1999-01-01" not in benchmark_dates
    assert benchmark_dates <= stock_dates


def test_benchmark_empty_when_provider_returns_no_data(fake_providers, fixed_now):
    fmp = FakeFMPProvider(
        ohlcv=fake_providers.fmp._ohlcv, overrides={"benchmark_ohlcv": []}
    )
    providers = Providers(fmp=fmp, edgar=fake_providers.edgar, finnhub=fake_providers.finnhub, fred=fake_providers.fred)

    packet = build_packet("NVDA", providers, fixed_now)

    assert packet.market_data.benchmark == []
    assert packet.market_data.sector == []


# --- hash stability -----------------------------------------------------


def test_hash_stable_across_rebuilds_with_same_inputs(fake_providers, fixed_now):
    packet_a = build_packet("NVDA", fake_providers, fixed_now)
    packet_b = build_packet("NVDA", fake_providers, fixed_now)

    assert packet_a.packet_hash == packet_b.packet_hash
    assert packet_a.packet_hash != ""


def test_hash_changes_when_one_input_changes(fake_providers, fixed_now):
    packet_a = build_packet("NVDA", fake_providers, fixed_now)

    mutated_profile = copy.deepcopy(fake_providers.fmp._get("profile"))
    mutated_profile[0]["price"] = mutated_profile[0]["price"] + 1.0
    mutated_fmp = FakeFMPProvider(
        ohlcv=fake_providers.fmp._ohlcv, overrides={"profile": mutated_profile}
    )
    mutated_providers = Providers(
        fmp=mutated_fmp, edgar=fake_providers.edgar, finnhub=fake_providers.finnhub, fred=fake_providers.fred
    )
    packet_b = build_packet("NVDA", mutated_providers, fixed_now)

    assert packet_a.packet_hash != packet_b.packet_hash


def test_hash_excludes_itself_from_the_hashed_payload(fake_providers, fixed_now):
    """Sanity check: packet_hash is not self-referential garbage — it's a
    real 64-char hex sha256 digest."""
    packet = build_packet("NVDA", fake_providers, fixed_now)

    assert len(packet.packet_hash) == 64
    int(packet.packet_hash, 16)  # raises ValueError if not hex


# --- staleness table ------------------------------------------------------


def test_daily_market_staleness_fresh_for_recent_ohlcv(fake_providers, fixed_now):
    packet = build_packet("NVDA", fake_providers, fixed_now)

    assert packet.staleness["daily_market"] == "FRESH"


def test_quarterly_fundamentals_staleness_stale_for_old_filings(fake_providers, fixed_now):
    # fixture quarterly statements are dated ~2025-04, `fixed_now` is
    # 2026-07 -> well past the 120-day quarterly_fundamentals threshold.
    packet = build_packet("NVDA", fake_providers, fixed_now)

    assert packet.staleness["quarterly_fundamentals"] == "STALE"


def test_quarterly_fundamentals_staleness_fresh_for_recent_filing(fixed_now):
    recent_date = (fixed_now - timedelta(days=10)).date().isoformat()
    income_q = [{**row, "date": recent_date} for row in FakeFMPProvider()._get("income_quarterly")]
    ohlcv = generate_ohlcv_sessions(end=(fixed_now - timedelta(days=1)).date())
    fmp = FakeFMPProvider(ohlcv=ohlcv, overrides={"income_quarterly": income_q})
    providers = Providers(fmp=fmp, edgar=FakeEdgarProvider(), finnhub=FakeFinnhubProvider(), fred=FakeFredProvider())

    packet = build_packet("NVDA", providers, fixed_now)

    assert packet.staleness["quarterly_fundamentals"] == "FRESH"


def test_consensus_staleness_stale_for_old_earnings_print(fake_providers, fixed_now):
    # fixture fmp earnings_calendar's latest actual print is 2026-05-27,
    # 50 days before fixed_now (2026-07-16) -> past the 7-day threshold.
    packet = build_packet("NVDA", fake_providers, fixed_now)

    assert packet.staleness["consensus"] == "STALE"


def test_peer_set_staleness_stale_for_old_13f(fake_providers, fixed_now):
    # fixture institutional_holders dateReported is 2026-03-31, well past
    # the 90-day peer_set threshold relative to fixed_now.
    packet = build_packet("NVDA", fake_providers, fixed_now)

    assert packet.staleness["peer_set"] == "STALE"


# --- facts table --------------------------------------------------------


def test_facts_table_revenue_reconciled_from_fmp_and_edgar(fake_providers, fixed_now):
    packet = build_packet("NVDA", fake_providers, fixed_now)

    revenue = packet.facts_table["revenue"]
    assert revenue.is_valid
    assert revenue.value == 130497000000


def test_facts_table_diluted_shares_reconciled(fake_providers, fixed_now):
    packet = build_packet("NVDA", fake_providers, fixed_now)

    shares = packet.facts_table["diluted_shares"]
    assert shares.is_valid
    assert shares.value == 24700000000


def test_facts_table_cash_reconciled(fake_providers, fixed_now):
    packet = build_packet("NVDA", fake_providers, fixed_now)

    cash = packet.facts_table["cash"]
    assert cash.is_valid
    assert cash.value == 8589000000


def test_facts_table_total_debt_reconciled(fake_providers, fixed_now):
    packet = build_packet("NVDA", fake_providers, fixed_now)

    total_debt = packet.facts_table["total_debt"]
    assert total_debt.is_valid
    assert total_debt.value == 9710000000


def test_facts_table_total_debt_includes_edgar_operating_lease_liability(fixed_now):
    """FMP's totalDebt aggregate includes ASC-842 operating-lease
    liabilities; EDGAR's LongTermDebtNoncurrent+DebtCurrent alone must
    not be compared against it directly -- that produces a spurious
    CONFLICTED verdict on real lease-heavy filers (observed live on
    NVDA). EDGAR's own OperatingLeaseLiability{Noncurrent,Current}
    (present value, already on the balance sheet) must be added in
    before reconciling."""
    companyfacts = copy.deepcopy(FakeEdgarProvider()._companyfacts)
    end = "2025-01-26"  # matches the fixture's latest annual period end
    companyfacts["facts"]["us-gaap"]["OperatingLeaseLiabilityNoncurrent"] = {
        "units": {"USD": [{"end": end, "val": 500000000}]}
    }
    companyfacts["facts"]["us-gaap"]["OperatingLeaseLiabilityCurrent"] = {
        "units": {"USD": [{"end": end, "val": 100000000}]}
    }
    # 8460M (LT) + 1250M (current) + 500M (lease noncurrent) + 100M (lease current) = 10,310M
    balance_annual_with_lease = copy.deepcopy(FakeFMPProvider()._get("balance_annual"))
    balance_annual_with_lease[0]["totalDebt"] = 10310000000

    providers = Providers(
        fmp=FakeFMPProvider(
            ohlcv=generate_ohlcv_sessions(end=(fixed_now - timedelta(days=1)).date()),
            overrides={"balance_annual": balance_annual_with_lease},
        ),
        edgar=FakeEdgarProvider(companyfacts=companyfacts),
        finnhub=FakeFinnhubProvider(),
        fred=FakeFredProvider(),
    )

    packet = build_packet("NVDA", providers, fixed_now)

    total_debt = packet.facts_table["total_debt"]
    assert total_debt.is_valid
    assert total_debt.value == 10310000000


def test_facts_table_price_is_fmp_only(fake_providers, fixed_now):
    packet = build_packet("NVDA", fake_providers, fixed_now)

    price = packet.facts_table["price"]
    assert price.is_valid
    assert price.value == 120.5
    assert price.source_name == "FMP"


def test_facts_table_revenue_conflicted_when_sources_disagree(fixed_now):
    companyfacts = FakeEdgarProvider()._companyfacts
    import copy as _copy

    conflicted_facts = _copy.deepcopy(companyfacts)
    conflicted_facts["facts"]["us-gaap"]["Revenues"]["units"]["USD"][1]["val"] = 200000000000  # >5% off FMP's 130.497B
    providers = Providers(
        fmp=FakeFMPProvider(ohlcv=generate_ohlcv_sessions(end=(fixed_now - timedelta(days=1)).date())),
        edgar=FakeEdgarProvider(companyfacts=conflicted_facts),
        finnhub=FakeFinnhubProvider(),
        fred=FakeFredProvider(),
    )

    packet = build_packet("NVDA", providers, fixed_now)

    assert packet.facts_table["revenue"].is_null
    assert packet.facts_table["revenue"].state == "CONFLICTED"


# --- hard rejects ---------------------------------------------------------


def test_missing_currency_rejects(fake_providers, fixed_now):
    no_currency_profile = copy.deepcopy(fake_providers.fmp._get("profile"))
    del no_currency_profile[0]["currency"]
    fmp = FakeFMPProvider(ohlcv=fake_providers.fmp._ohlcv, overrides={"profile": no_currency_profile})
    providers = Providers(fmp=fmp, edgar=fake_providers.edgar, finnhub=fake_providers.finnhub, fred=fake_providers.fred)

    with pytest.raises(PacketRejected, match="currency"):
        build_packet("NVDA", providers, fixed_now)


def test_missing_timestamps_rejects_when_no_market_data_at_all(fixed_now):
    fmp = FakeFMPProvider(ohlcv=[])
    finnhub = FakeFinnhubProvider(overrides={"quote": None})
    providers = Providers(fmp=fmp, edgar=FakeEdgarProvider(), finnhub=finnhub, fred=FakeFredProvider())

    with pytest.raises(PacketRejected, match="timestamp"):
        build_packet("NVDA", providers, fixed_now)


def test_no_diluted_share_count_from_any_source_rejects(fixed_now):
    income_annual_no_shares = [
        {k: v for k, v in row.items() if k != "weightedAverageShsOutDil"}
        for row in FakeFMPProvider()._get("income_annual")
    ]
    income_quarterly_no_shares = [
        {k: v for k, v in row.items() if k != "weightedAverageShsOutDil"}
        for row in FakeFMPProvider()._get("income_quarterly")
    ]
    companyfacts_no_shares = copy.deepcopy(FakeEdgarProvider()._companyfacts)
    del companyfacts_no_shares["facts"]["us-gaap"]["WeightedAverageNumberOfDilutedSharesOutstanding"]
    del companyfacts_no_shares["facts"]["dei"]["EntityCommonStockSharesOutstanding"]

    fmp = FakeFMPProvider(
        ohlcv=generate_ohlcv_sessions(end=(fixed_now - timedelta(days=1)).date()),
        overrides={
            "income_annual": income_annual_no_shares,
            "income_quarterly": income_quarterly_no_shares,
        },
    )
    edgar = FakeEdgarProvider(companyfacts=companyfacts_no_shares)
    providers = Providers(fmp=fmp, edgar=edgar, finnhub=FakeFinnhubProvider(), fred=FakeFredProvider())

    with pytest.raises(PacketRejected, match="diluted share"):
        build_packet("NVDA", providers, fixed_now)


def test_diluted_shares_falls_back_to_edgar_basic_shares_when_no_weighted_tag(fixed_now):
    """EDGAR-only fallback: no FMP diluted count, no EDGAR weighted-diluted
    tag, but EDGAR's dei:EntityCommonStockSharesOutstanding is present ->
    should NOT reject."""
    income_annual_no_shares = [
        {k: v for k, v in row.items() if k != "weightedAverageShsOutDil"}
        for row in FakeFMPProvider()._get("income_annual")
    ]
    income_quarterly_no_shares = [
        {k: v for k, v in row.items() if k != "weightedAverageShsOutDil"}
        for row in FakeFMPProvider()._get("income_quarterly")
    ]
    companyfacts_basic_only = copy.deepcopy(FakeEdgarProvider()._companyfacts)
    del companyfacts_basic_only["facts"]["us-gaap"]["WeightedAverageNumberOfDilutedSharesOutstanding"]

    fmp = FakeFMPProvider(
        ohlcv=generate_ohlcv_sessions(end=(fixed_now - timedelta(days=1)).date()),
        overrides={
            "income_annual": income_annual_no_shares,
            "income_quarterly": income_quarterly_no_shares,
        },
    )
    edgar = FakeEdgarProvider(companyfacts=companyfacts_basic_only)
    providers = Providers(fmp=fmp, edgar=edgar, finnhub=FakeFinnhubProvider(), fred=FakeFredProvider())

    packet = build_packet("NVDA", providers, fixed_now)

    assert packet is not None


# --- top-level packet shape -----------------------------------------------


def test_packet_is_a_valid_pydantic_packet(fake_providers, fixed_now):
    packet = build_packet("NVDA", fake_providers, fixed_now)

    assert isinstance(packet, Packet)
    assert packet.security.ticker == "NVDA"
    assert packet.security.reporting_currency == "USD"
    assert packet.security.valuation_currency == "USD"
    assert packet.analysis.knowledge_timestamp == fixed_now.isoformat()
    assert packet.analysis.market_timestamp is not None


def test_knowledge_timestamp_uses_the_now_parameter_not_wall_clock(fake_providers):
    frozen = datetime(2020, 1, 1, tzinfo=timezone.utc)
    ohlcv = generate_ohlcv_sessions(end=(frozen - timedelta(days=1)).date())
    providers = Providers(
        fmp=FakeFMPProvider(ohlcv=ohlcv), edgar=FakeEdgarProvider(), finnhub=FakeFinnhubProvider(), fred=FakeFredProvider()
    )

    packet = build_packet("NVDA", providers, frozen)

    assert packet.analysis.knowledge_timestamp == frozen.isoformat()
