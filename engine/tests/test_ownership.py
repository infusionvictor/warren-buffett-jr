"""Tests for insider (Form 4) and 13F institutional ownership processing."""

import json
from datetime import date
from pathlib import Path

from wbj.ownership import (
    MATERIAL_USD,
    _direction,
    insider_summary,
    institutional_summary,
)

FIX = Path(__file__).parent / "fixtures" / "fmp"
TODAY = date(2026, 7, 17)


def _load(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def test_direction_classification():
    assert _direction("P-Purchase") == "buy"
    assert _direction("S-Sale") == "sell"
    assert _direction("A-Award") == "other"
    assert _direction("M-Exercise") == "other"
    assert _direction(None) == "other"


def test_insider_summary_flags_material_sells():
    s = insider_summary(_load("insider_trades.json"), today=TODAY)
    assert s is not None
    # Huang 120000*118.45=14.214M and Kress 45000*112.30=5.0535M, both > $1M.
    assert len(s["material_sells"]) == 2
    assert s["material_buys"] == []
    assert abs(s["total_material_sell_usd"] - (14_214_000 + 5_053_500)) < 1
    assert s["net_material_usd"] < 0            # net selling
    assert s["material_threshold"] == MATERIAL_USD
    assert s["top_transactions"][0]["insider"] == "Huang Jen-Hsun"  # largest first


def test_insider_summary_threshold_and_window():
    trades = [
        {"reportingName": "Small Buyer", "transactionType": "P-Purchase",
         "securitiesTransacted": 1000, "price": 100.0, "transactionDate": "2026-07-01"},   # $100k < $1M
        {"reportingName": "Big Buyer", "transactionType": "P-Purchase",
         "securitiesTransacted": 50000, "price": 100.0, "transactionDate": "2026-07-01"},  # $5M
        {"reportingName": "Old Seller", "transactionType": "S-Sale",
         "securitiesTransacted": 50000, "price": 100.0, "transactionDate": "2020-01-01"},  # out of window
    ]
    s = insider_summary(trades, today=TODAY, months=12)
    assert [b["insider"] for b in s["material_buys"]] == ["Big Buyer"]  # small excluded
    assert s["material_sells"] == []                                    # old excluded
    assert s["net_material_usd"] == 5_000_000


def test_insider_summary_none_on_empty():
    assert insider_summary([], today=TODAY) is None
    assert insider_summary(None, today=TODAY) is None


def test_institutional_summary_ranks_and_values():
    inst = institutional_summary(_load("institutional_holders.json"), price=118.45)
    assert inst is not None
    assert inst["top_holders"][0]["holder"] == "Vanguard Group Inc"  # most shares first
    assert inst["top_holders"][0]["value_usd"] == round(1_580_000_000 * 118.45)
    assert inst["n_increasing"] == 1 and inst["n_decreasing"] == 1
    assert inst["net_change_shares"] == 12_500_000 - 3_200_000
    assert inst["as_of"] == "2026-03-31"


def test_institutional_summary_without_price():
    inst = institutional_summary(_load("institutional_holders.json"))
    assert inst["top_holders"][0]["value_usd"] is None  # no price -> no value
    assert inst["total_shares"] == 1_580_000_000 + 1_340_000_000
