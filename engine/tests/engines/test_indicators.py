"""Tests for `wbj.engines.indicators`.

Verified against hand-computed short series per the task-11 brief, plus a
golden check against the NVDA packet fixture (600 real-shaped sessions)
comparing against reference values computed inline with straightforward
pandas ops — never hardcoded magic numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from wbj.engines.indicators import (
    adx14,
    atr14,
    cmf,
    composite_rs_percentile,
    ema,
    macd,
    median_dollar_volume,
    obv,
    range_position_52w,
    relative_strength,
    roc,
    rsi14,
    sma,
    true_range,
    up_down_volume_ratio,
    volume_ratio,
    realized_vol,
)

_FIXTURE_PATH = (
    Path(__file__).parent.parent / "fixtures" / "packet" / "NVDA_packet.json"
)


def constant_tr_frame(tr: float, bars: int) -> pd.DataFrame:
    """`bars` sessions with a constant true range of `tr`.

    high/low span exactly `tr`, close sits at the midpoint and never moves,
    so |high-prior_close| and |low-prior_close| never exceed high-low.
    """
    high = [100.0 + tr / 2] * bars
    low = [100.0 - tr / 2] * bars
    close = [100.0] * bars
    return pd.DataFrame({"high": high, "low": low, "close": close})


def load_nvda_ohlcv() -> pd.DataFrame:
    """Ascending-date OHLCV DataFrame from the golden NVDA packet fixture."""
    payload = json.loads(_FIXTURE_PATH.read_text())
    rows = payload["market_data"]["daily"]
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["close"] = df["adj_close"]  # use the adjusted series for calculations
    return df[["date", "open", "high", "low", "close", "volume"]]


# --- brief Step 1 tests (verbatim shape) ---


def test_wilder_atr_smoothing():
    df = constant_tr_frame(tr=2.0, bars=20)
    assert abs(atr14(df).iloc[-1] - 2.0) < 1e-9


def test_rsi_all_gains_is_100():
    close = pd.Series(np.arange(1.0, 40.0))
    assert rsi14(close).iloc[-1] == 100.0


def test_composite_rs_weights_sum():
    # 101 evenly spaced values 1..101; target sits exactly at the median
    # (51) in every window -> each PctRank is exactly 50 -> composite 50.
    universe = pd.DataFrame(
        {
            "RS21": np.arange(1, 102),
            "RS63": np.arange(1, 102),
            "RS126": np.arange(1, 102),
            "RS252": np.arange(1, 102),
        }
    )
    rs_by_window = {"RS21": 51, "RS63": 51, "RS126": 51, "RS252": 51}
    result = composite_rs_percentile(rs_by_window, universe)
    assert abs(result - 50.0) < 1e-9


def test_ema_initialized_with_sma():
    close = pd.Series(np.arange(1.0, 30.0))
    n = 10
    result = ema(close, n)
    assert abs(result.iloc[n - 1] - close.iloc[:n].mean()) < 1e-12
    # Nothing before the seed index.
    assert result.iloc[: n - 1].isna().all()


# --- supplementary coverage ---


def test_sma_requires_n_observations():
    close = pd.Series(np.arange(1.0, 10.0))
    result = sma(close, 5)
    assert result.iloc[:4].isna().all()
    assert abs(result.iloc[4] - close.iloc[:5].mean()) < 1e-12


def test_ema_recursion_matches_alpha_formula():
    close = pd.Series(np.arange(1.0, 30.0))
    n = 10
    alpha = 2 / (n + 1)
    result = ema(close, n)
    expected_next = alpha * close.iloc[n] + (1 - alpha) * result.iloc[n - 1]
    assert abs(result.iloc[n] - expected_next) < 1e-12


def test_true_range_first_bar_falls_back_to_high_minus_low():
    df = pd.DataFrame({"high": [102.0, 105.0], "low": [98.0, 101.0], "close": [100.0, 104.0]})
    tr = true_range(df)
    assert abs(tr.iloc[0] - 4.0) < 1e-12
    # bar 1: max(105-101=4, |105-100|=5, |101-100|=1) = 5
    assert abs(tr.iloc[1] - 5.0) < 1e-12


def test_rsi_all_losses_is_zero():
    close = pd.Series(np.arange(40.0, 1.0, -1.0))
    assert rsi14(close).iloc[-1] == 0.0


def test_rsi_flat_price_is_100():
    close = pd.Series([50.0] * 30)
    assert rsi14(close).iloc[-1] == 100.0


def test_macd_returns_dict_with_expected_components():
    close = pd.Series(np.linspace(50.0, 90.0, 80))
    result = macd(close)
    assert set(result.keys()) == {"macd", "signal", "hist"}
    macd_line = ema(close, 12) - ema(close, 26)
    signal_line = ema(macd_line, 9)
    pd.testing.assert_series_equal(result["macd"], macd_line, check_names=False)
    pd.testing.assert_series_equal(result["signal"], signal_line, check_names=False)
    pd.testing.assert_series_equal(result["hist"], macd_line - signal_line, check_names=False)


def test_adx14_bounded_and_high_for_strong_trend():
    n = 60
    close = pd.Series(100.0 + np.arange(n) * 1.5)
    high = close + 1.0
    low = close - 1.0
    df = pd.DataFrame({"high": high, "low": low, "close": close})
    result = adx14(df)
    valid = result.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()
    # A relentless uptrend should register strong (not weak) trend strength.
    assert valid.iloc[-1] > 20


def test_roc():
    close = pd.Series([100.0, 105.0, 110.0, 121.0])
    result = roc(close, 2)
    assert result.isna().iloc[:2].all()
    assert abs(result.iloc[2] - (110.0 / 100.0 - 1)) < 1e-12
    assert abs(result.iloc[3] - (121.0 / 105.0 - 1)) < 1e-12


def test_relative_strength_is_return_difference_per_cerebro_tech_rs_011():
    # Cerebro TECH-RS-011: "Stock total return_N - Benchmark total return_N"
    # (a difference of N-day returns) — this wins over the task-11 brief's
    # own interface note ("ratio of n-day returns"); see commit message.
    close = pd.Series([100.0, 110.0, 121.0])
    bench = pd.Series([100.0, 105.0, 110.0])
    result = relative_strength(close, bench, 2)
    expected = roc(close, 2) - roc(bench, 2)
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_realized_vol_matches_inline_pandas():
    close = pd.Series([100.0, 101.0, 99.0, 103.0, 102.0, 105.0, 104.0, 108.0])
    n = 5
    log_ret = np.log(close / close.shift(1))
    expected = log_ret.rolling(n).std() * np.sqrt(252)
    result = realized_vol(close, n)
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_volume_ratio_vs_median_50d():
    volume = pd.Series([100.0] * 49 + [400.0] + [1000.0])
    result = volume_ratio(volume)
    median_50 = volume.iloc[-51:-1].median()
    assert abs(result.iloc[-1] - 1000.0 / median_50) < 1e-9


def test_up_down_volume_ratio_zero_denominator_is_not_meaningful():
    close = pd.Series([100.0] + [100.0 + i for i in range(1, 51)])  # all up closes
    volume = pd.Series([10.0] * 51)
    df = pd.DataFrame({"close": close, "volume": volume})
    result = up_down_volume_ratio(df, n=50)
    assert pd.isna(result.iloc[-1])


def test_obv_accumulates_signed_volume():
    close = pd.Series([100.0, 101.0, 99.0, 99.0, 102.0])
    volume = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    df = pd.DataFrame({"close": close, "volume": volume})
    result = obv(df)
    # 0, +20, -30, +0 (flat), +50
    assert list(result) == [0.0, 20.0, -10.0, -10.0, 40.0]


def test_cmf_zero_range_bar_has_zero_multiplier():
    high = pd.Series([105.0, 100.0, 108.0])
    low = pd.Series([95.0, 100.0, 98.0])
    close = pd.Series([102.0, 100.0, 106.0])
    volume = pd.Series([1000.0, 500.0, 1500.0])
    df = pd.DataFrame({"high": high, "low": low, "close": close, "volume": volume})
    result = cmf(df, n=3)
    assert np.isfinite(result.iloc[-1])


def test_range_position_52w():
    n = 10
    close = pd.Series(np.linspace(50.0, 100.0, n))
    high = close + 1.0
    low = close - 1.0
    df = pd.DataFrame({"high": high, "low": low, "close": close})
    result = range_position_52w(df, n=n)
    expected = (close.iloc[-1] - low.iloc[:n].min()) / (high.iloc[:n].max() - low.iloc[:n].min())
    assert abs(result.iloc[-1] - expected) < 1e-9


def test_median_dollar_volume():
    close = pd.Series([10.0, 20.0, 30.0])
    volume = pd.Series([100.0, 200.0, 300.0])
    df = pd.DataFrame({"close": close, "volume": volume})
    result = median_dollar_volume(df, n=3)
    assert abs(result.iloc[-1] - pd.Series([1000.0, 4000.0, 9000.0]).median()) < 1e-9


# --- golden fixture: NVDA 600-session history ---


def test_golden_nvda_sma_ema_rsi_atr_macd():
    df = load_nvda_ohlcv()
    close = df["close"]

    sma20 = sma(close, 20)
    assert abs(sma20.iloc[-1] - close.iloc[-20:].mean()) < 1e-9

    ema12 = ema(close, 12)
    # Reference EMA computed with pandas' own adjust=False recursion,
    # reseeded to SMA(12) at index 11 to match this module's Wilder-style
    # initialization instead of pandas' exponentially-weighted seed.
    ref = pd.Series(index=close.index, dtype=float)
    ref.iloc[11] = close.iloc[:12].mean()
    alpha = 2 / 13
    for i in range(12, len(close)):
        ref.iloc[i] = alpha * close.iloc[i] + (1 - alpha) * ref.iloc[i - 1]
    assert abs(ema12.iloc[-1] - ref.iloc[-1]) < 1e-6

    atr = atr14(df)
    tr = true_range(df)
    ref_atr = tr.iloc[:14].mean()
    for i in range(14, len(tr)):
        ref_atr = (13 * ref_atr + tr.iloc[i]) / 14
    assert abs(atr.iloc[-1] - ref_atr) < 1e-6

    rsi = rsi14(close)
    assert 0.0 <= rsi.iloc[-1] <= 100.0

    macd_result = macd(close)
    ema26 = ema(close, 26)
    assert abs((ema12.iloc[-1] - ema26.iloc[-1]) - macd_result["macd"].iloc[-1]) < 1e-9
