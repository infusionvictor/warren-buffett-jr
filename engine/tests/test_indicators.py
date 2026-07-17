"""Tests for the pure technical-indicator math."""

import math

from wbj import indicators as ind


def test_sma_basic_and_too_short():
    assert ind.sma([1, 2, 3, 4], 2) == 3.5
    assert ind.sma([1, 2, 3, 4], 4) == 2.5
    assert ind.sma([1, 2], 5) is None


def test_roc():
    assert abs(ind.roc([100, 110], 1) - 0.1) < 1e-9
    assert abs(ind.roc([100, 105, 120], 2) - 0.20) < 1e-9
    assert ind.roc([100], 1) is None  # not enough history


def test_linreg_slope():
    assert ind.linreg_slope([1, 2, 3, 4]) == 1.0
    assert ind.linreg_slope([4, 3, 2, 1]) == -1.0
    assert ind.linreg_slope([5]) is None


def test_rsi_extremes():
    assert ind.rsi(list(range(1, 30))) == 100.0  # only gains
    assert ind.rsi(list(range(30, 1, -1))) == 0.0  # only losses (avg_gain 0)


def test_realized_vol_positive():
    # A gently oscillating series has small but positive annualized vol.
    closes = [100 + math.sin(i / 3) for i in range(80)]
    v = ind.realized_vol(closes)
    assert v is not None and v > 0


def test_pct_position_52w():
    assert ind.pct_position_52w([10, 20, 30]) == 1.0   # latest is the high
    assert ind.pct_position_52w([30, 20, 10]) == 0.0   # latest is the low
    assert ind.pct_position_52w([10, 20, 15]) == 0.5   # midway
    assert ind.pct_position_52w([5, 5, 5]) is None      # flat range


def test_up_down_volume_and_obv():
    # 51 points, mostly up days, constant volume -> ratio > 1, OBV rising.
    closes = [100 + i for i in range(51)]
    vols = [1_000_000.0] * 51
    assert ind.up_down_volume_ratio(closes, vols) is None  # no down days to divide by
    # Introduce a couple of down days.
    closes2 = closes[:]
    closes2[25] = closes2[24] - 1
    closes2[40] = closes2[39] - 1
    ratio = ind.up_down_volume_ratio(closes2, vols)
    assert ratio is not None and ratio > 1
    assert ind.obv_slope(closes2, vols) > 0


def test_avg_dollar_volume():
    closes = [10.0] * 50
    vols = [1_000_000.0] * 50
    assert ind.avg_dollar_volume(closes, vols) == 10_000_000.0
