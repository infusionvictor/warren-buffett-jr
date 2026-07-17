"""Pure technical-indicator math on price/volume series.

Stdlib + math only (no pandas dependency) so these stay trivially testable
offline. Every function takes plain lists of floats (oldest first) and returns
plain floats or None when there is not enough history — never a fabricated
value. Callers turn None into an honest MISSING Value per the Cerebro rule
"sin evidencia, no hay número".
"""

from __future__ import annotations

import math


def sma(values: list[float], n: int) -> float | None:
    """Simple moving average of the last `n` points, or None if too short."""
    if n <= 0 or len(values) < n:
        return None
    return sum(values[-n:]) / n


def roc(values: list[float], n: int) -> float | None:
    """Rate of change (return) over the last `n` sessions: v[-1]/v[-1-n] - 1."""
    if n <= 0 or len(values) <= n:
        return None
    past = values[-1 - n]
    if past <= 0:
        return None
    return values[-1] / past - 1


def linreg_slope(values: list[float]) -> float | None:
    """Ordinary-least-squares slope of `values` vs. their index (per step).

    Positive => the series is rising over the window. None if < 2 points.
    """
    m = len(values)
    if m < 2:
        return None
    xs = range(m)
    mean_x = (m - 1) / 2.0
    mean_y = sum(values) / m
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return None
    return num / den


def sma_slope_annualized(values: list[float], n: int, window: int = 63) -> float | None:
    """Annualized % slope of the `n`-period SMA over the last `window` steps.

    Builds the SMA series, fits a line to its last `window` points, and
    expresses the slope as a fraction of the latest SMA per ~252 sessions.
    Returns None without enough history for both the SMA and the window.
    """
    if len(values) < n + window:
        return None
    sma_series = [sum(values[i - n:i]) / n for i in range(n, len(values) + 1)]
    tail = sma_series[-window:]
    slope = linreg_slope(tail)
    if slope is None or tail[-1] <= 0:
        return None
    return slope * 252.0 / tail[-1]


def rsi(values: list[float], n: int = 14) -> float | None:
    """Wilder's Relative Strength Index (0-100), or None if too short."""
    if len(values) <= n:
        return None
    gains, losses = 0.0, 0.0
    for i in range(-n, 0):
        change = values[i] - values[i - 1]
        if change >= 0:
            gains += change
        else:
            losses -= change
    avg_gain, avg_loss = gains / n, losses / n
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def realized_vol(values: list[float], n: int = 63) -> float | None:
    """Annualized realized volatility from daily log returns over `n` sessions."""
    if len(values) < n + 1:
        return None
    rets = []
    for i in range(-n, 0):
        prev = values[i - 1]
        if prev <= 0 or values[i] <= 0:
            return None
        rets.append(math.log(values[i] / prev))
    m = len(rets)
    mean = sum(rets) / m
    var = sum((r - mean) ** 2 for r in rets) / (m - 1)
    return math.sqrt(var) * math.sqrt(252.0)


def pct_position_52w(values: list[float]) -> float | None:
    """Where the latest close sits in its trailing range: 0=low, 1=high.

    Uses up to the last 252 sessions. None if fewer than 2 points or a
    degenerate (flat) range.
    """
    window = values[-252:]
    if len(window) < 2:
        return None
    lo, hi = min(window), max(window)
    if hi <= lo:
        return None
    return (window[-1] - lo) / (hi - lo)


def obv_slope(closes: list[float], volumes: list[float], window: int = 50) -> float | None:
    """Sign-and-magnitude of the On-Balance-Volume trend, normalized.

    Builds OBV (cumulative signed volume), fits a line to its last `window`
    points, and normalizes the slope by mean volume so the result is a small
    dimensionless number comparable across tickers. None without volume or
    enough history.
    """
    if len(closes) != len(volumes) or len(closes) < window + 1:
        return None
    obv = [0.0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv.append(obv[-1] + volumes[i])
        elif closes[i] < closes[i - 1]:
            obv.append(obv[-1] - volumes[i])
        else:
            obv.append(obv[-1])
    tail = obv[-window:]
    slope = linreg_slope(tail)
    mean_vol = sum(volumes[-window:]) / window
    if slope is None or mean_vol <= 0:
        return None
    return slope / mean_vol


def up_down_volume_ratio(closes: list[float], volumes: list[float], window: int = 50) -> float | None:
    """Sum of up-day volume / sum of down-day volume over `window` sessions.

    >1 means accumulation (volume concentrated on up days). None without
    volume, enough history, or any down-day volume to divide by.
    """
    if len(closes) != len(volumes) or len(closes) < window + 1:
        return None
    up_vol, down_vol = 0.0, 0.0
    for i in range(len(closes) - window, len(closes)):
        if closes[i] > closes[i - 1]:
            up_vol += volumes[i]
        elif closes[i] < closes[i - 1]:
            down_vol += volumes[i]
    if down_vol <= 0:
        return None
    return up_vol / down_vol


def avg_dollar_volume(closes: list[float], volumes: list[float], window: int = 50) -> float | None:
    """Average daily dollar volume (close * volume) over `window` sessions."""
    if len(closes) != len(volumes) or len(closes) < window:
        return None
    pairs = list(zip(closes[-window:], volumes[-window:]))
    return sum(c * v for c, v in pairs) / window
