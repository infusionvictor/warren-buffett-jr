"""Technical indicator library.

Pure pandas/numpy calculations over adjusted daily OHLCV, per `Cerebro/
04_technical_momentum/FORMULAS.md` (TECH-001..021, 034..040 — the
authoritative source for exact parameters; see the module-level
`_CEREBRO_DISCREPANCIES` note below for the one place this module diverges
from the task-11 brief's own interface note in favor of Cerebro).

Every function takes plain pandas `Series`/`DataFrame` and returns a plain
`Series` (or `float` for `composite_rs_percentile`) — no `Value` wrapping.
Callers that need null-state lineage (e.g. the technical specialist, a
later task) wrap individual point-in-time results in `wbj.core.nullstates
.Value` at the call site; vectorizing the underlying math here keeps this
module simple and independently testable.

`DataFrame` inputs are expected to carry lowercase columns `open`, `high`,
`low`, `close`, `volume` — `close` already the split/dividend-adjusted
series (matching `wbj.schemas.packet.OHLCVRow.adj_close`), sorted
ascending by date (oldest first). `Series` inputs (e.g. to `sma`/`ema`)
are any single adjusted price/volume column in that same ascending order.

Discrepancy vs. the task-11 brief (documented here per the task
instructions; Cerebro wins): the brief's own interface note describes
`relative_strength` as a "ratio of n-day returns", but Cerebro's
TECH-RS-011 defines it as a *difference* of N-day total returns (`Stock
total return_N - Benchmark total return_N`). This module implements the
Cerebro definition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "sma",
    "ema",
    "true_range",
    "atr14",
    "rsi14",
    "macd",
    "adx14",
    "roc",
    "relative_strength",
    "composite_rs_percentile",
    "realized_vol",
    "volume_ratio",
    "up_down_volume_ratio",
    "obv",
    "cmf",
    "range_position_52w",
    "median_dollar_volume",
]

# Cerebro TECH-RSC-013 composite relative-strength percentile weights.
_RS_WEIGHTS: dict[str, float] = {
    "RS21": 0.35,
    "RS63": 0.25,
    "RS126": 0.25,
    "RS252": 0.15,
}


def sma(close: pd.Series, n: int) -> pd.Series:
    """Simple moving average (TECH-SMA-002): mean(Close over N sessions).

    Requires N valid observations — NaN until the window fills.
    """
    return close.rolling(window=n, min_periods=n).mean()


def _wilder_smooth(series: pd.Series, n: int) -> pd.Series:
    """Generic Wilder smoothing: seed = mean of the first N valid values,
    then `out_t = ((n-1)*out_{t-1} + x_t) / n`.

    Leading NaNs (e.g. from a `.diff()`-derived input series) are dropped
    before locating the seed window, then the result is reindexed back
    onto the original (NaN-prefixed) index — this is the shared
    implementation behind `atr14`, `rsi14`'s average gain/loss, and
    `adx14`'s +DM/-DM/DX smoothing, all of which use the same Wilder
    recursion at possibly different starting positions.
    """
    result = pd.Series(np.nan, index=series.index, dtype=float)
    valid = series.dropna()
    if len(valid) < n:
        return result

    values = valid.to_numpy(dtype=float)
    out = np.empty(len(values), dtype=float)
    out[: n - 1] = np.nan
    out[n - 1] = values[:n].mean()
    for i in range(n, len(values)):
        out[i] = ((n - 1) * out[i - 1] + values[i]) / n

    result.loc[valid.index] = out
    return result


def ema(close: pd.Series, n: int) -> pd.Series:
    """Exponential moving average (TECH-EMA-003).

    `alpha = 2/(n+1)`, initialized with the SMA of the first N (valid)
    values rather than pandas' default exponential seed. Leading NaNs in
    `close` (e.g. an EMA-of-an-EMA such as the MACD signal line) are
    skipped when locating the seed window.
    """
    result = pd.Series(np.nan, index=close.index, dtype=float)
    valid = close.dropna()
    if len(valid) < n:
        return result

    alpha = 2.0 / (n + 1)
    values = valid.to_numpy(dtype=float)
    out = np.empty(len(values), dtype=float)
    out[: n - 1] = np.nan
    out[n - 1] = values[:n].mean()
    for i in range(n, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]

    result.loc[valid.index] = out
    return result


def true_range(df: pd.DataFrame) -> pd.Series:
    """True range (TECH-TR-005): max(H-L, |H-PriorClose|, |L-PriorClose|).

    The first bar has no prior close; `DataFrame.max(skipna=True)` (the
    pandas default) naturally falls back to `High-Low` for that row since
    the other two candidates are NaN.
    """
    high, low, close = df["high"], df["low"], df["close"]
    prior_close = close.shift(1)
    candidates = pd.concat(
        [high - low, (high - prior_close).abs(), (low - prior_close).abs()],
        axis=1,
    )
    return candidates.max(axis=1)


def atr14(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder ATR14 (TECH-ATR-006): `ATR_t = ((n-1)*ATR_t-1 + TR_t) / n`,
    seeded with the mean of the first N true-range values."""
    return _wilder_smooth(true_range(df), n)


def rsi14(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI14 (TECH-RSI-007): `100 - 100/(1 + AvgGain/AvgLoss)`.

    Wilder-smoothed average gain/loss (not a simple average). A zero
    average loss (including the flat-price case where both average gain
    and average loss are zero) yields RSI 100, per Cerebro's explicit rule.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = _wilder_smooth(gain, n)
    avg_loss = _wilder_smooth(loss, n)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.where(avg_loss != 0, 100.0)


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> dict[str, pd.Series]:
    """MACD (TECH-MACD-008): EMA12-EMA26, Signal=EMA9(MACD), Histogram=MACD-Signal."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return {"macd": macd_line, "signal": signal_line, "hist": hist}


def adx14(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder ADX14 (TECH-DMI-009): Wilder average of
    `DX = 100*|+DI - -DI| / (+DI + -DI)`, where `+DI`/`-DI` are Wilder-
    smoothed directional movement scaled by Wilder-smoothed true range.

    Measures trend strength (0-100), not direction.
    """
    high, low = df["high"], df["low"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index
    )
    # up_move/down_move are NaN on the first bar; keep plus_dm/minus_dm NaN
    # there too so _wilder_smooth's leading-NaN handling seeds correctly.
    plus_dm.iloc[0] = np.nan
    minus_dm.iloc[0] = np.nan

    atr = atr14(df, n)
    plus_dm_smooth = _wilder_smooth(plus_dm, n)
    minus_dm_smooth = _wilder_smooth(minus_dm, n)

    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return _wilder_smooth(dx, n)


def roc(close: pd.Series, n: int) -> pd.Series:
    """Rate of change (TECH-ROC-010): `Close_t / Close_t-N - 1`."""
    return close / close.shift(n) - 1


def relative_strength(close: pd.Series, bench: pd.Series, n: int) -> pd.Series:
    """Benchmark relative return (TECH-RS-011): Stock N-day return minus
    Benchmark N-day return (a percentage-point excess return).

    See the module docstring's discrepancy note: Cerebro's difference
    definition wins over the task-11 brief's "ratio of n-day returns"
    interface note.
    """
    return roc(close, n) - roc(bench, n)


def _percentile_rank(value: float, distribution: np.ndarray) -> float:
    """Percentile rank of `value` within `distribution`, 0-100 scale.

    Ties split the credit (a value equal to k other points counts as
    "beating" half of them) — the same "mean" convention as
    `scipy.stats.percentileofscore(kind="mean")`.
    """
    if len(distribution) == 0:
        return float("nan")
    less = np.sum(distribution < value)
    equal = np.sum(distribution == value)
    return float((less + 0.5 * equal) / len(distribution) * 100)


def composite_rs_percentile(rs_by_window: dict[str, float], universe: pd.DataFrame) -> float:
    """Composite relative-strength percentile (TECH-RSC-013):

    `0.35*PctRank(RS21) + 0.25*PctRank(RS63) + 0.25*PctRank(RS126) + 0.15*PctRank(RS252)`

    `rs_by_window` carries this security's own RS value for each window
    (keys `"RS21"`, `"RS63"`, `"RS126"`, `"RS252"`); `universe` carries the
    point-in-time RS distribution for the full comparison universe in
    matching columns, one row per universe member.
    """
    composite = 0.0
    for window, weight in _RS_WEIGHTS.items():
        value = rs_by_window[window]
        distribution = universe[window].dropna().to_numpy(dtype=float)
        composite += weight * _percentile_rank(value, distribution)
    return composite


def realized_vol(close: pd.Series, n: int) -> pd.Series:
    """Annualized realized volatility (TECH-VOL-018): stdev(log returns_N)*sqrt(252)."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(n).std() * np.sqrt(252)


def volume_ratio(volume: pd.Series, n: int = 50) -> pd.Series:
    """Volume ratio (TECH-VR-014): Volume_t / median(Volume over prior N sessions)."""
    return volume / volume.rolling(n).median()


def up_down_volume_ratio(df: pd.DataFrame, n: int = 50) -> pd.Series:
    """Up/down volume ratio (TECH-UDV-015):
    sum(Volume on up closes, N) / sum(Volume on down closes, N).

    A zero denominator (no down-volume in the window) is not meaningful —
    returned as NaN rather than +inf.
    """
    close, volume = df["close"], df["volume"]
    delta = close.diff()
    up_vol = volume.where(delta > 0, 0.0)
    down_vol = volume.where(delta < 0, 0.0)
    down_sum = down_vol.rolling(n).sum()
    up_sum = up_vol.rolling(n).sum()
    return up_sum / down_sum.where(down_sum != 0, np.nan)


def obv(df: pd.DataFrame) -> pd.Series:
    """On-balance volume (TECH-OBV-016): `OBV_t = OBV_t-1 + sign(Close_t-Close_t-1)*Volume_t`.

    Use slope/divergence, not absolute level.
    """
    close, volume = df["close"], df["volume"]
    sign = np.sign(close.diff()).fillna(0.0)
    return (sign * volume).cumsum()


def cmf(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Chaikin money flow (TECH-CMF-017):
    `sum(((2C-H-L)/(H-L))*Volume, N) / sum(Volume, N)`.

    If High == Low, that bar's money-flow multiplier is 0.
    """
    high, low, close, volume = df["high"], df["low"], df["close"], df["volume"]
    rng = high - low
    mfm = ((2 * close - high - low) / rng).where(rng != 0, 0.0)
    mfv = mfm * volume
    return mfv.rolling(n).sum() / volume.rolling(n).sum()


def range_position_52w(df: pd.DataFrame, n: int = 252) -> pd.Series:
    """52-week range position (TECH-52W-036):
    `(Close - Nd low) / (Nd high - Nd low)`.

    Not meaningful (NaN) if the N-session range is zero.
    """
    high, low, close = df["high"], df["low"], df["close"]
    low_n = low.rolling(n).min()
    high_n = high.rolling(n).max()
    rng = high_n - low_n
    pos = (close - low_n) / rng
    return pos.where(rng != 0, np.nan)


def median_dollar_volume(df: pd.DataFrame, n: int = 63) -> pd.Series:
    """Median dollar volume (TECH-LIQ-040): median(Close*Volume over N sessions)."""
    dollar_volume = df["close"] * df["volume"]
    return dollar_volume.rolling(n).median()
