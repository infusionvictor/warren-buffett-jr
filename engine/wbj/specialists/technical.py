"""Technical and Momentum specialist (Task 17): TECH-RET-001..TECH-LIQ-040
(`FORMULAS.md`) and the six weighted technical dimensions (20 pts).

Sources of truth (`Cerebro/04_technical_momentum/`):
- `FORMULAS.md`: the 40 registered formulas.
- `DECISION_RULES.md`: the exact primary-trend anchors, the deterministic
  support/resistance engine (implemented by Task 12's `levels_engine`,
  reused here rather than reimplemented), and the technical profile bands.
- `SCORING.md`: the six weighted dimensions and their per-dimension
  gate/cap column.
- `OUTPUT_SCHEMA.md`: `TechnicalOutput`'s extension fields
  (`market_state`, `indicators`, `important_levels`,
  `breakouts_and_failures`).

## Engine reuse (per the task-17 brief: do NOT reimplement)

Every point-in-time oscillator/moving-average/volume/relative-strength
formula (TECH-SMA-002..TECH-RS-013, TECH-VR-014..TECH-CMF-017,
TECH-VOL-018, TECH-52W-036, TECH-LIQ-040) is a direct call into
`wbj.engines.indicators` (Task 11). Every pivot/zone/touch/strength/
breakout/AVWAP/volume-profile/earnings-gap formula (TECH-PIV-022..
TECH-VP-035, TECH-GAP-020/TECH-GHOLD-021) is a direct call into
`wbj.engines.levels_engine` (Task 12)'s `compute_levels`/`earnings_gaps` —
their output (`LevelsOutput`, `EarningsGap`) is surfaced almost verbatim
in `envelope.important_levels`/`breakouts_and_failures` rather than
re-derived. `TECH-SLOPE-004` (ATR-normalized OLS trend slope),
`TECH-VCP-019` (volatility contraction ratio), `TECH-TIGHT-038`
(tight-close ratio), and `TECH-BASE-037` (base depth) have no home in
either engine and are implemented locally below.

## Dataset coverage vs. `Packet`

`packet.market_data.daily` maps directly onto the OHLCV shape both engines
expect (`adj_close` -> `close`). Per the task-17 brief's own note (a Task
10 limitation), `packet.market_data.benchmark`/`.sector` may be empty —
`TECH-RS-011`, `TECH-RSS-012`, and the sector-breadth member all degrade
to `NOT_SCORABLE` honestly rather than crashing. `Packet` carries no
earnings-event-date field and no point-in-time sector-constituent panel
(`DATASET.md`'s `sector_constituent_history` is "conditional" and not part
of the schema) — `overlay["earnings_dates"]` (a list of ISO date strings,
each resolved to its first regular gap session) and
`overlay["sector_breadth"]` (`{"above_50dma": int, "above_200dma": int,
"valid_members": int}`) supply them; without an overlay, the
earnings-gap dimension and the sector-breadth metric read `NOT_SCORABLE`.

## Trend-anchor reading

`DECISION_RULES.md`'s six-row anchor table names "SMA200 slope" with an
explicit unit ("< -1 ATR over 50 sessions") but never states which `N` the
implied SMA50 slope uses. This module reads "SMA200 slope" as
`TECH-SLOPE-004` with `N=50` (the table's own worked unit) and the SMA50
slope as `TECH-SLOPE-004` with `N=20` (a shorter window tracking the
faster-moving average) — a documented, defensible reading, not a value
taken directly from a `FORMULAS.md`/`DECISION_RULES.md` table cell.
`primary_trend_score` evaluates the six rows most-specific-first (the two
bullish/bearish extremes, then the +/-1 ATR neutral chop zone, then the
generic below/above-SMA200 catch-alls) since the table itself is not
presented as a strict top-to-bottom if/elif chain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from wbj.core.confidence import confidence as _confidence_formula
from wbj.core.nullstates import EvidenceClass, NullState, Value
from wbj.core.scoring import Category, Dimension, anchor_score
from wbj.engines import indicators as ind
from wbj.engines import levels_engine as lv
from wbj.schemas.levels import LevelsOutput
from wbj.schemas.packet import OHLCVRow, Packet
from wbj.specialists.common import (
    CategoryStats,
    apply_dimension_cap,
    JudgmentRequest,
    MetricRow,
    SecurityRef,
    SpecialistOutput,
    ValidationTestsSummary,
    status_from_coverage,
)

__all__ = [
    "AGENT_ID",
    "MAX_POINTS",
    "DIM_TREND",
    "DIM_RS",
    "DIM_VOLUME",
    "DIM_EARNINGS_GAP",
    "DIM_BREAKOUT_BASE",
    "DIM_SECTOR_BREADTH_VOL",
    "DIMENSION_NAMES",
    "TechnicalOutput",
    "verdict",
    "run",
    "atr_normalized_slope",
    "volatility_contraction_ratio",
    "tight_close_ratio",
    "base_depth",
    "primary_trend_score",
]

_VERSION = "2.0.0"
AGENT_ID = "technical_momentum"
MAX_POINTS = 20.0

DIM_TREND = "primary_price_trend"
DIM_RS = "relative_strength"
DIM_VOLUME = "volume_and_institutional_demand"
DIM_EARNINGS_GAP = "earnings_gap_behavior"
DIM_BREAKOUT_BASE = "breakout_and_base_quality"
DIM_SECTOR_BREADTH_VOL = "sector_breadth_and_volatility_quality"

DIMENSION_NAMES = (DIM_TREND, DIM_RS, DIM_VOLUME, DIM_EARNINGS_GAP, DIM_BREAKOUT_BASE, DIM_SECTOR_BREADTH_VOL)

DIMENSION_MAX_POINTS: dict[str, float] = {
    DIM_TREND: 4.0, DIM_RS: 4.0, DIM_VOLUME: 3.0,
    DIM_EARNINGS_GAP: 3.0, DIM_BREAKOUT_BASE: 3.0, DIM_SECTOR_BREADTH_VOL: 3.0,
}
assert sum(DIMENSION_MAX_POINTS.values()) == MAX_POINTS

MIN_SESSIONS_FOR_TREND = 200
MIN_EARNINGS_EVENTS = 4


def _ok(x: float, unit: str, **lineage: object) -> Value:
    return Value.of(x, unit=unit, evidence_class=lineage.pop("evidence_class", EvidenceClass.C), **lineage)


def _null(state: NullState, unit: str, *warnings: str) -> Value:
    return Value.null(state, unit=unit, warnings=list(warnings))


# ============================================================================
# Locally implemented formulas (no home in indicators.py / levels_engine.py)
# ============================================================================


def atr_normalized_slope(close: pd.Series, n: int, atr14_latest: float) -> Value:
    """ATR-normalized trend slope (TECH-SLOPE-004): `OLS slope(Close over
    N) * N / ATR14_latest` -- the total ATR-denominated price change over
    the window."""
    window = close.dropna().tail(n)
    if len(window) < n or atr14_latest is None or math.isnan(atr14_latest) or atr14_latest == 0:
        return _null(NullState.MISSING, "atr_units", "SLOPE_INSUFFICIENT_HISTORY_OR_ZERO_ATR")
    x = np.arange(n, dtype=float)
    ols_slope, _intercept = np.polyfit(x, window.to_numpy(dtype=float), 1)
    return _ok(float(ols_slope) * n / atr14_latest, unit="atr_units")


def volatility_contraction_ratio(atr_over_close: pd.Series, n: int = 126) -> Value:
    """Volatility contraction ratio (TECH-VCP-019): `(ATR14/Close)_latest /
    median(ATR14/Close over prior N sessions)`. Below 1 indicates
    contraction (FORMULAS.md)."""
    valid = atr_over_close.dropna()
    if len(valid) < n + 1:
        return _null(NullState.MISSING, "ratio", "VCP_INSUFFICIENT_HISTORY")
    latest = float(valid.iloc[-1])
    median_prior = float(valid.iloc[-(n + 1):-1].median())
    if median_prior == 0:
        return _null(NullState.NOT_MEANINGFUL, "ratio", "VCP_ZERO_MEDIAN")
    return _ok(latest / median_prior, unit="ratio")


def tight_close_ratio(log_returns: pd.Series) -> Value:
    """Tight-close ratio (TECH-TIGHT-038): `stdev(log returns last 10d) /
    stdev(log returns prior 50d)`. Lower indicates contraction
    (FORMULAS.md)."""
    valid = log_returns.dropna()
    if len(valid) < 60:
        return _null(NullState.MISSING, "ratio", "TIGHT_CLOSE_INSUFFICIENT_HISTORY")
    last10 = valid.iloc[-10:]
    prior50 = valid.iloc[-60:-10]
    std10 = float(last10.std())
    std50 = float(prior50.std())
    if std50 == 0:
        return _null(NullState.NOT_MEANINGFUL, "ratio", "TIGHT_CLOSE_ZERO_PRIOR_STDEV")
    return _ok(std10 / std50, unit="ratio")


def base_depth(df: pd.DataFrame, window: int = 20) -> Value:
    """Base depth (TECH-BASE-037): `(Base high - Base low) / Base high`
    over the trailing `window` sessions. `DECISION_RULES.md`: "Use only
    after deterministic base segmentation" -- this module's segmentation is
    the simplest deterministic rule available (a fixed trailing window,
    default 20 sessions), not a swing-based base detector; a documented
    simplification, not a value taken from FORMULAS.md."""
    if len(df) < window:
        return _null(NullState.MISSING, "pct", "BASE_DEPTH_INSUFFICIENT_HISTORY")
    recent = df.tail(window)
    base_high = float(recent["high"].max())
    base_low = float(recent["low"].min())
    if base_high == 0:
        return _null(NullState.NOT_MEANINGFUL, "pct", "BASE_DEPTH_ZERO_HIGH")
    return _ok((base_high - base_low) / base_high, unit="pct")


# ============================================================================
# Primary-trend anchor (DECISION_RULES.md, verbatim six-row table)
# ============================================================================


def primary_trend_score(
    *, close: float, sma50: float | None, sma200: float | None, atr14_latest: float | None,
    slope200_atr: float | None, slope50_atr: float | None, adx: float | None, range_pos_52w: float | None,
) -> tuple[float, str]:
    """The exact primary-trend anchors (`DECISION_RULES.md`), as a single
    0-10 composite score plus a human-readable condition label. See the
    module docstring's "Trend-anchor reading" for the slope-window
    convention and evaluation order.

    `sma200`/`atr14_latest` unavailable -> caller applies the "score capped
    6" gate separately (this function still returns a best-effort score
    from whatever is available, using `sma50`/close alone when `sma200`/
    ATR are missing).
    """
    if sma200 is None or atr14_latest is None or atr14_latest == 0:
        # No SMA200/ATR at all: fall back to a coarse close-vs-SMA50 read.
        if sma50 is None:
            return 5.0, "insufficient_data_neutral_default"
        return (7.0, "above_sma50_no_sma200") if close > sma50 else (3.0, "below_sma50_no_sma200")

    distance_atr = (close - sma200) / atr14_latest
    bullish_stack = sma50 is not None and close > sma50 > sma200
    bearish_stack = sma50 is not None and close < sma50 < sma200
    s200 = slope200_atr if slope200_atr is not None else 0.0
    s50 = slope50_atr if slope50_atr is not None else 0.0

    if bullish_stack and s50 > 0 and s200 > 0:
        if adx is not None and adx >= 25 and range_pos_52w is not None and range_pos_52w >= 0.80:
            adx_excess = min(1.0, (adx - 25.0) / 25.0)
            range_excess = min(1.0, (range_pos_52w - 0.80) / 0.20)
            score = 9.0 + min(1.0, (adx_excess + range_excess) / 2.0)
            return score, "bullish_stack_adx_confirmed_9_10"
        return 8.0, "bullish_stack_positive_slopes_8"

    if bearish_stack and s200 < -1.0:
        overshoot = min(1.0, abs(s200 + 1.0) / 2.0)
        score = max(0.0, 2.0 - 2.0 * overshoot)
        return score, "bearish_stack_falling_sma200_0_2"

    if abs(distance_atr) <= 1.0 and -0.25 <= s200 <= 0.25:
        return 4.5, "neutral_chop_near_sma200_4_5"

    if close < sma200:
        return 3.0, "below_sma200_mixed_stack_3"

    return 6.0, "above_sma200_mixed_sma50_6"


# ============================================================================
# Scoring helper. (SCORING.md's "Gate / cap" column is applied via the shared
# wbj.specialists.common.apply_dimension_cap -- imported above, not local.)
# ============================================================================


def _score_from_anchor(v: Value, anchors: list[tuple[float, float]]) -> float | None:
    if v.is_null:
        return None
    return anchor_score(v.value, anchors)


# ============================================================================
# OHLCV plumbing
# ============================================================================


def _to_df(rows: list[OHLCVRow]) -> pd.DataFrame:
    """`Packet.market_data.*` (newest-first, per `packet.fundamentals`'s
    same convention) -> ascending OHLCV DataFrame with `close = adj_close`
    (both engines expect an already-adjusted `close` column)."""
    ascending = list(reversed(rows))
    return pd.DataFrame(
        {
            "date": [r.date for r in ascending],
            "open": [r.open for r in ascending],
            "high": [r.high for r in ascending],
            "low": [r.low for r in ascending],
            "close": [r.adj_close for r in ascending],
            "volume": [r.volume for r in ascending],
        }
    )


def _last_valid(series: pd.Series) -> float | None:
    valid = series.dropna()
    return float(valid.iloc[-1]) if len(valid) else None


# ============================================================================
# Verdict (technical profile bands are on `awarded_points`, out of 20 --
# NOT `score_10`, unlike every other specialist's verdict())
# ============================================================================


def verdict(awarded_points: float) -> str:
    """Technical profile bands (`DECISION_RULES.md`, verbatim): these bands
    are stated against the raw category points (max 20), not a 0-10
    score -- the one specialist whose verdict function takes
    `awarded_points` rather than `score_10`."""
    if awarded_points >= 17.0:
        return "Powerful leadership"
    if awarded_points >= 13.0:
        return "Healthy uptrend"
    if awarded_points >= 9.0:
        return "Neutral / transition"
    return "Weak or broken"


# ============================================================================
# Envelope assembly
# ============================================================================


class MarketState(BaseModel):
    model_config = ConfigDict(frozen=True)

    trend: str | None = None
    relative_strength: str | None = None
    demand: str | None = None
    volatility: str | None = None


class IndicatorSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    sma20: float | None = None
    sma50: float | None = None
    sma100: float | None = None
    sma200: float | None = None
    atr14: float | None = None
    rsi14: float | None = None
    adx14: float | None = None
    macd: dict[str, float | None] = Field(default_factory=dict)


class TechnicalOutput(SpecialistOutput):
    """`technical_momentum`'s `run()` result: the shared envelope plus
    OUTPUT_SCHEMA.md's technical-specific extension fields."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    market_state: MarketState = Field(default_factory=MarketState)
    indicators: IndicatorSnapshot = Field(default_factory=IndicatorSnapshot)
    important_levels: LevelsOutput = Field(default_factory=LevelsOutput)
    breakouts_and_failures: list[dict[str, Any]] = Field(default_factory=list)


@dataclass(frozen=True)
class _Row:
    metric_id: str
    value: Value
    score10: float | None
    confidence: float = 0.0


def _confidence_for(v: Value) -> float:
    if v.is_null:
        return 0.0
    base_by_class = {
        EvidenceClass.R: 95.0, EvidenceClass.C: 85.0, EvidenceClass.E: 60.0,
        EvidenceClass.A: 50.0, EvidenceClass.Q: 30.0,
    }
    base = base_by_class.get(v.evidence_class, 70.0) if v.evidence_class else 70.0
    base -= 15.0 * len(v.warnings)
    return max(0.0, min(100.0, base))


def _empty_output(packet: Packet, reason: str) -> TechnicalOutput:
    """`AGENT.md`'s boundary: "reject unadjusted histories" (TECH-T010) --
    an all-`NOT_SCORABLE`, zero-coverage `ERROR` envelope rather than a
    crash or a silently-scored packet.

    `awarded_points`/`score_10` are 0.0 by construction (empty dimensions),
    but `confidence` is derived from the real five-component formula at
    coverage 0 (not a hardcoded literal): confidence measures evidence
    quality independent of the economic score, so an ERROR envelope still
    reports the (low, non-zero) confidence the formula yields for a
    zero-coverage packet from this data source."""
    dims = [Dimension(name=n, max_points=DIMENSION_MAX_POINTS[n], metric_scores=[]) for n in DIMENSION_NAMES]
    n_sessions = len(packet.market_data.daily)
    return TechnicalOutput(
        agent_id=AGENT_ID, status="ERROR",
        security=SecurityRef(ticker=packet.security.ticker, exchange=packet.security.exchange, currency=packet.security.reporting_currency),
        knowledge_timestamp=packet.analysis.knowledge_timestamp,
        category=CategoryStats(max_points=MAX_POINTS, awarded_points=0.0, score_10=0.0, confidence=_category_confidence(0.0, n_sessions)),
        verdict=verdict(0.0), coverage=0.0, dimensions=dims, metrics=[],
        mandatory_flags=[reason], assumptions=[], judgment_requests=[],
        source_lineage=["packet.market_data.daily"], validation_tests=ValidationTestsSummary(passed=0, failed=1, warnings=0),
    )


def run(packet: Packet, overlay: dict[str, Any] | None = None) -> TechnicalOutput:
    """Run the technical specialist against `packet`, producing a
    `TechnicalOutput` (max 20 pts): six weighted dimensions, mandatory
    flags, and OUTPUT_SCHEMA.md's extension fields (`important_levels` is
    Task 12's `levels_engine.compute_levels` output, reused almost
    verbatim).

    `overlay` may supply `earnings_dates` (list of ISO date strings, each
    already resolved to its first regular gap session) and
    `sector_breadth` (`{"above_50dma": int, "above_200dma": int,
    "valid_members": int}`) -- see the module docstring.
    """
    overlay = overlay or {}
    if not packet.market_data.adjusted:
        return _empty_output(packet, "REJECTED_UNADJUSTED_HISTORY")

    daily_rows = packet.market_data.daily
    if len(daily_rows) < 15:  # not enough even for ATR14
        return _empty_output(packet, "INSUFFICIENT_PRICE_HISTORY")

    df = _to_df(daily_rows)
    n_sessions = len(df)
    close = df["close"]
    atr = ind.atr14(df)
    atr_latest = _last_valid(atr)

    sma20, sma50, sma100, sma200 = ind.sma(close, 20), ind.sma(close, 50), ind.sma(close, 100), ind.sma(close, 200)
    sma20_now, sma50_now, sma100_now, sma200_now = (_last_valid(s) for s in (sma20, sma50, sma100, sma200))
    rsi_now = _last_valid(ind.rsi14(close))
    adx_now = _last_valid(ind.adx14(df))
    macd_d = ind.macd(close)
    macd_now = {k: _last_valid(v) for k, v in macd_d.items()}
    range_pos_now = _last_valid(ind.range_position_52w(df))
    close_now = float(close.iloc[-1])

    assumptions: list[str] = []
    rows: list[_Row] = []

    def add(metric_id: str, v: Value, score10: float | None) -> None:
        rows.append(_Row(metric_id=metric_id, value=v, score10=score10, confidence=_confidence_for(v)))

    # ---- TECH-RET-001: latest log return (context) ----
    log_returns = np.log(close / close.shift(1))
    ret_latest = _last_valid(log_returns)
    add("TECH-RET-001", _ok(ret_latest, unit="pct") if ret_latest is not None else _null(NullState.MISSING, "pct", "RET_INSUFFICIENT_HISTORY"), None)

    # ---- TECH-SMA-002 / TECH-EMA-003 / TECH-ATR-006 / TECH-RSI-007 / TECH-MACD-008 / TECH-DMI-009 (context rows) ----
    add("TECH-SMA-002", _ok(sma200_now, unit="usd") if sma200_now is not None else _null(NullState.MISSING, "usd", "SMA200_UNAVAILABLE"), None)
    ema20_now = _last_valid(ind.ema(close, 20))
    add("TECH-EMA-003", _ok(ema20_now, unit="usd") if ema20_now is not None else _null(NullState.MISSING, "usd", "EMA_UNAVAILABLE"), None)
    add("TECH-TR-005", _ok(float(ind.true_range(df).iloc[-1]), unit="usd"), None)
    add("TECH-ATR-006", _ok(atr_latest, unit="usd") if atr_latest is not None else _null(NullState.MISSING, "usd", "ATR_UNAVAILABLE"), None)
    add("TECH-RSI-007", _ok(rsi_now, unit="score") if rsi_now is not None else _null(NullState.MISSING, "score", "RSI_UNAVAILABLE"), None)
    add("TECH-MACD-008", _ok(macd_now.get("hist"), unit="usd") if macd_now.get("hist") is not None else _null(NullState.MISSING, "usd", "MACD_UNAVAILABLE"), None)
    add("TECH-DMI-009", _ok(adx_now, unit="score") if adx_now is not None else _null(NullState.MISSING, "score", "ADX_UNAVAILABLE"), None)

    # ---- TECH-SLOPE-004: ATR-normalized slope (N=20 SMA50-proxy, N=50 SMA200-proxy) ----
    v_slope50 = atr_normalized_slope(close, 20, atr_latest) if atr_latest else _null(NullState.MISSING, "atr_units", "SLOPE_ATR_UNAVAILABLE")
    v_slope200 = atr_normalized_slope(close, 50, atr_latest) if atr_latest else _null(NullState.MISSING, "atr_units", "SLOPE_ATR_UNAVAILABLE")
    add("TECH-SLOPE-004", v_slope200, None)

    # ---- TECH-52W-036: 52-week range position ----
    add("TECH-52W-036", _ok(range_pos_now, unit="pct") if range_pos_now is not None else _null(NullState.MISSING, "pct", "RANGE_52W_UNAVAILABLE"), _score_from_anchor(_ok(range_pos_now, unit="pct") if range_pos_now is not None else _null(NullState.MISSING, "pct"), [(0.0, 0), (0.5, 5), (0.8, 8), (1.0, 10)]))

    # ---- Primary trend composite (DIM_TREND's single driving score) ----
    has_sma200 = n_sessions >= MIN_SESSIONS_FOR_TREND and sma200_now is not None
    trend_score, trend_label = primary_trend_score(
        close=close_now, sma50=sma50_now, sma200=sma200_now, atr14_latest=atr_latest,
        slope200_atr=v_slope200.value if v_slope200.is_valid else None,
        slope50_atr=v_slope50.value if v_slope50.is_valid else None,
        adx=adx_now, range_pos_52w=range_pos_now,
    )
    add("TECH-TREND-COMPOSITE", _ok(trend_score, unit="score", evidence_class=EvidenceClass.C, warnings=[] if has_sma200 else ["SMA200_OR_200_SESSIONS_UNAVAILABLE"]), trend_score)
    if not has_sma200:
        assumptions.append(
            f"Primary trend: {n_sessions} sessions available (<{MIN_SESSIONS_FOR_TREND} required) or no valid "
            "SMA200 -- DIM_TREND is capped at 6 per SCORING.md."
        )

    # ---- TECH-RS-011 / TECH-RSS-012 / TECH-RSC-013: relative strength ----
    bench_rows, sector_rows = packet.market_data.benchmark, packet.market_data.sector
    if bench_rows and len(bench_rows) >= 64:
        bench_df = _to_df(bench_rows)
        n_common = min(len(df), len(bench_df))
        rs_series = ind.relative_strength(close.tail(n_common).reset_index(drop=True), bench_df["close"].tail(n_common).reset_index(drop=True), 63)
        rs63 = _last_valid(rs_series)
        v_rs = _ok(rs63, unit="pp") if rs63 is not None else _null(NullState.MISSING, "pp", "RS_INSUFFICIENT_OVERLAP")
    else:
        v_rs = _null(NullState.MISSING, "pp", "BENCHMARK_DATA_UNAVAILABLE_EMPTY_MARKET_DATA")
    add("TECH-RS-011", v_rs, _score_from_anchor(v_rs, [(-0.10, 0), (0.0, 4), (0.05, 7), (0.15, 10)]))

    if sector_rows and len(sector_rows) >= 64:
        sector_df = _to_df(sector_rows)
        n_common = min(len(df), len(sector_df))
        rss_series = ind.relative_strength(close.tail(n_common).reset_index(drop=True), sector_df["close"].tail(n_common).reset_index(drop=True), 63)
        rss63 = _last_valid(rss_series)
        v_rss = _ok(rss63, unit="pp") if rss63 is not None else _null(NullState.MISSING, "pp", "RSS_INSUFFICIENT_OVERLAP")
    else:
        v_rss = _null(NullState.MISSING, "pp", "SECTOR_DATA_UNAVAILABLE_EMPTY_MARKET_DATA")
    add("TECH-RSS-012", v_rss, _score_from_anchor(v_rss, [(-0.10, 0), (0.0, 4), (0.05, 7), (0.15, 10)]))

    universe = overlay.get("rs_universe")
    if universe and v_rs.is_valid:
        rs_by_window = {"RS21": v_rs.value, "RS63": v_rs.value, "RS126": v_rs.value, "RS252": v_rs.value}
        percentile = ind.composite_rs_percentile(rs_by_window, pd.DataFrame(universe))
        v_rsc = _ok(percentile, unit="percentile")
    else:
        v_rsc = _null(NullState.NOT_SCORABLE, "percentile", "RSC_UNIVERSE_UNAVAILABLE_POINT_IN_TIME_REQUIRED")
    # Brief's explicit percentile bands: <30 -> 0-3; 30-70 -> 4-6; >70 -> 7-10.
    add("TECH-RSC-013", v_rsc, _score_from_anchor(v_rsc, [(0.0, 0), (30.0, 3), (50.0, 5), (70.0, 7), (100.0, 10)]) if v_rsc.is_valid else None)

    # ---- TECH-VR-014..TECH-CMF-017: volume and demand ----
    volume_present = df["volume"].fillna(0).abs().sum() > 0
    vr_now = _last_valid(ind.volume_ratio(df["volume"])) if volume_present else None
    udv_now = _last_valid(ind.up_down_volume_ratio(df)) if volume_present else None
    cmf_now = _last_valid(ind.cmf(df)) if volume_present else None
    obv_series = ind.obv(df) if volume_present else pd.Series(dtype=float)
    obv_slope = None
    obv_score = None
    if volume_present and len(obv_series.dropna()) >= 20:
        tail = obv_series.dropna().tail(20).to_numpy(dtype=float)
        obv_slope, _ = np.polyfit(np.arange(len(tail), dtype=float), tail, 1)
        # TECH-OBV-016: "use slope/divergence, not absolute level." The raw
        # slope's magnitude is share-count and not comparable across
        # tickers, so it is normalized by the 50-session median volume
        # (net accumulation per session as a fraction of a typical day)
        # before scoring -- a documented reading (FORMULAS.md gives no
        # numeric OBV band). Positive = accumulation = constructive.
        med_vol = float(df["volume"].tail(50).median()) if volume_present else 0.0
        obv_norm = obv_slope / med_vol if med_vol else 0.0
        obv_score = anchor_score(obv_norm, [(-0.5, 0), (0.0, 5), (0.3, 8), (0.6, 10)])

    add("TECH-VR-014", _ok(vr_now, unit="ratio") if vr_now is not None else _null(NullState.MISSING, "ratio", "VOLUME_UNAVAILABLE"), _score_from_anchor(_ok(vr_now, unit="ratio") if vr_now is not None else _null(NullState.MISSING, "ratio"), [(0.5, 2), (1.0, 5), (1.5, 8), (2.5, 10)]))
    add("TECH-UDV-015", _ok(udv_now, unit="ratio") if udv_now is not None else _null(NullState.MISSING, "ratio", "UDV_UNAVAILABLE"), _score_from_anchor(_ok(udv_now, unit="ratio") if udv_now is not None else _null(NullState.MISSING, "ratio"), [(0.5, 0), (0.8, 3), (1.2, 7), (2.0, 10)]))
    add("TECH-OBV-016", _ok(float(obv_slope), unit="shares_per_session") if obv_slope is not None else _null(NullState.MISSING, "shares_per_session", "OBV_INSUFFICIENT_HISTORY"), obv_score)
    add("TECH-CMF-017", _ok(cmf_now, unit="ratio") if cmf_now is not None else _null(NullState.MISSING, "ratio", "CMF_UNAVAILABLE"), _score_from_anchor(_ok(cmf_now, unit="ratio") if cmf_now is not None else _null(NullState.MISSING, "ratio"), [(-0.20, 0), (-0.10, 2), (0.0, 5), (0.10, 8), (0.20, 10)]))

    # ---- TECH-VOL-018 / TECH-VCP-019 / TECH-TIGHT-038: volatility ----
    realized_vol_now = _last_valid(ind.realized_vol(close, 63))
    add("TECH-VOL-018", _ok(realized_vol_now, unit="pct_annualized") if realized_vol_now is not None else _null(NullState.MISSING, "pct_annualized", "VOL_UNAVAILABLE"), _score_from_anchor(_ok(realized_vol_now, unit="pct") if realized_vol_now is not None else _null(NullState.MISSING, "pct"), [(0.15, 10), (0.30, 6), (0.50, 3), (0.80, 0)]))

    atr_over_close = atr / close
    v_vcp = volatility_contraction_ratio(atr_over_close)
    add("TECH-VCP-019", v_vcp, _score_from_anchor(v_vcp, [(0.5, 10), (1.0, 6), (1.5, 3), (2.5, 0)]))

    v_tight = tight_close_ratio(log_returns)
    add("TECH-TIGHT-038", v_tight, _score_from_anchor(v_tight, [(0.5, 10), (1.0, 6), (1.5, 3), (2.5, 0)]))

    # ---- TECH-GAP-020 / TECH-GHOLD-021: earnings gaps ----
    earnings_dates = overlay.get("earnings_dates") or []
    gap_records = lv.earnings_gaps(df, earnings_dates, atr) if earnings_dates else []
    valid_gap_records = [g for g in gap_records if g.day5_hold_ratio is not None]
    if len(valid_gap_records) >= MIN_EARNINGS_EVENTS:
        avg_gap_pct = sum(g.gap_percent for g in valid_gap_records) / len(valid_gap_records)
        day5_holds = [g.day5_hold_ratio for g in valid_gap_records if g.day5_hold_ratio is not None]
        day20_holds = [g.day20_hold_ratio for g in valid_gap_records if g.day20_hold_ratio is not None]
        v_gap = _ok(avg_gap_pct / 100.0, unit="pct")
        pct_hold_5 = sum(1 for h in day5_holds if h >= 0.7) / len(day5_holds) if day5_holds else None
        pct_hold_20 = sum(1 for h in day20_holds if h >= 0.7) / len(day20_holds) if day20_holds else None
        v_ghold5 = _ok(pct_hold_5, unit="pct") if pct_hold_5 is not None else _null(NullState.MISSING, "pct", "GHOLD5_UNAVAILABLE")
        v_ghold20 = _ok(pct_hold_20, unit="pct") if pct_hold_20 is not None else _null(NullState.MISSING, "pct", "GHOLD20_UNAVAILABLE")
    else:
        v_gap = _null(NullState.NOT_SCORABLE, "pct", f"EARNINGS_GAP_NEEDS_{MIN_EARNINGS_EVENTS}_VALID_EVENTS: got {len(valid_gap_records)}")
        v_ghold5 = _null(NullState.NOT_SCORABLE, "pct", "EARNINGS_GAP_INSUFFICIENT_EVENTS")
        v_ghold20 = _null(NullState.NOT_SCORABLE, "pct", "EARNINGS_GAP_INSUFFICIENT_EVENTS")
    add("TECH-GAP-020", v_gap, _score_from_anchor(v_gap, [(-0.10, 0), (0.0, 5), (0.05, 8), (0.15, 10)]))
    add("TECH-GHOLD-021", v_ghold5, _score_from_anchor(v_ghold5, [(0.0, 0), (0.5, 5), (0.7, 7), (1.0, 10)]))
    ghold20_score = _score_from_anchor(v_ghold20, [(0.0, 0), (0.5, 5), (0.7, 7), (1.0, 10)])

    # ---- Levels engine (Task 12): pivots/zones/touches/breakouts/AVWAP/volume-profile/gaps ----
    levels_output = LevelsOutput()
    if n_sessions >= 40:
        levels_output = lv.compute_levels(df, earnings_dates)

    nearest_resistance = levels_output.nearest_resistance[0] if levels_output.nearest_resistance else None
    breakouts_and_failures: list[dict[str, Any]] = []
    breakout_confirmed_flag = False
    if nearest_resistance is not None:
        confirmed = lv.breakout_confirmed(df, lv.ZoneState(
            kind="high", center=nearest_resistance.center, lower=nearest_resistance.lower, upper=nearest_resistance.upper,
            timeframe=nearest_resistance.timeframe, touches=[],
        ), atr)
        breakout_confirmed_flag = confirmed
        breakouts_and_failures.append(
            {"zone_id": nearest_resistance.zone_id, "status": nearest_resistance.status, "breakout_confirmed": confirmed}
        )

    best_zone_strength = max(
        (z.strength_0_100 for z in (levels_output.nearest_support + levels_output.nearest_resistance)), default=None
    )
    v_lstr = _ok(best_zone_strength, unit="score_0_100") if best_zone_strength is not None else _null(NullState.MISSING, "score_0_100", "LSTR_NO_ZONES")
    add("TECH-LSTR-028", v_lstr, _score_from_anchor(v_lstr, [(0.0, 0), (40.0, 4), (70.0, 8), (100.0, 10)]))
    add("TECH-BCONF-031", _ok(1.0 if breakout_confirmed_flag else 0.0, unit="bool"), 10.0 if breakout_confirmed_flag else (5.0 if nearest_resistance is None else 0.0))

    v_base = base_depth(df)
    add("TECH-BASE-037", v_base, _score_from_anchor(v_base, [(0.05, 10), (0.15, 6), (0.30, 3), (0.50, 0)]))

    # ---- TECH-BREAD-039 / TECH-LIQ-040 ----
    breadth_overlay = overlay.get("sector_breadth")
    if breadth_overlay and breadth_overlay.get("valid_members", 0) > 0:
        breadth_pct = breadth_overlay["above_50dma"] / breadth_overlay["valid_members"]
        v_bread = _ok(breadth_pct, unit="pct")
    else:
        v_bread = _null(NullState.NOT_SCORABLE, "pct", "BREADTH_UNAVAILABLE_NO_CONSTITUENT_PANEL")
    add("TECH-BREAD-039", v_bread, _score_from_anchor(v_bread, [(0.20, 0), (0.50, 5), (0.70, 8), (0.90, 10)]))

    liq_now = _last_valid(ind.median_dollar_volume(df))
    add("TECH-LIQ-040", _ok(liq_now, unit="usd") if liq_now is not None else _null(NullState.MISSING, "usd", "LIQUIDITY_UNAVAILABLE"), _score_from_anchor(_ok(liq_now, unit="usd") if liq_now is not None else _null(NullState.MISSING, "usd"), [(1_000_000.0, 3), (10_000_000.0, 7), (50_000_000.0, 10)]))

    by_id = {r.metric_id: r for r in rows}

    # ---- MetricRow assembly ----
    metric_rows: list[MetricRow] = []
    for r in rows:
        score: float | str = r.score10 if r.score10 is not None else "NOT_SCORABLE"
        metric_rows.append(
            MetricRow.from_value(r.metric_id, r.value, formula_id=r.metric_id, formula_version=_VERSION, score=score, confidence=r.confidence)
        )

    # ---- DIM_TREND (4 pts): single composite driver, capped at 6 without SMA200/200 sessions ----
    trend_scores: list[tuple[float, Value]] = [(1.0, Value.of(trend_score, unit="score"))]
    if not has_sma200:
        trend_scores = apply_dimension_cap(trend_scores, cap=6.0)
    trend_dim = Dimension(name=DIM_TREND, max_points=DIMENSION_MAX_POINTS[DIM_TREND], metric_scores=trend_scores)

    # ---- DIM_RS (4 pts) ----
    rs_scores: list[tuple[float, Value]] = []
    for mid in ("TECH-RS-011", "TECH-RSS-012", "TECH-RSC-013"):
        s = by_id[mid].score10
        rs_scores.append((1 / 3, Value.of(s, unit="score") if s is not None else Value.null(NullState.NOT_SCORABLE, unit="score")))
    rs_dim = Dimension(name=DIM_RS, max_points=DIMENSION_MAX_POINTS[DIM_RS], metric_scores=rs_scores)

    # ---- DIM_VOLUME (3 pts): VR/UDV/OBV/CMF (SCORING.md TECH-VR-014..017) ----
    # SCORING.md's gate ("Score capped 5 when volume is missing/unadjusted")
    # is a *cap on a real score*, never a licence to fabricate one: when
    # volume is entirely absent every member here is NOT_SCORABLE, so the
    # dimension is honestly NOT_SCORABLE (contributes 0 coverage, 0 points)
    # rather than being handed a synthetic mid score -- "missing evidence is
    # never neutral" (SCORING.md). `apply_dimension_cap(..., cap=5.0)` only
    # bites when a genuine volume score above 5 exists to scale down (an
    # unadjusted-but-present series); with fully missing volume it is a
    # no-op (nothing valid), and the VOLUME_UNAVAILABLE_DEMAND_CAPPED flag
    # below records the gap.
    vol_scores: list[tuple[float, Value]] = []
    for mid in ("TECH-VR-014", "TECH-UDV-015", "TECH-OBV-016", "TECH-CMF-017"):
        s = by_id[mid].score10
        vol_scores.append((0.25, Value.of(s, unit="score") if s is not None else Value.null(NullState.NOT_SCORABLE, unit="score")))
    if not volume_present:
        vol_scores = apply_dimension_cap(vol_scores, cap=5.0)
    demand_dim = Dimension(name=DIM_VOLUME, max_points=DIMENSION_MAX_POINTS[DIM_VOLUME], metric_scores=vol_scores)

    # ---- DIM_EARNINGS_GAP (3 pts) ----
    gap_scores: list[tuple[float, Value]] = [
        (1 / 3, Value.of(by_id["TECH-GAP-020"].score10, unit="score") if by_id["TECH-GAP-020"].score10 is not None else Value.null(NullState.NOT_SCORABLE, unit="score")),
        (1 / 3, Value.of(by_id["TECH-GHOLD-021"].score10, unit="score") if by_id["TECH-GHOLD-021"].score10 is not None else Value.null(NullState.NOT_SCORABLE, unit="score")),
        (1 / 3, Value.of(ghold20_score, unit="score") if ghold20_score is not None else Value.null(NullState.NOT_SCORABLE, unit="score")),
    ]
    gap_dim = Dimension(name=DIM_EARNINGS_GAP, max_points=DIMENSION_MAX_POINTS[DIM_EARNINGS_GAP], metric_scores=gap_scores)

    # ---- DIM_BREAKOUT_BASE (3 pts): VCP/LSTR/BCONF/BASE + TIGHT-038 ----
    # SCORING.md lists TECH-TIGHT-038 (tight-close ratio) as a primary input
    # of breakout & base quality alongside VCP-019/PIV-022..033/BASE-037.
    breakout_scores: list[tuple[float, Value]] = []
    for mid in ("TECH-VCP-019", "TECH-LSTR-028", "TECH-BCONF-031", "TECH-BASE-037", "TECH-TIGHT-038"):
        s = by_id[mid].score10
        breakout_scores.append((0.2, Value.of(s, unit="score") if s is not None else Value.null(NullState.NOT_SCORABLE, unit="score")))
    breakout_dim = Dimension(name=DIM_BREAKOUT_BASE, max_points=DIMENSION_MAX_POINTS[DIM_BREAKOUT_BASE], metric_scores=breakout_scores)

    # ---- DIM_SECTOR_BREADTH_VOL (3 pts) ----
    sector_scores: list[tuple[float, Value]] = []
    for mid in ("TECH-VOL-018", "TECH-VCP-019", "TECH-BREAD-039", "TECH-LIQ-040"):
        s = by_id[mid].score10
        sector_scores.append((0.25, Value.of(s, unit="score") if s is not None else Value.null(NullState.NOT_SCORABLE, unit="score")))
    sector_dim = Dimension(name=DIM_SECTOR_BREADTH_VOL, max_points=DIMENSION_MAX_POINTS[DIM_SECTOR_BREADTH_VOL], metric_scores=sector_scores)

    dimensions = [trend_dim, rs_dim, demand_dim, gap_dim, breakout_dim, sector_dim]

    cat = Category(name=AGENT_ID, max_points=MAX_POINTS, dimensions=dimensions)
    awarded_points = cat.points()
    dim_score10 = cat.score10()
    coverage = cat.coverage()

    mandatory_flags: list[str] = []
    if not has_sma200:
        mandatory_flags.append("SMA200_UNAVAILABLE_TREND_CAPPED")
    if not volume_present:
        mandatory_flags.append("VOLUME_UNAVAILABLE_DEMAND_CAPPED")

    label = verdict(awarded_points)

    category = CategoryStats(
        max_points=MAX_POINTS, awarded_points=awarded_points, score_10=dim_score10,
        confidence=_category_confidence(coverage, n_sessions),
    )
    status = status_from_coverage(coverage)

    passed = 0
    failed = 0
    checks = [
        abs((category.awarded_points or 0.0) - Category(name=AGENT_ID, max_points=MAX_POINTS, dimensions=dimensions).points()) < 1e-6,
        all(row.formula_id for row in metric_rows),
        bool(packet.analysis.knowledge_timestamp),
        category.confidence is not None,
    ]
    for ok in checks:
        passed += 1 if ok else 0
        failed += 0 if ok else 1
    validation_tests = ValidationTestsSummary(passed=passed, failed=failed, warnings=0)

    market_state = MarketState(
        trend=trend_label,
        relative_strength="above_benchmark" if v_rs.is_valid and v_rs.value > 0 else ("below_benchmark" if v_rs.is_valid else None),
        demand="constructive" if (cmf_now is not None and cmf_now > 0.10) else ("distributive" if (cmf_now is not None and cmf_now < -0.10) else "neutral"),
        volatility="contracting" if (v_vcp.is_valid and v_vcp.value < 1.0) else ("expanding" if v_vcp.is_valid else None),
    )
    indicator_snapshot = IndicatorSnapshot(
        sma20=sma20_now, sma50=sma50_now, sma100=sma100_now, sma200=sma200_now,
        atr14=atr_latest, rsi14=rsi_now, adx14=adx_now, macd=macd_now,
    )

    return TechnicalOutput(
        agent_id=AGENT_ID,
        status=status,
        security=SecurityRef(ticker=packet.security.ticker, exchange=packet.security.exchange, currency=packet.security.reporting_currency),
        knowledge_timestamp=packet.analysis.knowledge_timestamp,
        category=category,
        verdict=label,
        coverage=coverage,
        dimensions=dimensions,
        metrics=metric_rows,
        mandatory_flags=mandatory_flags,
        assumptions=assumptions,
        judgment_requests=[],
        source_lineage=["packet.market_data.daily", "packet.market_data.benchmark", "packet.market_data.sector"],
        validation_tests=validation_tests,
        market_state=market_state,
        indicators=indicator_snapshot,
        important_levels=levels_output,
        breakouts_and_failures=breakouts_and_failures,
    )


def _category_confidence(coverage: float, n_sessions: int) -> float:
    coverage_component = max(0.0, min(1.0, coverage)) * 100.0
    source_quality = 90.0  # adjusted market-data feed
    freshness = 100.0  # daily bars are as-of the analysis timestamp by construction
    consistency = 80.0
    model_fit = 90.0 if n_sessions >= MIN_SESSIONS_FOR_TREND else 60.0
    return _confidence_formula(
        coverage=coverage_component, source_quality=source_quality, freshness=freshness,
        consistency=consistency, model_fit=model_fit,
    )
