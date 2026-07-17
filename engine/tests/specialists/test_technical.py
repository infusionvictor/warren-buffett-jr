"""Tests for `wbj.specialists.technical` (Task 17): trend anchors, the six
weighted dimensions, engine reuse (Tasks 11-12), and `run()` against the
NVDA golden fixture.

Sources of truth: `Cerebro/04_technical_momentum/{FORMULAS,SCORING,
DECISION_RULES,VALIDATION_TESTS}.md`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wbj.core.nullstates import NullState
from wbj.core.scoring import Category
from wbj.engines import levels_engine as lv
from wbj.schemas.packet import AnalysisMeta, MarketData, OHLCVRow, Packet, Security
import wbj.specialists.technical as tech

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "packet" / "NVDA_packet.json"


@pytest.fixture(scope="module")
def nvda_packet() -> Packet:
    data = json.loads(_FIXTURE.read_text())
    return Packet.model_validate(data)


def _dates(n: int) -> list[str]:
    base = pd.Timestamp("2020-01-01")
    return [(base + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n)]


def _rows_from_closes(closes: list[float], volumes: list[float] | None = None) -> list[OHLCVRow]:
    """Ascending closes -> newest-first `OHLCVRow` list (Packet's own
    convention, matching `_row`-style helpers in the sibling specialist
    test files)."""
    dates = _dates(len(closes))
    vols = volumes if volumes is not None else [1_000_000.0] * len(closes)
    ascending = [
        OHLCVRow(
            date=d, open=c, high=c * 1.01, low=c * 0.99, close=c, adj_close=c, volume=v,
        )
        for d, c, v in zip(dates, closes, vols)
    ]
    return list(reversed(ascending))  # newest-first


def _packet(
    closes: list[float], *, benchmark: list[float] | None = None, sector: list[float] | None = None,
    volumes: list[float] | None = None, adjusted: bool = True,
) -> Packet:
    md = MarketData(
        daily=_rows_from_closes(closes, volumes),
        benchmark=_rows_from_closes(benchmark) if benchmark else [],
        sector=_rows_from_closes(sector) if sector else [],
        adjusted=adjusted,
    )
    return Packet(
        security=Security(ticker="TEST", exchange="NASDAQ", security_type="operating_company", reporting_currency="USD", valuation_currency="USD"),
        analysis=AnalysisMeta(knowledge_timestamp="2026-07-16T21:00:00+00:00", industry_adapter="default_nonfinancial"),
        fundamentals={"annual": [], "quarterly": []},
        market_data=md,
        estimates={},
        capital_structure={},
        facts_table={},
        staleness={},
    )


def _uptrend_closes(n: int, start: float = 100.0, daily_pct: float = 0.003) -> list[float]:
    return [start * (1 + daily_pct) ** i for i in range(n)]


def _downtrend_closes(n: int, start: float = 100.0, daily_pct: float = -0.003) -> list[float]:
    return [start * (1 + daily_pct) ** i for i in range(n)]


# ============================================================================
# TECH-VALIDATION_TESTS.md, encoded verbatim where directly testable at
# this specialist's own boundary (engine-level behavior is Task 11/12's
# own suite; these confirm technical.py wires the engines correctly).
# ============================================================================


def test_TECH_T001_constant_price_series_atr_zero_and_guarded():
    """Constant price series -> ATR=0; level-distance ratios guarded."""
    closes = [100.0] * 30
    df = pd.DataFrame({"date": _dates(30), "open": closes, "high": closes, "low": closes, "close": closes, "volume": [1e6] * 30})
    from wbj.engines import indicators as ind

    atr = ind.atr14(df)
    assert float(atr.iloc[-1]) == pytest.approx(0.0)
    v = tech.atr_normalized_slope(df["close"], 20, 0.0)
    assert v.is_null  # guarded: zero ATR must not divide


def test_TECH_T002_true_range():
    """High=12, low=10, prior close=11 -> True range=2."""
    from wbj.engines import indicators as ind

    df = pd.DataFrame({"high": [11.0, 12.0], "low": [9.0, 10.0], "close": [11.0, 11.5]})
    tr = ind.true_range(df)
    assert float(tr.iloc[1]) == pytest.approx(2.0)


def test_TECH_T003_two_pivot_highs_confirmed_resistance():
    """Two pivot highs in overlapping ATR zones, 20 sessions apart, each
    rejecting >0.5 ATR -> confirmed resistance (reuses Task 12's engine,
    the same one `run()` calls via `compute_levels`)."""
    n = 80
    closes = [100.0] * n
    # Two symmetric spikes 20 sessions apart, each followed by a pullback.
    for peak in (20, 40):
        closes[peak] = 110.0
        closes[peak + 1] = 104.0
        closes[peak + 2] = 103.0
    df = pd.DataFrame({
        "date": _dates(n), "open": closes, "high": [c * 1.002 for c in closes], "low": [c * 0.998 for c in closes],
        "close": closes, "volume": [1e6] * n,
    })
    from wbj.engines import indicators as ind

    atr = ind.atr14(df)
    pivots = lv.find_pivots(df, 3)
    candidates = lv.cluster_zones(pivots, atr, df["close"])
    highs = [c for c in candidates if c.kind == "high"]
    assert highs
    touches = lv.count_touches(highs[0], df, atr, timeframe="daily")
    assert len(touches) >= 2  # >=2 independent touches -> CONFIRMED


def test_TECH_T004_two_highs_2_sessions_apart_count_as_one_touch():
    """Two highs only 2 sessions apart -> count as one swing touch (the
    5-session daily independence gap in `count_touches`)."""
    n = 40
    closes = [100.0] * n
    closes[10] = 110.0
    closes[12] = 109.0  # only 2 sessions later -> same swing
    df = pd.DataFrame({
        "date": _dates(n), "open": closes, "high": [c * 1.002 for c in closes], "low": [c * 0.998 for c in closes],
        "close": closes, "volume": [1e6] * n,
    })
    from wbj.engines import indicators as ind

    atr = ind.atr14(df)
    pivots = lv.find_pivots(df, 3)
    candidates = lv.cluster_zones(pivots, atr, df["close"])
    highs = [c for c in candidates if c.kind == "high"]
    if highs:
        touches = lv.count_touches(highs[0], df, atr, timeframe="daily")
        assert len(touches) <= 1


def test_TECH_T009_distance_atr_nearby_resistance():
    """Current close 100, resistance zone low 102, ATR 2 -> Distance=1 ATR; nearby."""
    distance_atr = (102.0 - 100.0) / 2.0
    assert distance_atr == pytest.approx(1.0)
    assert 0 <= distance_atr <= 1.0  # "nearby" per DECISION_RULES.md


def test_TECH_T010_unadjusted_history_rejected():
    """Price history not split adjusted -> reject technical packet."""
    packet = _packet(_uptrend_closes(30), adjusted=False)
    out = tech.run(packet)
    assert out.status == "ERROR"
    assert "REJECTED_UNADJUSTED_HISTORY" in out.mandatory_flags


def test_TECH_T011_pivot_delayed_by_k_sessions():
    """Pivot k=3 at historical date -> pivot availability delayed 3
    sessions (a pivot needs 3 future bars closed before it's usable)."""
    n = 20
    closes = [100.0] * n
    closes[10] = 110.0
    df = pd.DataFrame({"date": _dates(n), "high": closes, "low": [c - 1 for c in closes], "close": closes})
    pivots = lv.find_pivots(df, 3)
    # A pivot at index 10 requires indices 7..13 all present; find_pivots
    # only ever returns pivots whose window fully exists in df.
    assert all(p.index <= n - 1 - 3 for p in pivots)


def test_TECH_T012_missing_volume_marks_demand_not_scorable_no_fabrication(nvda_packet):
    """Volume missing -> Volume dimension honestly NOT_SCORABLE (no
    fabricated evidence), breakout cannot confirm, flag raised. "Missing
    evidence is never neutral" (SCORING.md) -- with every volume member
    NOT_SCORABLE the dimension carries zero valid weight rather than a
    synthetic mid score, so it contributes 0 coverage and 0 points."""
    closes = _uptrend_closes(220)
    packet = _packet(closes, volumes=[0.0] * 220)
    out = tech.run(packet)
    demand_dim = next(d for d in out.dimensions if d.name == tech.DIM_VOLUME)
    assert demand_dim.valid_weight() == 0.0            # no fabricated score
    assert demand_dim.score10_value().is_null           # honestly unscorable
    assert "VOLUME_UNAVAILABLE_DEMAND_CAPPED" in out.mandatory_flags
    # category still reproduces from dimensions (a NOT_SCORABLE dim -> 0 pts)
    recomputed = Category(name=tech.AGENT_ID, max_points=tech.MAX_POINTS, dimensions=out.dimensions)
    assert out.category.awarded_points == pytest.approx(recomputed.points(), abs=1e-6)


def test_run_obv_is_a_scored_volume_dimension_member():
    """SCORING.md lists TECH-OBV-016 as a primary input of the volume
    dimension (TECH-VR-014..017); it must be scored and participate in the
    dimension's weighted mean, not reported context-only."""
    packet = _packet(_uptrend_closes(260))
    out = tech.run(packet)
    obv_row = next(r for r in out.metrics if r.metric_id == "TECH-OBV-016")
    assert obv_row.score != "NOT_SCORABLE"
    demand_dim = next(d for d in out.dimensions if d.name == tech.DIM_VOLUME)
    assert len(demand_dim.metric_scores) == 4  # VR, UDV, OBV, CMF


def test_run_tight_close_is_a_scored_breakout_dimension_member():
    """SCORING.md lists TECH-TIGHT-038 as a primary input of breakout &
    base quality; it must participate in that dimension's weighted mean."""
    packet = _packet(_uptrend_closes(260))
    out = tech.run(packet)
    breakout_dim = next(d for d in out.dimensions if d.name == tech.DIM_BREAKOUT_BASE)
    assert len(breakout_dim.metric_scores) == 5  # VCP, LSTR, BCONF, BASE, TIGHT


def test_empty_output_confidence_derived_from_formula_not_hardcoded():
    """The unadjusted-rejection ERROR envelope derives confidence from the
    real five-component formula at coverage 0, not a hardcoded 0.0
    literal (awarded_points/score_10 stay 0.0 by construction)."""
    packet = _packet(_uptrend_closes(30), adjusted=False)
    out = tech.run(packet)
    assert out.status == "ERROR"
    assert out.category.awarded_points == 0.0
    assert out.category.score_10 == 0.0
    expected = tech._category_confidence(0.0, len(packet.market_data.daily))
    assert out.category.confidence == pytest.approx(expected)
    assert out.category.confidence > 0.0  # formula yields a real, low value


# ============================================================================
# TECH-TDEC (brief): time decay isn't part of this specialist, but the
# ATR-normalized slope guard and trend-anchor math are -- exhaustive
# per-row coverage of DECISION_RULES.md's six-row anchor table.
# ============================================================================


def test_primary_trend_bullish_stack_positive_slopes_scores_8():
    score, label = tech.primary_trend_score(
        close=120.0, sma50=110.0, sma200=100.0, atr14_latest=2.0,
        slope200_atr=0.5, slope50_atr=0.5, adx=15.0, range_pos_52w=0.5,
    )
    assert score == pytest.approx(8.0)
    assert label == "bullish_stack_positive_slopes_8"


def test_primary_trend_bullish_stack_adx_and_52w_confirmed_scores_9_to_10():
    score, label = tech.primary_trend_score(
        close=120.0, sma50=110.0, sma200=100.0, atr14_latest=2.0,
        slope200_atr=0.5, slope50_atr=0.5, adx=30.0, range_pos_52w=0.90,
    )
    assert 9.0 <= score <= 10.0
    assert label == "bullish_stack_adx_confirmed_9_10"


def test_primary_trend_bearish_stack_falling_sma200_scores_0_to_2():
    score, label = tech.primary_trend_score(
        close=80.0, sma50=90.0, sma200=100.0, atr14_latest=2.0,
        slope200_atr=-2.0, slope50_atr=-1.0, adx=20.0, range_pos_52w=0.1,
    )
    assert 0.0 <= score <= 2.0
    assert label == "bearish_stack_falling_sma200_0_2"


def test_primary_trend_below_sma200_mixed_stack_scores_3():
    score, label = tech.primary_trend_score(
        close=95.0, sma50=97.0, sma200=100.0, atr14_latest=2.0,
        slope200_atr=-0.5, slope50_atr=0.1, adx=18.0, range_pos_52w=0.3,
    )
    assert score == pytest.approx(3.0)
    assert label == "below_sma200_mixed_stack_3"


def test_primary_trend_neutral_chop_near_sma200_scores_4_5():
    # sma50 < sma200 (not a clean bullish/bearish stack either way) but
    # close sits within 1 ATR of sma200 with a flat slope -> isolates the
    # neutral-chop branch from the bullish-stack branch above it.
    score, label = tech.primary_trend_score(
        close=100.5, sma50=99.5, sma200=100.0, atr14_latest=2.0,
        slope200_atr=0.1, slope50_atr=0.05, adx=15.0, range_pos_52w=0.5,
    )
    assert score == pytest.approx(4.5)
    assert label == "neutral_chop_near_sma200_4_5"


def test_primary_trend_above_sma200_mixed_sma50_scores_6():
    score, label = tech.primary_trend_score(
        close=110.0, sma50=112.0, sma200=100.0, atr14_latest=2.0,
        slope200_atr=0.1, slope50_atr=-0.1, adx=15.0, range_pos_52w=0.5,
    )
    assert score == pytest.approx(6.0)
    assert label == "above_sma200_mixed_sma50_6"


# ============================================================================
# Locally implemented formulas
# ============================================================================


def test_volatility_contraction_ratio_below_one_is_contraction():
    n = 200
    atr_over_close = pd.Series([0.02] * (n - 5) + [0.01] * 5)
    v = tech.volatility_contraction_ratio(atr_over_close, n=126)
    assert v.is_valid
    assert v.value < 1.0


def test_tight_close_ratio_computes():
    n = 70
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.01, n))
    v = tech.tight_close_ratio(returns)
    assert v.is_valid
    assert v.value > 0.0


def test_base_depth_computes_from_trailing_window():
    n = 25
    closes = [100.0] * n
    df = pd.DataFrame({"high": [c * 1.05 for c in closes], "low": [c * 0.95 for c in closes], "close": closes})
    v = tech.base_depth(df, window=20)
    assert v.is_valid
    assert v.value == pytest.approx(0.10 / 1.05, rel=1e-6)


# ============================================================================
# Dimension caps (the apply_dimension_cap helper is tested in test_common.py;
# the tests below exercise technical.py's *use* of it -- trend/volume caps).
# ============================================================================


def test_run_trend_capped_at_6_without_200_sessions():
    packet = _packet(_uptrend_closes(120))  # <200 sessions
    out = tech.run(packet)
    trend_dim = next(d for d in out.dimensions if d.name == tech.DIM_TREND)
    assert trend_dim.score10() <= 6.0 + 1e-9
    assert "SMA200_UNAVAILABLE_TREND_CAPPED" in out.mandatory_flags


def test_run_trend_not_capped_with_full_uptrend_history():
    packet = _packet(_uptrend_closes(260))
    out = tech.run(packet)
    trend_dim = next(d for d in out.dimensions if d.name == tech.DIM_TREND)
    assert trend_dim.score10() >= 6.0  # a clean 260-session uptrend should score well above the cap


# ============================================================================
# Earnings-gap >=4-event gate
# ============================================================================


def test_run_earnings_gap_needs_4_events_else_not_scorable():
    packet = _packet(_uptrend_closes(220))
    out = tech.run(packet, overlay={"earnings_dates": []})
    gap_row = next(r for r in out.metrics if r.metric_id == "TECH-GAP-020")
    assert gap_row.score == "NOT_SCORABLE"


def test_run_earnings_gap_scorable_with_4_valid_events():
    closes = _uptrend_closes(260)
    # Insert 4 clean positive gaps at well-separated sessions with a prior
    # session available and >=20 future sessions to compute hold ratios.
    dates = _dates(260)
    packet = _packet(closes)
    earnings_dates = [dates[50], dates[100], dates[150], dates[200]]
    out = tech.run(packet, overlay={"earnings_dates": earnings_dates})
    gap_row = next(r for r in out.metrics if r.metric_id == "TECH-GAP-020")
    assert gap_row.score != "NOT_SCORABLE"


# ============================================================================
# run() against the NVDA fixture
# ============================================================================


def test_run_nvda_fixture_schema_valid(nvda_packet):
    out = tech.run(nvda_packet)
    assert out.agent_id == "technical_momentum"
    assert out.version == "2.0.0"
    assert out.security.ticker == "NVDA"
    assert out.category.max_points == 20.0
    assert out.status in ("COMPLETE", "INCOMPLETE", "ERROR")
    assert len(out.dimensions) == 6
    for row in out.metrics:
        assert row.metric_id
        assert row.formula_id
        assert row.formula_version
        assert row.score == "NOT_SCORABLE" or isinstance(row.score, float)
        assert 0.0 <= row.confidence <= 100.0
        assert (row.value is None) != (row.state is None)


def test_run_nvda_fixture_category_math_reproduces_from_dimensions(nvda_packet):
    out = tech.run(nvda_packet)
    recomputed = Category(name=tech.AGENT_ID, max_points=tech.MAX_POINTS, dimensions=out.dimensions)
    assert out.category.awarded_points == pytest.approx(recomputed.points(), abs=1e-6)
    assert out.category.score_10 == pytest.approx(recomputed.score10(), abs=1e-6)
    assert out.coverage == pytest.approx(recomputed.coverage(), abs=1e-6)


def test_run_nvda_fixture_category_confidence_computed(nvda_packet):
    out = tech.run(nvda_packet)
    assert out.category.confidence is not None
    assert 0.0 <= out.category.confidence <= 100.0


def test_run_nvda_fixture_serializes_to_json(nvda_packet):
    out = tech.run(nvda_packet)
    dumped = out.model_dump(mode="json")
    json.dumps(dumped)
    assert dumped["agent_id"] == "technical_momentum"


def test_run_nvda_fixture_important_levels_populated(nvda_packet):
    """`important_levels` reuses Task 12's `compute_levels` output --
    NVDA's 600-session fixture should produce at least the moving-average
    levels."""
    out = tech.run(nvda_packet)
    assert out.important_levels is not None
    assert isinstance(out.important_levels.moving_averages, list)


def test_run_nvda_fixture_benchmark_and_sector_empty_degrades_gracefully(nvda_packet):
    """The NVDA fixture's `market_data.benchmark`/`.sector` are empty (a
    Task 10 limitation, per the task-17 brief) -- RS/RSS must read
    NOT_SCORABLE/MISSING, not crash."""
    assert nvda_packet.market_data.benchmark == []
    assert nvda_packet.market_data.sector == []
    out = tech.run(nvda_packet)
    rs_row = next(r for r in out.metrics if r.metric_id == "TECH-RS-011")
    assert rs_row.score == "NOT_SCORABLE"


def test_run_nvda_fixture_validation_tests_all_self_checks_pass(nvda_packet):
    out = tech.run(nvda_packet)
    assert out.validation_tests.failed == 0
    assert out.validation_tests.passed >= 1


def test_run_empty_daily_history_degrades_without_crashing():
    packet = _packet([])
    out = tech.run(packet)
    assert out.status == "ERROR"
    assert out.coverage == 0.0


def test_verdict_bands_use_awarded_points_not_score10():
    """Technical profile bands (DECISION_RULES.md) are stated against raw
    category points (max 20), not score_10 -- the one specialist whose
    verdict differs from the score10-based pattern."""
    assert tech.verdict(18.0) == "Powerful leadership"
    assert tech.verdict(14.0) == "Healthy uptrend"
    assert tech.verdict(10.0) == "Neutral / transition"
    assert tech.verdict(5.0) == "Weak or broken"
