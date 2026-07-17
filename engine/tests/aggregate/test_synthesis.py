"""Tests for `wbj.aggregate.synthesis` (Task 21): `synthesize_levels`.

Sources of truth: `Cerebro/00_main_agent/PRICE_LEVEL_SYNTHESIS.md`,
`Cerebro/00_main_agent/VALIDATION_TESTS.md` (MAIN-008).
"""

from __future__ import annotations

import pytest

from wbj.aggregate.synthesis import (
    FORBIDDEN_PHRASES,
    LevelReference,
    confluence_tolerance,
    synthesize_levels,
)
from wbj.schemas.levels import (
    AVWAPLevel,
    EarningsGap,
    LevelsOutput,
    MovingAverageLevel,
    Touch,
    Zone,
)
from wbj.specialists.valuation import ReferenceBands, ReverseDCFSummary

from .conftest import make_technical, make_valuation

PRICE = 100.0
ATR = 2.0  # confluence_tolerance(2.0, 100.0) == max(1.0, 0.75) == 1.0


def _support_zone(lower=90.0, upper=92.0, status="confirmed") -> Zone:
    return Zone(
        zone_id="daily-support-91.00", type="support", lower=lower, center=(lower + upper) / 2, upper=upper,
        timeframe="daily", status=status, strength_0_100=70.0,
        touches=[Touch(date="2026-06-01", pivot_price=lower + 0.1, rejection_atr=0.6, volume_ratio=1.4)],
        distance_percent=-8.0, distance_atr=-4.0,
        confirmation_rule="Confirmed by >=2 independent touches.",
        invalidation_rule="Broken by a confirmed close beyond the zone with volume follow-through.",
    )


def _resistance_zone(lower=108.0, upper=110.0, status="confirmed", timeframe="daily") -> Zone:
    return Zone(
        zone_id=f"{timeframe}-resistance-109.00", type="resistance", lower=lower, center=(lower + upper) / 2, upper=upper,
        timeframe=timeframe, status=status, strength_0_100=65.0,
        touches=[Touch(date="2026-05-20", pivot_price=upper - 0.1, rejection_atr=0.7, volume_ratio=1.6)],
        distance_percent=8.0, distance_atr=4.0,
        confirmation_rule="Confirmed by >=2 independent touches.",
        invalidation_rule="Broken by a confirmed close beyond the zone with volume follow-through.",
    )


def _full_levels_output() -> LevelsOutput:
    return LevelsOutput(
        nearest_support=[_support_zone(), _support_zone(lower=80.0, upper=82.0, status="candidate")],
        nearest_resistance=[_resistance_zone(), _resistance_zone(lower=118.0, upper=120.0, timeframe="weekly")],
        moving_averages=[
            MovingAverageLevel(label="SMA20", value=101.0),
            MovingAverageLevel(label="SMA50", value=98.0),
            MovingAverageLevel(label="SMA100", value=95.0),
            MovingAverageLevel(label="SMA200", value=90.0),
        ],
        avwaps=[AVWAPLevel(anchor_date="2026-04-01", anchor_reason="major swing low", value=97.0)],
        earnings_gaps=[
            EarningsGap(
                event_date="2026-05-01", prior_close=95.0, gap_open=99.0, gap_high=100.0, gap_low=98.5,
                gap_midpoint=97.0, gap_percent=0.042, material=True, fill_status="partially_filled",
            )
        ],
    )


def _full_valuation_reference_bands() -> ReferenceBands:
    return ReferenceBands(bear=85.0, base=105.0, bull=125.0, margin_of_safety_15pct=89.25, margin_of_safety_25pct=78.75)


def _full_reverse_dcf() -> ReverseDCFSummary:
    return ReverseDCFSummary(current_price=PRICE, implied_revenue_cagr=0.18, implied_margin=0.22, implied_high_growth_years=5)


def test_confluence_tolerance_formula():
    assert confluence_tolerance(atr=2.0, price=100.0) == 1.0  # max(0.5*2.0, 0.0075*100) = max(1.0, 0.75)
    assert confluence_tolerance(atr=0.1, price=1000.0) == 7.5  # max(0.05, 7.5)


# ============================================================================
# MAIN-008: support and fair-value band overlap within tolerance -> confluence
# flag, not an averaged value
# ============================================================================


def test_MAIN_008_support_and_fair_value_confluence():
    tech = make_technical(important_levels=LevelsOutput(nearest_support=[_support_zone(lower=90.0, upper=92.0)]))
    val = make_valuation(reference_bands=ReferenceBands(base=93.0))  # within tolerance (1.0) of the zone's upper edge

    result = synthesize_levels(tech, val, price=PRICE, atr=ATR)

    assert len(result.confluences) >= 1
    conf = result.confluences[0]
    assert "Nearest support zone" in conf.level_labels[0] or "Nearest support zone" in conf.level_labels[1]
    assert any("Base-case" in lbl for lbl in conf.level_labels)


def test_synthesis_never_averages_technical_and_intrinsic():
    tech = make_technical(important_levels=LevelsOutput(nearest_support=[_support_zone(lower=90.0, upper=92.0)]))
    val = make_valuation(reference_bands=ReferenceBands(base=93.0))

    result = synthesize_levels(tech, val, price=PRICE, atr=ATR)

    support_level = next(lv for lv in result.levels if lv.level_class == "nearest_confirmed_support_zone")
    intrinsic_level = next(lv for lv in result.levels if lv.level_class == "intrinsic_value_scenario")

    # Each level's own numbers are untouched -- never blended with the other lens.
    assert support_level.zone_low == 90.0
    assert support_level.zone_high == 92.0
    assert intrinsic_level.value == 93.0

    # The confluence span is the union of the two independent bounds, not
    # some computed mean (e.g. NOT (91 + 93) / 2 == 92).
    conf = result.confluences[0]
    assert conf.low == 90.0
    assert conf.high == 93.0
    mean_of_centers = (91.0 + 93.0) / 2
    assert conf.low != mean_of_centers
    assert conf.high != mean_of_centers


def test_no_confluence_when_references_are_far_apart():
    tech = make_technical(important_levels=LevelsOutput(nearest_support=[_support_zone(lower=50.0, upper=52.0)]))
    val = make_valuation(reference_bands=ReferenceBands(base=93.0))
    result = synthesize_levels(tech, val, price=PRICE, atr=ATR)
    assert result.confluences == []


def test_confluence_requires_at_least_one_technical_reference():
    """Two valuation-only references must never confluence with each
    other (PRICE_LEVEL_SYNTHESIS.md: "at least one reference must be
    technical")."""
    tech = make_technical(important_levels=LevelsOutput())
    val = make_valuation(
        reference_bands=ReferenceBands(bear=100.0, base=100.4)  # well within tolerance of each other
    )
    result = synthesize_levels(tech, val, price=PRICE, atr=ATR)
    assert result.confluences == []


def test_confluence_boundary_exactly_at_tolerance_counts():
    tol = confluence_tolerance(atr=ATR, price=PRICE)  # 1.0
    tech = make_technical(important_levels=LevelsOutput(moving_averages=[MovingAverageLevel(label="SMA20", value=100.0)]))
    val = make_valuation(reference_bands=ReferenceBands(bear=100.0 + 2 * tol))  # exactly at the overlap boundary
    result = synthesize_levels(tech, val, price=PRICE, atr=ATR)
    assert len(result.confluences) == 1


def test_confluence_boundary_just_outside_tolerance_does_not_count():
    tol = confluence_tolerance(atr=ATR, price=PRICE)  # 1.0
    tech = make_technical(important_levels=LevelsOutput(moving_averages=[MovingAverageLevel(label="SMA20", value=100.0)]))
    val = make_valuation(reference_bands=ReferenceBands(bear=100.0 + 2 * tol + 0.05))
    result = synthesize_levels(tech, val, price=PRICE, atr=ATR)
    assert result.confluences == []


# ============================================================================
# All 12 required level classes
# ============================================================================


def test_all_12_level_classes_represented_with_full_data():
    tech = make_technical(
        important_levels=_full_levels_output(),
        breakouts_and_failures=[{"zone_id": "daily-resistance-109.00", "status": "confirmed", "breakout_confirmed": True}],
    )
    val = make_valuation(reference_bands=_full_valuation_reference_bands(), reverse_dcf=_full_reverse_dcf())

    result = synthesize_levels(tech, val, price=PRICE, atr=ATR)
    classes = {lv.level_class for lv in result.levels}

    expected = {
        "current_adjusted_close",  # 1
        "nearest_confirmed_support_zone",  # 2
        "nearest_confirmed_resistance_zone",  # 3
        "breakout_trigger_or_failure",  # 4
        # 5 (role_reversal_retest_zone) only appears when a zone is role_reversed
        "moving_average",  # 6
        "anchored_vwap",  # 7
        "earnings_gap_boundary",  # 8
        "weekly_zone",  # 9
        "intrinsic_value_scenario",  # 10
        "reverse_dcf_implied_assumptions",  # 11
        "margin_of_safety_band",  # 12
    }
    assert expected <= classes


def test_role_reversal_retest_zone_when_validated():
    tech = make_technical(
        important_levels=LevelsOutput(nearest_support=[_support_zone(status="role_reversed")])
    )
    val = make_valuation()
    result = synthesize_levels(tech, val, price=PRICE, atr=ATR)
    classes = {lv.level_class for lv in result.levels}
    assert "role_reversal_retest_zone" in classes


def test_moving_averages_carry_distance_in_percent_units_not_fraction():
    tech = make_technical(important_levels=LevelsOutput(moving_averages=[MovingAverageLevel(label="SMA20", value=110.0)]))
    val = make_valuation()
    result = synthesize_levels(tech, val, price=PRICE, atr=ATR)
    ma = next(lv for lv in result.levels if lv.level_class == "moving_average")
    # (110-100)/100 * 100 = 10.0 (percent units, matching Task 12's Zone.distance_percent convention)
    assert ma.distance_percent == pytest.approx(10.0)
    assert ma.distance_atr == pytest.approx(5.0)  # (110-100)/2.0


# ============================================================================
# Language whitelist
# ============================================================================


def test_forbidden_phrases_are_rejected():
    for phrase in FORBIDDEN_PHRASES:
        with pytest.raises(ValueError):
            LevelReference(level_class="x", label=f"This is a {phrase}", source="technical")


def test_generated_labels_never_contain_forbidden_phrases():
    tech = make_technical(important_levels=_full_levels_output())
    val = make_valuation(reference_bands=_full_valuation_reference_bands(), reverse_dcf=_full_reverse_dcf())
    result = synthesize_levels(tech, val, price=PRICE, atr=ATR)
    texts: list[str] = []
    for lv in result.levels:
        texts.extend(t for t in (lv.label, lv.confirmation, lv.invalidation, lv.note) if t)
    texts.extend(c.note for c in result.confluences)
    all_text = " ".join(texts).lower()
    for phrase in FORBIDDEN_PHRASES:
        assert phrase not in all_text
