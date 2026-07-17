"""Price-level synthesis (Task 21): `synthesize_levels`.

Source of truth: `Cerebro/00_main_agent/PRICE_LEVEL_SYNTHESIS.md` -- the 12
required level classes, the confluence rule, the distance formulas, and
the report-language whitelist.

## Design

`synthesize_levels(technical_output, valuation_output, price, atr)` --
the exact signature the task brief pins -- reads Task 17's
`TechnicalOutput.important_levels` (Task 12's `LevelsOutput`) and Task 19's
`ValuationOutput.reference_bands` / `.reverse_dcf`, and produces one
`LevelReference` per populated level class plus the `ConfluenceZone`s
between them. It never averages a technical zone with an intrinsic-value
reference (PRICE_LEVEL_SYNTHESIS.md, verbatim: "They are not interchangeable
and must not be averaged mechanically") -- confluence detection only
*flags* overlapping independent references; `ConfluenceZone.low`/`.high`
is the union span of the participating references' own bounds, never a
computed mean, and every `LevelReference`'s own `value`/`zone_low`/
`zone_high` is copied straight from its source, untouched.

## Documented discrepancy: `distance_percent` units

PRICE_LEVEL_SYNTHESIS.md states `Distance_percent = (Level -
CurrentPrice) / CurrentPrice` (a raw fraction, e.g. `0.05` for 5%). Task
12's already-implemented `wbj.engines.levels_engine.compute_levels` (which
this module must not modify) instead computes `Zone.distance_percent` as
that fraction *times 100* (a percentage number, e.g. `5.0`). Rather than
mix units within one synthesized table -- unreadable, and silently wrong
if a caller ever sums/compares two entries -- every `distance_percent`
this module computes locally (moving averages, AVWAPs, earnings-gap
boundaries, valuation bands) follows Task 12's already-shipped convention
(percentage units, not a fraction) for consistency across the whole table.
Flagged here and in the Task 21 commit message; Cerebro's own formula
text would otherwise win per the task instructions, but it cannot be
reconciled with already-frozen Task 12 code without modifying that module,
which is out of scope.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from wbj.schemas.levels import Zone
from wbj.specialists.technical import TechnicalOutput
from wbj.specialists.valuation import ValuationOutput

__all__ = [
    "FORBIDDEN_PHRASES",
    "LevelReference",
    "ConfluenceZone",
    "LevelSynthesis",
    "confluence_tolerance",
    "synthesize_levels",
]

Source = Literal["technical", "valuation", "current_price"]

# PRICE_LEVEL_SYNTHESIS.md "Language rules": allowed vocabulary is
# reference/zone/confirmation/invalidation/scenario value; these three
# phrases are explicitly forbidden (they read as a promise, not a
# reference).
FORBIDDEN_PHRASES: tuple[str, ...] = ("guaranteed target", "must hold", "certain floor")


def _check_language(text: str | None) -> None:
    if not text:
        return
    lowered = text.lower()
    for phrase in FORBIDDEN_PHRASES:
        if phrase in lowered:
            raise ValueError(
                f"PRICE_LEVEL_SYNTHESIS.md language rule violated: forbidden phrase {phrase!r} in {text!r}"
            )


class LevelReference(BaseModel):
    """One row of PRICE_LEVEL_SYNTHESIS.md's "Final table schema" (Rank is
    the row's position in `LevelSynthesis.levels`, not stored per-row)."""

    model_config = ConfigDict(frozen=True)

    level_class: str
    label: str
    source: Source
    zone_low: float | None = None
    zone_high: float | None = None
    value: float | None = None
    distance_percent: float | None = None
    distance_atr: float | None = None
    strength_or_confidence: float | None = None
    status: str | None = None
    confirmation: str | None = None
    invalidation: str | None = None
    note: str | None = None

    def model_post_init(self, __context: object) -> None:
        for text in (self.label, self.confirmation, self.invalidation, self.note):
            _check_language(text)


class ConfluenceZone(BaseModel):
    """A confluence flag between >=2 independent `LevelReference`s
    (PRICE_LEVEL_SYNTHESIS.md "Confluence rules"). `low`/`high` is the
    union span of the participating references' own bounds -- never an
    averaged value."""

    model_config = ConfigDict(frozen=True)

    level_labels: list[str]
    low: float
    high: float
    note: str

    def model_post_init(self, __context: object) -> None:
        _check_language(self.note)


class LevelSynthesis(BaseModel):
    """`synthesize_levels`'s result."""

    model_config = ConfigDict(frozen=True)

    current_price: float
    atr14: float
    levels: list[LevelReference] = Field(default_factory=list)
    confluences: list[ConfluenceZone] = Field(default_factory=list)


def confluence_tolerance(atr: float, price: float) -> float:
    """`max(0.50*ATR14, 0.75% of current price)` (PRICE_LEVEL_SYNTHESIS.md
    "Confluence rules", verbatim)."""
    return max(0.50 * atr, 0.0075 * price)


def _point_distance(value: float, price: float, atr: float) -> tuple[float, float | None]:
    distance_percent = (value - price) / price * 100.0 if price else 0.0
    distance_atr = (value - price) / atr if atr else None
    return distance_percent, distance_atr


def _zone_level(level_class: str, label: str, zone: Zone) -> LevelReference:
    """A `LevelReference` copied straight from an already-computed Task-12
    `Zone` -- reuses `zone.distance_percent`/`zone.distance_atr` as-is
    (see module docstring's units note) rather than recomputing them."""
    return LevelReference(
        level_class=level_class,
        label=f"{label} ({zone.zone_id})",
        source="technical",
        zone_low=zone.lower,
        zone_high=zone.upper,
        distance_percent=zone.distance_percent,
        distance_atr=zone.distance_atr,
        strength_or_confidence=zone.strength_0_100,
        status=zone.status,
        confirmation=zone.confirmation_rule,
        invalidation=zone.invalidation_rule,
    )


def _nearest_confirmed(zones: list[Zone]) -> Zone | None:
    for z in zones:
        if z.status in ("confirmed", "strong", "role_reversed"):
            return z
    return zones[0] if zones else None


def _bounds(level: LevelReference) -> tuple[float, float] | None:
    if level.zone_low is not None and level.zone_high is not None:
        return level.zone_low, level.zone_high
    if level.value is not None:
        return level.value, level.value
    return None


def _find_confluences(levels: list[LevelReference], price: float, atr: float) -> list[ConfluenceZone]:
    """Confluence rule: two independent references overlap within
    `confluence_tolerance`, with at least one of the pair technical.
    `current_price` itself is excluded (a level is never "confluent" with
    the price it's measured against)."""
    tol = confluence_tolerance(atr, price)
    out: list[ConfluenceZone] = []
    n = len(levels)
    for i in range(n):
        a = levels[i]
        if a.source == "current_price":
            continue
        bounds_a = _bounds(a)
        if bounds_a is None:
            continue
        for j in range(i + 1, n):
            b = levels[j]
            if b.source == "current_price":
                continue
            if a.source != "technical" and b.source != "technical":
                continue  # need >=1 technical reference
            bounds_b = _bounds(b)
            if bounds_b is None:
                continue
            lo_a, hi_a = bounds_a[0] - tol, bounds_a[1] + tol
            lo_b, hi_b = bounds_b[0] - tol, bounds_b[1] + tol
            if lo_a <= hi_b and lo_b <= hi_a:
                out.append(
                    ConfluenceZone(
                        level_labels=[a.label, b.label],
                        low=min(bounds_a[0], bounds_b[0]),
                        high=max(bounds_a[1], bounds_b[1]),
                        note=(
                            f"Confluence reference: {a.label!r} and {b.label!r} overlap within "
                            f"tolerance ({tol:.4f}); both zone/scenario values are retained "
                            "independently, not averaged."
                        ),
                    )
                )
    return out


def synthesize_levels(
    technical_output: TechnicalOutput,
    valuation_output: ValuationOutput,
    price: float,
    atr: float,
) -> LevelSynthesis:
    """The 12 required level classes (PRICE_LEVEL_SYNTHESIS.md), assembled
    from Task 17's `important_levels`/`breakouts_and_failures` (technical
    lens) and Task 19's `reference_bands`/`reverse_dcf` (intrinsic-value
    lens), plus the confluence zones between them. A class with no
    available data simply contributes no rows -- it is never fabricated or
    imputed."""
    levels: list[LevelReference] = []

    # 1. Current adjusted close.
    levels.append(
        LevelReference(
            level_class="current_adjusted_close",
            label="Current adjusted close",
            source="current_price",
            value=price,
            distance_percent=0.0,
            distance_atr=0.0,
        )
    )

    important_levels = technical_output.important_levels

    # 2. Nearest confirmed support zone.
    nearest_support = _nearest_confirmed(important_levels.nearest_support)
    if nearest_support is not None:
        levels.append(_zone_level("nearest_confirmed_support_zone", "Nearest support zone", nearest_support))

    # 3. Nearest confirmed resistance zone.
    nearest_resistance = _nearest_confirmed(important_levels.nearest_resistance)
    if nearest_resistance is not None:
        levels.append(_zone_level("nearest_confirmed_resistance_zone", "Nearest resistance zone", nearest_resistance))

    # 4. Confirmed breakout trigger and failed-breakout level.
    all_zones_by_id = {z.zone_id: z for z in (*important_levels.nearest_support, *important_levels.nearest_resistance)}
    for b in technical_output.breakouts_and_failures:
        zone = all_zones_by_id.get(b.get("zone_id"))
        if zone is None:
            continue
        label = "Confirmed breakout trigger" if b.get("breakout_confirmed") else "Failed-breakout level"
        levels.append(_zone_level("breakout_trigger_or_failure", label, zone))

    # 5. Role-reversal retest zone, when validated.
    for z in (*important_levels.nearest_support, *important_levels.nearest_resistance):
        if z.status == "role_reversed":
            levels.append(_zone_level("role_reversal_retest_zone", "Role-reversal retest zone", z))

    # 6. 20-, 50-, 100-, and 200-session moving averages.
    for ma in important_levels.moving_averages:
        dp, da = _point_distance(ma.value, price, atr)
        levels.append(
            LevelReference(
                level_class="moving_average", label=ma.label, source="technical",
                value=ma.value, distance_percent=dp, distance_atr=da,
            )
        )

    # 7. Selected anchored VWAP levels.
    for a in important_levels.avwaps:
        dp, da = _point_distance(a.value, price, atr)
        levels.append(
            LevelReference(
                level_class="anchored_vwap",
                label=f"AVWAP anchored {a.anchor_date} ({a.anchor_reason})",
                source="technical", value=a.value, distance_percent=dp, distance_atr=da,
            )
        )

    # 8. Material earnings-gap boundaries and midpoint.
    for g in important_levels.earnings_gaps:
        if not g.material:
            continue
        for boundary_label, boundary_value in (
            ("gap open", g.gap_open), ("gap high", g.gap_high),
            ("gap low", g.gap_low), ("gap midpoint", g.gap_midpoint),
        ):
            dp, da = _point_distance(boundary_value, price, atr)
            levels.append(
                LevelReference(
                    level_class="earnings_gap_boundary",
                    label=f"Earnings gap {g.event_date} {boundary_label}",
                    source="technical", value=boundary_value, distance_percent=dp, distance_atr=da,
                    status=g.fill_status,
                )
            )

    # 9. Weekly support/resistance zones.
    for z in (*important_levels.nearest_support, *important_levels.nearest_resistance):
        if z.timeframe == "weekly":
            levels.append(_zone_level("weekly_zone", f"Weekly {z.type} zone", z))

    # 10. Bear, base, and bull intrinsic-value ranges.
    rb = valuation_output.reference_bands
    for name, val in (("Bear", rb.bear), ("Base", rb.base), ("Bull", rb.bull)):
        if val is None:
            continue
        dp, da = _point_distance(val, price, atr)
        levels.append(
            LevelReference(
                level_class="intrinsic_value_scenario",
                label=f"{name}-case intrinsic-value reference (scenario value)",
                source="valuation", value=val, distance_percent=dp, distance_atr=da,
            )
        )

    # 11. Reverse-DCF current-price implied assumptions.
    rdcf = valuation_output.reverse_dcf
    if rdcf.implied_revenue_cagr is not None or rdcf.implied_margin is not None:
        parts = []
        if rdcf.implied_revenue_cagr is not None:
            parts.append(f"implied revenue CAGR {rdcf.implied_revenue_cagr:.2%}")
        if rdcf.implied_margin is not None:
            parts.append(f"implied operating margin {rdcf.implied_margin:.2%}")
        levels.append(
            LevelReference(
                level_class="reverse_dcf_implied_assumptions",
                label="Reverse-DCF current-price implied assumptions (reference only)",
                source="valuation",
                value=rdcf.current_price,
                note="; ".join(parts),
            )
        )

    # 12. Margin-of-safety reference bands.
    for name, val in (("15%", rb.margin_of_safety_15pct), ("25%", rb.margin_of_safety_25pct)):
        if val is None:
            continue
        dp, da = _point_distance(val, price, atr)
        levels.append(
            LevelReference(
                level_class="margin_of_safety_band",
                label=f"Margin-of-safety reference band ({name})",
                source="valuation", value=val, distance_percent=dp, distance_atr=da,
            )
        )

    confluences = _find_confluences(levels, price, atr)

    return LevelSynthesis(current_price=price, atr14=atr, levels=levels, confluences=confluences)
