"""Raw score, descriptive bands, total confidence, and profile gates
(Task 21).

Source of truth: `Cerebro/00_main_agent/SCORING_AND_GATES.md`. Every
threshold below is copied verbatim from that file's "Profile gates"
section; see each `_*_gate` function's docstring for the exact clause it
implements.

## Design: `apply_gates(raw_total, cats, confidences, overrides)`

The task brief pins this exact signature. `overrides` is the `list[Override]`
returned by `wbj.aggregate.overrides.apply_overrides` -- rather than
re-deriving "is Risk <=4/15" or "is any category coverage <0.70" a second
time from raw numbers, this module reads those two conditions off
`OVERRIDE_4_RISK_FLOOR` / `OVERRIDE_6_COVERAGE_GATE_INELIGIBLE` (one source
of truth for each condition, matching `overrides.py`'s own reuse
discipline), while still taking `cats`/`confidences` directly for the gate
math that SCORING_AND_GATES.md states purely in terms of category points
(momentum/quality/value's numeric thresholds) since those aren't
overrides at all -- they're the gates themselves.

## Documented gap: raw total in [50, 60)

SCORING_AND_GATES.md's profile-gate rows cover raw >= 60 (Conditional/
Watch), several non-raw-total Speculative conditions, and raw < 50
(Avoid/Wait) -- but a raw total of, say, 55 with no override and no
Speculative trigger is not assigned a profile by any rule in that file
(its own descriptive-band table separately labels 50-59.99 "Weak", but
"a raw band is not the final profile"). This module closes that gap by
falling back to the `GATE_WEAK` label (`"Weak / Wait (no gate passed)"`)
for that range, flagged here and in the Task 21 commit message as an
interpretation beyond the literal spec rather than a value taken directly
from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from wbj.aggregate.overrides import (
    OVERRIDE_1_CAPITAL_DEPENDENCE,
    OVERRIDE_2_ROIC_BELOW_WACC,
    OVERRIDE_3_SOLVENCY_WARNING,
    OVERRIDE_4_RISK_FLOOR,
    OVERRIDE_5_PREMIUM_BREAKDOWN,
    OVERRIDE_6_COVERAGE_GATE_INELIGIBLE,
    Override,
)
from wbj.core.confidence import total_confidence as _total_confidence_formula

__all__ = [
    "CATEGORY_ORDER",
    "CategoryPoints",
    "CategoryConfidences",
    "ProfileResult",
    "GATE_MOMENTUM",
    "GATE_QUALITY",
    "GATE_VALUE",
    "GATE_CONDITIONAL",
    "GATE_SPECULATIVE",
    "GATE_AVOID",
    "GATE_WEAK",
    "raw_total",
    "descriptive_band",
    "total_confidence",
    "apply_gates",
]

CATEGORY_ORDER = ("business", "financial", "market", "technical", "risk", "valuation")

GATE_MOMENTUM = "Momentum Candidate"
GATE_QUALITY = "Quality Opportunity"
GATE_VALUE = "Value Opportunity"
GATE_CONDITIONAL = "Conditional / Watch"
GATE_SPECULATIVE = "Speculative"
GATE_AVOID = "Avoid / Wait"
# Fallback label for the documented [50, 60) gap (see module docstring):
# raw in [50,60) with no override and no Speculative trigger is assigned no
# profile by any SCORING_AND_GATES.md rule.
GATE_WEAK = "Weak / Wait (no gate passed)"

_TOTAL_CONFIDENCE_SPECULATIVE_FLOOR = 60.0


@dataclass(frozen=True)
class CategoryPoints:
    """Each category's awarded points, on its own fixed 0-max scale
    (business 0-20, financial 0-15, market 0-20, technical 0-20, risk
    0-15, valuation 0-10) -- i.e. `SpecialistOutput.category.awarded_points`
    for each of the six specialists."""

    business: float
    financial: float
    market: float
    technical: float
    risk: float
    valuation: float

    def as_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in CATEGORY_ORDER}


@dataclass(frozen=True)
class CategoryConfidences:
    """Each category's confidence, 0-100 -- i.e.
    `SpecialistOutput.category.confidence` for each of the six
    specialists."""

    business: float
    financial: float
    market: float
    technical: float
    risk: float
    valuation: float

    def as_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in CATEGORY_ORDER}


@dataclass(frozen=True)
class ProfileResult:
    """`apply_gates`'s result: the final profile label plus every input
    SCORING_AND_GATES.md says a report must show beside it."""

    label: str
    raw_score: float
    total_confidence: float
    descriptive_band: str
    passed_gates: list[str] = field(default_factory=list)
    failed_gates: list[str] = field(default_factory=list)
    overrides: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def raw_total(points) -> float:
    """`sum` of the six category awarded points.

    Per MAIN-002: `raw_total([16, 10.5, 18, 16, 9, 7]) == 76.5`. Accepts
    either a bare iterable of 6 numbers (MAIN-002's own call shape) or a
    `CategoryPoints`.
    """
    if isinstance(points, CategoryPoints):
        values = points.as_dict().values()
    else:
        values = points
    return float(sum(values))


def descriptive_band(raw: float) -> str:
    """Raw-score descriptive band (SCORING_AND_GATES.md table, verbatim).
    Informational only -- "a raw band is not the final profile."."""
    if raw >= 90:
        return "Elite raw score"
    if raw >= 80:
        return "Strong raw score"
    if raw >= 70:
        return "Conditional raw score"
    if raw >= 60:
        return "Mixed / wait"
    if raw >= 50:
        return "Weak"
    return "Avoid on raw score"


def total_confidence(confidences: CategoryConfidences) -> float:
    """`sum(category_max_points * category_confidence) / 100`
    (SCORING_AND_GATES.md "Total confidence"). Delegates to Task 5's
    `wbj.core.confidence.total_confidence`, the one existing implementation
    of this exact formula (weighted by the fixed category *maxima*, not by
    awarded points)."""
    return _total_confidence_formula(confidences.as_dict())


def _momentum_gate(raw: float, cats: CategoryPoints, technical_confidence: float) -> tuple[bool, list[str]]:
    """Momentum Candidate (SCORING_AND_GATES.md, verbatim):
    `raw>=78, Technical>=17/20, Market>=16/20, Business+Financial>=28/35,
    Risk>=8/15, Technical confidence>=70`."""
    reasons: list[str] = []
    if not raw >= 78:
        reasons.append(f"raw_total<78 (got {raw!r})")
    if not cats.technical >= 17:
        reasons.append(f"technical<17 (got {cats.technical!r})")
    if not cats.market >= 16:
        reasons.append(f"market<16 (got {cats.market!r})")
    bus_fin = cats.business + cats.financial
    if not bus_fin >= 28:
        reasons.append(f"business+financial<28 (got {bus_fin!r})")
    if not cats.risk >= 8:
        reasons.append(f"risk<8 (got {cats.risk!r})")
    if not technical_confidence >= 70:
        reasons.append(f"technical_confidence<70 (got {technical_confidence!r})")
    return (len(reasons) == 0, reasons)


def _quality_gate(raw: float, cats: CategoryPoints) -> tuple[bool, list[str]]:
    """Quality Opportunity (SCORING_AND_GATES.md, verbatim):
    `raw>=80, Business>=16/20, Financial>=11/15, Risk>=10/15,
    Valuation>=5/10, Technical>=12/20`."""
    reasons: list[str] = []
    if not raw >= 80:
        reasons.append(f"raw_total<80 (got {raw!r})")
    if not cats.business >= 16:
        reasons.append(f"business<16 (got {cats.business!r})")
    if not cats.financial >= 11:
        reasons.append(f"financial<11 (got {cats.financial!r})")
    if not cats.risk >= 10:
        reasons.append(f"risk<10 (got {cats.risk!r})")
    if not cats.valuation >= 5:
        reasons.append(f"valuation<5 (got {cats.valuation!r})")
    if not cats.technical >= 12:
        reasons.append(f"technical<12 (got {cats.technical!r})")
    return (len(reasons) == 0, reasons)


def _value_gate(raw: float, cats: CategoryPoints) -> tuple[bool, list[str]]:
    """Value Opportunity (SCORING_AND_GATES.md, verbatim):
    `raw>=75, Valuation>=8/10, Business>=13/20, Risk>=10/15,
    Technical>=9/20`."""
    reasons: list[str] = []
    if not raw >= 75:
        reasons.append(f"raw_total<75 (got {raw!r})")
    if not cats.valuation >= 8:
        reasons.append(f"valuation<8 (got {cats.valuation!r})")
    if not cats.business >= 13:
        reasons.append(f"business<13 (got {cats.business!r})")
    if not cats.risk >= 10:
        reasons.append(f"risk<10 (got {cats.risk!r})")
    if not cats.technical >= 9:
        reasons.append(f"technical<9 (got {cats.technical!r})")
    return (len(reasons) == 0, reasons)


def apply_gates(
    raw_total: float,
    cats: CategoryPoints,
    confidences: CategoryConfidences,
    overrides: list[Override],
    *,
    runway_unfunded: bool = False,
    pre_profit: bool = False,
    valuation_mandatory_flags: list[str] | None = None,
) -> ProfileResult:
    """Apply SCORING_AND_GATES.md's profile gates and mandatory overrides
    to a frozen raw score, producing the single final `ProfileResult`.

    Two Speculative bullets are not derivable from
    `cats`/`confidences`/`overrides` alone, so they are explicit keyword-only
    inputs (matching the `runway_unfunded` precedent):

    - `runway_unfunded` -- Speculative's "financing runway is less than 12
      months without committed funding" bullet. No override or
      category-point carries a runway signal.
    - `pre_profit` + `valuation_mandatory_flags` -- Speculative's "pre-profit
      valuation depends on a low-confidence terminal value" bullet. The
      low-confidence-terminal-value signal already exists in the codebase as
      `valuation.py`'s `HIGH_TERMINAL_SENSITIVITY` mandatory flag (raised
      when `TERMINAL_VALUE_SHARE_ABOVE_75PCT` fires); this gate reads it
      from `valuation_mandatory_flags`. It only forces Speculative when the
      company is also `pre_profit` (net loss), because SCORING_AND_GATES.md
      scopes the bullet to "pre-profit valuation". A caller (Task 24's
      orchestrator) sets `pre_profit` from the financial specialist's net-
      loss signal and passes `valuation.mandatory_flags` straight through.

    All three default to their no-op value, matching every existing MAIN-* /
    gate test, none of which exercises these bullets.
    """
    override_ids = {o.id for o in overrides}
    conf_total = total_confidence(confidences)
    band = descriptive_band(raw_total)
    warnings = [o.reason for o in overrides if o.id == OVERRIDE_3_SOLVENCY_WARNING]
    override_id_list = [o.id for o in overrides]

    momentum_ok, momentum_reasons = _momentum_gate(raw_total, cats, confidences.technical)
    quality_ok, quality_reasons = _quality_gate(raw_total, cats)
    value_ok, value_reasons = _value_gate(raw_total, cats)

    if OVERRIDE_2_ROIC_BELOW_WACC in override_ids:
        quality_reasons.append(f"{OVERRIDE_2_ROIC_BELOW_WACC}: ROIC below WACC prevents Elite/Quality")
        quality_ok = False

    # --- Avoid/Wait: an override demands it, or raw_total < 50 ---
    forced_avoid_reasons: list[str] = []
    if OVERRIDE_5_PREMIUM_BREAKDOWN in override_ids:
        forced_avoid_reasons.append(f"{OVERRIDE_5_PREMIUM_BREAKDOWN}: valuation<=4/10 and technical<=8/20")
    if raw_total < 50:
        forced_avoid_reasons.append(f"raw_total<50 (got {raw_total!r})")
    if forced_avoid_reasons:
        return ProfileResult(
            label=GATE_AVOID,
            raw_score=raw_total,
            total_confidence=conf_total,
            descriptive_band=band,
            passed_gates=[],
            failed_gates=forced_avoid_reasons,
            overrides=override_id_list,
            warnings=warnings,
        )

    # --- Speculative: risk<=4/15, total confidence<60, a critical category
    # incomplete (coverage override), capital-dependence override, a
    # pre-profit low-confidence terminal value, or unfunded runway.
    #
    # OVERRIDE_1_CAPITAL_DEPENDENCE routes here (Speculative), never to
    # Avoid: SCORING_AND_GATES.md caps a capital-dependent name at
    # "Avoid/Speculative", and Avoid is reached only via the independent
    # raw_total<50 rule (evaluated in the block above) -- so a capital-
    # dependent name with raw>=50 lands in Speculative, and one with raw<50
    # in Avoid, exactly reproducing the "Avoid/Speculative" cap without
    # override 1 itself ever forcing Avoid. ---
    critical_incomplete = OVERRIDE_6_COVERAGE_GATE_INELIGIBLE in override_ids
    forced_speculative_reasons: list[str] = []
    if OVERRIDE_1_CAPITAL_DEPENDENCE in override_ids:
        forced_speculative_reasons.append(f"{OVERRIDE_1_CAPITAL_DEPENDENCE}: capital dependence")
    if OVERRIDE_4_RISK_FLOOR in override_ids or cats.risk <= 4.0:
        forced_speculative_reasons.append(f"risk<=4/15 (got {cats.risk!r})")
    if conf_total < _TOTAL_CONFIDENCE_SPECULATIVE_FLOOR:
        forced_speculative_reasons.append(f"total_confidence<60 (got {conf_total!r})")
    if critical_incomplete:
        forced_speculative_reasons.append(f"{OVERRIDE_6_COVERAGE_GATE_INELIGIBLE}: a critical category is incomplete")
    low_confidence_terminal_value = (
        pre_profit
        and valuation_mandatory_flags is not None
        and "HIGH_TERMINAL_SENSITIVITY" in valuation_mandatory_flags
    )
    if low_confidence_terminal_value:
        forced_speculative_reasons.append(
            "pre-profit valuation depends on a low-confidence terminal value "
            "(valuation HIGH_TERMINAL_SENSITIVITY)"
        )
    if runway_unfunded:
        forced_speculative_reasons.append("financing runway<12 months without committed funding")

    if forced_speculative_reasons:
        return ProfileResult(
            label=GATE_SPECULATIVE,
            raw_score=raw_total,
            total_confidence=conf_total,
            descriptive_band=band,
            passed_gates=[],
            failed_gates=momentum_reasons + quality_reasons + value_reasons,
            overrides=override_id_list,
            warnings=warnings + forced_speculative_reasons,
        )

    passed = []
    if momentum_ok:
        passed.append(GATE_MOMENTUM)
    if quality_ok:
        passed.append(GATE_QUALITY)
    if value_ok:
        passed.append(GATE_VALUE)

    # Reasons for every gate that did NOT pass, kept even when another gate
    # did pass -- an auditor reading `ProfileResult` should be able to see
    # *why* e.g. Quality Opportunity was denied even when Momentum
    # Candidate carried the day.
    not_passed_reasons = (
        (momentum_reasons if not momentum_ok else [])
        + (quality_reasons if not quality_ok else [])
        + (value_reasons if not value_ok else [])
    )

    if passed:
        return ProfileResult(
            label=passed[0],
            raw_score=raw_total,
            total_confidence=conf_total,
            descriptive_band=band,
            passed_gates=passed,
            failed_gates=not_passed_reasons,
            overrides=override_id_list,
            warnings=warnings,
        )

    failed = momentum_reasons + quality_reasons + value_reasons
    if raw_total >= 60:
        return ProfileResult(
            label=GATE_CONDITIONAL,
            raw_score=raw_total,
            total_confidence=conf_total,
            descriptive_band=band,
            passed_gates=[],
            failed_gates=failed,
            overrides=override_id_list,
            warnings=warnings,
        )

    # 50 <= raw_total < 60: see module docstring's documented gap.
    return ProfileResult(
        label=GATE_WEAK,
        raw_score=raw_total,
        total_confidence=conf_total,
        descriptive_band=band,
        passed_gates=[],
        failed_gates=failed,
        overrides=override_id_list,
        warnings=warnings,
    )
