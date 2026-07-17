"""Handoff validation and mandatory overrides (Task 21).

Sources of truth:
- `Cerebro/shared/HANDOFF_CONTRACT.md`: `validate_handoff`'s five rejection
  rules.
- `Cerebro/00_main_agent/SCORING_AND_GATES.md`'s "Mandatory overrides"
  section: the exact 7 override conditions, verbatim.
- `Cerebro/00_main_agent/VALIDATION_TESTS.md` (MAIN-003, MAIN-004,
  MAIN-005, MAIN-006, MAIN-007, MAIN-009, MAIN-010).

## Design: `AggregateInputs`

The task brief's signature is `apply_overrides(outputs) -> list[Override]`,
where "outputs" are "the 6 SpecialistOutputs". Override 7 (a facts-table
conflict) needs `Packet.facts_table` (`dict[str, Value]`), which is not
itself a field of any `SpecialistOutput` -- no specialist's envelope
surfaces "this fact was CONFLICTED" as a distinct top-level signal (e.g.
`ValuationOutput._fact()` silently treats a CONFLICTED `Value` the same as
MISSING: `v.is_valid` is `False` either way). Rather than reach back into
`Packet` (not part of this task's "build on" list, and not itself an
output the main agent aggregates), `AggregateInputs` bundles the 6
required outputs plus an *optional* `facts_table` the caller may pass
through from the frozen `Packet` when available -- `apply_overrides`
degrades gracefully (skips override 7's facts-table check) when it is
omitted. This is a deliberate interface extension beyond the brief's
literal "outputs" wording, documented here and in the Task 21 commit
message rather than silently added.

## Design: overrides reuse each specialist's own mandatory-flag math

Overrides 1, 2, and 3 are conditions the relevant specialist (Task 14
`financial.py`, Task 15 `business.py`, Task 18 `risk.py`) already computes
exactly per its own `DECISION_RULES.md`/`SCORING.md` and records on
`mandatory_flags` (financial's `OVERRIDE_1_LOSS_NEGATIVE_FCF_EXTERNAL_
DEPENDENCE` / `OVERRIDE_2_ROIC_BELOW_WACC`; business's
`VALUE_DESTRUCTION`; risk's `SOLVENCY_WARNING`). Reusing those flags here
-- rather than re-deriving net income/FCF/ROIC/WACC/interest-coverage from
scratch a second time -- keeps one source of truth per condition and
avoids the two layers silently disagreeing. Overrides 4, 5, and 6 are
purely main-agent-level (category-points and coverage thresholds fixed by
`SCORING_AND_GATES.md`, not owned by any one specialist) and are computed
directly here from `category.awarded_points` / `coverage`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from wbj.core.nullstates import NullState, Value
from wbj.core.scoring import COVERAGE_USABLE, Category
from wbj.specialists.business import BusinessOutput
from wbj.specialists.common import SpecialistOutput
from wbj.specialists.financial import FinancialOutput
from wbj.specialists.market import MarketOutput
from wbj.specialists.risk import RiskOutput
from wbj.specialists.technical import TechnicalOutput
from wbj.specialists.valuation import ValuationOutput

__all__ = [
    "AggregateInputs",
    "Override",
    "OVERRIDE_1_CAPITAL_DEPENDENCE",
    "OVERRIDE_2_ROIC_BELOW_WACC",
    "OVERRIDE_3_SOLVENCY_WARNING",
    "OVERRIDE_4_RISK_FLOOR",
    "OVERRIDE_5_PREMIUM_BREAKDOWN",
    "OVERRIDE_6_COVERAGE_GATE_INELIGIBLE",
    "OVERRIDE_7_DATA_CONFLICT_SUPPRESS_PER_SHARE",
    "apply_overrides",
    "validate_handoff",
    "is_handoff_valid",
]

# Materiality thresholds, verbatim from SCORING_AND_GATES.md's "Mandatory
# overrides" list.
RISK_FLOOR_MAX_POINTS = 4.0  # override 4: Risk 0-4/15
VALUATION_BREAKDOWN_MAX_POINTS = 4.0  # override 5: Valuation 0-4/10
TECHNICAL_BREAKDOWN_MAX_POINTS = 8.0  # override 5: Technical 0-8/20
CORE_CATEGORY_COVERAGE_FLOOR = COVERAGE_USABLE  # override 6: 0.70

# Facts-table fields override 7 treats as material, per HANDOFF_CONTRACT.md
# ("a required override flag is omitted... price levels lack touch dates,
# zone bounds..."; SCORING_AND_GATES.md override 7: "share-count, debt,
# cash, or price conflict").
_MATERIAL_FACTS_FIELDS = ("diluted_shares", "total_debt", "cash", "price")

OVERRIDE_1_CAPITAL_DEPENDENCE = "OVERRIDE_1_CAPITAL_DEPENDENCE"
OVERRIDE_2_ROIC_BELOW_WACC = "OVERRIDE_2_ROIC_BELOW_WACC"
OVERRIDE_3_SOLVENCY_WARNING = "OVERRIDE_3_SOLVENCY_WARNING"
OVERRIDE_4_RISK_FLOOR = "OVERRIDE_4_RISK_FLOOR"
OVERRIDE_5_PREMIUM_BREAKDOWN = "OVERRIDE_5_PREMIUM_BREAKDOWN"
OVERRIDE_6_COVERAGE_GATE_INELIGIBLE = "OVERRIDE_6_COVERAGE_GATE_INELIGIBLE"
OVERRIDE_7_DATA_CONFLICT_SUPPRESS_PER_SHARE = "OVERRIDE_7_DATA_CONFLICT_SUPPRESS_PER_SHARE"


@dataclass(frozen=True)
class Override:
    """One triggered mandatory override.

    `effect` is a short machine-readable tag a caller (e.g. `gates.py`) can
    switch on without re-parsing `reason`'s prose.
    """

    id: str
    effect: str
    reason: str


@dataclass(frozen=True)
class AggregateInputs:
    """The 6 frozen `SpecialistOutput`s this task aggregates, plus the
    optional facts-table needed for override 7. See the module docstring's
    "Design: `AggregateInputs`" note."""

    business: BusinessOutput
    financial: FinancialOutput
    market: MarketOutput
    technical: TechnicalOutput
    risk: RiskOutput
    valuation: ValuationOutput
    facts_table: dict[str, Value] = field(default_factory=dict)

    def by_category(self) -> dict[str, SpecialistOutput]:
        """`{category_name: output}` in `CATEGORY_WEIGHTS` key order."""
        return {
            "business": self.business,
            "financial": self.financial,
            "market": self.market,
            "technical": self.technical,
            "risk": self.risk,
            "valuation": self.valuation,
        }

    def __iter__(self) -> Iterator[SpecialistOutput]:
        return iter(self.by_category().values())


# ============================================================================
# validate_handoff (HANDOFF_CONTRACT.md)
# ============================================================================


def validate_handoff(output: SpecialistOutput) -> list[str]:
    """Return the list of handoff-rejection reasons for `output` (empty if
    the packet is acceptable), per `HANDOFF_CONTRACT.md`'s five rules:

    1. category points do not reproduce from dimension scores;
    2. a score lacks a formula ID;
    3. the knowledge timestamp is absent;
    4. confidence and/or coverage are absent;
    5. (technical only) price levels lack touch dates, zone bounds, or ATR
       distance.

    `Zone.lower/center/upper` are non-optional pydantic fields, so "zone
    bounds" are always structurally present once a `Zone` exists; this
    function's zone check instead enforces the touch-date and ATR-distance
    parts of rule 5 for every zone whose status implies it has qualifying
    touches (`confirmed`/`strong`/`role_reversed` -- a bare `candidate`
    zone is not yet claiming to be touch-confirmed).
    """
    reasons: list[str] = []

    if output.dimensions:
        recomputed = Category(
            name=output.agent_id, max_points=output.category.max_points, dimensions=output.dimensions
        ).points()
        awarded = output.category.awarded_points
        if awarded is None or abs(recomputed - awarded) > 1e-6:
            reasons.append(
                "CATEGORY_POINTS_DO_NOT_REPRODUCE: "
                f"dimensions recompute to {recomputed!r}, envelope claims awarded_points={awarded!r}"
            )
    else:
        reasons.append("NO_DIMENSIONS_TO_VALIDATE_CATEGORY_POINTS")

    for m in output.metrics:
        if not m.formula_id:
            reasons.append(f"METRIC_MISSING_FORMULA_ID: {m.metric_id!r}")

    if not output.knowledge_timestamp:
        reasons.append("MISSING_KNOWLEDGE_TIMESTAMP")

    if output.category.confidence is None:
        reasons.append("MISSING_CONFIDENCE")

    if output.coverage is None:
        reasons.append("MISSING_COVERAGE")

    if isinstance(output, TechnicalOutput):
        reasons.extend(_validate_levels(output))

    return reasons


def _validate_levels(output: TechnicalOutput) -> list[str]:
    reasons: list[str] = []
    zones = [*output.important_levels.nearest_support, *output.important_levels.nearest_resistance]
    for zone in zones:
        if zone.status not in ("confirmed", "strong", "role_reversed"):
            continue
        if not zone.touches:
            reasons.append(f"ZONE_MISSING_TOUCHES: {zone.zone_id!r} status={zone.status!r} claims confirmation but has no touches")
            continue
        for t in zone.touches:
            if not t.date:
                reasons.append(f"ZONE_TOUCH_MISSING_DATE: {zone.zone_id!r}")
        if zone.distance_atr is None:
            reasons.append(f"ZONE_MISSING_ATR_DISTANCE: {zone.zone_id!r}")
    return reasons


def is_handoff_valid(output: SpecialistOutput) -> bool:
    """`True` iff `validate_handoff(output)` is empty."""
    return not validate_handoff(output)


# ============================================================================
# apply_overrides (SCORING_AND_GATES.md "Mandatory overrides")
# ============================================================================


def apply_overrides(inputs: AggregateInputs) -> list[Override]:
    """All 7 mandatory overrides, evaluated against the 6 frozen
    `SpecialistOutput`s (+ optional facts-table). Returns only the
    overrides that actually trigger, in SCORING_AND_GATES.md's numbered
    order."""
    out: list[Override] = []

    # --- Override 1: capital dependence -> caps Avoid/Speculative ---
    fin_flags = set(inputs.financial.mandatory_flags)
    if "OVERRIDE_1_LOSS_NEGATIVE_FCF_EXTERNAL_DEPENDENCE" in fin_flags:
        out.append(
            Override(
                id=OVERRIDE_1_CAPITAL_DEPENDENCE,
                effect="CAP_AVOID_SPECULATIVE",
                reason=(
                    "Net loss + negative FCF + dependence on external capital "
                    "(financial_analysis OVERRIDE_1_LOSS_NEGATIVE_FCF_EXTERNAL_DEPENDENCE): "
                    "final profile capped at Avoid/Speculative."
                ),
            )
        )

    # --- Override 2: ROIC < WACC -> no Elite/Quality/Excellent-business ---
    biz_flags = set(inputs.business.mandatory_flags)
    roic_below_wacc = (
        "OVERRIDE_2_ROIC_BELOW_WACC" in fin_flags or "VALUE_DESTRUCTION" in biz_flags
    )
    if roic_below_wacc:
        out.append(
            Override(
                id=OVERRIDE_2_ROIC_BELOW_WACC,
                effect="NO_ELITE_QUALITY",
                reason=(
                    "ROIC below WACC (financial_analysis OVERRIDE_2_ROIC_BELOW_WACC "
                    f"and/or business_analysis VALUE_DESTRUCTION): Elite, Quality Opportunity, "
                    "or Excellent-business classification is unavailable."
                ),
            )
        )

    # --- Override 3: interest coverage < 1.5x -> solvency warning, always ---
    risk_flags = set(inputs.risk.mandatory_flags)
    solvency_warning = "SOLVENCY_WARNING" in risk_flags or any(
        "SOLVENCY_WARNING" in m.warnings for m in inputs.financial.metrics
    )
    if solvency_warning:
        out.append(
            Override(
                id=OVERRIDE_3_SOLVENCY_WARNING,
                effect="PROMINENT_WARNING",
                reason="Interest coverage below 1.5x: solvency warning must appear prominently in every report.",
            )
        )

    # --- Override 4: Risk 0-4/15 -> caps Speculative ---
    risk_points = inputs.risk.category.awarded_points
    if risk_points is not None and risk_points <= RISK_FLOOR_MAX_POINTS:
        out.append(
            Override(
                id=OVERRIDE_4_RISK_FLOOR,
                effect="CAP_SPECULATIVE",
                reason=f"Risk category awarded_points={risk_points!r} <= {RISK_FLOOR_MAX_POINTS}/15: profile capped at Speculative.",
            )
        )

    # --- Override 5: Valuation 0-4/10 AND Technical 0-8/20 -> Wait/Avoid ---
    val_points = inputs.valuation.category.awarded_points
    tech_points = inputs.technical.category.awarded_points
    if (
        val_points is not None
        and tech_points is not None
        and val_points <= VALUATION_BREAKDOWN_MAX_POINTS
        and tech_points <= TECHNICAL_BREAKDOWN_MAX_POINTS
    ):
        out.append(
            Override(
                id=OVERRIDE_5_PREMIUM_BREAKDOWN,
                effect="WAIT_AVOID",
                reason=(
                    f"Valuation={val_points!r} <= {VALUATION_BREAKDOWN_MAX_POINTS}/10 and "
                    f"Technical={tech_points!r} <= {TECHNICAL_BREAKDOWN_MAX_POINTS}/20: profile becomes Wait/Avoid."
                ),
            )
        )

    # --- Override 6: any core category coverage < 0.70 -> gate-ineligible ---
    below_coverage = [
        name
        for name, output in inputs.by_category().items()
        if output.coverage is None or output.coverage < CORE_CATEGORY_COVERAGE_FLOOR
    ]
    if below_coverage:
        out.append(
            Override(
                id=OVERRIDE_6_COVERAGE_GATE_INELIGIBLE,
                effect="GATE_INELIGIBLE",
                reason=(
                    f"Category coverage below {CORE_CATEGORY_COVERAGE_FLOOR:.0%} for: "
                    f"{', '.join(below_coverage)}. No profile gate may pass."
                ),
            )
        )

    # --- Override 7: unresolved material facts-table conflict -> suppress per-share valuation ---
    conflicted_fields = [
        field_name
        for field_name in _MATERIAL_FACTS_FIELDS
        if (v := inputs.facts_table.get(field_name)) is not None and v.state == NullState.CONFLICTED
    ]
    if not conflicted_fields:
        # Fall back to scanning the valuation specialist's own metric rows,
        # in case a caller didn't pass `facts_table` through.
        conflicted_fields = sorted(
            {m.metric_id for m in inputs.valuation.metrics if m.state == NullState.CONFLICTED}
        )
    if conflicted_fields:
        out.append(
            Override(
                id=OVERRIDE_7_DATA_CONFLICT_SUPPRESS_PER_SHARE,
                effect="SUPPRESS_PER_SHARE",
                reason=(
                    f"Unresolved material data conflict in: {', '.join(conflicted_fields)}. "
                    "Per-share valuation publication is suppressed until resolved."
                ),
            )
        )

    return out
