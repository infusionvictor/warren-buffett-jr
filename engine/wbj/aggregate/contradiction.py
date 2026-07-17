"""Contradiction resolution (Task 21).

Source of truth: `Cerebro/00_main_agent/CONTRADICTION_RESOLUTION.md`'s
"Common combinations" table -- six rows, reproduced verbatim below as the
`combination`/`interpretation` pair each `Contradiction` carries, alongside
this module's own `label` (the "Main-agent label" column).

## Resolution rule 1: never mutates scores

`contradictions()` is a pure read over already-frozen category scores (and,
for row 6, already-frozen valuation numbers) -- it returns descriptive
`Contradiction` objects and touches nothing. Per CONTRADICTION_RESOLUTION.md
rule 1 ("Keep each category score unchanged"), no caller should ever feed
its output back into a score.

## Documented interpretation: "strong" / "weak" / "low"

CONTRADICTION_RESOLUTION.md's table names qualitative levels ("strong
business", "weak technical", "low risk score") without pinning down a
number. This module reads every category's 0-10 `score10` (already the
codebase's shared 0-10 scale: `wbj.core.scoring.Category.score10()`, and
each specialist's own `verdict()` bands consistently treat `>=8.0` as the
top band and `<4.0` as the bottom) and uses `STRONG_THRESHOLD = 7.0`
(comfortably inside every specialist's second-best band) / `WEAK_THRESHOLD
= 4.0` (each specialist's own worst-band boundary) as the two anchors for
"strong" / "weak" respectively. Row 5's "strong total" reads the raw 0-100
total against SCORING_AND_GATES.md's own "Strong raw score" band (80-89.99,
i.e. `raw_total >= 80.0`) rather than inventing a separate number. Both are
this module's own dated (2.0.0) reading, not a value taken directly from
CONTRADICTION_RESOLUTION.md -- flagged here and in the Task 21 commit
message.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "STRONG_THRESHOLD",
    "WEAK_THRESHOLD",
    "STRONG_RAW_TOTAL_THRESHOLD",
    "Contradiction",
    "CategoryScore10s",
    "ReverseDCFContext",
    "contradictions",
]

STRONG_THRESHOLD = 7.0
WEAK_THRESHOLD = 4.0
STRONG_RAW_TOTAL_THRESHOLD = 80.0

# Row 6 ("DCF high, reverse DCF demanding") thresholds: this module's own
# reading, documented for the same reason as STRONG/WEAK above.
_DCF_HIGH_UPSIDE = 0.20  # base-case fair value >=20% above current price
_REVERSE_DCF_DEMANDING_SPREAD = 0.05  # implied growth >=5pp above the reference growth


@dataclass(frozen=True)
class Contradiction:
    """One row of CONTRADICTION_RESOLUTION.md's "Common combinations"
    table, matched against the current scores."""

    combination: str
    interpretation: str
    label: str


@dataclass(frozen=True)
class CategoryScore10s:
    """Each category's 0-10 `score10` -- i.e.
    `wbj.core.scoring.Category.score10()` for each of the six specialists
    (equivalently, `10 * awarded_points / max_points`)."""

    business: float
    financial: float
    market: float
    technical: float
    risk: float
    valuation: float


@dataclass(frozen=True)
class ReverseDCFContext:
    """Optional context for row 6 ("DCF high, reverse DCF demanding").
    All fields default `None`; the row is skipped unless every field this
    row needs is supplied."""

    base_case_upside_pct: float | None = None  # (base_per_share - price) / price
    reverse_dcf_implied_growth: float | None = None
    reference_growth: float | None = None  # consensus growth, or the base-case scenario's own growth assumption

    def dcf_high(self) -> bool:
        return self.base_case_upside_pct is not None and self.base_case_upside_pct >= _DCF_HIGH_UPSIDE

    def reverse_dcf_demanding(self) -> bool:
        if self.reverse_dcf_implied_growth is None or self.reference_growth is None:
            return False
        return self.reverse_dcf_implied_growth - self.reference_growth >= _REVERSE_DCF_DEMANDING_SPREAD


def _strong(x: float) -> bool:
    return x >= STRONG_THRESHOLD


def _weak(x: float) -> bool:
    return x < WEAK_THRESHOLD


def contradictions(
    cats: CategoryScore10s,
    raw_total: float,
    *,
    reverse_dcf: ReverseDCFContext | None = None,
) -> list[Contradiction]:
    """The 6-row lookup table, evaluated against `cats`/`raw_total` (+ the
    optional reverse-DCF context for row 6). Rows are independent -- more
    than one, or none, may match. Never mutates any input."""
    out: list[Contradiction] = []

    if _strong(cats.business) and _weak(cats.technical):
        out.append(
            Contradiction(
                combination="Strong business, weak technical",
                interpretation="Quality may be intact; timing is unconfirmed",
                label="Quality watch / wait for confirmation",
            )
        )

    if _weak(cats.business) and _strong(cats.technical):
        out.append(
            Contradiction(
                combination="Weak business, strong technical",
                interpretation="Price leadership without durable economics",
                label="Speculative momentum only",
            )
        )

    if _strong(cats.valuation) and _weak(cats.technical):
        out.append(
            Contradiction(
                combination="Strong valuation, weak technical",
                interpretation="Possible value trap",
                label="Value watch",
            )
        )

    if _weak(cats.valuation) and _strong(cats.market) and _strong(cats.technical):
        out.append(
            Contradiction(
                combination="Expensive valuation, strong growth and technical",
                interpretation="Premium still validated",
                label="Momentum candidate if gates pass",
            )
        )

    if raw_total >= STRONG_RAW_TOTAL_THRESHOLD and _weak(cats.risk):
        out.append(
            Contradiction(
                combination="Strong total, low risk score",
                interpretation="Aggregate hides survival risk",
                label="Apply risk override",
            )
        )

    if reverse_dcf is not None and reverse_dcf.dcf_high() and reverse_dcf.reverse_dcf_demanding():
        out.append(
            Contradiction(
                combination="DCF high, reverse DCF demanding",
                interpretation="Model assumptions may be optimistic",
                label="Lower valuation confidence",
            )
        )

    return out
