"""`FinalReport` pydantic schema (Task 21).

Source of truth: `Cerebro/00_main_agent/FINAL_REPORT_SCHEMA.md`.

## Documented discrepancy: `executive_thesis` has 5 keys, but 7 sentences

FINAL_REPORT_SCHEMA.md's YAML stub names exactly 5 `executive_thesis`
keys (`business_quality`, `growth_engine`, `market_validation`,
`valuation_message`, `primary_risk`), but the same document's "Required
executive-summary sentences" section lists 7 distinct sentence topics
(economic function, value-creation durability, growth funding, market
validation, price-implied assumptions, nearest support/resistance +
intrinsic-value references, and the primary invalidation risk). The task
brief itself says `executive_thesis (7 sentences)`. This module keeps all
5 of the schema's named keys (mapped onto the topics they clearly name --
sentences 1, 3, 4, 5, 7) and adds two more fields,
`value_creation_durability` (sentence 2) and `key_levels_summary`
(sentence 6), so `ExecutiveThesis` is a strict superset of the schema's
named shape that also satisfies the "7 sentences" requirement. Flagged
here and in the Task 21 commit message per the task instructions ("Cerebro
wins on conflicts; note discrepancies").

## `build_final_report`: narrative sentences are supplied, not generated

`executive_thesis`, `thesis_killers`' free-text framing, and
`monitoring_triggers` are narrative prose. Per this project's own
non-negotiable rule ("Sin evidencia, no hay número... Sin fórmula, no hay
conclusión" -- `CLAUDE.md`) and the existing codebase's judgment-overlay
precedent (Task 20: a specialist defers anything it "cannot answer
mechanically" rather than guessing), `build_final_report` accepts a
caller-supplied `ExecutiveThesis` rather than fabricating sentences from
the numeric outputs -- composing investment narrative from frozen scores
is the report-renderer's job (Task 23), not this task's. Every other
`FinalReport` field *is* mechanically assembled here from the 6
`SpecialistOutput`s, `wbj.aggregate.gates.ProfileResult`,
`wbj.aggregate.contradiction.Contradiction`, and
`wbj.aggregate.synthesis.LevelSynthesis` -- the Task 21 deliverables.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wbj.aggregate.contradiction import Contradiction
from wbj.aggregate.gates import ProfileResult
from wbj.aggregate.overrides import AggregateInputs
from wbj.aggregate.synthesis import LevelReference, LevelSynthesis
from wbj.specialists.valuation import ReverseDCFSummary as _ValuationReverseDCFSummary
from wbj.specialists.valuation import ScenarioSummary

__all__ = [
    "REPORT_VERSION",
    "ReportSecurity",
    "ReportProfile",
    "CategoryScorecardEntry",
    "CategoryScorecard",
    "ExecutiveThesis",
    "AuditSummary",
    "FinalReport",
    "build_final_report",
]

REPORT_VERSION = "2.0.0"


class ReportSecurity(BaseModel):
    """`report.security` (FINAL_REPORT_SCHEMA.md, verbatim)."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    exchange: str
    currency: str
    analysis_timestamp: str
    knowledge_timestamp: str


class ReportProfile(BaseModel):
    """`report.profile` (FINAL_REPORT_SCHEMA.md, verbatim)."""

    model_config = ConfigDict(frozen=True)

    label: str
    raw_score: float
    total_confidence: float
    passed_gates: list[str] = Field(default_factory=list)
    failed_gates: list[str] = Field(default_factory=list)
    overrides: list[str] = Field(default_factory=list)

    @classmethod
    def from_profile_result(cls, result: ProfileResult) -> "ReportProfile":
        return cls(
            label=result.label,
            raw_score=result.raw_score,
            total_confidence=result.total_confidence,
            passed_gates=list(result.passed_gates),
            failed_gates=list(result.failed_gates),
            overrides=list(result.overrides),
        )


class CategoryScorecardEntry(BaseModel):
    """One row of `report.category_scorecard` (FINAL_REPORT_SCHEMA.md,
    verbatim: `{points, max, confidence}`)."""

    model_config = ConfigDict(frozen=True)

    points: float | None
    max: float
    confidence: float | None


class CategoryScorecard(BaseModel):
    """`report.category_scorecard` -- the fixed six categories and their
    maxima, per `SCORING_AND_GATES.md`'s "Fixed category maximums"."""

    model_config = ConfigDict(frozen=True)

    business: CategoryScorecardEntry
    financial: CategoryScorecardEntry
    market: CategoryScorecardEntry
    technical: CategoryScorecardEntry
    risk: CategoryScorecardEntry
    valuation: CategoryScorecardEntry

    @classmethod
    def from_aggregate_inputs(cls, inputs: AggregateInputs) -> "CategoryScorecard":
        by_cat = inputs.by_category()
        return cls(
            **{
                name: CategoryScorecardEntry(
                    points=output.category.awarded_points,
                    max=output.category.max_points,
                    confidence=output.category.confidence,
                )
                for name, output in by_cat.items()
            }
        )


class ExecutiveThesis(BaseModel):
    """The 7 required executive-summary sentences (FINAL_REPORT_SCHEMA.md
    "Required executive-summary sentences"). See the module docstring's
    discrepancy note for why this has 7 fields where the schema's own
    `executive_thesis` stub names only 5."""

    model_config = ConfigDict(frozen=True)

    business_quality: str  # 1. what the company economically does
    value_creation_durability: str  # 2. why value creation is or is not durable
    growth_engine: str  # 3. what is funding growth
    market_validation: str  # 4. whether the market currently validates the thesis
    valuation_message: str  # 5. what assumptions the current price appears to require
    key_levels_summary: str  # 6. nearest support/resistance and intrinsic-value references
    primary_risk: str  # 7. the single most important invalidation risk

    def as_sentences(self) -> list[str]:
        """The 7 sentences, in FINAL_REPORT_SCHEMA.md's numbered order."""
        return [
            self.business_quality,
            self.value_creation_durability,
            self.growth_engine,
            self.market_validation,
            self.valuation_message,
            self.key_levels_summary,
            self.primary_risk,
        ]


class AuditSummary(BaseModel):
    """`report.audit` (FINAL_REPORT_SCHEMA.md, verbatim)."""

    model_config = ConfigDict(frozen=True)

    packet_hashes: dict[str, str] = Field(default_factory=dict)
    formula_versions: list[str] = Field(default_factory=list)
    validation_summary: dict[str, Any] = Field(default_factory=dict)


class FinalReport(BaseModel):
    """The full `FINAL_REPORT_SCHEMA.md` document."""

    model_config = ConfigDict(frozen=True)

    report_version: str = REPORT_VERSION
    security: ReportSecurity
    profile: ReportProfile
    category_scorecard: CategoryScorecard
    executive_thesis: ExecutiveThesis
    important_levels: list[LevelReference] = Field(default_factory=list)
    valuation_scenarios: list[ScenarioSummary] = Field(default_factory=list)
    reverse_dcf: _ValuationReverseDCFSummary = Field(default_factory=_ValuationReverseDCFSummary)
    thesis_killers: list[str] = Field(default_factory=list)
    monitoring_triggers: list[str] = Field(default_factory=list)
    missing_or_conflicted_data: list[str] = Field(default_factory=list)
    audit: AuditSummary = Field(default_factory=AuditSummary)

    @model_validator(mode="after")
    def _report_version_pinned(self) -> "FinalReport":
        if self.report_version != REPORT_VERSION:
            raise ValueError(f"FinalReport.report_version must be {REPORT_VERSION!r}, got {self.report_version!r}")
        return self


def _collect_thesis_killers(inputs: AggregateInputs) -> list[str]:
    """Merges every specialist's own thesis-killer list (business's
    `three_thesis_killers`, market's `three_growth_thesis_killers`, and
    risk's `thesis_killers` -- a `list[dict]`, reduced to its `description`/
    `killer`/first-string-value per entry) into one deduplicated list,
    preserving order."""
    out: list[str] = []
    for k in inputs.business.three_thesis_killers:
        if k not in out:
            out.append(k)
    for k in inputs.market.three_growth_thesis_killers:
        if k not in out:
            out.append(k)
    for entry in inputs.risk.thesis_killers:
        text = entry.get("description") or entry.get("killer") or entry.get("name")
        if text is None:
            text = next((str(v) for v in entry.values() if isinstance(v, str)), None)
        if text and text not in out:
            out.append(text)
    return out


def build_final_report(
    *,
    inputs: AggregateInputs,
    profile: ProfileResult,
    contradictions: list[Contradiction],
    levels: LevelSynthesis,
    executive_thesis: ExecutiveThesis,
    exchange: str,
    currency: str,
    analysis_timestamp: str,
    packet_hashes: dict[str, str] | None = None,
    formula_versions: list[str] | None = None,
) -> FinalReport:
    """Mechanically assemble a `FinalReport` from the Task 21 building
    blocks (`AggregateInputs`, a `ProfileResult`, `contradictions()`'s
    output, and a `LevelSynthesis`) plus narrative prose the caller
    supplies. See the module docstring's "narrative sentences are
    supplied, not generated" note for why `executive_thesis` is a required
    argument rather than derived here.
    """
    security = inputs.business.security
    knowledge_timestamp = max(
        (o.knowledge_timestamp for o in inputs if o.knowledge_timestamp), default=analysis_timestamp
    )

    missing_or_conflicted: list[str] = []
    for o in _overrides_for_report(profile):
        missing_or_conflicted.append(o)
    for name, output in inputs.by_category().items():
        for flag in output.mandatory_flags:
            entry = f"{name}: {flag}"
            if entry not in missing_or_conflicted:
                missing_or_conflicted.append(entry)

    monitoring_triggers = [c.label for c in contradictions]

    return FinalReport(
        security=ReportSecurity(
            ticker=security.ticker,
            exchange=exchange,
            currency=currency,
            analysis_timestamp=analysis_timestamp,
            knowledge_timestamp=knowledge_timestamp,
        ),
        profile=ReportProfile.from_profile_result(profile),
        category_scorecard=CategoryScorecard.from_aggregate_inputs(inputs),
        executive_thesis=executive_thesis,
        important_levels=list(levels.levels),
        valuation_scenarios=list(inputs.valuation.scenarios),
        reverse_dcf=inputs.valuation.reverse_dcf,
        thesis_killers=_collect_thesis_killers(inputs),
        monitoring_triggers=monitoring_triggers,
        missing_or_conflicted_data=missing_or_conflicted,
        audit=AuditSummary(
            packet_hashes=packet_hashes or {},
            formula_versions=formula_versions or [],
            validation_summary={
                "passed_gates": list(profile.passed_gates),
                "failed_gates": list(profile.failed_gates),
                "overrides": list(profile.overrides),
                "warnings": list(profile.warnings),
            },
        ),
    )


def _overrides_for_report(profile: ProfileResult) -> list[str]:
    return list(profile.warnings)
