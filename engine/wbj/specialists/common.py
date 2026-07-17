"""Common `SpecialistOutput` envelope shared by every Cerebro specialist
(Tasks 14-19).

Sources of truth (Cerebro/):
- `shared/OUTPUT_CONTRACT.md`: the metric-row field contract every category
  agent must satisfy (`metric_id, value, unit, period, formula, score,
  evidence_class, source, confidence, warnings`) and the narrative rule.
- `shared/HANDOFF_CONTRACT.md`: the top-level specialist -> main-agent
  envelope (`agent_id, version, ticker/security, knowledge_timestamp,
  status, category, coverage, dimensions, metrics, mandatory_flags,
  assumptions, source_lineage, validation_tests`) and the main agent's
  handoff-rejection rules this envelope must satisfy by construction.
- `02_financial_analysis/OUTPUT_SCHEMA.md` (representative of every
  `0N_*_analysis/OUTPUT_SCHEMA.md`): the exact top-level shape implemented
  here; every specialist's `OUTPUT_SCHEMA.md` repeats this same envelope
  verbatim before adding its own category-specific extension fields.

Design decisions Tasks 15-19 inherit from this module (documented once
here, rather than re-litigated per specialist):

1. `MetricRow` flattens `OUTPUT_CONTRACT.md`'s ten required columns onto
   one model rather than nesting a `Value` inside a row. Every wbj
   computation already produces a `wbj.core.nullstates.Value`, so
   `MetricRow.from_value` is the one place that maps a `Value` plus the
   score/confidence bookkeeping a bare `Value` doesn't carry onto a
   contract-shaped row. `value`/`state` mirror `Value`'s own "exactly one
   of" invariant, so a row can never claim both a number and a null
   reason (or neither).
2. `status` is `Literal["COMPLETE", "INCOMPLETE", "ERROR"]`. Every Cerebro
   `OUTPUT_SCHEMA.md` shows only the happy-path `status: COMPLETE` in its
   example; `INCOMPLETE`/`ERROR` are this module's own (documented)
   extension of that literal, needed because a real packet's
   `Category.coverage()` is not always >= 0.85
   (`MISSING_DATA_POLICY.md`'s "complete" threshold). `status_from_coverage`
   is the one place that maps a coverage ratio to a status so every
   specialist's `run()` applies the same bands.
3. `rescore()` is a thin wrapper around `wbj.core.scoring.Category`, not a
   parallel implementation of the point math. Task 20 (judgment overlay)
   rebuilds each affected `Dimension.metric_scores` tuple list once
   `JudgmentRequest` answers are merged in, then calls
   `rescore(output, dimensions=updated_dimensions)` to regenerate
   `category`/`coverage`/`status` rather than recomputing
   `Category.points()`/`score10()`/`coverage()` itself — one source of
   truth (Task 4's `wbj.core.scoring.Category`) for the point math, reused
   here and by every specialist.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wbj.core.nullstates import EvidenceClass, NullState, Value
from wbj.core.scoring import COVERAGE_COMPLETE, Category, Dimension

VERSION = "2.0.0"

Status = Literal["COMPLETE", "INCOMPLETE", "ERROR"]


class SecurityRef(BaseModel):
    """`envelope.security` — per HANDOFF_CONTRACT.md / OUTPUT_SCHEMA.md."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    exchange: str
    currency: str


class CategoryStats(BaseModel):
    """`envelope.category` — point maximum, awarded points, 0-10 score, and
    confidence."""

    model_config = ConfigDict(frozen=True)

    max_points: float
    awarded_points: float | None = None
    score_10: float | None = None
    confidence: float | None = None


class ValidationTestsSummary(BaseModel):
    """`envelope.validation_tests` — pass/fail/warning counts from the
    specialist's own internal self-checks (handoff-shape sanity checks,
    e.g. "category points reproduce from dimension scores" per
    HANDOFF_CONTRACT.md's rejection rules) — distinct from, and much
    smaller than, the specialist's own pytest `VALIDATION_TESTS.md` suite."""

    model_config = ConfigDict(frozen=True)

    passed: int = 0
    failed: int = 0
    warnings: int = 0


class MetricRow(BaseModel):
    """One `OUTPUT_CONTRACT.md`-shaped metric row.

    Exactly one of `value`/`state` is set, mirroring `Value`. `score` is a
    0-10 float or the literal string `"NOT_SCORABLE"`, per
    OUTPUT_CONTRACT.md's `score: 0-10 or NOT_SCORABLE`.

    `evidence_class` is `None` on a null row (`state` set, e.g.
    MISSING/NOT_SCORABLE): a value that doesn't exist has no provenance
    class to report, and `MetricRow.from_value` copies it straight from the
    source `Value` (which is itself `None` for a null). Downstream
    consumers (and Tasks 15-19) should read `evidence_class` as "the
    provenance of this row's number, if it has one" and not expect an R/C/E/
    A/Q on a MISSING/NOT_SCORABLE row.
    """

    model_config = ConfigDict(frozen=True)

    metric_id: str
    value: float | None = None
    state: NullState | None = None
    unit: str = ""
    period: str | None = None
    formula_id: str
    formula_version: str
    score: float | Literal["NOT_SCORABLE"]
    evidence_class: EvidenceClass | None = None
    source: str | None = None
    confidence: float
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _exactly_one_of_value_or_state(self) -> "MetricRow":
        if (self.value is None) == (self.state is None):
            raise ValueError(
                "MetricRow requires exactly one of `value` or `state`, "
                f"got value={self.value!r} state={self.state!r}"
            )
        return self

    @classmethod
    def from_value(
        cls,
        metric_id: str,
        v: Value,
        *,
        formula_id: str,
        formula_version: str,
        score: float | Literal["NOT_SCORABLE"],
        confidence: float,
        source: str | None = None,
    ) -> "MetricRow":
        """Build a `MetricRow` from a computed `Value` plus the score/
        confidence bookkeeping a bare `Value` doesn't carry.

        `source` defaults to the `Value`'s own lineage (`source_name` or
        `source_locator`) when not given explicitly.
        """
        return cls(
            metric_id=metric_id,
            value=v.value,
            state=v.state,
            unit=v.unit,
            period=v.period,
            formula_id=formula_id,
            formula_version=formula_version,
            score=score,
            evidence_class=v.evidence_class,
            source=source if source is not None else (v.source_name or v.source_locator),
            confidence=confidence,
            warnings=list(v.warnings),
        )


class JudgmentRequest(BaseModel):
    """A question this specialist cannot answer mechanically, deferred to
    the Task 20 judgment overlay rather than guessed at — per Cerebro's
    no-speculation rule (e.g. `02_financial_analysis/AGENT.md`: "A claim
    that cannot be tied to a formula result, reported evidence, or
    explicitly disclosed assumption is context only and receives no
    score").
    """

    model_config = ConfigDict(frozen=True)

    request_id: str
    agent_id: str
    metric_id: str
    question: str
    schema_hint: str


class SpecialistOutput(BaseModel):
    """The shared specialist -> main-agent envelope (`HANDOFF_CONTRACT.md` /
    every `0N_*_analysis/OUTPUT_SCHEMA.md`'s common prefix). Each
    specialist's own output model subclasses this and adds its
    category-specific extension fields (e.g. `financial.py`'s
    `core_27_metrics`).

    `verdict` is the category's qualitative label (e.g. financial's
    "Excellent financial health" / ... / "Weak / high financial risk"
    band). It is the ONLY place a mandatory override may cap the outcome:
    `category.awarded_points`/`score_10` must always reproduce from
    `dimensions` (HANDOFF_CONTRACT.md rejects a packet whose category
    points don't, with no exception for overrides), so an override that
    "caps the verdict at Bad/Avoid" or "prevents an Excellent verdict"
    lowers this label (and records a `mandatory_flags` entry), never the
    points. Tasks 15-19 must follow the same discipline.

    `judgment_slots` maps a `JudgmentRequest.metric_id` to the exact
    `(dimension_name, slot_index)` of the `NOT_SCORABLE`
    `Dimension.metric_scores` entry that request feeds, when there is one.
    It exists because `wbj.core.scoring.Dimension.metric_scores` is a bare
    `list[(weight, Value)]` carrying no `metric_id` — so, given only the
    frozen envelope, there is otherwise no way for the Task 20 judgment
    overlay to know which dimension slot a judged answer should replace and
    rescore. A specialist that registers a judgment-only metric *as a
    dimension member* (e.g. `financial.py`'s `FIN-GR-004`/`FIN-GR-005`,
    members of `revenue_quality_and_growth` via its `_DIMENSION_MEMBERS`
    table) populates this map so `wbj.overlay.merge.merge_overlay` can move
    `category.awarded_points`/`coverage`, not just the flat `metrics` row.
    A judgment metric with no dimension slot (a mandatory context-only list
    like "three thesis killers", or a metric a specialist hasn't wired to a
    slot yet) simply has no entry here and is applied to the flat `metrics`
    row only. Defaults empty; entirely backward compatible for any output
    that doesn't set it.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    agent_id: str
    version: str = VERSION
    status: Status
    security: SecurityRef
    knowledge_timestamp: str
    category: CategoryStats
    verdict: str | None = None
    coverage: float | None = None
    dimensions: list[Dimension] = Field(default_factory=list)
    metrics: list[MetricRow] = Field(default_factory=list)
    mandatory_flags: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    judgment_requests: list[JudgmentRequest] = Field(default_factory=list)
    judgment_slots: dict[str, tuple[str, int]] = Field(default_factory=dict)
    source_lineage: list[str] = Field(default_factory=list)
    validation_tests: ValidationTestsSummary = Field(default_factory=ValidationTestsSummary)


def apply_dimension_cap(
    metric_scores: list[tuple[float, Value]], *, cap: float
) -> list[tuple[float, Value]]:
    """Scale every valid `Value` in `metric_scores` by a common factor so
    the resulting weighted mean is `min(original_weighted_mean, cap)`.

    The single shared implementation of `SCORING.md`'s dimension-level
    "Gate / cap" column (e.g. business's "moat capped at 6 without positive
    spread", market's "narrative-only catalyst capped at 3", technical's
    volume/SMA200 caps, valuation's low-confidence MOS cap). Applied
    directly to the numbers `Category(dimensions)` reproduces from, so the
    cap is part of the deterministic point math -- distinct from a
    `capped_verdict`-style label-only override (see
    `SpecialistOutput.verdict`). A no-op when the dimension is not scorable,
    empty, or already at/below `cap`. Pure and stateless: every specialist
    that has a numeric cap imports this one definition (risk_analysis has
    none, so it does not).
    """
    total = sum(w for w, _ in metric_scores)
    valid = sum(w for w, v in metric_scores if v.is_valid)
    if valid <= 0 or total <= 0:
        return metric_scores
    weighted_mean = sum(w * v.value for w, v in metric_scores if v.is_valid) / valid
    if weighted_mean <= cap:
        return metric_scores
    factor = cap / weighted_mean
    out: list[tuple[float, Value]] = []
    for w, v in metric_scores:
        if v.is_valid:
            out.append((w, Value.of(v.value * factor, unit=v.unit, evidence_class=v.evidence_class, warnings=list(v.warnings))))
        else:
            out.append((w, v))
    return out


def status_from_coverage(coverage: float) -> Status:
    """Map a `Category.coverage()` ratio to an envelope `status`.

    Per `MISSING_DATA_POLICY.md`'s coverage bands: `>=0.85` complete,
    `[0.70, 0.85)` usable-with-caveat, `<0.70` incomplete. This module
    additionally treats exactly-zero coverage (no scorable dimension at
    all) as `ERROR` rather than `INCOMPLETE`, since there is nothing
    usable to hand off — a documented extension of Cerebro's two-band
    policy, needed because `HANDOFF_CONTRACT.md`'s three-status literal
    isn't itself pinned to coverage numbers.
    """
    if coverage <= 0:
        return "ERROR"
    if coverage >= COVERAGE_COMPLETE:
        return "COMPLETE"
    return "INCOMPLETE"


def rescore(
    output: SpecialistOutput,
    *,
    dimensions: list[Dimension] | None = None,
    metrics: list[MetricRow] | None = None,
) -> SpecialistOutput:
    """Recompute `category`/`coverage`/`status` from (possibly
    judgment-updated) dimensions, returning a new `SpecialistOutput`.

    Reuses `wbj.core.scoring.Category`'s point math (Task 4) rather than
    re-deriving it: Task 20's judgment overlay merges `JudgmentRequest`
    answers into updated `Dimension.metric_scores` tuples, then calls this
    once to regenerate the envelope's category totals. `confidence` is
    left untouched (judgment answers change *what* was scored, not how
    much we trust the underlying evidence) unless the caller updates it
    separately on the returned output.

    `metrics`, if given, replaces `output.metrics` (e.g. once a
    `JudgmentRequest` row's placeholder `NOT_SCORABLE` score is replaced
    with the merged answer) but is not required — a caller that only
    changed dimension weights/scores may omit it.
    """
    dims = dimensions if dimensions is not None else output.dimensions
    cat = Category(name=output.agent_id, max_points=output.category.max_points, dimensions=dims)
    coverage = cat.coverage()
    new_category = CategoryStats(
        max_points=output.category.max_points,
        awarded_points=cat.points(),
        score_10=cat.score10(),
        confidence=output.category.confidence,
    )
    update: dict[str, Any] = {
        "dimensions": dims,
        "category": new_category,
        "coverage": coverage,
        "status": status_from_coverage(coverage),
    }
    if metrics is not None:
        update["metrics"] = metrics
    return output.model_copy(update=update)
