"""Task 20: judgment overlay — lets a Claude sub-agent fill in the
`JudgmentRequest`s the 6 Cerebro specialists (Tasks 15-19) leave
`NOT_SCORABLE` (moat classification, catalyst probability, TAM tier,
thesis killers, ...), then folds the answers back into a rescored
`SpecialistOutput`.

Two entry points:

- `collect_requests(outputs)` gathers every `JudgmentRequest` across the 6
  specialist outputs, for writing `judgment_requests.json` (the file a
  Claude sub-agent reads and answers).
- `merge_overlay(outputs, judgments)` takes those answers back
  (`wbj.schemas.overlay.Judgment`), validates each against its request's
  `schema_hint`, and returns a new list of `SpecialistOutput` with the
  judged metrics scored and `category`/`coverage`/`status` recomputed via
  `wbj.specialists.common.rescore` (Task 14) — never by re-deriving the
  point math here.

Design decisions (documented once here rather than re-litigated per call
site):

1. **Locating the Dimension slot a `JudgmentRequest` feeds — by metric_id,
   via `SpecialistOutput.judgment_slots`.** `wbj.core.scoring.Dimension.
   metric_scores` is a bare `list[(weight, Value)]` — it carries no
   `metric_id`, so there is no way, given only a frozen `Dimension`, to
   recover which slot a given metric occupied. Task 20 therefore reads
   `SpecialistOutput.judgment_slots` (added to `common.py` for this task):
   a `{metric_id -> (dimension_name, slot_index)}` map a specialist
   populates for every judgment-only metric it registered *as a dimension
   member*. `merge_overlay` looks up the answered request's `metric_id`
   there and replaces exactly that `(weight, Value)` slot (same weight)
   with the judged score, then calls `rescore`.

   A judgment whose `metric_id` has no `judgment_slots` entry (a mandatory
   context-only list like business/market's "three thesis killers", or a
   metric a specialist hasn't wired to a slot yet) still updates the flat
   `output.metrics` row (see point 2), but moves no dimension math —
   correctly, since there is no slot for it to move. `financial.py` wires
   its `FIN-GR-004`/`FIN-GR-005` (both `revenue_quality_and_growth`
   members); the other specialists' judgment metrics are either genuinely
   context-only or await a Task 24 dimension slot (reported there).

2. **The flat `metrics` list.** `MetricRow.metric_id` *is* a stable key
   (financial.py's judgment-only rows already follow "request.metric_id ==
   row.metric_id" exactly, e.g. `FIN-GR-004`), so `merge_overlay` always
   replaces the `MetricRow` whose `metric_id` matches the judgment
   request's `metric_id` (creating one if the specialist never registered
   a placeholder row for it), independent of whether a dimension slot was
   found.

3. **Answer -> 0-10 score.** `Judgment.answer` is `float | str | dict`
   (`wbj.schemas.overlay.Answer`). A numeric answer is used directly
   (clamped to [0, 10]). A string answer is scored only when its request's
   `schema_hint` enum is a *known* ordered qualitative ladder
   (`_ENUM_SCORE_TABLE`) — the exact vocabularies the 6 specialists emit,
   each mapped to the 0-10 score its own specialist's methodology implies
   (financial's `BAD/GOOD/EXCELLENT` == band 0/1/2 == 0/5/10; business's
   moat `Wide/Narrow/None` == 10/5/0). This is deliberately *not* a
   positional "first option = best" convention: the two real ladders point
   in opposite directions, so position alone would silently mis-score one
   of them. An enum not in the table, a `dict` answer, or any answer that
   can't be reduced to a single number is *never* scored — per this
   project's own rule (CLAUDE.md: "Una afirmación cualitativa solo puede
   incluirse como contexto; jamás se convierte en score salvo que una regla
   del Cerebro lo defina explícitamente"): the answer is recorded as
   context (rationale/source in the row's warnings and in
   `output.assumptions`), but the metric stays `NOT_SCORABLE` and no
   dimension slot is touched.

4. **Rejection vs. hard error.** An unknown `request_id` (one that
   doesn't correspond to any `JudgmentRequest` across `outputs`) is a
   caller/programming error — `merge_overlay` raises
   `UnknownJudgmentRequestError` for it. A judgment that resolves to a
   real request but is missing `evidence_class`/`source`, or whose answer
   fails its request's `schema_hint` check, is a plausible/expected bad
   sub-agent answer — `merge_overlay` *silently skips* it (that output is
   returned unchanged for that judgment) rather than raising, so one
   malformed answer in a batch doesn't sink the rest.

5. **Output hash.** The brief says "assigns new output hash if the
   envelope carries one." `SpecialistOutput` (Task 14, `common.py`) has no
   hash field today, so this is a no-op — nothing to reassign.

6. **Confidence.** `category.confidence` is left untouched (matching
   `rescore()`'s own documented behavior) since no Cerebro formula defines
   how a judgment shifts category-level confidence. The *per-row*
   `MetricRow.confidence`, however, is **recomputed from the judgment's own
   `evidence_class`** (`_CONFIDENCE_BY_EVIDENCE_CLASS`: R=90, C=80, E=65,
   A=45, Q=30), *not* inherited from the specialist's `NOT_SCORABLE`
   placeholder row — whose confidence is 0.0 ("we knew nothing"), the exact
   opposite of what a sourced, evidence-classed judgment now carries.
"""

from __future__ import annotations

import re
from collections import defaultdict

from wbj.core.nullstates import EvidenceClass, NullState, Value
from wbj.core.scoring import Dimension
from wbj.schemas.overlay import Answer, Judgment
from wbj.specialists.common import JudgmentRequest, MetricRow, SpecialistOutput, rescore

# Per-row confidence derived from a judgment's own evidence class (0-100),
# so an evidenced answer carries real confidence instead of inheriting the
# 0.0 of the specialist's "we couldn't score it" placeholder row. Mirrors
# financial.py's `_confidence_for` ordering (R>C>E>A>Q), tuned down a notch
# since a judgment is an out-of-band estimate, not a mechanically computed
# value.
_CONFIDENCE_BY_EVIDENCE_CLASS: dict[EvidenceClass, float] = {
    EvidenceClass.R: 90.0,
    EvidenceClass.C: 80.0,
    EvidenceClass.E: 65.0,
    EvidenceClass.A: 45.0,
    EvidenceClass.Q: 30.0,
}

# The ordered qualitative ladders the 6 specialists actually emit as
# `schema_hint="one of ..."`, each mapped to the 0-10 score its own
# specialist's methodology implies (documented, not positional guessing --
# see module docstring point 3). Keyed by the frozenset of options so the
# lookup is order-insensitive.
_ENUM_SCORE_TABLE: dict[frozenset[str], dict[str, float]] = {
    frozenset({"BAD", "GOOD", "EXCELLENT"}): {"BAD": 0.0, "GOOD": 5.0, "EXCELLENT": 10.0},
    frozenset({"Wide", "Narrow", "None"}): {"Wide": 10.0, "Narrow": 5.0, "None": 0.0},
}


class UnknownJudgmentRequestError(ValueError):
    """Raised by `merge_overlay` when a `Judgment.request_id` doesn't match
    any `JudgmentRequest` across the given outputs."""

    def __init__(self, request_id: str) -> None:
        super().__init__(f"no JudgmentRequest with request_id={request_id!r} in the given outputs")
        self.request_id = request_id


def collect_requests(outputs: list[SpecialistOutput]) -> list[JudgmentRequest]:
    """Gather every `JudgmentRequest` across the 6 specialist outputs, in
    output order, for writing `judgment_requests.json`."""
    return [req for output in outputs for req in output.judgment_requests]


# --- schema_hint validation & answer -> score -------------------------------

_ENUM_RE = re.compile(r"one of ([A-Za-z0-9_]+(?:\s*\|\s*[A-Za-z0-9_]+)+)", re.IGNORECASE)
_ARRAY_HINT_RE = re.compile(r"\barray of\b", re.IGNORECASE)
# Scalar-number hints only: "integer 1-5", "float 0-10", "number", a bare
# "0-10" range. Deliberately does NOT include "probability" -- that word
# appears *inside* market.py's dict hint "{probability: 0-1, ...}", and a
# dict hint is matched first (see `schema_hint_ok`) so the range/word tokens
# inside its braces never reach this check.
_NUMERIC_HINT_RE = re.compile(r"\binteger\b|\bfloat\b|\bnumber\b|\b\d+\s*-\s*\d+\b", re.IGNORECASE)


def _enum_options(schema_hint: str) -> list[str] | None:
    m = _ENUM_RE.search(schema_hint)
    if not m:
        return None
    return [opt.strip() for opt in m.group(1).split("|")]


def _is_dict_hint(schema_hint: str) -> bool:
    """True for object-shaped hints like market.py's
    `"{probability: 0-1, impact: usd, evidence_quality: 0-1}"` — a brace at
    the start, or a `key: value` colon. (The `array of {...}` hints have no
    colon and are matched by the array branch first, so this never
    misfires on them.)"""
    s = schema_hint.strip()
    return s.startswith("{") or ":" in s


def schema_hint_ok(schema_hint: str, answer: Answer) -> bool:
    """Best-effort structural check that `answer` is shaped the way
    `schema_hint` describes. Deliberately loose (a sanity check on an LLM's
    free-text answer, not a formal schema language). Checked in an order
    that resolves the real-hint ambiguities:

    1. `"array of ..."`      -> `{"items": [...]}`
    2. `"one of A|B|C"`      -> a string that is one of the options
    3. object-shaped (`"{...}"` / contains `:`) -> a `dict`
    4. scalar-number hint    -> a real (non-bool) number
    5. anything else         -> permissive (True)

    The dict check precedes the numeric check so a hint like market.py's
    `"{probability: 0-1, ...}"` — which contains number-range tokens inside
    its braces — is validated as a dict, not mis-forced to a scalar number
    (the bug this ordering fixes).
    """
    if _ARRAY_HINT_RE.search(schema_hint):
        return isinstance(answer, dict) and isinstance(answer.get("items"), list)
    options = _enum_options(schema_hint)
    if options is not None:
        return isinstance(answer, str) and answer.strip() in options
    if _is_dict_hint(schema_hint):
        return isinstance(answer, dict)
    if _NUMERIC_HINT_RE.search(schema_hint):
        return isinstance(answer, (int, float)) and not isinstance(answer, bool)
    return True


def _score_from_answer(schema_hint: str, answer: Answer) -> tuple[float | None, str | None]:
    """Convert a validated answer into a `(score, note)` pair. `score` is
    `None` when the answer can't be reduced to a single 0-10 number (see
    module docstring point 3); `note` documents how the score was derived
    (or why it wasn't), to be recorded on the resulting `MetricRow`.
    """
    if isinstance(answer, bool):  # bool is an int subclass; not a real numeric answer
        return None, None
    if isinstance(answer, (int, float)):
        score = max(0.0, min(10.0, float(answer)))
        return score, f"JUDGMENT_SCORE_FROM_NUMERIC_ANSWER: {answer!r} -> {score:.4f}"
    options = _enum_options(schema_hint)
    if options and isinstance(answer, str) and answer.strip() in options:
        table = _ENUM_SCORE_TABLE.get(frozenset(options))
        if table is not None:
            score = table[answer.strip()]
            return score, (
                f"JUDGMENT_SCORE_FROM_ENUM_LADDER: {answer!r} in {options!r} -> {score:.4f}"
            )
        return None, (
            f"JUDGMENT_ENUM_LADDER_UNKNOWN: {options!r} has no defined 0-10 ordering; "
            "recorded as context only"
        )
    return None, "JUDGMENT_ANSWER_NOT_REDUCIBLE_TO_SCORE: recorded as context only"


# --- merge --------------------------------------------------------------


def merge_overlay(outputs: list[SpecialistOutput], judgments: list[Judgment]) -> list[SpecialistOutput]:
    """Apply `judgments` (Claude sub-agent answers) to `outputs`,
    rescoring each affected specialist via `wbj.specialists.common.rescore`.

    Returns a new list, same order/length as `outputs`; outputs with no
    accepted judgment are returned unchanged (same object). See the module
    docstring for the unknown-request-id / rejection rules.
    """
    requests_by_id: dict[str, tuple[int, JudgmentRequest]] = {}
    for output_index, output in enumerate(outputs):
        for req in output.judgment_requests:
            requests_by_id[req.request_id] = (output_index, req)

    accepted: dict[int, list[tuple[JudgmentRequest, Judgment]]] = defaultdict(list)
    for judgment in judgments:
        if judgment.request_id not in requests_by_id:
            raise UnknownJudgmentRequestError(judgment.request_id)
        if judgment.evidence_class is None or not judgment.source:
            continue  # soft-reject: missing evidence_class/source (see module docstring, point 4)
        output_index, req = requests_by_id[judgment.request_id]
        if not schema_hint_ok(req.schema_hint, judgment.answer):
            continue  # soft-reject: answer doesn't match schema_hint
        accepted[output_index].append((req, judgment))

    new_outputs = list(outputs)
    for output_index, updates in accepted.items():
        new_outputs[output_index] = _apply_updates(outputs[output_index], updates)
    return new_outputs


def _apply_updates(
    output: SpecialistOutput, updates: list[tuple[JudgmentRequest, Judgment]]
) -> SpecialistOutput:
    metrics_by_id: dict[str, MetricRow] = {row.metric_id: row for row in output.metrics}
    dimensions = list(output.dimensions)
    dim_index_by_name = {d.name: i for i, d in enumerate(dimensions)}
    new_assumptions = list(output.assumptions)

    for req, judgment in updates:
        score, note = _score_from_answer(req.schema_hint, judgment.answer)
        confidence = _CONFIDENCE_BY_EVIDENCE_CLASS.get(judgment.evidence_class, 50.0)
        warnings = [w for w in (note,) if w]
        if judgment.rationale:
            warnings.append(f"JUDGMENT_RATIONALE: {judgment.rationale}")

        value = (
            Value.of(
                score,
                unit="score",
                evidence_class=judgment.evidence_class,
                source_name=judgment.source,
                warnings=warnings,
            )
            if score is not None
            else Value.null(
                NullState.NOT_SCORABLE,
                unit="score",
                evidence_class=judgment.evidence_class,
                source_name=judgment.source,
                warnings=warnings,
            )
        )

        existing = metrics_by_id.get(req.metric_id)
        metrics_by_id[req.metric_id] = MetricRow.from_value(
            req.metric_id,
            value,
            formula_id=existing.formula_id if existing else req.metric_id,
            formula_version=existing.formula_version if existing else "judgment-overlay",
            score=score if score is not None else "NOT_SCORABLE",
            confidence=confidence,
            source=judgment.source,
        )

        new_assumptions.append(
            f"judgment {req.request_id!r} (metric={req.metric_id!r}) answered by sub-agent: "
            f"{judgment.answer!r} (evidence_class={judgment.evidence_class}, source={judgment.source!r})"
        )

        # Move the dimension math only when the judgment both scores AND maps
        # to a known dimension slot (SpecialistOutput.judgment_slots).
        slot = output.judgment_slots.get(req.metric_id)
        if score is not None and slot is not None:
            dim_name, slot_index = slot
            di = dim_index_by_name.get(dim_name)
            if di is not None:
                dimensions[di] = _replace_slot(dimensions[di], slot_index, score, judgment)

    merged = rescore(output, dimensions=dimensions, metrics=list(metrics_by_id.values()))
    return merged.model_copy(update={"assumptions": new_assumptions})


def _replace_slot(dimension: Dimension, slot_index: int, score: float, judgment: Judgment) -> Dimension:
    """Replace `dimension.metric_scores[slot_index]` with a scored `Value`,
    keeping the same weight. No-op if `slot_index` is out of range (a stale
    `judgment_slots` entry — defended against rather than trusted blindly).
    """
    scores = list(dimension.metric_scores)
    if not (0 <= slot_index < len(scores)):
        return dimension
    weight, _ = scores[slot_index]
    scores[slot_index] = (
        weight,
        Value.of(
            score,
            unit="score",
            evidence_class=judgment.evidence_class,
            source_name=judgment.source,
        ),
    )
    return dimension.model_copy(update={"metric_scores": scores})
