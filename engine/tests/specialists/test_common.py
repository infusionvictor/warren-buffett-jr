"""Tests for `wbj.specialists.common` (Task 14): the shared
`SpecialistOutput` envelope, `MetricRow`, `JudgmentRequest`, and `rescore`.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from wbj.core.nullstates import EvidenceClass, NullState, Value
from wbj.core.scoring import Category, Dimension
from wbj.specialists.common import (
    VERSION,
    CategoryStats,
    JudgmentRequest,
    MetricRow,
    SecurityRef,
    SpecialistOutput,
    ValidationTestsSummary,
    apply_dimension_cap,
    rescore,
    status_from_coverage,
)


# ============================================================================
# apply_dimension_cap: the single shared SCORING.md "Gate / cap" mechanism
# (business/market/technical/valuation all import this one definition).
# ============================================================================


def test_apply_dimension_cap_scales_valid_scores_to_hit_cap_exactly():
    scores = [(0.5, Value.of(10.0, unit="score")), (0.5, Value.of(8.0, unit="score"))]
    capped = apply_dimension_cap(scores, cap=6.0)
    weighted = sum(w * v.value for w, v in capped if v.is_valid)
    total_w = sum(w for w, v in capped if v.is_valid)
    assert weighted / total_w == pytest.approx(6.0)


def test_apply_dimension_cap_noop_when_already_below_cap():
    scores = [(1.0, Value.of(3.0, unit="score"))]
    capped = apply_dimension_cap(scores, cap=6.0)
    assert capped[0][1].value == pytest.approx(3.0)


def test_apply_dimension_cap_noop_when_nothing_valid():
    """Every member NOT_SCORABLE -> no valid weight to scale, returned
    unchanged (the honest 'missing evidence is never neutral' path -- the
    cap never fabricates a score)."""
    scores = [(0.5, Value.null(NullState.NOT_SCORABLE, unit="score")), (0.5, Value.null(NullState.NOT_SCORABLE, unit="score"))]
    capped = apply_dimension_cap(scores, cap=5.0)
    assert capped == scores


def test_apply_dimension_cap_preserves_null_members_and_warnings():
    scores = [
        (0.5, Value.of(10.0, unit="score", warnings=["W"])),
        (0.5, Value.null(NullState.NOT_SCORABLE, unit="score")),
    ]
    capped = apply_dimension_cap(scores, cap=4.0)
    assert capped[0][1].value == pytest.approx(4.0)   # scaled to the cap (only valid member)
    assert "W" in capped[0][1].warnings                # warnings carried through
    assert capped[1][1].is_null                        # null member untouched


def _security() -> SecurityRef:
    return SecurityRef(ticker="NVDA", exchange="NASDAQ", currency="USD")


# --- MetricRow ---------------------------------------------------------


def test_metric_row_from_value_valid():
    v = Value.of(0.12, unit="pct", period="FY2025", evidence_class=EvidenceClass.C, source_name="FMP")
    row = MetricRow.from_value(
        "FIN-GR-001", v, formula_id="FIN-GR-001", formula_version="2.0.0", score=10.0, confidence=90.0
    )
    assert row.value == 0.12
    assert row.state is None
    assert row.unit == "pct"
    assert row.period == "FY2025"
    assert row.score == 10.0
    assert row.evidence_class == EvidenceClass.C
    assert row.source == "FMP"
    assert row.confidence == 90.0
    assert row.warnings == []


def test_metric_row_from_value_null_carries_state_and_not_scorable():
    v = Value.null(NullState.MISSING, unit="pct", warnings=["NO_DATA"])
    row = MetricRow.from_value(
        "FIN-BS-020", v, formula_id="FIN-BS-020", formula_version="2.0.0", score="NOT_SCORABLE", confidence=0.0
    )
    assert row.value is None
    assert row.state == NullState.MISSING
    assert row.score == "NOT_SCORABLE"
    assert row.warnings == ["NO_DATA"]


def test_metric_row_rejects_both_value_and_state():
    with pytest.raises(ValidationError):
        MetricRow(
            metric_id="x",
            value=1.0,
            state=NullState.MISSING,
            formula_id="x",
            formula_version="1",
            score=5.0,
            confidence=50.0,
        )


def test_metric_row_rejects_neither_value_nor_state():
    with pytest.raises(ValidationError):
        MetricRow(metric_id="x", formula_id="x", formula_version="1", score=5.0, confidence=50.0)


def test_metric_row_source_defaults_to_value_lineage():
    v = Value.of(1.0, unit="x", source_locator="10-K:item8")
    row = MetricRow.from_value("m", v, formula_id="m", formula_version="1", score=5.0, confidence=50.0)
    assert row.source == "10-K:item8"


def test_metric_row_explicit_source_overrides_value_lineage():
    v = Value.of(1.0, unit="x", source_name="FMP")
    row = MetricRow.from_value(
        "m", v, formula_id="m", formula_version="1", score=5.0, confidence=50.0, source="override"
    )
    assert row.source == "override"


# --- status_from_coverage ------------------------------------------------


def test_status_from_coverage_bands():
    assert status_from_coverage(0.0) == "ERROR"
    assert status_from_coverage(0.5) == "INCOMPLETE"
    assert status_from_coverage(0.70) == "INCOMPLETE"
    assert status_from_coverage(0.849) == "INCOMPLETE"
    assert status_from_coverage(0.85) == "COMPLETE"
    assert status_from_coverage(1.0) == "COMPLETE"


# --- SpecialistOutput envelope shape -------------------------------------


def test_specialist_output_minimal_construction():
    out = SpecialistOutput(
        agent_id="financial_analysis",
        status="COMPLETE",
        security=_security(),
        knowledge_timestamp="2026-07-16T21:00:00+00:00",
        category=CategoryStats(max_points=15.0, awarded_points=10.0, score_10=6.67, confidence=88.0),
        coverage=0.9,
    )
    assert out.version == VERSION
    assert out.dimensions == []
    assert out.metrics == []
    assert out.judgment_requests == []
    assert out.validation_tests == ValidationTestsSummary(passed=0, failed=0, warnings=0)


def test_judgment_request_shape():
    jr = JudgmentRequest(
        request_id="jr-1",
        agent_id="financial_analysis",
        metric_id="FIN-GR-004",
        question="Classify organic growth quality given the reconciliation bridge.",
        schema_hint="one of BAD|GOOD|EXCELLENT",
    )
    assert jr.metric_id == "FIN-GR-004"


# --- rescore ---------------------------------------------------------


def _base_output(dims: list[Dimension]) -> SpecialistOutput:
    cat = Category(name="financial_analysis", max_points=15.0, dimensions=dims)
    return SpecialistOutput(
        agent_id="financial_analysis",
        status=status_from_coverage(cat.coverage()),
        security=_security(),
        knowledge_timestamp="2026-07-16T21:00:00+00:00",
        category=CategoryStats(
            max_points=15.0, awarded_points=cat.points(), score_10=cat.score10(), confidence=80.0
        ),
        coverage=cat.coverage(),
        dimensions=dims,
    )


def test_rescore_recomputes_category_from_updated_dimensions():
    d1 = Dimension(name="a", max_points=3.0, metric_scores=[(1.0, Value.of(5.0, unit="score"))])
    out = _base_output([d1])
    assert out.category.awarded_points == pytest.approx(1.5)  # 3 * 5/10

    d1_updated = Dimension(name="a", max_points=3.0, metric_scores=[(1.0, Value.of(10.0, unit="score"))])
    rescored = rescore(out, dimensions=[d1_updated])

    assert rescored.category.awarded_points == pytest.approx(3.0)
    # category max_points is 15 (financial's full category size) but only
    # one 3-point dimension is registered here, so score_10 = 10*3/15 = 2.0
    assert rescored.category.score_10 == pytest.approx(2.0)
    assert rescored.coverage == pytest.approx(1.0)
    assert rescored.dimensions == [d1_updated]
    # original is untouched (frozen model, new instance returned)
    assert out.category.awarded_points == pytest.approx(1.5)


def test_rescore_updates_status_when_coverage_crosses_band():
    d1 = Dimension(
        name="a",
        max_points=3.0,
        metric_scores=[
            (0.5, Value.null(NullState.MISSING)),
            (0.5, Value.of(8.0, unit="score")),
        ],
    )  # 50% valid -> NOT_SCORABLE dimension -> coverage 0.5 -> INCOMPLETE
    out = _base_output([d1])
    assert out.status == "INCOMPLETE"

    d1_complete = Dimension(name="a", max_points=3.0, metric_scores=[(1.0, Value.of(8.0, unit="score"))])
    rescored = rescore(out, dimensions=[d1_complete])
    assert rescored.status == "COMPLETE"


def test_rescore_replaces_metrics_when_given():
    d1 = Dimension(name="a", max_points=3.0, metric_scores=[(1.0, Value.of(5.0, unit="score"))])
    out = _base_output([d1])
    v = Value.of(1.0, unit="x")
    new_row = MetricRow.from_value("m", v, formula_id="m", formula_version="1", score=5.0, confidence=50.0)
    rescored = rescore(out, metrics=[new_row])
    assert rescored.metrics == [new_row]
    assert out.metrics == []


def test_rescore_without_dimensions_or_metrics_keeps_existing_and_recomputes():
    d1 = Dimension(name="a", max_points=3.0, metric_scores=[(1.0, Value.of(5.0, unit="score"))])
    out = _base_output([d1])
    rescored = rescore(out)
    assert rescored.dimensions == out.dimensions
    assert rescored.category.awarded_points == pytest.approx(out.category.awarded_points)
