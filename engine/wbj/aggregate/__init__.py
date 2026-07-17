"""Aggregation (Task 21): overrides, gates, contradictions, and price-level
synthesis -- the main-agent scoring logic layered on top of the 6 frozen
Cerebro specialist outputs.

Sources of truth (`Cerebro/`): `00_main_agent/SCORING_AND_GATES.md`,
`00_main_agent/CONTRADICTION_RESOLUTION.md`,
`00_main_agent/PRICE_LEVEL_SYNTHESIS.md`, `shared/HANDOFF_CONTRACT.md`.
See each submodule's docstring for the specific rules it implements and
any documented discrepancy/interpretation.
"""

from __future__ import annotations

from wbj.aggregate.contradiction import (
    Contradiction,
    CategoryScore10s,
    ReverseDCFContext,
    contradictions,
)
from wbj.aggregate.gates import (
    CategoryConfidences,
    CategoryPoints,
    GATE_AVOID,
    GATE_CONDITIONAL,
    GATE_MOMENTUM,
    GATE_QUALITY,
    GATE_SPECULATIVE,
    GATE_VALUE,
    GATE_WEAK,
    ProfileResult,
    apply_gates,
    descriptive_band,
    raw_total,
    total_confidence,
)
from wbj.aggregate.overrides import (
    AggregateInputs,
    Override,
    apply_overrides,
    is_handoff_valid,
    validate_handoff,
)
from wbj.aggregate.synthesis import (
    ConfluenceZone,
    LevelReference,
    LevelSynthesis,
    confluence_tolerance,
    synthesize_levels,
)

__all__ = [
    # overrides.py
    "AggregateInputs",
    "Override",
    "apply_overrides",
    "validate_handoff",
    "is_handoff_valid",
    # gates.py
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
    # contradiction.py
    "Contradiction",
    "CategoryScore10s",
    "ReverseDCFContext",
    "contradictions",
    # synthesis.py
    "LevelReference",
    "ConfluenceZone",
    "LevelSynthesis",
    "confluence_tolerance",
    "synthesize_levels",
]
