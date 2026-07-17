"""Formula registry and calculation-convention helpers.

Every formula that computes a metric for the wbj compute engine registers
itself here via `register_formula`, and is invoked through `run_formula`,
which produces a Cerebro-shaped result object, unwraps `Value` inputs,
propagates null states without crashing, and never lets a raw exception
escape (it is converted to a `status="ERROR"` result instead).

Sources of truth (Cerebro/shared/):
- FORMULA_REGISTRY.md: formula-result object shape, registration
  requirements (stable id, version, inputs/units, missing-data behavior).
- CALCULATION_CONVENTIONS.md: denominators, growth rates, annualization,
  winsorization.

Discrepancy vs. Cerebro (documented per task instructions, also noted in
the commit message): FORMULA_REGISTRY.md's example result object uses a
single `status: VALID` value. The wbj Task 3 brief instead specifies
`status: Literal["OK", "NULL", "ERROR"]` as an interface later tasks
depend on by exact name, so this module implements the brief's three-way
status rather than Cerebro's single `VALID` sentinel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from wbj.core.nullstates import EvidenceClass, NullState, Value


@dataclass(frozen=True)
class Formula:
    """A registered formula: metadata plus the callable that computes it."""

    id: str
    version: str
    unit: str
    inputs: list[str]
    min_history: int
    missing_behavior: NullState
    fn: Callable[..., float]


REGISTRY: dict[str, Formula] = {}


class FormulaResult(BaseModel):
    """Cerebro-shaped result of running a registered formula."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    formula_id: str
    formula_version: str
    inputs: dict[str, object]
    result: Value
    status: Literal["OK", "NULL", "ERROR"]
    warnings: list[str] = []


def register_formula(
    *,
    id: str,
    version: str,
    unit: str,
    inputs: list[str],
    min_history: int = 0,
    missing_behavior: NullState = NullState.MISSING,
) -> Callable[[Callable[..., float]], Callable[..., float]]:
    """Decorator that registers `fn` into `REGISTRY` under `id`.

    The decorated function is returned unchanged (still directly callable);
    it is looked up and invoked through `run_formula`.
    """

    def decorator(fn: Callable[..., float]) -> Callable[..., float]:
        REGISTRY[id] = Formula(
            id=id,
            version=version,
            unit=unit,
            inputs=list(inputs),
            min_history=min_history,
            missing_behavior=missing_behavior,
            fn=fn,
        )
        return fn

    return decorator


def run_formula(formula_id: str, data: dict[str, "Value | object"]) -> FormulaResult:
    """Look up `formula_id` in `REGISTRY` and run it against `data`.

    - Any declared input missing from `data` entirely -> NULL result using
      the formula's `missing_behavior`.
    - Any declared input present as a null `Value` -> NULL result carrying
      that same null state (the input's own state wins over
      `missing_behavior`, since the caller already told us why).
    - Any exception raised by the formula body -> ERROR result, exception
      message captured in `warnings`.
    - A successful numeric return is wrapped in `Value.of(..., unit=formula.unit,
      evidence_class=EvidenceClass.C)`.
    """
    if formula_id not in REGISTRY:
        raise KeyError(f"No formula registered with id {formula_id!r}")
    formula = REGISTRY[formula_id]

    inputs_snapshot = {name: data[name] for name in formula.inputs if name in data}
    warnings: list[str] = []

    kwargs: dict[str, float] = {}
    for name in formula.inputs:
        if name not in data:
            return FormulaResult(
                formula_id=formula.id,
                formula_version=formula.version,
                inputs=inputs_snapshot,
                result=Value.null(formula.missing_behavior, unit=formula.unit),
                status="NULL",
                warnings=warnings,
            )
        raw = data[name]
        if isinstance(raw, Value):
            if raw.is_null:
                return FormulaResult(
                    formula_id=formula.id,
                    formula_version=formula.version,
                    inputs=inputs_snapshot,
                    result=Value.null(raw.state, unit=formula.unit),
                    status="NULL",
                    warnings=warnings,
                )
            kwargs[name] = raw.value
        else:
            kwargs[name] = raw

    try:
        raw_result = formula.fn(**kwargs)
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any formula-body error -> ERROR result
        return FormulaResult(
            formula_id=formula.id,
            formula_version=formula.version,
            inputs=inputs_snapshot,
            result=Value.null(NullState.NOT_SCORABLE, unit=formula.unit),
            status="ERROR",
            warnings=[f"{type(exc).__name__}: {exc}"],
        )

    return FormulaResult(
        formula_id=formula.id,
        formula_version=formula.version,
        inputs=inputs_snapshot,
        result=Value.of(raw_result, unit=formula.unit, evidence_class=EvidenceClass.C),
        status="OK",
        warnings=warnings,
    )


# --- Calculation-convention helpers (Cerebro/shared/CALCULATION_CONVENTIONS.md) ---
#
# Each helper accepts either a plain float or a `Value` for its numeric
# arguments (unwrapping and propagating null states as needed) and always
# returns a `Value`.


def _unwrap(x: float | Value, unit: str = "") -> Value:
    return x if isinstance(x, Value) else Value.of(x, unit=unit)


def yoy(cur: float | Value, prior: float | Value) -> Value:
    """Year-over-year growth: (Current / Prior) - 1.

    Propagates null states from either input. A zero (immaterial) prior is
    `NOT_MEANINGFUL` per CALCULATION_CONVENTIONS.md's denominator rule.
    """
    cur_v, prior_v = _unwrap(cur), _unwrap(prior)
    if cur_v.is_null:
        return cur_v
    if prior_v.is_null:
        return prior_v
    if prior_v.value == 0:
        return Value.null(
            NullState.NOT_MEANINGFUL,
            unit="pct",
            warnings=["YOY_ZERO_DENOMINATOR"],
        )
    return Value.of(cur_v.value / prior_v.value - 1, unit="pct", evidence_class=EvidenceClass.C)


def cagr(end: float | Value, begin: float | Value, years: float) -> Value:
    """Compound annual growth rate: (End / Begin)^(1/years) - 1.

    Per CALCULATION_CONVENTIONS.md: "Do not calculate CAGR across a sign
    change or from a non-positive beginning value." Both cases return
    `NOT_MEANINGFUL` with an explanatory warning instead of raising.
    """
    end_v, begin_v = _unwrap(end), _unwrap(begin)
    if end_v.is_null:
        return end_v
    if begin_v.is_null:
        return begin_v
    if begin_v.value <= 0:
        return Value.null(
            NullState.NOT_MEANINGFUL,
            unit="pct",
            warnings=["CAGR_NONPOSITIVE_BEGIN"],
        )
    if end_v.value < 0:
        return Value.null(
            NullState.NOT_MEANINGFUL,
            unit="pct",
            warnings=["CAGR_SIGN_CHANGE"],
        )
    if years <= 0:
        return Value.null(
            NullState.NOT_MEANINGFUL,
            unit="pct",
            warnings=["CAGR_NONPOSITIVE_YEARS"],
        )
    ratio = end_v.value / begin_v.value
    return Value.of(ratio ** (1.0 / years) - 1, unit="pct", evidence_class=EvidenceClass.C)


def avg_balance(begin: float | Value | None, end: float | Value) -> Value:
    """Average of beginning and ending balance-sheet values.

    Per CALCULATION_CONVENTIONS.md: "If only ending values exist, label the
    result END_BALANCE_PROXY and reduce confidence." Falls back to `end`
    (tagged with the `END_BALANCE_PROXY` warning) when `begin` is missing
    (`None`) or is itself a null `Value`.
    """
    end_v = _unwrap(end)
    if end_v.is_null:
        return end_v
    if begin is None:
        return Value.of(
            end_v.value,
            unit=end_v.unit,
            warnings=["END_BALANCE_PROXY"],
            evidence_class=EvidenceClass.C,
        )
    begin_v = _unwrap(begin)
    if begin_v.is_null:
        return Value.of(
            end_v.value,
            unit=end_v.unit,
            warnings=["END_BALANCE_PROXY"],
            evidence_class=EvidenceClass.C,
        )
    return Value.of(
        (begin_v.value + end_v.value) / 2,
        unit=end_v.unit,
        evidence_class=EvidenceClass.C,
    )


def annualize_vol(daily_std: float | Value) -> Value:
    """Annualize a daily volatility figure by scaling by sqrt(252).

    Per CALCULATION_CONVENTIONS.md's Annualization section.
    """
    v = _unwrap(daily_std)
    if v.is_null:
        return v
    return Value.of(v.value * math.sqrt(252), unit=v.unit, evidence_class=EvidenceClass.C)


def winsorize(
    series: Sequence[float | Value],
    lower: float = 0.05,
    upper: float = 0.95,
) -> list[Value]:
    """Winsorize a peer-distribution series at the given percentiles.

    Per CALCULATION_CONVENTIONS.md: "Peer distributions may be winsorized
    at the 5th and 95th percentiles for robust ranking. The raw value must
    remain in the audit trail." Null entries pass through unchanged and are
    excluded from the percentile computation. Clamped entries keep the raw
    value in `source_locator` (`"raw=<value>"`) and carry a `"WINSORIZED"`
    warning; untouched entries are returned as plain valid `Value`s.
    """
    unwrapped = [x if isinstance(x, Value) else Value.of(x, unit="") for x in series]
    numeric = [v.value for v in unwrapped if v.is_valid]

    if not numeric:
        return list(unwrapped)

    lower_bound = float(np.percentile(numeric, lower * 100))
    upper_bound = float(np.percentile(numeric, upper * 100))

    out: list[Value] = []
    for v in unwrapped:
        if v.is_null:
            out.append(v)
            continue
        raw = v.value
        clamped = min(max(raw, lower_bound), upper_bound)
        if clamped != raw:
            out.append(
                Value.of(
                    clamped,
                    unit=v.unit,
                    evidence_class=EvidenceClass.C,
                    source_locator=f"raw={raw}",
                    warnings=["WINSORIZED"],
                )
            )
        else:
            out.append(Value.of(clamped, unit=v.unit, evidence_class=EvidenceClass.C))
    return out
