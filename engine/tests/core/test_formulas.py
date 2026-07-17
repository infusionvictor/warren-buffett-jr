import pytest

from wbj.core.formulas import (
    REGISTRY,
    annualize_vol,
    avg_balance,
    cagr,
    register_formula,
    run_formula,
    winsorize,
    yoy,
)
from wbj.core.nullstates import EvidenceClass, NullState, Value


# --- brief Step 1 tests (verbatim) ---


def test_register_and_run():
    @register_formula(id="TST-001", version="2.0.0", unit="pct", inputs=["a", "b"])
    def growth(a: float, b: float) -> float:
        return a / b - 1

    r = run_formula("TST-001", {"a": 110.0, "b": 100.0})
    assert r.status == "OK" and abs(r.result.value - 0.10) < 1e-12
    assert r.formula_id == "TST-001" and r.formula_version == "2.0.0"


def test_missing_input_returns_null_not_crash():
    r = run_formula("TST-001", {"a": 110.0, "b": Value.null(NullState.MISSING)})
    assert r.status == "NULL" and r.result.state == NullState.MISSING


def test_cagr_rejects_sign_change():
    r = cagr(end=50.0, begin=-10.0, years=3)
    assert r.is_null and r.state == NullState.NOT_MEANINGFUL


def test_yoy():
    assert abs(yoy(120.0, 100.0).value - 0.20) < 1e-12


# --- supplementary coverage ---


def test_register_formula_stores_in_registry():
    @register_formula(id="TST-002", version="1.0.0", unit="usd", inputs=["x"])
    def identity(x: float) -> float:
        return x

    assert "TST-002" in REGISTRY
    f = REGISTRY["TST-002"]
    assert f.id == "TST-002" and f.version == "1.0.0" and f.unit == "usd"
    assert f.inputs == ["x"] and f.min_history == 0
    assert f.missing_behavior == NullState.MISSING


def test_run_formula_unknown_id_raises_keyerror():
    with pytest.raises(KeyError):
        run_formula("NOPE-999", {})


def test_run_formula_missing_key_entirely_uses_missing_behavior():
    @register_formula(
        id="TST-003",
        version="1.0.0",
        unit="usd",
        inputs=["a"],
        missing_behavior=NullState.NOT_APPLICABLE,
    )
    def f(a: float) -> float:
        return a

    r = run_formula("TST-003", {})
    assert r.status == "NULL" and r.result.state == NullState.NOT_APPLICABLE


def test_run_formula_result_has_evidence_class_c():
    r = run_formula("TST-001", {"a": 110.0, "b": 100.0})
    assert r.result.evidence_class == EvidenceClass.C
    assert r.result.unit == "pct"


def test_run_formula_catches_exception_and_returns_error():
    @register_formula(id="TST-004", version="1.0.0", unit="usd", inputs=["a", "b"])
    def divide(a: float, b: float) -> float:
        return a / b

    r = run_formula("TST-004", {"a": 1.0, "b": 0.0})
    assert r.status == "ERROR"
    assert r.warnings and len(r.warnings) >= 1


def test_run_formula_propagates_conflicted_null_state():
    @register_formula(id="TST-005", version="1.0.0", unit="usd", inputs=["a"])
    def f(a: float) -> float:
        return a

    r = run_formula("TST-005", {"a": Value.null(NullState.CONFLICTED)})
    assert r.status == "NULL" and r.result.state == NullState.CONFLICTED


def test_yoy_zero_prior_is_not_meaningful():
    r = yoy(50.0, 0.0)
    assert r.is_null and r.state == NullState.NOT_MEANINGFUL


def test_yoy_accepts_value_inputs_and_propagates_null():
    r = yoy(Value.null(NullState.STALE), Value.of(100.0, unit="usd"))
    assert r.is_null and r.state == NullState.STALE


def test_cagr_basic():
    r = cagr(end=133.1, begin=100.0, years=3)
    assert r.is_valid and abs(r.value - 0.10) < 1e-9


def test_cagr_rejects_nonpositive_begin():
    r = cagr(end=10.0, begin=0.0, years=2)
    assert r.is_null and r.state == NullState.NOT_MEANINGFUL


def test_cagr_rejects_negative_end_with_positive_begin():
    r = cagr(end=-5.0, begin=10.0, years=2)
    assert r.is_null and r.state == NullState.NOT_MEANINGFUL


def test_avg_balance_averages_when_both_present():
    r = avg_balance(100.0, 200.0)
    assert r.is_valid and abs(r.value - 150.0) < 1e-12
    assert "END_BALANCE_PROXY" not in r.warnings


def test_avg_balance_falls_back_to_end_with_proxy_warning_when_begin_missing():
    r = avg_balance(None, 200.0)
    assert r.is_valid and r.value == 200.0
    assert "END_BALANCE_PROXY" in r.warnings


def test_avg_balance_falls_back_when_begin_is_null_value():
    r = avg_balance(Value.null(NullState.MISSING), 200.0)
    assert r.is_valid and r.value == 200.0
    assert "END_BALANCE_PROXY" in r.warnings


def test_annualize_vol_multiplies_by_sqrt_252():
    r = annualize_vol(0.01)
    assert r.is_valid and abs(r.value - (0.01 * (252**0.5))) < 1e-12


def test_annualize_vol_propagates_null():
    r = annualize_vol(Value.null(NullState.MISSING))
    assert r.is_null and r.state == NullState.MISSING


def test_winsorize_clamps_outliers_and_preserves_raw_in_audit_trail():
    series = [1.0] + [50.0] * 18 + [1000.0]  # 20 points, extreme low/high tails
    out = winsorize(series, 0.05, 0.95)
    assert len(out) == len(series)
    assert all(v.is_valid for v in out)
    # low outlier clamped up, high outlier clamped down
    assert out[0].value > 1.0
    assert out[-1].value < 1000.0
    assert "WINSORIZED" in out[0].warnings
    assert "raw=1.0" in (out[0].source_locator or "")
    # interior values untouched
    assert out[1].value == 50.0
    assert "WINSORIZED" not in out[1].warnings


def test_winsorize_passes_through_null_values():
    series = [10.0, Value.null(NullState.MISSING), 30.0]
    out = winsorize(series, 0.05, 0.95)
    assert out[1].is_null and out[1].state == NullState.MISSING
