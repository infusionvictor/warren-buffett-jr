"""Tests for the prediction-tracking memory loop."""

import json
from datetime import date

from wbj.memoria import evaluate, load_predictions, save_prediction, track

TARGETS = {
    "status": "ok", "price": 100.0, "growth_base": 0.15, "pe_now": 25.0,
    "horizon": "12 meses",
    "scenarios": [
        {"key": "bear", "target": 80.0, "upside": -0.20, "assumptions": "x"},
        {"key": "base", "target": 120.0, "upside": 0.20, "assumptions": "x"},
        {"key": "bull", "target": 150.0, "upside": 0.50, "assumptions": "x"},
    ],
}
SCORECARD = {"overall_10": 7.5, "evidence_points_covered": 50}


def test_save_and_load_roundtrip(tmp_path):
    p = save_prediction(tmp_path, "nvda", date(2026, 1, 15), SCORECARD, TARGETS)
    assert p is not None and p.name == "prediccion.json"
    preds = load_predictions(tmp_path)
    assert len(preds) == 1
    assert preds[0]["ticker"] == "NVDA" and preds[0]["base"] == 120.0
    assert preds[0]["score10"] == 7.5


def test_save_skips_not_scorable(tmp_path):
    assert save_prediction(tmp_path, "X", date(2026, 1, 1), SCORECARD,
                           {"status": "not_scorable"}) is None


def test_evaluate_in_progress_deviation():
    pred = {"ticker": "T", "date": "2026-01-01", "price": 100.0,
            "bear": 80.0, "base": 120.0, "bull": 150.0, "horizon_days": 365}
    # ~half the horizon elapsed; base implies +20%/yr -> prorated ~ +10%
    r = evaluate(pred, price_now=115.0, today=date(2026, 7, 2))
    assert r["outcome"] == "en_curso" and not r["mature"]
    assert abs(r["base_prorated"] - 0.20 * (182 / 365)) < 1e-3
    assert abs(r["deviation"] - (0.15 - 0.20 * 182 / 365)) < 1e-3


def test_evaluate_mature_outcomes():
    pred = {"ticker": "T", "date": "2025-01-01", "price": 100.0,
            "bear": 80.0, "base": 120.0, "bull": 150.0, "horizon_days": 365}
    today = date(2026, 7, 1)
    assert evaluate(pred, 130.0, today)["outcome"] == "en_rango"
    assert evaluate(pred, 160.0, today)["outcome"] == "supero_bull"
    assert evaluate(pred, 70.0, today)["outcome"] == "rompio_bear"


def test_track_writes_calibracion(tmp_path):
    reports, memoria = tmp_path / "Reportes", tmp_path / "Memoria"
    save_prediction(reports, "AAA", date(2025, 1, 1), SCORECARD, TARGETS)
    save_prediction(reports, "BBB", date(2026, 6, 1), SCORECARD, TARGETS)
    prices = {"AAA": 130.0, "BBB": 105.0}
    s = track(reports, memoria, lambda t: prices.get(t), today=date(2026, 7, 17))
    assert s["total"] == 2 and s["maduras"] == 1
    assert s["hit_rate"] == 1.0  # AAA at 130 is inside [80, 150]
    text = (memoria / "calibracion.md").read_text(encoding="utf-8")
    assert "AAA" in text and "en rango" in text and "en curso" in text


def test_track_skips_tickers_without_price(tmp_path):
    reports, memoria = tmp_path / "Reportes", tmp_path / "Memoria"
    save_prediction(reports, "AAA", date(2026, 1, 1), SCORECARD, TARGETS)
    s = track(reports, memoria, lambda t: None, today=date(2026, 7, 17))
    assert s["total"] == 0 and s["hit_rate"] is None


def test_load_ignores_malformed(tmp_path):
    d = tmp_path / "XXX" / "2026-01-01"
    d.mkdir(parents=True)
    (d / "prediccion.json").write_text("{corrupted")
    assert load_predictions(tmp_path) == []
