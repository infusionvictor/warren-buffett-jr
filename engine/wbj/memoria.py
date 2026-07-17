"""Agent memory: persist predictions and evaluate them against reality.

The honest learning loop (Cerebro: BACKTESTING_AND_CALIBRATION.md):

1. Every analysis SAVES its prediction (price, targets, score, date) to
   Reportes/<TICKER>/<date>/prediccion.json — the seed of memory.
2. `wbj track` HARVESTS: compares every saved prediction against today's
   price, measures bias, and writes Memoria/calibracion.md.
3. The orchestrator (Claude) reads Memoria/ before each analysis and
   records qualitative lessons after it — code handles numbers, the
   agent handles judgment.

No prediction is ever edited or deleted: misses are the learning signal.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

MATURITY_DAYS = 365  # targets are 12-month — only then is a hit/miss final


def save_prediction(reports_dir: Path, ticker: str, day: date,
                    scorecard: dict, targets: dict) -> Path | None:
    """Persist one analysis' numeric prediction. Returns path or None."""
    if targets.get("status") != "ok":
        return None
    out_dir = reports_dir / ticker.upper() / day.isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    by = {s["key"]: s["target"] for s in targets["scenarios"]}
    record = {
        "ticker": ticker.upper(),
        "date": day.isoformat(),
        "price": targets["price"],
        "bear": by["bear"], "base": by["base"], "bull": by["bull"],
        "growth_base": targets["growth_base"],
        "pe_now": targets["pe_now"],
        "score10": scorecard.get("overall_10"),
        "evidence_points": scorecard.get("evidence_points_covered"),
        "horizon_days": MATURITY_DAYS,
    }
    path = out_dir / "prediccion.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def load_predictions(reports_dir: Path) -> list[dict]:
    """All saved predictions, oldest first. Ignores malformed files."""
    preds = []
    if not reports_dir.exists():
        return preds
    for p in sorted(reports_dir.glob("*/*/prediccion.json")):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
            if all(k in rec for k in ("ticker", "date", "price", "bear", "base", "bull")):
                preds.append(rec)
        except (json.JSONDecodeError, OSError):
            continue
    return preds


def evaluate(pred: dict, price_now: float, today: date) -> dict:
    """Judge one prediction against the current price.

    - deviation: actual return so far MINUS the pro-rated base-scenario
      return (positive = the stock is running ahead of our base case).
    - mature (>= horizon): final verdict — inside [bear, bull] = "en rango".
    """
    d0 = date.fromisoformat(pred["date"])
    days = (today - d0).days
    p0 = pred["price"]
    change = price_now / p0 - 1
    base_ret = pred["base"] / p0 - 1
    prorated = base_ret * min(days, MATURITY_DAYS) / MATURITY_DAYS
    mature = days >= pred.get("horizon_days", MATURITY_DAYS)

    if mature:
        if price_now > pred["bull"]:
            outcome = "supero_bull"
        elif price_now < pred["bear"]:
            outcome = "rompio_bear"
        else:
            outcome = "en_rango"
    else:
        outcome = "en_curso"

    return {
        "ticker": pred["ticker"], "date": pred["date"], "days": days,
        "price_then": p0, "price_now": round(price_now, 2),
        "change": round(change, 4), "base_prorated": round(prorated, 4),
        "deviation": round(change - prorated, 4),
        "mature": mature, "outcome": outcome,
        "bear": pred["bear"], "base": pred["base"], "bull": pred["bull"],
    }


def track(reports_dir: Path, memoria_dir: Path, price_fn, today: date) -> dict:
    """Evaluate every saved prediction and write Memoria/calibracion.md.

    price_fn(ticker) -> float | None (injected so tests stay offline).
    """
    rows = []
    for pred in load_predictions(reports_dir):
        price_now = price_fn(pred["ticker"])
        if price_now is None:
            continue
        rows.append(evaluate(pred, price_now, today))

    mature = [r for r in rows if r["mature"]]
    in_range = [r for r in mature if r["outcome"] == "en_rango"]
    summary = {
        "as_of": today.isoformat(),
        "total": len(rows),
        "maduras": len(mature),
        "en_rango": len(in_range),
        "hit_rate": round(len(in_range) / len(mature), 3) if mature else None,
        "sesgo_medio": round(sum(r["deviation"] for r in rows) / len(rows), 4) if rows else None,
        "rows": rows,
    }

    memoria_dir.mkdir(parents=True, exist_ok=True)
    (memoria_dir / "calibracion.md").write_text(_render(summary), encoding="utf-8")
    return summary


def _render(s: dict) -> str:
    lines = [
        "# Calibración del agente",
        "",
        f"**Actualizado:** {s['as_of']} · generado por `wbj track` — no editar a mano.",
        "",
        f"- Predicciones registradas: **{s['total']}** ({s['maduras']} maduras ≥12 meses)",
    ]
    if s["hit_rate"] is not None:
        lines.append(f"- Acierto en rango Bear–Bull (maduras): **{s['hit_rate']:.0%}**")
    if s["sesgo_medio"] is not None:
        adj = ("optimista — los targets Medio van por encima de la realidad"
               if s["sesgo_medio"] < 0 else
               "conservador — la realidad va por encima del escenario Medio")
        lines.append(f"- Sesgo medio vs escenario Medio (prorrateado): **{s['sesgo_medio']:+.1%}** ({adj})")
    lines += ["", "| Ticker | Fecha | Días | Precio→Hoy | Real | Medio esperado | Desvío | Estado |",
              "|---|---|---|---|---|---|---|---|"]
    for r in s["rows"]:
        estado = {"en_curso": "⏳ en curso", "en_rango": "✅ en rango",
                  "supero_bull": "🚀 superó Bull", "rompio_bear": "🔻 rompió Bear"}[r["outcome"]]
        lines.append(
            f"| {r['ticker']} | {r['date']} | {r['days']} "
            f"| ${r['price_then']:,.2f}→${r['price_now']:,.2f} "
            f"| {r['change']:+.1%} | {r['base_prorated']:+.1%} "
            f"| {r['deviation']:+.1%} | {estado} |"
        )
    lines += ["", "> Regla de aprendizaje: si el sesgo medio supera ±10% con ≥10 predicciones,",
              "> ajustar los factores de escenario en `engine/wbj/targets.py` (_SCENARIOS)",
              "> y registrar el cambio en `Memoria/errores.md`.", ""]
    return "\n".join(lines)
