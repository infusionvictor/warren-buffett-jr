"""Bull / Base / Bear 12-month price targets + plain-Spanish narrative.

Quick, transparent model (every assumption declared, per the visualization
rules: never a single line, always labeled assumptions):

    target = EPS x (1 + growth_scenario) x (PE_now x multiple_factor)

- EPS from EDGAR (latest FY net income / diluted shares)
- growth anchored on 5y net-income CAGR (fallback: revenue CAGR), clamped
- multiple anchored on TODAY's P/E (needs live price; Stooq, free/no key)
- No price or negative EPS -> honest NOT_SCORABLE, never invented numbers.
"""

from __future__ import annotations

import httpx

_FMP_QUOTE_URL = "https://financialmodelingprep.com/stable/quote-short"
_YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{t}"
_YAHOO_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh) warren-buffett-jr"}

# Scenario definitions: (growth delta vs base, multiple factor, label)
_SCENARIOS = {
    "bear": {"g_delta": -0.08, "pe_factor": 0.75, "label": "Bear"},
    "base": {"g_delta": 0.0, "pe_factor": 1.00, "label": "Medio"},
    "bull": {"g_delta": +0.05, "pe_factor": 1.15, "label": "Bull"},
}
_G_FLOOR, _G_CAP = -0.10, 0.40


def live_price(
    ticker: str,
    fmp_api_key: str | None = None,
    client: httpx.Client | None = None,
) -> float | None:
    """Latest market price. FMP stable API first (key), Yahoo chart fallback
    (keyless). None on total failure — targets then degrade to NOT_SCORABLE."""
    own = client is None
    client = client or httpx.Client(timeout=8.0)
    try:
        if fmp_api_key:
            try:
                r = client.get(_FMP_QUOTE_URL, params={"symbol": ticker.upper(), "apikey": fmp_api_key})
                if r.status_code == 200:
                    rows = r.json()
                    if isinstance(rows, list) and rows and isinstance(rows[0].get("price"), (int, float)):
                        return float(rows[0]["price"])
            except (httpx.HTTPError, ValueError):
                pass
        try:
            r = client.get(
                _YAHOO_URL.format(t=ticker.upper()),
                params={"range": "1d", "interval": "1d"},
                headers=_YAHOO_HEADERS,
            )
            if r.status_code != 200:
                return None
            meta = r.json()["chart"]["result"][0]["meta"]
            price = meta.get("regularMarketPrice")
            return float(price) if isinstance(price, (int, float)) else None
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
            return None
    finally:
        if own:
            client.close()


def _cagr(series: list[dict], years: int = 5) -> float | None:
    rows = series[-years:]
    if len(rows) < 3:
        return None
    begin, end = rows[0]["val"], rows[-1]["val"]
    if begin <= 0 or end <= 0:
        return None
    return (end / begin) ** (1 / (len(rows) - 1)) - 1


def price_targets(packet: dict, price: float | None) -> dict:
    """12-month Bull/Base/Bear targets with declared assumptions."""
    a = packet["annual"]
    ni, rev, shares = a["net_income"], a["revenue"], a.get("diluted_shares", [])
    ni_l = ni[-1]["val"] if ni else None
    sh_l = shares[-1]["val"] if shares else None

    if price is None:
        return {"status": "not_scorable", "reason": "sin precio de mercado disponible"}
    if not ni_l or not sh_l or ni_l <= 0:
        return {"status": "not_scorable", "reason": "sin ganancias positivas por acción (EPS)"}

    eps = ni_l / sh_l
    pe_now = price / eps
    g = _cagr(ni) if _cagr(ni) is not None else _cagr(rev)
    g_src = "utilidades 5a" if _cagr(ni) is not None else "ingresos 5a"
    if g is None:
        return {"status": "not_scorable", "reason": "historial insuficiente para estimar crecimiento"}
    g = max(_G_FLOOR, min(_G_CAP, g))

    rows = []
    for key, s in _SCENARIOS.items():
        gs = max(_G_FLOOR, min(_G_CAP, g + s["g_delta"]))
        pe = pe_now * s["pe_factor"]
        target = round(eps * (1 + gs) * pe, 2)
        rows.append({
            "key": key,
            "label": s["label"],
            "target": target,
            "upside": round(target / price - 1, 4),
            "assumptions": (
                f"crecimiento {gs:+.0%} ({g_src}) · "
                f"multiplo {pe:.1f}x ({s['pe_factor']:.0%} del P/E actual)"
            ),
        })

    return {
        "status": "ok",
        "price": round(price, 2),
        "eps": round(eps, 2),
        "pe_now": round(pe_now, 1),
        "growth_base": round(g, 4),
        "horizon": "12 meses",
        "scenarios": rows,
        "disclaimer": (
            "Rangos de referencia con supuestos declarados — clasificacion de "
            "research, no es asesoria de inversion ni promesa de retorno."
        ),
    }


def price_history(
    ticker: str,
    client: httpx.Client | None = None,
) -> list[dict]:
    """Daily closes for the last year (Yahoo, keyless) for the chart:
    [{"time": "YYYY-MM-DD", "value": close}, ...]. Empty list on failure."""
    from datetime import datetime, timezone

    own = client is None
    client = client or httpx.Client(timeout=8.0)
    try:
        r = client.get(
            _YAHOO_URL.format(t=ticker.upper()),
            params={"range": "1y", "interval": "1d"},
            headers=_YAHOO_HEADERS,
        )
        if r.status_code != 200:
            return []
        result = r.json()["chart"]["result"][0]
        stamps = result.get("timestamp") or []
        closes = result["indicators"]["quote"][0].get("close") or []
        out = []
        for ts, c in zip(stamps, closes):
            if isinstance(c, (int, float)):
                day = datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()
                out.append({"time": day, "value": round(float(c), 2)})
        return out
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
        return []
    finally:
        if own:
            client.close()


# ---------------------------------------------------------------- narrative

def _fmt_b(x: float) -> str:
    return f"${x / 1e9:,.1f}B" if abs(x) >= 1e9 else f"${x / 1e6:,.0f}M"


def narrative(packet: dict, scorecard: dict, targets: dict) -> list[str]:
    """Plain-Spanish sentences explaining the company from the data."""
    a = packet["annual"]
    rev, ni = a["revenue"], a["net_income"]
    ocf, capex = a["operating_cash_flow"], a["capex"]
    debt, eq = a["long_term_debt"], a["equity"]
    out: list[str] = []

    if rev:
        r = rev[-1]["val"]
        if len(rev) >= 2 and rev[-2]["val"]:
            g = r / rev[-2]["val"] - 1
            trend = ("crecen con fuerza" if g > 0.10 else
                     "crecen moderadamente" if g > 0.02 else
                     "estan casi planas" if g > -0.02 else "estan cayendo")
            out.append(f"Vendio {_fmt_b(r)} el ultimo año fiscal; las ventas {trend} ({g:+.1%}).")
        else:
            out.append(f"Vendio {_fmt_b(r)} el ultimo año fiscal.")

    if rev and ni and rev[-1]["val"]:
        m = ni[-1]["val"] / rev[-1]["val"]
        adj = ("muy rentable" if m > 0.20 else "rentable" if m > 0.10 else
               "con margenes ajustados" if m > 0.03 else "con margenes debiles" if m > 0 else "con perdidas")
        out.append(f"De cada $100 que vende, ${m * 100:,.0f} son ganancia neta — es {adj}.")

    if ocf and capex and rev and rev[-1]["val"]:
        fcf = ocf[-1]["val"] - capex[-1]["val"]
        if fcf > 0:
            out.append(f"Genera efectivo real: {_fmt_b(fcf)} de flujo de caja libre al año.")
        else:
            out.append(f"Quema efectivo: {_fmt_b(fcf)} de flujo de caja libre — vigilar financiamiento.")

    if debt and eq and eq[-1]["val"]:
        de = debt[-1]["val"] / eq[-1]["val"]
        adj = ("deuda baja" if de < 0.5 else "deuda manejable" if de < 1.0 else
               "deuda moderada" if de < 2.0 else "deuda alta")
        out.append(f"Debe ${de:,.2f} por cada $1 de capital propio — {adj}.")

    if scorecard.get("overall_10") is not None:
        out.append(
            f"Puntaje rapido: {scorecard['overall_10']}/10, calculado con "
            f"{scorecard['evidence_points_covered']} de 100 puntos de evidencia "
            "(las categorias de mercado, tecnico y valuacion se activan cuando "
            "sus motores esten listos)."
        )

    if targets.get("status") == "ok":
        base = next(r for r in targets["scenarios"] if r["key"] == "base")
        direction = "arriba" if base["upside"] >= 0 else "abajo"
        out.append(
            f"Si sigue creciendo como hasta ahora y el mercado paga el mismo "
            f"multiplo, la accion valdria ~${base['target']:,.2f} en 12 meses "
            f"({abs(base['upside']):.0%} {direction} del precio actual de ${targets['price']:,.2f})."
        )
    return out
