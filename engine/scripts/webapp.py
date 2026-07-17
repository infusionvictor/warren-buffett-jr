"""WBJ mini web app: search any US-listed company, analyze on demand.

Usage: .venv/bin/python scripts/webapp.py  ->  http://localhost:8765
Stdlib http.server only — no extra dependencies.
Card-dashboard UI (light, big numbers, plain-Spanish explanations) modeled
on the reference screenshots in Referencias/.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from wbj.cli import _build_packet, _compute
from wbj.config import load_settings
from wbj.providers.cache import Cache
from wbj.providers.edgar import (
    _EDGAR_HEADERS,
    _GLOBAL_CACHE_TICKER,
    _MAX_AGE_TICKERS,
    TICKERS_URL,
    EdgarProvider,
)
from wbj.screener import screen as run_screen
from wbj.targets import live_price, narrative, price_history, price_targets

PORT = 8765
_lock = threading.Lock()

settings = load_settings()
edgar = EdgarProvider(settings, Cache(settings.cache_dir))


def ticker_map() -> list[dict]:
    payload = edgar.get_json(
        TICKERS_URL, {}, "tickers", _GLOBAL_CACHE_TICKER,
        max_age_days=_MAX_AGE_TICKERS, headers=_EDGAR_HEADERS,
    )
    if not isinstance(payload, dict):
        return []
    return [e for e in payload.values() if isinstance(e, dict)]


def search(q: str, limit: int = 8) -> list[dict]:
    q = q.strip().upper()
    if not q:
        return []
    exact, prefix, name = [], [], []
    for e in ticker_map():
        t = str(e.get("ticker", "")).upper()
        n = str(e.get("title", "")).upper()
        row = {"ticker": t, "name": e.get("title", "")}
        if t == q:
            exact.append(row)
        elif t.startswith(q):
            prefix.append(row)
        elif q in n:
            name.append(row)
    return (exact + prefix + name)[:limit]


def _history(packet: dict) -> list[dict]:
    """Yearly revenue + net margin for the charts."""
    rev = {r["end"]: r["val"] for r in packet["annual"]["revenue"]}
    ni = {r["end"]: r["val"] for r in packet["annual"]["net_income"]}
    rows = []
    for end in sorted(rev)[-6:]:
        rows.append({
            "year": end[:4],
            "revenue": rev[end],
            "margin": (ni[end] / rev[end]) if end in ni and rev[end] else None,
        })
    return rows


def analyze(ticker: str) -> dict:
    packet = _build_packet(ticker)
    result = _compute(packet)
    price = live_price(ticker, fmp_api_key=settings.fmp_api_key)
    targets = price_targets(packet, price)
    result["targets"] = targets
    result["narrative"] = narrative(packet, result["scorecard"], targets)
    result["history"] = _history(packet)
    result["chart"] = price_history(ticker)
    return result


PAGE = """<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Warren Buffett Jr — Scorecard</title>
<style>
  :root { color-scheme: light;
    --page:#edeff5; --card:#ffffff; --ink:#16181d; --ink2:#5b6270; --muted:#9aa1ad;
    --grid:#eef0f4; --green:#22a06b; --green-bg:#e7f6ef; --purple:#7c5cfc;
    --purple-bg:#f0ecfe; --orange:#f5a623; --orange-bg:#fef4e2; --red:#e5484d;
    --red-bg:#fdecec; --blue:#3b82f6; --blue-bg:#e8f1fe; }
  * { margin:0; box-sizing:border-box; }
  body { background:var(--page); color:var(--ink);
    font-family:system-ui,-apple-system,"Segoe UI",sans-serif; padding:36px 20px 60px; }
  .wrap { max-width:1040px; margin:0 auto; }
  .kicker { font-size:12px; letter-spacing:.14em; text-transform:uppercase;
    color:var(--muted); font-weight:600; margin-bottom:6px; }
  h1 { font-size:26px; margin-bottom:16px; letter-spacing:-.02em; }
  .topbar { display:flex; gap:12px; align-items:stretch; max-width:760px; }
  .searchbox { position:relative; flex:1; max-width:560px; }
  .discover { font:inherit; font-size:14.5px; font-weight:700; padding:0 22px;
    border:0; border-radius:14px; background:var(--purple); color:#fff; cursor:pointer;
    box-shadow:0 1px 3px rgba(20,22,30,.15); white-space:nowrap; }
  .discover:hover { filter:brightness(1.08); }
  table.scr { width:100%; border-collapse:collapse; margin-top:14px; font-size:13.5px; }
  table.scr th { text-align:left; font-size:11.5px; text-transform:uppercase;
    letter-spacing:.08em; color:var(--muted); padding:0 10px 10px 0; font-weight:600; }
  table.scr td { padding:11px 10px 11px 0; border-top:1px solid var(--grid);
    font-variant-numeric:tabular-nums; }
  table.scr td.nm { max-width:230px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  table.scr .tk { font-weight:800; cursor:pointer; color:var(--purple); }
  table.scr .num { text-align:right; }
  .scorepill { display:inline-flex; align-items:center; gap:8px; }
  .scorepill .mini { width:64px; height:7px; background:var(--grid); border-radius:5px; overflow:hidden; }
  .scorepill .mini i { display:block; height:100%; background:var(--green); border-radius:5px; }
  input { width:100%; font:inherit; font-size:16px; padding:14px 18px;
    border-radius:14px; border:1px solid transparent; background:var(--card);
    color:var(--ink); outline:none; box-shadow:0 1px 3px rgba(20,22,30,.07); }
  input:focus { border-color:var(--purple); }
  .sugg { position:absolute; top:calc(100% + 8px); left:0; right:0; z-index:5;
    background:var(--card); border-radius:14px; overflow:hidden;
    box-shadow:0 12px 32px rgba(20,22,30,.16); display:none; }
  .sugg button { display:flex; gap:12px; width:100%; text-align:left; font:inherit;
    font-size:14px; padding:12px 18px; border:0; background:none; color:var(--ink);
    cursor:pointer; border-bottom:1px solid var(--grid); }
  .sugg button:last-child { border-bottom:none; }
  .sugg button:hover { background:var(--grid); }
  .sugg b { min-width:56px; } .sugg span { color:var(--ink2); }
  #status { margin:20px 2px; color:var(--ink2); font-size:14px; }
  .grid { display:grid; grid-template-columns:repeat(12, 1fr); gap:18px;
    margin-top:22px; display:none; }
  .card { background:var(--card); border-radius:18px; padding:24px;
    box-shadow:0 1px 3px rgba(20,22,30,.06); }
  .c-hero { grid-column:span 7; } .c-words { grid-column:span 5; }
  .c-chart { grid-column:span 12; background:#0e1113; color:#e8eaed; }
  .c-score { grid-column:span 5; } .c-target { grid-column:span 7; }
  @media (max-width:860px) { .c-hero,.c-words,.c-chart,.c-score,.c-target { grid-column:span 12; } }
  .c-chart h2 { color:#fff; } .c-chart .sub { color:#8b929c; }
  .chart-head { display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; margin:8px 0 4px; }
  .chart-head .px { font-size:34px; font-weight:800; letter-spacing:-.02em; }
  .chart-head .chg { font-size:14px; font-weight:700; }
  .chart-head .chg.up { color:#26d07c; } .chart-head .chg.down { color:#ff5a5f; }
  .chart-head .rng { color:#8b929c; font-size:13px; }
  #tvchart { height:340px; margin-top:10px; }
  .periods { display:flex; gap:8px; margin-top:12px; }
  .periods button { font:inherit; font-size:13px; font-weight:700; padding:6px 16px;
    border:0; border-radius:99px; background:transparent; color:#8b929c; cursor:pointer; }
  .periods button.on { background:#1d2b22; color:#26d07c; }
  .tglegend { display:flex; gap:18px; margin-top:10px; font-size:12.5px; flex-wrap:wrap; }
  .tglegend span { display:inline-flex; align-items:center; gap:7px; color:#aab1bb; }
  .tglegend i { width:18px; border-top:2px dashed; display:inline-block; }
  .card h2 { font-size:16px; font-weight:700; }
  .card .sub { color:var(--muted); font-size:12.5px; margin-top:3px; }
  .hero-num { display:flex; align-items:baseline; gap:12px; margin:16px 0 4px; }
  .hero-num .n { font-size:38px; font-weight:800; letter-spacing:-.03em; }
  .hero-num .u { font-size:13px; color:var(--ink2); }
  .chip { font-size:12.5px; font-weight:700; padding:3px 9px; border-radius:99px; }
  .chip.up { color:var(--green); background:var(--green-bg); }
  .chip.down { color:var(--red); background:var(--red-bg); }
  .bars { display:flex; align-items:flex-end; gap:14px; height:130px; margin-top:18px; }
  .bar { flex:1; display:flex; flex-direction:column; justify-content:flex-end;
    align-items:center; gap:6px; height:100%; }
  .bar .stick { width:100%; max-width:38px; border-radius:8px 8px 4px 4px;
    background:var(--purple); transition:height .5s ease; }
  .bar.last .stick { background:var(--green); }
  .bar .y { font-size:11.5px; color:var(--muted); }
  .bar .v { font-size:11.5px; color:var(--ink2); font-weight:600; }
  .legend { display:flex; gap:16px; margin-top:12px; font-size:12.5px; color:var(--ink2); }
  .legend i { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; }
  ul.words { list-style:none; margin-top:14px; }
  ul.words li { display:flex; gap:10px; padding:9px 0; font-size:14px; line-height:1.5;
    border-bottom:1px solid var(--grid); }
  ul.words li:last-child { border-bottom:none; }
  ul.words .dot { flex:none; width:8px; height:8px; border-radius:50%; margin-top:7px; }
  .gaugebox { display:flex; align-items:center; gap:6px; flex-direction:column; margin:6px 0 12px; }
  .gauge { width:190px; height:110px; }
  .gauge .track { stroke:var(--grid); } .gauge .arc { stroke:var(--green); }
  .gauge-num { font-size:34px; font-weight:800; margin-top:-58px; }
  .gauge-of { font-size:12px; color:var(--muted); margin-bottom:16px; }
  .srow { display:grid; grid-template-columns:150px 1fr 52px; gap:10px;
    align-items:center; margin-bottom:10px; font-size:13px; }
  .srow .nm { color:var(--ink2); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .srow .v { text-align:right; font-weight:700; font-variant-numeric:tabular-nums; }
  .track2 { height:9px; background:var(--grid); border-radius:6px; overflow:hidden; }
  .fill2 { height:100%; border-radius:6px; transition:width .5s ease; }
  .srow.ns .nm,.srow.ns .v { color:var(--muted); font-weight:500; }
  .price-now { display:flex; align-items:baseline; gap:10px; margin:14px 0 4px; }
  .price-now .n { font-size:30px; font-weight:800; }
  .price-now .u { font-size:12.5px; color:var(--muted); }
  .trow { display:grid; grid-template-columns:86px 1fr 96px 90px; gap:12px;
    align-items:center; padding:13px 0; border-bottom:1px solid var(--grid); }
  .trow:last-of-type { border-bottom:none; }
  .tag { font-size:12.5px; font-weight:700; padding:4px 0; border-radius:10px; text-align:center; }
  .tag.bull { color:var(--green); background:var(--green-bg); }
  .tag.base { color:var(--blue); background:var(--blue-bg); }
  .tag.bear { color:var(--red); background:var(--red-bg); }
  .trow .as { font-size:12px; color:var(--muted); line-height:1.45; }
  .trow .tp { font-size:19px; font-weight:800; text-align:right; font-variant-numeric:tabular-nums; }
  .trow .up { text-align:right; }
  .range { position:relative; height:54px; margin:18px 6px 2px; }
  .range .line { position:absolute; top:24px; left:0; right:0; height:8px;
    border-radius:6px; background:linear-gradient(90deg, var(--red-bg), var(--grid), var(--green-bg)); }
  .range .pt { position:absolute; top:20px; width:16px; height:16px; border-radius:50%;
    border:3px solid #fff; box-shadow:0 1px 4px rgba(0,0,0,.25); transform:translateX(-50%); }
  .range .lb { position:absolute; top:0; font-size:11px; font-weight:700; transform:translateX(-50%); white-space:nowrap; }
  .range .lb.below { top:auto; bottom:0; font-weight:600; color:var(--ink2); }
  .foot { margin-top:22px; color:var(--muted); font-size:12px; line-height:1.6; max-width:760px; }
  .spin { display:inline-block; width:14px; height:14px; border:2px solid var(--grid);
    border-top-color:var(--purple); border-radius:50%; animation:r .7s linear infinite;
    vertical-align:-2px; margin-right:7px; }
  @keyframes r { to { transform:rotate(360deg); } }
</style></head><body><div class="wrap">
  <div class="kicker">Warren Buffett Jr · Motor de Análisis · SEC EDGAR en vivo</div>
  <h1>Busca una empresa</h1>
  <div class="topbar">
    <div class="searchbox">
      <input id="q" placeholder="Escribe un ticker o nombre — ej. NFLX, Disney, Coca-Cola…"
        autocomplete="off" autofocus>
      <div class="sugg" id="sugg"></div>
    </div>
    <button class="discover" id="discoverBtn">✨ Descubrir empresas</button>
  </div>
  <div id="status"></div>
  <div class="card" id="screenCard" style="display:none;margin-top:22px"></div>
  <div class="grid" id="grid">
    <div class="card c-hero" id="heroCard"></div>
    <div class="card c-words" id="wordsCard"></div>
    <div class="card c-chart" id="chartCard"></div>
    <div class="card c-score" id="scoreCard"></div>
    <div class="card c-target" id="targetCard"></div>
  </div>
  <div class="foot" id="foot"><b>Nota:</b> Puntaje rápido con datos oficiales de la SEC (EDGAR).
  Sin evidencia no hay número: las categorías pendientes se muestran como N/S, nunca se inventan.
  Los targets son rangos de referencia con supuestos declarados — clasificación de research,
  no es asesoría de inversión.</div>
</div>
<script>
const q = document.getElementById('q'), sugg = document.getElementById('sugg'),
      status = document.getElementById('status'), grid = document.getElementById('grid');
let timer = null;

q.addEventListener('input', () => {
  clearTimeout(timer);
  timer = setTimeout(async () => {
    const v = q.value.trim();
    if (!v) { sugg.style.display = 'none'; return; }
    const r = await fetch('/api/search?q=' + encodeURIComponent(v));
    const items = await r.json();
    sugg.innerHTML = items.map(i =>
      `<button data-t="${i.ticker}"><b>${i.ticker}</b><span>${i.name}</span></button>`).join('');
    sugg.style.display = items.length ? 'block' : 'none';
  }, 180);
});
q.addEventListener('keydown', e => {
  if (e.key === 'Enter') {
    const first = sugg.querySelector('button');
    run(first ? first.dataset.t : q.value.trim().toUpperCase());
  }
});
sugg.addEventListener('click', e => {
  const b = e.target.closest('button');
  if (b) run(b.dataset.t);
});

const fmtB = x => '$' + (x / 1e9).toFixed(1) + 'B';
const CAT_COLORS = { business:'var(--purple)', financial:'var(--green)',
  risk:'var(--blue)', market:'var(--orange)', technical:'var(--orange)', valuation:'var(--orange)' };
const WORD_COLORS = ['var(--purple)','var(--green)','var(--blue)','var(--orange)','var(--red)','var(--purple)'];

function heroHtml(d) {
  const h = d.history, last = h[h.length - 1];
  const maxRev = Math.max(...h.map(r => r.revenue));
  const yoy = d.metrics.revenue_yoy;
  const chip = typeof yoy === 'number'
    ? `<span class="chip ${yoy >= 0 ? 'up' : 'down'}">${yoy >= 0 ? '▲' : '▼'} ${(Math.abs(yoy) * 100).toFixed(1)}% vs año anterior</span>` : '';
  const bars = h.map((r, i) => `
    <div class="bar ${i === h.length - 1 ? 'last' : ''}">
      <span class="v">${(r.revenue / 1e9).toFixed(0)}</span>
      <div class="stick" data-h="${(r.revenue / maxRev * 100).toFixed(0)}" style="height:4%"></div>
      <span class="y">${r.year}</span>
    </div>`).join('');
  return `<h2>${d.entity} · ${d.ticker}</h2>
    <div class="sub">Año fiscal terminado ${d.fiscal_year_end} · Form 10-K · SEC EDGAR</div>
    <div class="hero-num"><span class="n">${fmtB(last.revenue)}</span>
      <span class="u">ventas anuales</span>${chip}</div>
    <div class="bars">${bars}</div>
    <div class="legend"><span><i style="background:var(--purple)"></i>Ventas por año (miles de millones $)</span>
      <span><i style="background:var(--green)"></i>Último año</span></div>`;
}

function wordsHtml(d) {
  const lis = d.narrative.map((s, i) =>
    `<li><span class="dot" style="background:${WORD_COLORS[i % WORD_COLORS.length]}"></span><span>${s}</span></li>`).join('');
  return `<h2>En palabras simples</h2>
    <div class="sub">Qué dicen los números, sin jerga</div><ul class="words">${lis}</ul>`;
}

function gaugeSvg(score) {
  const frac = Math.max(0, Math.min(1, score / 10));
  const R = 80, C = Math.PI * R;
  return `<svg class="gauge" viewBox="0 0 200 110">
    <path class="track" d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke-width="14" stroke-linecap="round"/>
    <path class="arc" d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke-width="14"
      stroke-linecap="round" stroke-dasharray="${(frac * C).toFixed(1)} ${C.toFixed(1)}"/>
  </svg>`;
}

function scoreHtml(d) {
  const sc = d.scorecard;
  const rows = sc.categories.map(r => {
    if (r.status === 'scored') {
      return `<div class="srow"><span class="nm">${r.label}</span>
        <div class="track2"><div class="fill2" data-w="${r.score10 * 10}"
          style="width:0%;background:${CAT_COLORS[r.key]}"></div></div>
        <span class="v">${r.score10.toFixed(1)}/10</span></div>`;
    }
    return `<div class="srow ns"><span class="nm" title="${r.reason}">${r.label}</span>
      <div class="track2"></div><span class="v">N/S</span></div>`;
  }).join('');
  const overall = sc.overall_10 === null ? '—' : sc.overall_10.toFixed(1);
  return `<h2>Puntaje de los agentes</h2>
    <div class="sub">1–10 por categoría · ${sc.evidence_points_covered}/100 puntos de evidencia</div>
    <div class="gaugebox">${gaugeSvg(sc.overall_10 ?? 0)}
      <div class="gauge-num">${overall}</div><div class="gauge-of">de 10 (rápido)</div></div>${rows}`;
}

function targetHtml(d) {
  const t = d.targets;
  if (t.status !== 'ok') {
    return `<h2>Precio objetivo — 12 meses</h2>
      <div class="sub">Bull / Medio / Bear</div>
      <p style="margin-top:16px;color:var(--ink2);font-size:14px">
      No se puede calcular: ${t.reason}.</p>`;
  }
  const order = ['bull', 'base', 'bear'];
  const by = {}; t.scenarios.forEach(s => by[s.key] = s);
  const rows = order.map(k => {
    const s = by[k];
    const cls = k === 'base' ? 'base' : k;
    const chip = `<span class="chip ${s.upside >= 0 ? 'up' : 'down'}">${s.upside >= 0 ? '+' : ''}${(s.upside * 100).toFixed(0)}%</span>`;
    return `<div class="trow"><span class="tag ${cls}">${s.label}</span>
      <span class="as">${s.assumptions}</span>
      <span class="tp">$${s.target.toLocaleString('en-US', {minimumFractionDigits: 2})}</span>
      <span class="up">${chip}</span></div>`;
  }).join('');
  const pts = [ {v: t.price, lb: 'Hoy', c: 'var(--ink)', below: true},
    {v: by.bear.target, lb: 'Bear', c: 'var(--red)'},
    {v: by.base.target, lb: 'Medio', c: 'var(--blue)'},
    {v: by.bull.target, lb: 'Bull', c: 'var(--green)'} ];
  const lo = Math.min(...pts.map(p => p.v)) * 0.97, hi = Math.max(...pts.map(p => p.v)) * 1.03;
  const pos = v => ((v - lo) / (hi - lo) * 100).toFixed(1) + '%';
  const marks = pts.map(p => `
    <span class="lb ${p.below ? 'below' : ''}" style="left:${pos(p.v)};color:${p.c}">${p.lb} $${p.v.toFixed(0)}</span>
    <span class="pt" style="left:${pos(p.v)};background:${p.c}"></span>`).join('');
  return `<h2>Precio objetivo — ${t.horizon}</h2>
    <div class="sub">Escenarios con supuestos declarados (nunca una sola línea)</div>
    <div class="price-now"><span class="n">$${t.price.toLocaleString('en-US', {minimumFractionDigits: 2})}</span>
      <span class="u">precio actual · EPS $${t.eps} · P/E ${t.pe_now}x</span></div>
    <div class="range"><div class="line"></div>${marks}</div>${rows}
    <div class="sub" style="margin-top:12px">${t.disclaimer}</div>`;
}

// --- Gráfica SVG propia (cero dependencias externas) ---------------------
let tvData = [], tvTargets = null;
const PERIODS = { '1M': 21, '3M': 63, '6M': 126, '1A': 9999 };

function renderChart(d) {
  const el = document.getElementById('chartCard');
  tvData = d.chart || [];
  tvTargets = d.targets && d.targets.status === 'ok' ? d.targets : null;
  if (!tvData.length) {
    el.innerHTML = `<h2>Gráfica y targets</h2>
      <div class="sub">Sin historial de precio disponible para ${d.ticker}.</div>`;
    return;
  }
  el.innerHTML = `<h2>${d.ticker} — precio y targets</h2>
    <div class="sub">Último año · targets a 12 meses con supuestos declarados</div>
    <div class="chart-head" id="chartHead"></div>
    <div id="tvchart"></div>
    <div class="periods" id="periods">
      ${Object.keys(PERIODS).map(p =>
        `<button data-p="${p}" class="${p === '1A' ? 'on' : ''}">${p}</button>`).join('')}
    </div>`;
  document.getElementById('periods').addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    document.querySelectorAll('.periods button').forEach(x => x.classList.toggle('on', x === b));
    setPeriod(b.dataset.p);
  });
  setPeriod('1A');
}

function setPeriod(p) {
  const slice = tvData.slice(-PERIODS[p]);
  const W = 1000, H = 340, PADL = 14, PADR = 150, PADT = 18, PADB = 30;
  const vals = slice.map(r => r.value);
  const tg = tvTargets ? tvTargets.scenarios.map(s => s.target) : [];
  let lo = Math.min(...vals, ...(tg.length ? tg : vals));
  let hi = Math.max(...vals, ...(tg.length ? tg : vals));
  const pad = (hi - lo) * 0.07 || 1; lo -= pad; hi += pad;
  const X = i => PADL + i / (slice.length - 1) * (W - PADL - PADR);
  const Y = v => PADT + (1 - (v - lo) / (hi - lo)) * (H - PADT - PADB);

  const pts = slice.map((r, i) => `${X(i).toFixed(1)},${Y(r.value).toFixed(1)}`).join(' ');
  const area = `M ${X(0).toFixed(1)},${(H - PADB)} L ${pts.replaceAll(' ', ' L ')} L ${X(slice.length - 1).toFixed(1)},${H - PADB} Z`;

  // grid + eje de fechas
  const grid = [0.25, 0.5, 0.75].map(f => {
    const y = PADT + f * (H - PADT - PADB);
    return `<line x1="${PADL}" x2="${W - PADR + 60}" y1="${y}" y2="${y}"
      stroke="rgba(139,146,156,.14)" stroke-width="1"/>`;
  }).join('');
  const dates = [0, Math.floor(slice.length / 2), slice.length - 1].map(i =>
    `<text x="${X(i)}" y="${H - 8}" fill="#8b929c" font-size="11.5"
      text-anchor="${i === 0 ? 'start' : i === slice.length - 1 ? 'end' : 'middle'}">${slice[i].time}</text>`).join('');

  // líneas de target punteadas con etiqueta tipo píldora (como la referencia)
  let tlines = '';
  if (tvTargets) {
    const by = {}; tvTargets.scenarios.forEach(s => by[s.key] = s);
    const defs = [
      ['bull', '#26d07c', 'rgba(38,208,124,.14)', 'Bull'],
      ['base', '#f5a623', 'rgba(245,166,35,.14)', 'Medio'],
      ['bear', '#ff5a5f', 'rgba(255,90,95,.14)', 'Bear'],
    ];
    tlines = defs.map(([k, c, bg, lb]) => {
      const s = by[k], y = Y(s.target);
      const label = `${lb} $${s.target.toFixed(0)} (${s.upside >= 0 ? '+' : ''}${(s.upside * 100).toFixed(0)}%)`;
      return `<line x1="${PADL}" x2="${W - PADR + 4}" y1="${y}" y2="${y}"
          stroke="${c}" stroke-width="1.3" stroke-dasharray="7 6" opacity=".9"/>
        <rect x="${W - PADR + 8}" y="${y - 12}" width="${PADR - 16}" height="24" rx="7"
          fill="${bg}" stroke="${c}" stroke-width="1"/>
        <text x="${W - PADR / 2}" y="${y + 4}" fill="${c}" font-size="12.5"
          font-weight="700" text-anchor="middle">${label}</text>`;
    }).join('');
  }

  document.getElementById('tvchart').innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block"
        font-family="system-ui,-apple-system,sans-serif">
      <defs><linearGradient id="ga" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="rgba(38,208,124,.28)"/>
        <stop offset="1" stop-color="rgba(38,208,124,.02)"/>
      </linearGradient></defs>
      ${grid}
      <path d="${area}" fill="url(#ga)"/>
      <polyline points="${pts}" fill="none" stroke="#26d07c" stroke-width="2.2"
        stroke-linejoin="round" stroke-linecap="round"/>
      ${tlines}${dates}
    </svg>`;

  const first = slice[0].value, last = slice[slice.length - 1].value;
  const chg = last - first, pct = chg / first * 100;
  document.getElementById('chartHead').innerHTML = `
    <span class="px">$${last.toLocaleString('en-US', {minimumFractionDigits: 2})}</span>
    <span class="chg ${chg >= 0 ? 'up' : 'down'}">${chg >= 0 ? '+' : ''}$${Math.abs(chg).toFixed(2)}
      (${chg >= 0 ? '+' : ''}${pct.toFixed(2)}%)</span>
    <span class="rng">${p}</span>`;
}

const screenCard = document.getElementById('screenCard');
document.getElementById('discoverBtn').addEventListener('click', async () => {
  grid.style.display = 'none'; screenCard.style.display = 'none';
  status.innerHTML = `<span class="spin"></span>Escaneando todo el universo de la SEC:
    empresas medianas ($0.8B–$30B), rentables y creciendo… la primera vez tarda 1–2 min.`;
  try {
    const r = await fetch('/api/screen');
    if (!r.ok) throw new Error((await r.json()).error || r.status);
    const rows = await r.json();
    const trs = rows.map((x, i) => `<tr>
      <td>${i + 1}</td>
      <td class="tk" data-t="${x.ticker}">${x.ticker}</td>
      <td class="nm">${x.name}</td>
      <td class="num">$${(x.revenue / 1e9).toFixed(1)}B</td>
      <td class="num">${(x.growth * 100).toFixed(0)}%</td>
      <td class="num">${(x.margin * 100).toFixed(0)}%</td>
      <td class="num">${x.target_base ? '$' + x.target_base.toFixed(0) +
        ' <span class="chip ' + (x.upside_base >= 0 ? 'up' : 'down') + '">' +
        (x.upside_base >= 0 ? '+' : '') + (x.upside_base * 100).toFixed(0) + '%</span>' : '—'}</td>
      <td><span class="scorepill"><span class="mini"><i style="width:${x.score10 * 10}%"></i></span>
        <b>${x.score10.toFixed(1)}</b></span></td>
    </tr>`).join('');
    screenCard.innerHTML = `<h2>Empresas descubiertas — top ${rows.length} por puntaje</h2>
      <div class="sub">Prefiltro: ventas $0.8B–$30B · margen neto > 8% · crecimiento > 5% ·
      datos SEC EDGAR · clasificación de research, no asesoría. Toca un ticker para el análisis completo.</div>
      <table class="scr"><thead><tr><th>#</th><th>Ticker</th><th>Empresa</th>
        <th style="text-align:right">Ventas</th><th style="text-align:right">Crec.</th>
        <th style="text-align:right">Margen</th><th style="text-align:right">Target medio</th>
        <th>Puntaje</th></tr></thead><tbody>${trs}</tbody></table>`;
    screenCard.style.display = 'block';
    status.textContent = '';
    screenCard.querySelectorAll('.tk').forEach(el =>
      el.addEventListener('click', () => { screenCard.style.display = 'none'; run(el.dataset.t); }));
  } catch (err) {
    status.innerHTML = `No pude completar el descubrimiento: ${err.message}`;
  }
});

async function run(t) {
  if (!t) return;
  sugg.style.display = 'none'; grid.style.display = 'none';
  q.value = t;
  status.innerHTML = `<span class="spin"></span>Leyendo reportes de la SEC y calculando <b>${t}</b>…`;
  try {
    const r = await fetch('/api/analyze?ticker=' + encodeURIComponent(t));
    if (!r.ok) throw new Error((await r.json()).error || r.status);
    const d = await r.json();
    document.getElementById('heroCard').innerHTML = heroHtml(d);
    document.getElementById('wordsCard').innerHTML = wordsHtml(d);
    document.getElementById('scoreCard').innerHTML = scoreHtml(d);
    document.getElementById('targetCard').innerHTML = targetHtml(d);
    grid.style.display = 'grid';
    renderChart(d);
    status.textContent = '';
    requestAnimationFrame(() => {
      document.querySelectorAll('.stick[data-h]').forEach(b => b.style.height = b.dataset.h + '%');
      document.querySelectorAll('.fill2[data-w]').forEach(f => f.style.width = f.dataset.w + '%');
    });
  } catch (err) {
    status.innerHTML = `No pude analizar <b>${t}</b>: ${err.message}`;
  }
}
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj: dict | list, code: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (stdlib API)
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        if url.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif url.path == "/api/search":
            self._json(search(qs.get("q", [""])[0]))
        elif url.path == "/api/screen":
            try:
                with _lock:
                    self._json(run_screen(limit=15))
            except Exception as e:
                self._json({"error": str(e)}, 500)
        elif url.path == "/api/analyze":
            ticker = qs.get("ticker", [""])[0].strip().upper()
            if not ticker:
                self._json({"error": "missing ticker"}, 400)
                return
            try:
                # One analysis at a time: providers share one httpx client/cache.
                with _lock:
                    result = analyze(ticker)
                self._json(result)
            except Exception as e:  # surface as JSON, keep server alive
                self._json({"error": str(e)}, 500)
        else:
            self._json({"error": "not found"}, 404)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[wbj] {self.address_string()} {fmt % args}")


if __name__ == "__main__":
    print(f"WBJ web app -> http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
