"""Generate Reportes/dashboard.html from all saved scores.json files.

Usage: .venv/bin/python scripts/dashboard.py
Re-run any time after `wbj analyze <TICKER>` to refresh the page.
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REPORTES = REPO / "Reportes"
OUT = REPORTES / "dashboard.html"


def latest_scores() -> list[dict]:
    rows = []
    for tdir in sorted(REPORTES.iterdir()):
        if not tdir.is_dir():
            continue
        dates = sorted(d for d in tdir.iterdir() if (d / "scores.json").exists())
        if not dates:
            continue
        rows.append(json.loads((dates[-1] / "scores.json").read_text(encoding="utf-8")))
    rows.sort(key=lambda r: r["scores"]["category"]["points"], reverse=True)
    return rows


def pct(x) -> str:
    return f"{x * 100:.1f}%" if isinstance(x, (int, float)) else str(x)


def row_html(i: int, r: dict) -> str:
    m, c = r["metrics"], r["scores"]["category"]
    dims = r["scores"]["dimensions"]
    prof = dims.get("Profitability", 0)
    grow = dims.get("Growth & Balance Sheet", 0)
    rev = f"${m['revenue_usd'] / 1e9:,.1f}B" if m.get("revenue_usd") else "—"
    bar = lambda v: (
        f'<div class="track"><div class="fill" style="width:{min(v,10)*10:.0f}%"></div></div>'
        if isinstance(v, (int, float)) else '<span class="ns">N/S</span>'
    )
    num = lambda v: f"{v:.1f}" if isinstance(v, (int, float)) else "N/S"
    return f"""
    <tr>
      <td class="rank">{i}</td>
      <td class="tk"><b>{r['ticker']}</b><span>{(r.get('entity') or '').title()}</span></td>
      <td class="n">{rev}</td>
      <td class="n">{pct(m['revenue_yoy'])}</td>
      <td class="n">{pct(m['net_margin'])}</td>
      <td class="n">{pct(m['fcf_margin'])}</td>
      <td class="meter">{bar(prof)}<em>{num(prof)}</em></td>
      <td class="meter">{bar(grow)}<em>{num(grow)}</em></td>
      <td class="n pts">{c['points']:.1f}<small>/15</small></td>
    </tr>"""


def main() -> None:
    rows = latest_scores()
    body = "".join(row_html(i + 1, r) for i, r in enumerate(rows))
    as_of = rows[0]["as_of"] if rows else ""
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WBJ — Results</title>
<style>
  :root {{ color-scheme: light;
    --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
    --grid:#e1e0d9; --border:rgba(11,11,11,.10); --blue:#2a78d6; --track:#f0efec; }}
  @media (prefers-color-scheme: dark) {{ :root {{ color-scheme: dark;
    --page:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --border:rgba(255,255,255,.10); --blue:#3987e5; --track:#262624; }} }}
  * {{ margin:0; box-sizing:border-box; }}
  body {{ background:var(--page); color:var(--ink);
    font-family:system-ui,-apple-system,"Segoe UI",sans-serif; padding:40px 24px; }}
  .wrap {{ max-width:1040px; margin:0 auto; }}
  .kicker {{ font-size:12px; letter-spacing:.14em; text-transform:uppercase;
    color:var(--muted); font-weight:600; margin-bottom:6px; }}
  h1 {{ font-size:24px; margin-bottom:4px; }}
  .sub {{ color:var(--ink2); font-size:13.5px; margin-bottom:24px; }}
  .card {{ background:var(--surface); border:1px solid var(--border); border-radius:12px;
    padding:8px 18px; overflow-x:auto; }}
  table {{ width:100%; border-collapse:collapse; font-size:13.5px; min-width:820px; }}
  th {{ text-align:left; color:var(--muted); font-size:11.5px; text-transform:uppercase;
    letter-spacing:.08em; font-weight:600; padding:12px 10px 8px; border-bottom:1px solid var(--grid); }}
  th.r {{ text-align:right; }}
  td {{ padding:11px 10px; border-bottom:1px solid var(--grid); vertical-align:middle; }}
  tr:last-child td {{ border-bottom:none; }}
  td.rank {{ color:var(--muted); font-weight:600; width:30px; }}
  td.tk span {{ display:block; color:var(--muted); font-size:11.5px; }}
  td.n {{ text-align:right; font-variant-numeric:tabular-nums; font-weight:600; }}
  td.pts {{ font-size:15px; }} td.pts small {{ color:var(--muted); font-weight:500; }}
  td.meter {{ min-width:110px; }}
  td.meter em {{ font-style:normal; font-size:11.5px; color:var(--ink2);
    font-variant-numeric:tabular-nums; }}
  .track {{ height:8px; background:var(--track); border-radius:4px; overflow:hidden; margin-bottom:3px; }}
  .fill {{ height:100%; border-radius:4px; background:var(--blue); }}
  .ns {{ color:var(--muted); font-size:11.5px; }}
  .foot {{ margin-top:18px; color:var(--muted); font-size:12px; line-height:1.6; max-width:80ch; }}
</style></head><body><div class="wrap">
  <div class="kicker">Warren Buffett Jr · Compute Engine · Live SEC EDGAR</div>
  <h1>Financial Scores — {len(rows)} companies</h1>
  <div class="sub">Latest 10-K per company · as of {as_of} · ranked by Financial category points (max 15)</div>
  <div class="card"><table>
    <thead><tr>
      <th></th><th>Company</th><th class="r">Revenue</th><th class="r">YoY</th>
      <th class="r">Net mgn</th><th class="r">FCF mgn</th>
      <th>Profitability /10</th><th>Growth &amp; BS /10</th><th class="r">Points</th>
    </tr></thead>
    <tbody>{body}</tbody>
  </table></div>
  <div class="foot"><b>MVP assumptions:</b> anchor bands are engine defaults; Financial is 1 of 6 categories
  (15/100 pts); missing data is never imputed (N/S = not scorable). Research classification only —
  not investment advice.</div>
</div></body></html>"""
    OUT.write_text(html, encoding="utf-8")
    print(f"dashboard -> {OUT}")


if __name__ == "__main__":
    main()
