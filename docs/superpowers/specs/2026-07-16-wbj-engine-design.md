# Design Spec: `wbj` — Warren Buffett Jr Compute Engine

**Date:** 2026-07-16
**Status:** Approved by Victor Gonzalez
**Methodology:** Ruta 2030 Wall Street Agent System v2.0.0 (`Cerebro/`)

## 1. Purpose

A Python package that implements every **deterministic** piece of the Cerebro v2.0.0 methodology: data fetching, packet validation, all ~200 registered formulas, scoring/gates/overrides, the institutional valuation engine, the important-levels engine, price-level synthesis, and final report rendering.

Division of labor:

- **Python (`wbj`)** — everything computable: data, math, scoring, charts, report assembly. Exact, repeatable, auditable.
- **Claude sub-agents** — everything requiring judgment: moat classification, catalyst probabilities, thesis killers, TAM tier assessment. Injected via the **judgment overlay** (§7).

Reports are written in **English**. Robinhood integration is out of scope for v1.

## 2. Architecture

Package lives at `engine/` inside the repo. The pipeline mirrors the Cerebro's Phases 0–8; one CLI subcommand per stage; JSON artifacts between stages; every artifact carries `knowledge_timestamp`, source lineage, and a SHA-256 hash (audit requirements of `ORCHESTRATION.md`).

| Command | Cerebro phase | Output |
|---|---|---|
| `wbj fetch NVDA` | — | raw provider responses → `engine/cache/NVDA/*.json` |
| `wbj packet NVDA` | 0–1 | validated input packet + common facts table (hashed) |
| `wbj compute NVDA` | 2–3 | 6 specialist output JSONs (frozen, hashed) + `judgment_requests.json` |
| `wbj aggregate NVDA` | 4–7 | raw total → overrides → gates → price-level synthesis |
| `wbj report NVDA` | 8 | `Reportes/NVDA/<YYYY-MM-DD>/` — report.md, report.json, charts/ |
| `wbj analyze NVDA` | all | end-to-end |

`wbj aggregate --overlay judgments.json` merges agent judgments and re-scores.

## 3. Package layout

```
engine/
├── pyproject.toml            # package: wbj, console script: wbj
├── wbj/
│   ├── cli.py                # typer CLI
│   ├── config.py             # loads ../API/.env; never logs key values
│   ├── providers/            # fmp.py, finnhub.py, fred.py, edgar.py, cache.py
│   ├── packet/               # builder.py, facts_table.py, validation.py, staleness.py
│   ├── core/                 # formulas.py, scoring.py, confidence.py,
│   │                         # conventions.py, nullstates.py
│   ├── specialists/          # business.py, financial.py, market.py,
│   │                         # technical.py, risk.py, valuation.py
│   ├── engines/              # valuation_engine.py, levels_engine.py
│   ├── aggregate/            # gates.py, overrides.py, contradiction.py, synthesis.py
│   ├── overlay/              # judgment request/merge logic
│   ├── report/               # render.py (markdown), charts.py (matplotlib)
│   └── schemas/              # pydantic models mirroring every OUTPUT_SCHEMA.md
└── tests/                    # pytest — encodes each VALIDATION_TESTS.md + fixtures
```

Dependencies: `typer, httpx, pandas, numpy, scipy, pydantic, python-dotenv, matplotlib, pytest`. Python 3.11+.

## 4. Data layer

**Providers**

| Provider | Data |
|---|---|
| FMP | 5y statements (annual + quarterly), adjusted OHLCV, profile, peers, insider Forms 4, 13F holders |
| FinnHub | consensus estimates + revision history, earnings calendar |
| FRED | risk-free rate, macro series |
| SEC EDGAR | companyfacts XBRL (tier-1 cross-check), filing acceptance timestamps (`filing_timestamp` clock) |

**Rules**

- Source hierarchy per `shared/SOURCE_HIERARCHY.md`: regulatory filings (EDGAR) outrank aggregators (FMP). Material conflict on shares/debt/cash/price (moves a category score ≥0.5/10, valuation midpoint ≥5%, an S/R zone ≥0.5 ATR, or a gate) → field marked `CONFLICTED`; both values retained in lineage; per-share valuation suppressed (Override 7).
- Every raw response cached to `engine/cache/<TICKER>/` with fetch timestamp → offline reruns, fixtures, staleness checks per `shared/DATA_POLICY.md` (daily data: 1 session; consensus: 7 days; quarterly fundamentals: 120 days; market-size studies: 18 months; peer set: 90 days).
- Keys loaded from `API/.env` (python-dotenv). Missing key → provider disabled with a warning; affected fields `MISSING`. Never crash; never print key values.

## 5. Core engines

**Formula registry** (`core/formulas.py`) — every formula is a registered object `{id, version, fn, inputs, units, min_history, missing_behavior}` returning the standard result `{formula_id, formula_version, inputs, result{value, unit}, status, warnings[]}`. IDs match the Cerebro exactly: BUS-001…030, FIN-001…033, MKT-001…025, TECH-001…040, RSK-001…035, VAL-001…044. Calculation conventions from `shared/CALCULATION_CONVENTIONS.md` (average balance-sheet denominators with `END_BALANCE_PROXY` fallback; CAGR never across sign changes; √252 annualization; 5th/95th peer winsorization).

**Scoring engine** (`core/scoring.py`) — piecewise-linear anchor interpolation clamped [0,10]; peer-relative percentile scoring (min 8 valid peers, else absolute fallback or `NOT_SCORABLE`); hybrid weighting; missing metrics reweighted only if ≥70% of metric weight valid; category `INCOMPLETE` below 70% valid dimension weight (gate-ineligible). `dimension_points = dimension_max × score/10`; `raw_total = Σ category_points` (0–100). Weights fixed: Business 20, Financial 15, Market 20, Technical 20, Risk 15, Valuation 10.

**Confidence engine** (`core/confidence.py`) — `0.30·coverage + 0.25·source_quality + 0.20·freshness + 0.15·consistency + 0.10·model_fit`, per category, from lineage metadata. Total confidence = points-weighted average. Confidence never alters a score; <50 → category cannot pass a gate.

**Null states** (`core/nullstates.py`) — a `Value` wrapper: every metric is a number-with-lineage or one of the six allowed null states (`MISSING`, `NOT_APPLICABLE`, `NOT_MEANINGFUL`, `CONFLICTED`, `STALE`, `NOT_SCORABLE`). No bare `None` in the pipeline. The prohibited-imputation list (`shared/MISSING_DATA_POLICY.md`) is enforced in code — those fields are never estimated.

## 6. Specialists & special sauces

Each specialist module (`specialists/*.py`) computes its formula set from the packet, scores its dimensions, applies its caps (e.g., no positive ROIC−WACC spread → moat capped at 6; largest customer >30% → durability capped at 6), and emits a pydantic object matching its `OUTPUT_SCHEMA.md`, including mandatory flags (`VALUE_DESTRUCTION`, `SOLVENCY_WARNING`, `CONCENTRATION_RED_FLAG`, `DILUTION_RED_FLAG`).

**Valuation engine** (`engines/valuation_engine.py`) — per `special_sauces/INSTITUTIONAL_VALUATION_ENGINE.md`:
normalized EBIT; R&D capitalization; operating-lease debt; NOPAT/ROIC/EVA block; bottom-up beta → WACC with ±100bp sensitivity; FCFF DCF with Gordon terminal value (terminal-share >75% → high-sensitivity warning); equity bridge → per-share; FCFE, economic-profit and residual-income cross-checks; reverse DCF (scipy solver for ≥2 implied unknowns); Bear/Base/Bull scenarios with declared assumptions; seeded Monte Carlo (numpy) reporting p10/p25/median/p75/p90 + seed/trials/correlations; reliability-weighted ensemble; margin-of-safety bands (15%/25%).

**Levels engine** (`engines/levels_engine.py`) — per `special_sauces/IMPORTANT_LEVELS_ENGINE.md`, exact parameters:
symmetric pivots (k=3 daily, k=2 weekly) + 1.5·ATR14 zigzag; zone tolerance `max(0.50·ATR14, 0.75% price)`; recency weighting `exp(−ln2·age/126)`; independent-touch rules (≥5 sessions apart daily); strength score 0–100 with the registered component weights; level states (Candidate/Confirmed/Strong/Broken/Role-reversed); breakout confirmation (close > zone_high + 0.25·ATR, volume ratio ≥1.5, persistence rule); AVWAP from approved anchors; approximate volume profile (POC/HVN/LVN); earnings-gap materiality `max(1.0 ATR, 3%)`; relevance ranking `0.45·strength + 0.25·recency + 0.20·confluence + 0.10·liquidity`.

Scenario assumptions (growth/margin per scenario) default to derived values (historical CAGR, consensus, fundamental growth = reinvestment × ROIC) and are agent-overridable via the overlay.

**Industry adapters** — v1 fully implements the general non-financial adapter. Banks, insurers, REITs, biotech, and early-stage are *detected* (sector/SIC) and flagged `ADAPTER_UNSUPPORTED`: affected metrics `NOT_APPLICABLE`, coverage and confidence reduced honestly, no mis-scoring.

## 7. Judgment overlay

`wbj compute` emits `judgment_requests.json`: every item code cannot score —

- moat classification + quantitative-evidence assessment
- catalyst probability / impact / evidence quality
- TAM source-tier assignment
- thesis killers (probability, impact, detectability, urgency)
- management track record, notable 13F holder interpretation
- qualitative regulatory/legal risk

Those metrics sit at `NOT_SCORABLE` with the coverage/confidence penalty the Cerebro dictates. Sub-agents write `judgments.json` (each answer with evidence class + source). `wbj aggregate --overlay judgments.json` validates entries against the same schemas, merges, re-scores, re-freezes with a new packet hash. Without an overlay the pipeline still completes — with lower coverage, honestly reported.

## 8. Aggregation

Applied in order, per `00_main_agent/SCORING_AND_GATES.md`:

1. **Raw total** — sum of six frozen category points; each packet re-validated against the handoff contract (dimension points must reproduce category points, all scores traceable to formula IDs).
2. **Mandatory overrides** (all 7): capital dependence → cap Avoid/Speculative; ROIC<WACC → no Elite/Quality; interest coverage <1.5× → solvency warning; Risk ≤4/15 → cap Speculative; Valuation ≤4 + Technical ≤8 → Wait/Avoid; category coverage <70% → gate-ineligible; unresolved data conflict → suppress per-share value.
3. **Profile gates** with exact thresholds (Momentum ≥78 raw + Technical ≥17 + …; Quality ≥80 raw + …; Value ≥75 raw + Valuation ≥8 + …; Conditional/Watch; Speculative; Avoid/Wait).
4. **Contradiction resolution** — lookup table from `CONTRADICTION_RESOLUTION.md`; scores never altered, contradictions surfaced as labeled information.
5. **Price-level synthesis** — 12 required level classes; confluence when two independent references overlap within `max(0.50·ATR14, 0.75% price)` with at least one technical; technical and intrinsic levels never averaged; language whitelist enforced (reference/zone/confirmation/invalidation — never "target"/"must hold").

## 9. Report

Output to `Reportes/<TICKER>/<YYYY-MM-DD>/`: `report.md`, `report.json` (validated against `FINAL_REPORT_SCHEMA.md`), `charts/*.png`.

Contents = final report schema **plus** the root-CLAUDE.md additions:

1. Research classification — good price? invest vs avoid (evidence-based classification, never an order)
2. If "avoid" → concrete revisit date or event
3. Price **ranges** by scenario with stated assumptions — never a single price
4. Notable 13F institutional holders + management track record
5. Insider buying/selling — only trades **>$1M USD total** flagged as important (Forms 4)
6. Profile-fit section: $25,000 capital, 30–60% max position, 3–5y horizon, aggressive/speculative, US-only
7. Audit appendix: packet hashes, formula versions, validation summary, missing/conflicted data

**Charts** (matplotlib, per the four visualization rules): price + levels + SMAs; scenario fan (solid historical, dotted projected, bands never single lines, assumptions labeled on-chart); category scorecard; valuation football field.

## 10. Error handling

- Provider failure → 3 retries with backoff → fields `MISSING`.
- Packet builder rejects only on Cerebro hard requirements (missing timestamps, units, currencies, period definitions, sources).
- Everything else degrades gracefully: compute what's computable, mark the rest with the correct null state, report coverage/confidence honestly. "Sin evidencia, no hay número."
- If insufficient data for a conclusion, the report states: "No tengo data suficiente para llegar a una conclusión de inversión" (rendered in English: "Insufficient data to reach an investment conclusion") and marks affected dimensions `NOT_SCORABLE`.

## 11. Testing

- Every row of every `VALIDATION_TESTS.md` (7 files, ~10 rows each) becomes a pytest case — e.g., MAIN-002 (16+10.5+18+16+9+7 = 76.5), MAIN-003 (Risk=4, total=90 → Speculative), MAIN-007 (coverage 0.65 → no gate).
- Indicator math verified against known series (Wilder RSI14/ATR14, MACD, ADX).
- Golden-file test: full pipeline offline on a cached fixture ticker; output compared to a committed golden report.json.
- Schema round-trip tests for every pydantic model.
- CI-ready: `pytest` runs with zero network access (fixtures only).

## 12. Out of scope (v1)

- Robinhood portfolio integration
- Real-options valuation (marked `NOT_SCORABLE` where the model-selection matrix would choose it)
- Non-general industry adapters (banks/insurers/REITs/biotech — detected and flagged, not scored)
- Backtesting harness (`shared/BACKTESTING_AND_CALIBRATION.md`) — engine is point-in-time-correct by design (five clocks respected) so a harness can be added later
