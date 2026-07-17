# Company Brief — plain-Spanish read of the scorecard for the web app

**Date:** 2026-07-17
**Status:** Approved — implementing

## Problem

When Victor analyzes a company he gets the 6-category scorecard and Bull/Base/Bear
targets, but not, in plain words: **what the score means**, **the probability the
price reaches each target**, **where the company is right now**, and **the key
things to watch**. He wants all four in the web app, below the scorecard.

## Design

A new pure module `wbj/brief.py` — `company_brief(packet, scorecard, targets) -> dict`
— assembles four sections from data already in the packet plus two extra FMP
feeds. No network I/O in the module (fully unit-testable); `_build_packet`
fetches the extras, the webapp renders the dict.

### Sections

1. **`interpretation`** — score meaning + research classification.
   - Per category: `score10` → plain band (`>=8` "muy fuerte", `6.5-8` "sólido",
     `5-6.5` "mixto", `3.5-5` "débil", `<3.5` "problemático", N/S "sin datos").
   - Overall: research classification `favorece` / `neutral` / `evitar` from
     `overall_10` (`>=7` favorece, `5-7` neutral, `<5` evitar) — labeled a
     research classification, never a buy/sell order (CLAUDE.md). When `evitar`,
     a concrete revisit condition (next earnings date, or "nuevo 10-K/10-Q").

2. **`probability`** — volatility-based reachability toward each target.
   - σ = annualized historical volatility from the packet's daily closes
     (`stdev(daily log returns) * sqrt(252)`, ~1y).
   - Horizon T = 1 year. Neutral assumption: median = current price (zero
     directional drift), so `ln(S_T/S0) ~ N(0, σ²T)`.
   - `P(S_T >= K) = Φ( ln(S0/K) / (σ·√T) )` for each scenario target K.
   - Also the **modal zone** (below Bear / Bear–Base / Base–Bull / above Bull)
     from the same distribution, and the declared assumptions (σ, horizon,
     "sin sesgo direccional").
   - N/S when targets are not `ok`, price missing, `< ~60` sessions of history,
     or σ = 0. Never an invented probability.

3. **`where`** — reuse `targets.narrative(packet, scorecard, targets)` verbatim.

4. **`watch`** — four blocks:
   - `levels`: entrada aprox. (precio actual), invalidación (Bear target),
     salida (Base / Bull targets). Reference, not an order.
   - `catalysts`: soonest earnings row dated after `as_of` (FMP earnings
     calendar) — date + estimated EPS/revenue.
   - `insiders`: Form 4 rows where `securitiesTransacted * price > $1M`,
     classified buy (`transactionType` starts "P") vs sell ("S"), most recent
     first, capped to a few. Empty list when none clear the threshold.
   - `risks`: 2–3 thesis-killers derived from the data (debt/equity high,
     negative FCF, falling net margin, low risk-category score, negative
     revenue growth).

### Data flow

- `_build_packet` (cli.py **and** webapp.py share the cli builder): extend the
  `market_data` block with `earnings` (`fmp.earnings_calendar`) and `insiders`
  (`fmp.insider_trades`) when `fmp.available`. Missing feed → that watch block
  is empty/absent, never fabricated.
- `webapp.analyze()`: `result["brief"] = company_brief(packet, scorecard, targets)`.
- Frontend: a new full-width card `c-brief` with a `briefHtml(d)` renderer —
  interpretation chips, a probability bar (4 zones, dotted = future) toward the
  targets, the "dónde está" sentences, and the four watch blocks.

### Field-drift tolerance (learned from Phase 1)

Live FMP `/stable/` names differ from fixtures. `brief.py` reads defensively:
earnings actual `epsActual`|`eps`; insider value from `securitiesTransacted *
price` (gifts/awards have `price == 0` → value 0 → filtered out).

## Honesty rules (Cerebro)

Every number carries a formula and declared assumptions. Probability is a
labeled statistical model under a neutral drift assumption — not a promise of
return. Any missing input → the affected block is N/S, never imputed.

## Testing (TDD)

`wbj/brief.py` pure functions:
- probability math: known σ, S0, K → known Φ (e.g. K = S0 → 0.50).
- interpretation bands and research classification thresholds.
- watch-list: insider $1M threshold (incl. price-0 gift excluded), next-earnings
  selection after `as_of`, risk derivation, all N/S paths.
- fixtures for earnings/insiders (live `/stable/` shape).

## Out of scope

- Full specialist engines (Phase 2 of quick-fmp-scoring).
- Barrier/touch probabilities (terminal-at-horizon only).
- Options-implied volatility (historical vol only).
