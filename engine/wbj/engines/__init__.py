"""Compute engines: pure pandas/numpy calculations over adjusted OHLCV.

Distinct from `wbj.core` (the `Value`/null-state and formula-registry
layer used by specialist scoring): engines in this package take and return
plain pandas `Series`/`DataFrame` (or `float`), with no `Value` wrapping,
so specialists can vectorize technical/levels/valuation math over full
history before wrapping individual point-in-time results in `Value` at the
call site.
"""

from __future__ import annotations
