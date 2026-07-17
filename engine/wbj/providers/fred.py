"""FRED (Federal Reserve Economic Data) provider.

Wraps the FRED `series/observations` endpoint for macro time series (e.g.
`DGS10`, the 10-year Treasury constant maturity rate) and exposes a
`risk_free_rate()` helper built on top of it. `FredProvider` is disabled
(`available == False`) when no API key is configured; every public
method then returns `None`/a null `Value` immediately without touching
the cache or the network. Requests and caching are delegated to
`wbj.providers.base.Provider.get_json` — this module only builds
URLs/params and picks cache keys / max_age_days.

FRED series are macro, not per-ticker, so they are cached under the
synthetic ticker `_macro` rather than under any equity symbol.

FRED's key is passed as the `api_key` query param; `base.Provider`
redacts `api_key` (along with `apikey`/`token`) from logged request
params.
"""

from __future__ import annotations

from typing import Any

from wbj.core.nullstates import NullState, Value
from wbj.providers.base import Provider

BASE_URL = "https://api.stlouisfed.org/fred"

_MACRO_TICKER = "_macro"

# max_age_days per cache key, per task brief: series 1.
_MAX_AGE_SERIES = 1

_RISK_FREE_SERIES_ID = "DGS10"


class FredProvider(Provider):
    """FRED macro data provider."""

    @property
    def available(self) -> bool:
        """True iff a FRED API key is configured."""
        return bool(self.settings and getattr(self.settings, "fred_api_key", None))

    def _params(self, **extra: Any) -> dict[str, Any]:
        params = {"api_key": self.settings.fred_api_key, "file_type": "json"}
        params.update(extra)
        return params

    def series(self, series_id: str, limit: int = 120) -> list | dict | None:
        """Most recent `limit` observations for `series_id`, newest first."""
        if not self.available:
            return None
        return self.get_json(
            f"{BASE_URL}/series/observations",
            self._params(series_id=series_id, sort_order="desc", limit=limit),
            f"fred_{series_id}",
            _MACRO_TICKER,
            max_age_days=_MAX_AGE_SERIES,
        )

    def risk_free_rate(self) -> Value:
        """Latest DGS10 (10y Treasury) observation, as a ratio (e.g. 0.0425).

        Returns `Value.null(NullState.MISSING, ...)` when the provider is
        unavailable, the fetch fails, or every recent observation is the
        FRED "." (not-yet-released) placeholder.
        """
        payload = self.series(_RISK_FREE_SERIES_ID)
        if not isinstance(payload, dict):
            return Value.null(
                NullState.MISSING,
                unit="ratio",
                source_name="FRED",
                source_locator=_RISK_FREE_SERIES_ID,
            )

        observations = payload.get("observations") or []
        for obs in observations:
            raw_value = obs.get("value")
            if raw_value is None or raw_value == ".":
                continue
            try:
                pct = float(raw_value)
            except (TypeError, ValueError):
                continue
            return Value.of(
                pct / 100.0,
                unit="ratio",
                source_name="FRED",
                source_locator=_RISK_FREE_SERIES_ID,
            )

        return Value.null(
            NullState.MISSING,
            unit="ratio",
            source_name="FRED",
            source_locator=_RISK_FREE_SERIES_ID,
        )
