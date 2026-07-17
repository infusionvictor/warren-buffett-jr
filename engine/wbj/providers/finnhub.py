"""FinnHub provider.

Wraps a slice of the FinnHub REST API: consensus EPS/revenue estimates,
the earnings calendar, and real-time quote. `FinnhubProvider` is disabled
(`available == False`) when no API key is configured; every public
method then returns `None` immediately without touching the cache or the
network. Requests and caching are delegated to
`wbj.providers.base.Provider.get_json` — this module only builds
URLs/params and picks cache keys / max_age_days per data type.

FinnHub's key is passed as the `token` query param; `base.Provider`
already redacts `token` from logged request params.
"""

from __future__ import annotations

from typing import Any

from wbj.providers.base import Provider

BASE_URL = "https://finnhub.io/api/v1"

# max_age_days per cache key, per task brief:
#   quote 1, estimates/revenue_estimates/earnings_calendar 7.
_MAX_AGE_QUOTE = 1
_MAX_AGE_ESTIMATES = 7
_MAX_AGE_CALENDAR = 7


class FinnhubProvider(Provider):
    """FinnHub data provider."""

    @property
    def available(self) -> bool:
        """True iff a FinnHub API key is configured."""
        return bool(self.settings and getattr(self.settings, "finnhub_api_key", None))

    def _params(self, **extra: Any) -> dict[str, Any]:
        params = {"token": self.settings.finnhub_api_key}
        params.update(extra)
        return params

    def estimates(self, t: str) -> list | dict | None:
        """Consensus EPS estimates (history + forward)."""
        if not self.available:
            return None
        return self.get_json(
            f"{BASE_URL}/stock/eps-estimate",
            self._params(symbol=t),
            "estimates",
            t,
            max_age_days=_MAX_AGE_ESTIMATES,
        )

    def revenue_estimates(self, t: str) -> list | dict | None:
        """Consensus revenue estimates (history + forward)."""
        if not self.available:
            return None
        return self.get_json(
            f"{BASE_URL}/stock/revenue-estimate",
            self._params(symbol=t),
            "revenue_estimates",
            t,
            max_age_days=_MAX_AGE_ESTIMATES,
        )

    def earnings_calendar(self, t: str) -> list | dict | None:
        """Upcoming/historical earnings calendar entries."""
        if not self.available:
            return None
        return self.get_json(
            f"{BASE_URL}/calendar/earnings",
            self._params(symbol=t),
            "earnings_calendar",
            t,
            max_age_days=_MAX_AGE_CALENDAR,
        )

    def quote(self, t: str) -> list | dict | None:
        """Real-time quote: current price, change, high/low/open, prev close."""
        if not self.available:
            return None
        return self.get_json(
            f"{BASE_URL}/quote",
            self._params(symbol=t),
            "quote",
            t,
            max_age_days=_MAX_AGE_QUOTE,
        )
