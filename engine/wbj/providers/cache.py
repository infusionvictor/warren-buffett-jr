"""Filesystem-backed JSON response cache for wbj providers.

File layout: `<cache_dir>/<TICKER>/<key>.json` containing
`{"fetched_at": iso8601 UTC, "payload": ...}`.

This module reads the wall clock (`datetime.now(timezone.utc)`) to stamp
and age cache entries — that is infrastructure bookkeeping, not analysis
math, and is exempt from the engine's null-state/lineage discipline
(see `wbj.core.nullstates`).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class Cache:
    """Filesystem-backed JSON cache, keyed by ticker and cache key."""

    def __init__(self, cache_dir: Path | str) -> None:
        self.cache_dir = Path(cache_dir)

    def _path(self, ticker: str, key: str) -> Path:
        return self.cache_dir / ticker / f"{key}.json"

    def _read_record(self, ticker: str, key: str) -> dict | None:
        path = self._path(ticker, key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def get(self, ticker: str, key: str) -> dict | None:
        """Return the cached payload for (ticker, key), or None if absent/corrupt."""
        record = self._read_record(ticker, key)
        if record is None:
            return None
        return record.get("payload")

    def put(self, ticker: str, key: str, payload: dict) -> None:
        """Write payload to cache, stamped with the current UTC time."""
        path = self._path(ticker, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
        path.write_text(json.dumps(record), encoding="utf-8")

    def age_days(self, ticker: str, key: str) -> float | None:
        """Return the cache entry's age in days, or None if absent/corrupt."""
        record = self._read_record(ticker, key)
        if record is None:
            return None
        try:
            fetched_at = datetime.fromisoformat(record["fetched_at"])
        except (KeyError, ValueError, TypeError):
            return None
        delta = datetime.now(timezone.utc) - fetched_at
        return delta.total_seconds() / 86400.0
