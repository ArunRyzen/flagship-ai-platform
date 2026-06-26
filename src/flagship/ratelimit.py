"""A simple per-client sliding-window rate limiter."""

from __future__ import annotations

from time import monotonic


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._hits: dict[str, list[float]] = {}

    def allow(self, key: str) -> bool:
        now = monotonic()
        recent = [t for t in self._hits.get(key, []) if t >= now - self._window]
        if len(recent) >= self._max:
            self._hits[key] = recent
            return False
        recent.append(now)
        self._hits[key] = recent
        return True
