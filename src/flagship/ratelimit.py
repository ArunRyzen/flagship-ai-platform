"""A simple per-client sliding-window rate limiter.

Beginner note: each client (keyed by IP) may make at most `max_requests` requests in any
rolling `window_seconds` window. We remember request timestamps per client; on each new
request we drop the ones older than the window and check whether the survivors already fill
the quota. This protects the service (and your API bill) from a single noisy caller.
"""

from __future__ import annotations

from time import monotonic


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._hits: dict[str, list[float]] = {}  # client key → timestamps of recent requests

    def allow(self, key: str) -> bool:
        """True if this client may proceed; False means "send back HTTP 429"."""
        now = monotonic()  # monotonic clock: immune to system-time jumps
        recent = [t for t in self._hits.get(key, []) if t >= now - self._window]
        if len(recent) >= self._max:
            self._hits[key] = recent  # keep the pruned list; do NOT record the denied hit
            return False
        recent.append(now)
        self._hits[key] = recent
        return True
