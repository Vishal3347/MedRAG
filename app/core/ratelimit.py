"""Tiny in-memory sliding-window rate limiter (protects the free-tier LLM quotas)."""
from __future__ import annotations

import time
from collections import deque
from typing import Callable


class SlidingWindowLimiter:
    def __init__(self, limit: int, window_s: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self.limit = limit
        self.window_s = window_s
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}

    def check(self, key: str) -> tuple[bool, float]:
        """Returns (allowed, retry_after_seconds)."""
        if self.limit <= 0:
            return True, 0.0
        now = self._clock()
        if len(self._hits) > 10_000:
            self._gc(now)
        q = self._hits.setdefault(key, deque())
        cutoff = now - self.window_s
        while q and q[0] <= cutoff:
            q.popleft()
        if len(q) >= self.limit:
            return False, max(0.0, q[0] + self.window_s - now)
        q.append(now)
        return True, 0.0

    def _gc(self, now: float) -> None:
        cutoff = now - self.window_s
        for key in [k for k, q in self._hits.items() if not q or q[-1] <= cutoff]:
            del self._hits[key]
