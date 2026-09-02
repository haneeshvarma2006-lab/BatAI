"""Per-tenant rate limiting and concurrency control.

Two distinct limits, because the costs are two orders of magnitude apart:

* a **token bucket** on request rate, for cheap CRUD calls;
* a **semaphore** on concurrent agent runs, because one run can occupy a model
  server slot for a minute and a tenant that opens fifty of them starves
  everyone else on the box.

Both are per-process. With more than one replica, the effective limit is
``replicas x configured`` — for a hard global limit these move behind Redis
(``INCR`` + ``EXPIRE`` for the bucket, a lease key for concurrency) without any
change to the call sites, which is why the surface here is intentionally small.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from bat.domain.errors import RateLimitError


@dataclass(slots=True)
class _Bucket:
    tokens: float
    updated_at: float


class TokenBucketLimiter:
    """Classic token bucket keyed by an arbitrary string (here: tenant id)."""

    __slots__ = ("_buckets", "_burst", "_lock", "_rate")

    def __init__(self, *, rate_per_second: float, burst: int) -> None:
        if rate_per_second <= 0 or burst <= 0:
            raise ValueError("rate_per_second and burst must be positive")
        self._rate = rate_per_second
        self._burst = float(burst)
        self._buckets: dict[str, _Bucket] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, key: str, *, cost: float = 1.0) -> None:
        """Consume ``cost`` tokens or raise :class:`RateLimitError`."""
        retry_after = await self._try_consume(key, cost)
        if retry_after is not None:
            raise RateLimitError(
                "rate limit exceeded",
                retry_after_seconds=retry_after,
                details={"limit_per_second": self._rate, "burst": int(self._burst)},
            )

    async def _try_consume(self, key: str, cost: float) -> float | None:
        """Return ``None`` on success, else the suggested retry delay."""
        now = time.monotonic()
        async with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=self._burst, updated_at=now)
                self._buckets[key] = bucket

            elapsed = now - bucket.updated_at
            bucket.tokens = min(self._burst, bucket.tokens + elapsed * self._rate)
            bucket.updated_at = now

            if bucket.tokens >= cost:
                bucket.tokens -= cost
                return None
            deficit = cost - bucket.tokens
            return round(deficit / self._rate, 3)

    async def evict_idle(self, *, older_than_s: float = 900.0) -> int:
        """Drop buckets untouched for a while, so keys cannot accumulate."""
        cutoff = time.monotonic() - older_than_s
        async with self._lock:
            stale = [k for k, b in self._buckets.items() if b.updated_at < cutoff]
            for key in stale:
                del self._buckets[key]
            return len(stale)


class ConcurrencyLimiter:
    """Caps simultaneous expensive operations per key, without queueing.

    Fails fast rather than blocking: a queued agent run holds a connection and
    burns the client's deadline for no benefit, so an over-limit tenant is told
    to retry instead. Because admission never waits, a counter guarded by the
    lock is the whole mechanism -- a semaphore would only add a check-then-wait
    race between the admission test and the acquire.
    """

    __slots__ = ("_in_flight", "_limit", "_lock")

    def __init__(self, *, limit: int) -> None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        self._limit = limit
        self._in_flight: dict[str, int] = {}
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def slot(self, key: str) -> AsyncIterator[None]:
        async with self._lock:
            current = self._in_flight.get(key, 0)
            if current >= self._limit:
                raise RateLimitError(
                    "too many concurrent agent runs for this tenant",
                    retry_after_seconds=1.0,
                    details={"max_concurrent_runs": self._limit},
                )
            self._in_flight[key] = current + 1
        try:
            yield
        finally:
            async with self._lock:
                remaining = self._in_flight.get(key, 1) - 1
                if remaining > 0:
                    self._in_flight[key] = remaining
                else:
                    # Drop the key entirely so idle tenants cost nothing.
                    self._in_flight.pop(key, None)

    def in_flight(self, key: str) -> int:
        return self._in_flight.get(key, 0)


@dataclass(slots=True)
class NullLimiter:
    """Disabled limiter, used when rate limiting is switched off."""

    calls: int = field(default=0)

    async def acquire(self, key: str, *, cost: float = 1.0) -> None:
        self.calls += 1
