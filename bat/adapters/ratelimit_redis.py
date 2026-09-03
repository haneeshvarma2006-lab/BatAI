"""Redis-backed rate limiting and concurrency control.

The in-process limiters in ``bat.ratelimit`` are correct but per-process, so the
effective limit is ``replicas x configured`` -- a tenant on a four-replica
deployment gets four times its quota, and the ceiling moves whenever the cluster
autoscales. These make the limit global.

Both operations are single Lua scripts. That matters more than it looks:
read-modify-write from the client is a lost-update race under concurrency, and
the race is *in the tenant's favour*, so the limit silently leaks exactly when
it is under the most pressure. Redis runs a script atomically, so check and
decrement cannot interleave.

Leases, not counters, for concurrency
-------------------------------------
An in-process semaphore is released by a ``finally`` block. Across a network,
the holder can die -- OOM kill, pod eviction, power loss -- and never release,
which would permanently shrink a tenant's capacity with no way to recover short
of manual intervention. So a slot is a *lease* in a sorted set, scored by expiry:
stale entries are dropped on the next admission check, and a crashed worker's
slot returns on its own.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from bat.domain.errors import RateLimitError, UpstreamError

logger = logging.getLogger("bat.ratelimit.redis")

#: Token bucket. KEYS[1] = bucket key.
#: ARGV = rate, burst, now (seconds, float), cost, ttl.
#: Returns {allowed, retry_after_seconds}.
_TOKEN_BUCKET_LUA = """
local key   = KEYS[1]
local rate  = tonumber(ARGV[1])
local burst = tonumber(ARGV[2])
local now   = tonumber(ARGV[3])
local cost  = tonumber(ARGV[4])
local ttl   = tonumber(ARGV[5])

local state = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(state[1])
local ts = tonumber(state[2])

if tokens == nil then
  tokens = burst
  ts = now
end

-- Refill for elapsed time, capped at the burst size. `math.max(0, ...)` guards
-- against a clock that went backwards between calls on different replicas.
local elapsed = math.max(0, now - ts)
tokens = math.min(burst, tokens + elapsed * rate)

local allowed = 0
local retry_after = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
else
  retry_after = (cost - tokens) / rate
end

redis.call('HSET', key, 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', key, ttl)
return {allowed, tostring(retry_after)}
"""

#: Concurrency lease. KEYS[1] = lease set. ARGV = now, limit, lease_id, ttl.
#: Returns 1 when admitted, 0 when the tenant is at its ceiling.
_LEASE_ACQUIRE_LUA = """
local key    = KEYS[1]
local now    = tonumber(ARGV[1])
local limit  = tonumber(ARGV[2])
local lease  = ARGV[3]
local ttl    = tonumber(ARGV[4])

-- Reap leases whose holder died without releasing.
redis.call('ZREMRANGEBYSCORE', key, '-inf', now)

if redis.call('ZCARD', key) >= limit then
  return 0
end

redis.call('ZADD', key, now + ttl, lease)
redis.call('EXPIRE', key, math.ceil(ttl) + 60)
return 1
"""


def _key(prefix: str, kind: str, identity: str) -> str:
    return f"{prefix}:{kind}:{identity}"


class RedisTokenBucketLimiter:
    """Global per-tenant request budget.

    Drop-in for :class:`bat.ratelimit.TokenBucketLimiter` -- same ``acquire``
    signature, so nothing at the call sites changes.
    """

    __slots__ = ("_burst", "_client", "_fail_open", "_prefix", "_rate", "_script", "_ttl")

    def __init__(
        self,
        client: Any,
        *,
        rate_per_second: float,
        burst: int,
        prefix: str = "bat:rl",
        idle_ttl_s: int = 900,
        fail_open: bool = True,
    ) -> None:
        if rate_per_second <= 0 or burst <= 0:
            raise ValueError("rate_per_second and burst must be positive")
        self._client = client
        self._rate = rate_per_second
        self._burst = burst
        self._prefix = prefix
        self._ttl = idle_ttl_s
        # A Redis outage should not take the whole API down with it. Failing
        # open means an outage costs rate limiting, not availability -- the
        # right trade when the limiter protects capacity rather than data.
        # Set False where the limit is a billing or abuse control.
        self._fail_open = fail_open
        self._script = client.register_script(_TOKEN_BUCKET_LUA)

    async def acquire(self, key: str, *, cost: float = 1.0) -> None:
        try:
            allowed, retry_after = await self._script(
                keys=[_key(self._prefix, "bucket", key)],
                args=[self._rate, self._burst, time.time(), cost, self._ttl],
            )
        except Exception as exc:
            if self._fail_open:
                logger.error(
                    "rate limiter unavailable; allowing request",
                    extra={"tenant_id": key, "error": str(exc)},
                )
                return
            raise UpstreamError(f"rate limiter unavailable: {exc}") from exc

        if not int(allowed):
            raise RateLimitError(
                "rate limit exceeded",
                retry_after_seconds=round(float(retry_after), 3),
                details={"limit_per_second": self._rate, "burst": self._burst},
            )

    async def verify(self) -> None:
        """Run the script once against a throwaway key.

        Without this, a Redis that lacks scripting -- or a typo in the Lua --
        fails on every request and is swallowed by `fail_open`, so the limiter
        silently stops limiting while looking healthy. Fail at boot instead.
        """
        probe = _key(self._prefix, "bucket", "__verify__")
        try:
            await self._script(
                keys=[probe], args=[self._rate, self._burst, time.time(), 0, 5]
            )
            await self._client.delete(probe)
        except Exception as exc:
            raise UpstreamError(
                f"Redis rate limiter script does not run: {exc}"
            ) from exc

    async def peek(self, key: str) -> float:
        """Tokens currently available. Diagnostics only; does not consume."""
        raw = await self._client.hget(_key(self._prefix, "bucket", key), "tokens")
        return float(raw) if raw is not None else float(self._burst)


class RedisConcurrencyLimiter:
    """Global cap on simultaneous agent runs per tenant.

    Drop-in for :class:`bat.ratelimit.ConcurrencyLimiter`.
    """

    __slots__ = ("_acquire_script", "_client", "_fail_open", "_lease_ttl", "_limit", "_prefix")

    def __init__(
        self,
        client: Any,
        *,
        limit: int,
        prefix: str = "bat:rl",
        lease_ttl_s: float = 300.0,
        fail_open: bool = True,
    ) -> None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        self._client = client
        self._limit = limit
        self._prefix = prefix
        # Must exceed the longest possible run, or a live run's lease expires
        # and the tenant is over its ceiling. Keep it above `agent.deadline_s`.
        self._lease_ttl = lease_ttl_s
        self._fail_open = fail_open
        self._acquire_script = client.register_script(_LEASE_ACQUIRE_LUA)

    @asynccontextmanager
    async def slot(self, key: str) -> AsyncIterator[None]:
        lease_id = uuid.uuid4().hex
        redis_key = _key(self._prefix, "runs", key)
        held = False

        try:
            admitted = await self._acquire_script(
                keys=[redis_key],
                args=[time.time(), self._limit, lease_id, self._lease_ttl],
            )
            held = bool(int(admitted))
        except Exception as exc:
            if not self._fail_open:
                raise UpstreamError(f"concurrency limiter unavailable: {exc}") from exc
            logger.error(
                "concurrency limiter unavailable; allowing run",
                extra={"tenant_id": key, "error": str(exc)},
            )
            yield
            return

        if not held:
            raise RateLimitError(
                "too many concurrent agent runs for this tenant",
                retry_after_seconds=1.0,
                details={"max_concurrent_runs": self._limit},
            )

        try:
            yield
        finally:
            try:
                await self._client.zrem(redis_key, lease_id)
            except Exception:
                # The lease expires on its own; losing the release is a delay,
                # not a leak, which is the whole point of using leases.
                logger.warning(
                    "could not release run lease; it will expire",
                    extra={"tenant_id": key, "lease_ttl_s": self._lease_ttl},
                )

    async def verify(self) -> None:
        """Acquire and release one probe lease, so a broken script fails at boot."""
        probe = _key(self._prefix, "runs", "__verify__")
        try:
            await self._acquire_script(
                keys=[probe], args=[time.time(), self._limit, "probe", 5]
            )
            await self._client.delete(probe)
        except Exception as exc:
            raise UpstreamError(
                f"Redis concurrency limiter script does not run: {exc}"
            ) from exc

    async def in_flight(self, key: str) -> int:
        """Live leases, excluding any that have expired."""
        redis_key = _key(self._prefix, "runs", key)
        await self._client.zremrangebyscore(redis_key, "-inf", time.time())
        return int(await self._client.zcard(redis_key))


async def create_redis_client(dsn: str) -> Any:
    """Connect and verify with a PING, so a bad DSN fails at startup."""
    import redis.asyncio as redis

    try:
        client = redis.from_url(dsn, decode_responses=True)
        await client.ping()
    except Exception as exc:
        raise UpstreamError(f"could not connect to Redis: {exc}") from exc
    return client
