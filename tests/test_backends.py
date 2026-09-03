"""Tests for the Postgres session store and the Redis limiters.

The Redis half runs against ``fakeredis[lua]``, which executes the Lua scripts
for real -- so atomicity, refill and lease expiry are genuinely exercised, not
mocked.

The Postgres half is **not** equivalently covered, and the docstrings say where
the line is: the SQL text has never been executed by a Postgres server. What is
tested here is everything around it -- cursor encoding, tenant arguments,
transaction boundaries, row mapping and error translation -- against a fake
connection that records statements. Read `TestPostgresQueries` as "the adapter
sends the right arguments", never as "the SQL is correct".
"""

from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime, timedelta
from typing import Any

import fakeredis.aioredis as fakeredis

from bat.adapters.ratelimit_redis import (
    RedisConcurrencyLimiter,
    RedisTokenBucketLimiter,
)
from bat.adapters.session_store_postgres import (
    SCHEMA,
    PostgresSessionStore,
    _decode_cursor,
    _encode_cursor,
    _to_message,
    _to_session,
)
from bat.domain.conversation import Message, Role, Session
from bat.domain.errors import ConflictError, NotFoundError, RateLimitError, ValidationError
from bat.ports.session_store import SessionStore
from bat.settings import AgentSettings, RateLimitSettings, Settings


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Redis limiters -- real Lua execution
# ---------------------------------------------------------------------------


class TestRedisTokenBucket(unittest.TestCase):
    def setUp(self) -> None:
        self.client = fakeredis.FakeRedis(decode_responses=True)

    def limiter(self, **kwargs: Any) -> RedisTokenBucketLimiter:
        kwargs.setdefault("rate_per_second", 100.0)
        kwargs.setdefault("burst", 3)
        return RedisTokenBucketLimiter(self.client, **kwargs)

    def test_verify_proves_the_script_runs(self) -> None:
        run(self.limiter().verify())

    def test_burst_is_enforced(self) -> None:
        limiter = self.limiter()

        async def scenario():
            for _ in range(3):
                await limiter.acquire("acme")
            await limiter.acquire("acme")

        with self.assertRaises(RateLimitError) as caught:
            run(scenario())
        self.assertGreater(caught.exception.retry_after_seconds, 0)

    def test_limit_is_per_tenant(self) -> None:
        limiter = self.limiter()

        async def scenario():
            for _ in range(3):
                await limiter.acquire("acme")
            await limiter.acquire("globex")  # must not be affected

        run(scenario())

    def test_tokens_refill_over_time(self) -> None:
        limiter = self.limiter()

        async def scenario():
            for _ in range(3):
                await limiter.acquire("acme")
            await asyncio.sleep(0.05)  # 100/s => plenty
            await limiter.acquire("acme")

        run(scenario())

    def test_concurrent_acquires_are_atomic(self) -> None:
        """A client-side read-modify-write would over-admit here."""
        limiter = self.limiter(rate_per_second=0.001, burst=10)

        async def scenario():
            results = await asyncio.gather(
                *(limiter.acquire("acme") for _ in range(50)),
                return_exceptions=True,
            )
            return sum(1 for r in results if r is None)

        self.assertEqual(run(scenario()), 10)

    def test_fail_open_allows_when_redis_is_down(self) -> None:
        limiter = RedisTokenBucketLimiter(
            _BrokenRedis(), rate_per_second=1, burst=1, fail_open=True
        )
        run(limiter.acquire("acme"))  # must not raise

    def test_fail_closed_rejects_when_redis_is_down(self) -> None:
        limiter = RedisTokenBucketLimiter(
            _BrokenRedis(), rate_per_second=1, burst=1, fail_open=False
        )
        with self.assertRaises(Exception):
            run(limiter.acquire("acme"))

    def test_verify_fails_loudly_even_when_fail_open(self) -> None:
        """Otherwise a broken script silently disables limiting forever."""
        limiter = RedisTokenBucketLimiter(
            _BrokenRedis(), rate_per_second=1, burst=1, fail_open=True
        )
        with self.assertRaises(Exception):
            run(limiter.verify())


class TestRedisConcurrency(unittest.TestCase):
    def setUp(self) -> None:
        self.client = fakeredis.FakeRedis(decode_responses=True)

    def test_cap_is_enforced_and_released(self) -> None:
        limiter = RedisConcurrencyLimiter(self.client, limit=2, lease_ttl_s=300.0)

        async def scenario():
            gate = asyncio.Event()

            async def hold():
                async with limiter.slot("acme"):
                    await gate.wait()

            held = [asyncio.create_task(hold()) for _ in range(2)]
            await asyncio.sleep(0.05)
            in_flight = await limiter.in_flight("acme")

            rejected = False
            try:
                async with limiter.slot("acme"):
                    pass
            except RateLimitError:
                rejected = True

            gate.set()
            await asyncio.gather(*held)
            return in_flight, rejected, await limiter.in_flight("acme")

        in_flight, rejected, after = run(scenario())
        self.assertEqual(in_flight, 2)
        self.assertTrue(rejected)
        self.assertEqual(after, 0)

    def test_a_crashed_holder_does_not_wedge_the_tenant(self) -> None:
        """The reason slots are leases rather than a counter."""
        limiter = RedisConcurrencyLimiter(self.client, limit=1, lease_ttl_s=0.2)

        async def scenario():
            leaked = limiter.slot("acme")
            await leaked.__aenter__()  # acquire, then never release

            blocked = False
            try:
                async with limiter.slot("acme"):
                    pass
            except RateLimitError:
                blocked = True

            await asyncio.sleep(0.35)
            async with limiter.slot("acme"):
                recovered = True
            return blocked, recovered

        blocked, recovered = run(scenario())
        self.assertTrue(blocked, "a live lease should block")
        self.assertTrue(recovered, "a stale lease should be reaped")


class _BrokenRedis:
    """A client whose every operation fails, for outage behaviour."""

    def register_script(self, _script: str) -> Any:
        async def fail(**_kwargs: Any) -> Any:
            raise ConnectionError("redis is down")

        return fail

    async def delete(self, *_args: Any) -> None:
        raise ConnectionError("redis is down")

    async def zrem(self, *_args: Any) -> None:
        raise ConnectionError("redis is down")


# ---------------------------------------------------------------------------
# Postgres store -- argument and mapping level only
# ---------------------------------------------------------------------------


class _FakeConnection:
    """Records statements and replays queued results. Executes no SQL."""

    def __init__(self, results: list[Any] | None = None) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.results = list(results or [])
        self.transactions = 0

    def _record(self, sql: str, args: tuple[Any, ...]) -> Any:
        self.calls.append((" ".join(sql.split()), args))
        return self.results.pop(0) if self.results else None

    async def execute(self, sql: str, *args: Any) -> Any:
        return self._record(sql, args)

    async def fetchval(self, sql: str, *args: Any) -> Any:
        return self._record(sql, args)

    async def fetchrow(self, sql: str, *args: Any) -> Any:
        return self._record(sql, args)

    async def fetch(self, sql: str, *args: Any) -> Any:
        return self._record(sql, args) or []

    def transaction(self) -> Any:
        connection = self

        class _Tx:
            async def __aenter__(self) -> None:
                connection.transactions += 1

            async def __aexit__(self, *_exc: Any) -> bool:
                return False

        return _Tx()

    async def __aenter__(self) -> _FakeConnection:
        return self

    async def __aexit__(self, *_exc: Any) -> bool:
        return False


def _store_with(connection: _FakeConnection, **kwargs: Any) -> PostgresSessionStore:
    store = PostgresSessionStore("postgresql://unused", **kwargs)
    store._pool = _FakePool(connection)
    return store


class _FakePool:
    def __init__(self, connection: _FakeConnection) -> None:
        self._connection = connection

    def acquire(self) -> _FakeConnection:
        return self._connection


def _row(**overrides: Any) -> dict[str, Any]:
    now = datetime.now(UTC)
    row = {
        "tenant_id": "acme",
        "id": "sess_" + "a" * 32,
        "principal_id": "u1",
        "title": "t",
        "created_at": now,
        "updated_at": now,
        "expires_at": None,
        "message_count": 2,
        "metadata": "{}",
    }
    row.update(overrides)
    return row


class TestPostgresQueries(unittest.TestCase):
    """Argument-level checks only -- the SQL itself is unverified here."""

    def test_satisfies_the_session_store_protocol(self) -> None:
        self.assertIsInstance(
            PostgresSessionStore("postgresql://unused"), SessionStore
        )

    def test_every_statement_carries_the_tenant(self) -> None:
        """The isolation invariant, since SQL has no structural partition."""
        # Queued in call order: INSERT returns an id, the listing returns no
        # rows, and the lookup misses.
        connection = _FakeConnection(results=["sess_new", [], None])
        store = _store_with(connection)

        run(store.create_session(tenant_id="acme", principal_id="u1"))
        run(store.list_sessions(tenant_id="acme"))
        with self.assertRaises(NotFoundError):
            run(store.get_session(tenant_id="acme", session_id="sess_x"))

        for sql, args in connection.calls:
            self.assertIn("tenant_id", sql, f"statement without a tenant filter: {sql}")
            self.assertIn("acme", args, f"tenant not bound: {sql}")

    def test_quota_rejection_becomes_a_conflict(self) -> None:
        connection = _FakeConnection(results=[None])  # INSERT matched no row
        store = _store_with(connection, max_sessions_per_principal=1)
        with self.assertRaises(ConflictError):
            run(store.create_session(tenant_id="acme", principal_id="u1"))

    def test_quota_is_checked_inside_the_insert(self) -> None:
        """A separate SELECT then INSERT would race two concurrent creates."""
        connection = _FakeConnection(results=["sess_x"])
        store = _store_with(connection)
        run(store.create_session(tenant_id="acme", principal_id="u1"))
        sql, _ = connection.calls[0]
        self.assertIn("INSERT INTO bat_sessions", sql)
        self.assertIn("count(*)", sql, "quota must be evaluated in the same statement")
        self.assertEqual(len(connection.calls), 1)

    def test_missing_session_is_not_found(self) -> None:
        store = _store_with(_FakeConnection(results=[None]))
        with self.assertRaises(NotFoundError):
            run(store.get_session(tenant_id="acme", session_id="sess_x"))

    def test_delete_of_absent_session_is_not_found(self) -> None:
        store = _store_with(_FakeConnection(results=[None]))
        with self.assertRaises(NotFoundError):
            run(store.delete_session(tenant_id="acme", session_id="sess_x"))

    def test_append_uses_one_transaction_and_bumps_atomically(self) -> None:
        connection = _FakeConnection(results=["sess_x", None, None])
        store = _store_with(connection)
        message = Message.create(
            session_id="sess_x", tenant_id="acme", role=Role.USER, content="hi"
        )
        run(store.append_message(tenant_id="acme", session_id="sess_x", message=message))

        self.assertEqual(connection.transactions, 1)
        update_sql = connection.calls[0][0]
        self.assertIn("message_count = message_count + 1", update_sql)

    def test_append_to_a_missing_session_is_not_found(self) -> None:
        store = _store_with(_FakeConnection(results=[None]))
        message = Message.create(
            session_id="sess_x", tenant_id="acme", role=Role.USER, content="hi"
        )
        with self.assertRaises(NotFoundError):
            run(
                store.append_message(
                    tenant_id="acme", session_id="sess_x", message=message
                )
            )

    def test_message_from_another_session_is_rejected(self) -> None:
        store = _store_with(_FakeConnection())
        message = Message.create(
            session_id="sess_other", tenant_id="acme", role=Role.USER, content="hi"
        )
        with self.assertRaises(ValidationError):
            run(
                store.append_message(
                    tenant_id="acme", session_id="sess_x", message=message
                )
            )

    def test_listing_rejects_a_silly_limit(self) -> None:
        store = _store_with(_FakeConnection())
        with self.assertRaises(ValidationError):
            run(store.list_sessions(tenant_id="acme", limit=1000))

    def test_listing_asks_for_one_extra_row_to_detect_more(self) -> None:
        connection = _FakeConnection(results=[[_row() for _ in range(4)]])
        store = _store_with(connection)
        page = run(store.list_sessions(tenant_id="acme", limit=3))
        self.assertEqual(connection.calls[0][1][-1], 4)
        self.assertTrue(page.has_more)
        self.assertEqual(len(page.items), 3)

    def test_schema_declares_composite_keys_and_cascade(self) -> None:
        self.assertIn("PRIMARY KEY (tenant_id, id)", SCHEMA)
        self.assertIn("ON DELETE CASCADE", SCHEMA)


class TestCursorRoundTrip(unittest.TestCase):
    def test_round_trip(self) -> None:
        session = Session.create(tenant_id="acme", principal_id="u1")
        created, session_id = _decode_cursor(_encode_cursor(session))
        self.assertEqual(session_id, session.id)
        self.assertEqual(created, session.created_at)

    def test_malformed_cursor_is_a_validation_error(self) -> None:
        with self.assertRaises(ValidationError):
            _decode_cursor("!!!not-base64!!!")


class TestRowMapping(unittest.TestCase):
    def test_session_row(self) -> None:
        session = _to_session(_row(title="Design", message_count=7))
        self.assertEqual(session.title, "Design")
        self.assertEqual(session.message_count, 7)
        self.assertEqual(session.tenant_id, "acme")

    def test_naive_timestamps_are_normalised_to_utc(self) -> None:
        naive = datetime(2026, 1, 1, 12, 0, 0)
        session = _to_session(_row(created_at=naive, updated_at=naive))
        self.assertEqual(session.created_at.tzinfo, UTC)

    def test_message_row_with_tool_calls(self) -> None:
        message = _to_message(
            {
                "tenant_id": "acme",
                "id": "msg_1",
                "session_id": "sess_1",
                "role": "assistant",
                "content": "",
                "created_at": datetime.now(UTC),
                "tool_calls": '[{"id":"c1","name":"calculator","arguments":{"expression":"1+1"}}]',
                "metadata": "{}",
            }
        )
        self.assertEqual(message.role, Role.ASSISTANT)
        self.assertEqual(message.tool_calls[0].name, "calculator")
        self.assertEqual(message.tool_calls[0].arguments, {"expression": "1+1"})

    def test_nameless_tool_call_is_dropped(self) -> None:
        message = _to_message(
            {
                "tenant_id": "acme",
                "id": "msg_1",
                "session_id": "sess_1",
                "role": "assistant",
                "content": "",
                "created_at": datetime.now(UTC),
                "tool_calls": '[{"id":"c1"}]',
                "metadata": "{}",
            }
        )
        self.assertEqual(message.tool_calls, ())


class TestProductionGuards(unittest.TestCase):
    def test_in_process_rate_limiting_is_refused_in_production(self) -> None:
        with self.assertRaises(Exception) as caught:
            Settings(environment="production")
        self.assertIn("rate_limit.backend='memory'", str(caught.exception))

    def test_lease_shorter_than_the_run_deadline_is_refused(self) -> None:
        """Otherwise a live run's slot is released while it is still running."""
        with self.assertRaises(Exception) as caught:
            Settings(
                environment="production",
                agent=AgentSettings(deadline_s=600.0),
                rate_limit=RateLimitSettings(
                    backend="redis", dsn="redis://x", lease_ttl_s=300.0
                ),
            )
        self.assertIn("lease_ttl_s", str(caught.exception))

    def test_redis_backend_requires_a_dsn(self) -> None:
        with self.assertRaises(Exception):
            RateLimitSettings(backend="redis")

    def test_postgres_backend_requires_a_dsn(self) -> None:
        from bat.settings import SessionSettings

        with self.assertRaises(Exception):
            SessionSettings(backend="postgres")


if __name__ == "__main__":
    unittest.main(verbosity=2)
