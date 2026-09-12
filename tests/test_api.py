"""End-to-end tests for the API module.

Written against stdlib :mod:`unittest` so they run with no extra dependency:

    python -m unittest discover -s tests -v

Coverage is aimed at the properties that actually matter for a multi-tenant
platform -- isolation, authentication, scopes, limits and streaming -- rather
than at line count.
"""

from __future__ import annotations

import json
import unittest

from fastapi.testclient import TestClient

from bat.api.app import create_app
from bat.api.security import generate_api_key
from bat.domain.tenancy import DEFAULT_SCOPES, Scope
from bat.settings import (
    AgentSettings,
    ApiKeyRecord,
    RateLimitSettings,
    SessionSettings,
    Settings,
)

ACME_KEY, ACME_DIGEST = generate_api_key()
GLOBEX_KEY, GLOBEX_DIGEST = generate_api_key()
READONLY_KEY, READONLY_DIGEST = generate_api_key()


def build_settings(**overrides) -> Settings:
    base = {
        "environment": "local",
        "json_logs": False,
        "log_level": "CRITICAL",
        "api_keys": (
            ApiKeyRecord(
                key_sha256=ACME_DIGEST,
                tenant_id="acme",
                principal_id="user-a",
                scopes=DEFAULT_SCOPES,
                label="acme primary",
            ),
            ApiKeyRecord(
                key_sha256=GLOBEX_DIGEST,
                tenant_id="globex",
                principal_id="user-g",
                scopes=DEFAULT_SCOPES,
            ),
            ApiKeyRecord(
                key_sha256=READONLY_DIGEST,
                tenant_id="acme",
                principal_id="user-ro",
                scopes=frozenset({Scope.SESSIONS_READ}),
            ),
        ),
        "session": SessionSettings(backend="memory", ttl_seconds=3600),
        "agent": AgentSettings(max_steps=3, deadline_s=10.0),
        "rate_limit": RateLimitSettings(
            enabled=True, requests_per_second=1000.0, burst=1000
        ),
    }
    base.update(overrides)
    # `_env_file=None` keeps the suite hermetic. Settings normally loads `.env`,
    # so without this a developer's local config (a model path, a Postgres DSN)
    # silently changes what the tests build -- and they fail, or worse pass, for
    # reasons that have nothing to do with the code.
    return Settings(_env_file=None, **base)


def auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


class ApiTestCase(unittest.TestCase):
    """Builds a fresh app per test class, so no state leaks between cases."""

    settings_overrides: dict = {}

    def setUp(self) -> None:
        self.client = TestClient(create_app(build_settings(**self.settings_overrides)))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def open_session(self, key: str = ACME_KEY, **body) -> str:
        response = self.client.post("/v1/sessions", json=body, headers=auth(key))
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["id"]


class TestHealth(ApiTestCase):
    def test_liveness_needs_no_credential(self) -> None:
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_every_response_carries_a_request_id(self) -> None:
        response = self.client.get("/healthz")
        self.assertTrue(response.headers.get("X-Request-Id"))

    def test_inbound_request_id_is_echoed_and_sanitised(self) -> None:
        response = self.client.get(
            "/healthz", headers={"X-Request-Id": "trace-123\r\ninjected: yes"}
        )
        echoed = response.headers["X-Request-Id"]
        self.assertNotIn("\n", echoed)
        self.assertTrue(echoed.startswith("trace-123"))

    def test_readiness_reports_dependency_state(self) -> None:
        body = self.client.get("/readyz").json()
        self.assertTrue(body["ready"])
        self.assertIn("session_store", body["checks"])


class TestAuthentication(ApiTestCase):
    def test_missing_credential_is_401(self) -> None:
        response = self.client.get("/v1/whoami")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["code"], "unauthenticated")

    def test_invalid_credential_is_401(self) -> None:
        response = self.client.get("/v1/whoami", headers=auth("bat_sk_not-a-real-key"))
        self.assertEqual(response.status_code, 401)

    def test_x_api_key_header_also_works(self) -> None:
        response = self.client.get("/v1/whoami", headers={"X-API-Key": ACME_KEY})
        self.assertEqual(response.status_code, 200)

    def test_credential_determines_tenant_not_the_header(self) -> None:
        """A caller cannot assert its way into another tenant."""
        response = self.client.get(
            "/v1/whoami", headers={**auth(ACME_KEY), "X-Tenant-Id": "globex"}
        )
        self.assertEqual(response.status_code, 403)

    def test_whoami_reports_the_resolved_grant(self) -> None:
        body = self.client.get("/v1/whoami", headers=auth(ACME_KEY)).json()
        self.assertEqual(body["tenant_id"], "acme")
        self.assertEqual(body["principal_id"], "user-a")
        self.assertIn("sessions:write", body["scopes"])

    def test_errors_use_problem_json(self) -> None:
        response = self.client.get("/v1/whoami")
        self.assertIn("application/problem+json", response.headers["content-type"])
        for field in ("type", "title", "status", "code", "detail"):
            self.assertIn(field, response.json())


class TestScopes(ApiTestCase):
    def test_read_only_key_cannot_create_a_session(self) -> None:
        response = self.client.post("/v1/sessions", json={}, headers=auth(READONLY_KEY))
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["code"], "forbidden")

    def test_read_only_key_cannot_invoke_the_agent(self) -> None:
        session_id = self.open_session()
        response = self.client.post(
            f"/v1/sessions/{session_id}/messages",
            json={"content": "hello"},
            headers=auth(READONLY_KEY),
        )
        self.assertEqual(response.status_code, 403)

    def test_read_only_key_can_list(self) -> None:
        response = self.client.get("/v1/sessions", headers=auth(READONLY_KEY))
        self.assertEqual(response.status_code, 200)


class TestSessions(ApiTestCase):
    def test_create_and_fetch(self) -> None:
        session_id = self.open_session(title="Design review")
        body = self.client.get(
            f"/v1/sessions/{session_id}", headers=auth(ACME_KEY)
        ).json()
        self.assertEqual(body["title"], "Design review")
        self.assertEqual(body["message_count"], 0)
        self.assertIsNotNone(body["expires_at"])

    def test_tenant_id_is_never_echoed(self) -> None:
        session_id = self.open_session()
        body = self.client.get(
            f"/v1/sessions/{session_id}", headers=auth(ACME_KEY)
        ).json()
        self.assertNotIn("tenant_id", body)

    def test_unknown_session_is_404(self) -> None:
        response = self.client.get(
            f"/v1/sessions/sess_{'0' * 32}", headers=auth(ACME_KEY)
        )
        self.assertEqual(response.status_code, 404)

    def test_malformed_session_id_is_422(self) -> None:
        response = self.client.get("/v1/sessions/not-an-id", headers=auth(ACME_KEY))
        self.assertEqual(response.status_code, 422)

    def test_unknown_body_fields_are_rejected(self) -> None:
        response = self.client.post(
            "/v1/sessions", json={"title": "x", "admin": True}, headers=auth(ACME_KEY)
        )
        self.assertEqual(response.status_code, 422)

    def test_delete_then_fetch_is_404(self) -> None:
        session_id = self.open_session()
        self.assertEqual(
            self.client.delete(
                f"/v1/sessions/{session_id}", headers=auth(ACME_KEY)
            ).status_code,
            204,
        )
        self.assertEqual(
            self.client.get(
                f"/v1/sessions/{session_id}", headers=auth(ACME_KEY)
            ).status_code,
            404,
        )

    def test_pagination_is_stable_and_complete(self) -> None:
        created = {self.open_session(title=f"s{i}") for i in range(7)}
        seen: list[str] = []
        cursor, pages = None, 0
        while True:
            params = {"limit": 3}
            if cursor:
                params["cursor"] = cursor
            page = self.client.get(
                "/v1/sessions", params=params, headers=auth(ACME_KEY)
            ).json()
            seen.extend(s["id"] for s in page["items"])
            pages += 1
            if not page["has_more"]:
                break
            cursor = page["next_cursor"]
        self.assertEqual(set(seen), created)
        self.assertEqual(len(seen), len(set(seen)), "pagination duplicated a row")
        self.assertEqual(pages, 3)

    def test_malformed_cursor_is_422(self) -> None:
        response = self.client.get(
            "/v1/sessions", params={"cursor": "!!!not-base64!!!"}, headers=auth(ACME_KEY)
        )
        self.assertEqual(response.status_code, 422)


class TestTenantIsolation(ApiTestCase):
    """The property the whole design exists to guarantee."""

    def test_cross_tenant_read_is_404_not_403(self) -> None:
        acme_session = self.open_session(ACME_KEY)
        response = self.client.get(
            f"/v1/sessions/{acme_session}", headers=auth(GLOBEX_KEY)
        )
        self.assertEqual(response.status_code, 404, "leaked existence across tenants")

    def test_cross_tenant_delete_is_404(self) -> None:
        acme_session = self.open_session(ACME_KEY)
        response = self.client.delete(
            f"/v1/sessions/{acme_session}", headers=auth(GLOBEX_KEY)
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            self.client.get(
                f"/v1/sessions/{acme_session}", headers=auth(ACME_KEY)
            ).status_code,
            200,
            "cross-tenant delete destroyed the owner's session",
        )

    def test_cross_tenant_message_send_is_404(self) -> None:
        acme_session = self.open_session(ACME_KEY)
        response = self.client.post(
            f"/v1/sessions/{acme_session}/messages",
            json={"content": "exfiltrate"},
            headers=auth(GLOBEX_KEY),
        )
        self.assertEqual(response.status_code, 404)

    def test_listings_never_cross_tenants(self) -> None:
        for _ in range(3):
            self.open_session(ACME_KEY)
        self.open_session(GLOBEX_KEY)
        globex = self.client.get(
            "/v1/sessions", params={"mine_only": False}, headers=auth(GLOBEX_KEY)
        ).json()
        self.assertEqual(len(globex["items"]), 1)

    def test_listing_defaults_to_the_calling_principal(self) -> None:
        self.open_session(ACME_KEY)  # owned by user-a
        body = self.client.get("/v1/sessions", headers=auth(READONLY_KEY)).json()
        self.assertEqual(body["items"], [], "user-ro saw another user's sessions")


class TestAgentTurns(ApiTestCase):
    def test_buffered_turn_persists_both_messages(self) -> None:
        session_id = self.open_session()
        response = self.client.post(
            f"/v1/sessions/{session_id}/messages",
            json={"content": "what is my status?"},
            headers=auth(ACME_KEY),
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["user_message"]["role"], "user")
        self.assertEqual(body["assistant_message"]["role"], "assistant")
        self.assertIn("what is my status?", body["assistant_message"]["content"])
        self.assertEqual(body["stop_reason"], "completed")

        transcript = self.client.get(
            f"/v1/sessions/{session_id}/messages", headers=auth(ACME_KEY)
        ).json()
        self.assertEqual([m["role"] for m in transcript["items"]], ["user", "assistant"])

    def test_history_is_carried_into_the_next_turn(self) -> None:
        session_id = self.open_session()
        for _ in range(2):
            self.client.post(
                f"/v1/sessions/{session_id}/messages",
                json={"content": "ping"},
                headers=auth(ACME_KEY),
            )
        session = self.client.get(
            f"/v1/sessions/{session_id}", headers=auth(ACME_KEY)
        ).json()
        self.assertEqual(session["message_count"], 4)

    def test_blank_content_is_rejected(self) -> None:
        session_id = self.open_session()
        response = self.client.post(
            f"/v1/sessions/{session_id}/messages",
            json={"content": "   "},
            headers=auth(ACME_KEY),
        )
        self.assertEqual(response.status_code, 422)

    def test_oversized_content_is_rejected(self) -> None:
        session_id = self.open_session()
        response = self.client.post(
            f"/v1/sessions/{session_id}/messages",
            json={"content": "x" * 40_000},
            headers=auth(ACME_KEY),
        )
        self.assertEqual(response.status_code, 422)

    def test_streaming_emits_tokens_then_final_then_done(self) -> None:
        session_id = self.open_session()
        with self.client.stream(
            "POST",
            f"/v1/sessions/{session_id}/messages/stream",
            json={"content": "stream please"},
            headers=auth(ACME_KEY),
        ) as response:
            self.assertEqual(response.status_code, 200)
            self.assertIn("text/event-stream", response.headers["content-type"])
            raw = "".join(response.iter_text())

        events = _parse_sse(raw)
        names = [name for name, _ in events]
        self.assertIn("token", names)
        self.assertIn("final", names)
        self.assertEqual(names[-1], "done")
        self.assertLess(names.index("final"), names.index("done"))

        streamed = "".join(
            json.loads(data)["text"] for name, data in events if name == "token"
        )
        final = next(json.loads(d) for n, d in events if n == "final")
        self.assertEqual(streamed, final["content"])

    def test_streamed_reply_is_persisted(self) -> None:
        session_id = self.open_session()
        with self.client.stream(
            "POST",
            f"/v1/sessions/{session_id}/messages/stream",
            json={"content": "remember this"},
            headers=auth(ACME_KEY),
        ) as response:
            "".join(response.iter_text())
        transcript = self.client.get(
            f"/v1/sessions/{session_id}/messages", headers=auth(ACME_KEY)
        ).json()
        self.assertEqual([m["role"] for m in transcript["items"]], ["user", "assistant"])


class TestRateLimiting(ApiTestCase):
    settings_overrides = {
        "rate_limit": RateLimitSettings(
            enabled=True, requests_per_second=1.0, burst=3, max_concurrent_runs=2
        )
    }

    def test_tenant_exceeding_its_budget_gets_429_with_retry_after(self) -> None:
        statuses = [
            self.client.get("/v1/whoami", headers=auth(ACME_KEY)).status_code
            for _ in range(6)
        ]
        self.assertIn(429, statuses)
        limited = next(
            r
            for r in (self.client.get("/v1/whoami", headers=auth(ACME_KEY)),)
            if r.status_code == 429
        )
        self.assertIn("Retry-After", limited.headers)
        self.assertEqual(limited.json()["code"], "rate_limited")

    def test_one_tenant_cannot_starve_another(self) -> None:
        for _ in range(6):
            self.client.get("/v1/whoami", headers=auth(ACME_KEY))
        response = self.client.get("/v1/whoami", headers=auth(GLOBEX_KEY))
        self.assertEqual(response.status_code, 200, "limiter is not per-tenant")


class TestConfigurationSafety(unittest.TestCase):
    def test_production_rejects_development_defaults(self) -> None:
        with self.assertRaises(Exception) as caught:
            Settings(_env_file=None, environment="production")
        message = str(caught.exception)
        self.assertIn("unsafe production configuration", message)
        for problem in ("api_keys", "model.model_path", "vector.mode", "session.backend"):
            self.assertIn(problem, message)

    def test_production_rejects_host_reaching_tools(self) -> None:
        """The tool defaults are production-safe, so assert the guard directly."""
        from bat.ports.tools import Authority, Isolation

        with self.assertRaises(Exception) as caught:
            Settings(
                _env_file=None,
                environment="production",
                agent=AgentSettings(max_tool_authority=Authority.HOST),
            )
        self.assertIn("max_tool_authority", str(caught.exception))

        with self.assertRaises(Exception) as caught:
            Settings(
                _env_file=None,
                environment="production",
                agent=AgentSettings(min_code_isolation=Isolation.IN_PROCESS),
            )
        self.assertIn("min_code_isolation", str(caught.exception))

    def test_api_key_records_store_only_digests(self) -> None:
        plaintext, digest = generate_api_key()
        record = ApiKeyRecord(
            key_sha256=digest, tenant_id="acme", principal_id="u1"
        )
        self.assertNotIn(plaintext, record.model_dump_json())

    def test_tenant_ids_are_constrained(self) -> None:
        with self.assertRaises(Exception):
            ApiKeyRecord(
                key_sha256="0" * 64, tenant_id="../etc/passwd", principal_id="u1"
            )


def _parse_sse(raw: str) -> list[tuple[str, str]]:
    """Parse an SSE body into ``(event_name, data)`` pairs, ignoring comments."""
    events: list[tuple[str, str]] = []
    for block in raw.split("\n\n"):
        name, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = line[len("data: ") :]
        if name is not None and data is not None:
            events.append((name, data))
    return events


if __name__ == "__main__":
    unittest.main(verbosity=2)
