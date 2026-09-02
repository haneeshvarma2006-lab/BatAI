# BAT

A multi-tenant agentic AI platform: FastAPI backend, RAG memory over ChromaDB,
and a policy-gated agent loop that runs tools under an explicit capability
model.

Full design rationale lives in **[ARCHITECTURE.md](ARCHITECTURE.md)**.

---

## Status

| Area | State |
| --- | --- |
| Tenancy, auth, scopes | **implemented** |
| Session + transcript store | **implemented** (in-memory adapter; Postgres pending) |
| HTTP API, SSE streaming, rate limits | **implemented** |
| Ports for RAG / LLM / tools / agent | **implemented** |
| Native `llama-cpp-python` inference | **in progress** — see [Roadmap](#roadmap) |
| RAG pipeline (Chroma adapter) | specified, ports landed |
| Agent loop + tool executor | specified, ports landed; a reference runner stands in |

37 tests pass. The agent currently answers via `ReferenceAgentRunner`, a
deterministic stand-in that exercises the whole streaming contract without
contacting a model. Swapping in the real loop is a one-line change in the
lifespan because both sit behind `AgentRunner`.

---

## Quickstart

```bash
pip install -r requirements.txt
```

Mint a key (the plaintext is shown once; only its SHA-256 digest is stored):

```bash
python -m bat.cli issue-key --tenant acme --principal alice
```

Put the printed record in `BAT_API_KEYS` (see [.env.example](.env.example)), then:

```bash
python -m bat.cli check-config
```

```bash
python -m bat.cli serve --reload
```

Interactive docs at `http://127.0.0.1:8000/docs`.

### Talk to it

```bash
curl -s http://127.0.0.1:8000/v1/whoami -H "Authorization: Bearer $BAT_KEY"
```

```bash
curl -s -X POST http://127.0.0.1:8000/v1/sessions -H "Authorization: Bearer $BAT_KEY" -H "Content-Type: application/json" -d '{"title":"first session"}'
```

```bash
curl -N -X POST http://127.0.0.1:8000/v1/sessions/$SESSION_ID/messages/stream -H "Authorization: Bearer $BAT_KEY" -H "Content-Type: application/json" -d '{"content":"hello"}'
```

---

## API

| Method | Path | Scope |
| --- | --- | --- |
| `GET` | `/healthz` | none — liveness |
| `GET` | `/readyz` | none — readiness |
| `GET` | `/v1/whoami` | any valid key |
| `POST` | `/v1/sessions` | `sessions:write` |
| `GET` | `/v1/sessions` | `sessions:read` |
| `GET` | `/v1/sessions/{id}` | `sessions:read` |
| `DELETE` | `/v1/sessions/{id}` | `sessions:write` |
| `GET` | `/v1/sessions/{id}/messages` | `sessions:read` |
| `POST` | `/v1/sessions/{id}/messages` | `agent:invoke` |
| `POST` | `/v1/sessions/{id}/messages/stream` | `agent:invoke` |

Auth is `Authorization: Bearer <key>` or `X-API-Key: <key>`.

Errors are RFC 9457 `application/problem+json` carrying `code`, `detail` and
`request_id`. Every response echoes `X-Request-Id`, which also appears on each
JSON log line for that request.

The streaming endpoint emits SSE frames named `token`, `tool_call`,
`tool_result`, `final`, `error`, and closes with `done`.

---

## Layout

```
bat/
  domain/       pure types: tenancy, sessions, messages, errors
  ports/        protocols: SessionStore, LLMClient, MemoryPipeline, Tool, AgentRunner
  adapters/     implementations behind those protocols
  api/          routers, dependencies, auth, middleware, error handlers
  settings.py   typed env-driven config with production hardening
  ratelimit.py  per-tenant token bucket + run concurrency cap
  cli.py        issue-key / check-config / serve
tests/          37 stdlib unittest cases
```

Dependencies point inward. The domain knows nothing about FastAPI; the API
knows nothing about ChromaDB.

---

## Tenancy

Every resource belongs to exactly one tenant, enforced by five rules:

1. One place mints identity — `ApiKeyAuthenticator`. Nothing downstream
   re-parses a header.
2. `TenantContext` is passed **explicitly**, never read from a global, so a
   missing isolation check is visible in a function signature.
3. `X-Tenant-Id` is checked against the credential, never trusted as a source.
   A caller cannot assert its way into another tenant.
4. Cross-tenant access returns **404, not 403** — a 403 would confirm the
   resource exists.
5. Storage is partitioned by tenant, so forgetting to filter is a missing key
   rather than a leak.

---

## Tool security

The agent's tool boundary is **default-deny**. A tool is unavailable unless its
name is in the tenant's allowlist; registration is not authorization.

Every tool declares an isolation level, and the deployment sets a floor:

| Level | Meaning | Allowed in the hosted product |
| --- | --- | --- |
| `NONE` | in-process, ambient authority | **no** — desktop build only |
| `PURE` | in-process, no I/O | dev only |
| `NETWORK` | egress to an allowlist | yes |
| `SUBPROCESS` | separate process, dropped privileges, rlimits | yes |
| `SANDBOX` | container/microVM per call | yes |

`Settings._harden` **refuses to boot production** below `SUBPROCESS`.

This is why the legacy tools in `tools/` cannot be registered as they stand.
`execute_python_code` calls `exec()` with full `__builtins__` (arbitrary RCE);
`automate_typing` drives the host's real keyboard; `search_and_open_file` and
`open_application` act on the server's desktop and shell. On a personal machine
that is the feature. Behind a shared API, with a model steerable by untrusted
retrieved text, it turns prompt injection into remote code execution.

Migration path for each is in [ARCHITECTURE.md §6](ARCHITECTURE.md).

---

## Configuration

All settings come from the environment with the `BAT_` prefix; nested fields use
`__` (e.g. `BAT_MODEL__NAME`). See [.env.example](.env.example).

`Settings._harden` fails the boot — loudly, before serving a request — on any
production config that only makes sense in development: no API keys, wildcard
CORS, in-memory sessions or vectors, or tool isolation below `SUBPROCESS`.

---

## Tests

```bash
python -m unittest discover -s tests -t . -v
```

No pytest required. Coverage targets the properties that matter for a
multi-tenant platform — isolation, auth, scopes, limits, pagination and
streaming — rather than line count.

---

## Roadmap

Next up: **replace Ollama with native in-process inference via
`llama-cpp-python`**, loading `.gguf` weights directly.

Three environment blockers found while scoping that work, recorded so they
don't need rediscovering:

1. **No `llama-cpp-python` wheel for Python 3.14.** The package resolves only
   as an sdist on this interpreter; `pip install --only-binary=:all:` finds no
   distribution. Either build from source or run the backend on Python 3.12,
   which has prebuilt wheels.
2. **No build toolchain present** — neither `cmake` nor MSVC `cl.exe` is on
   PATH, so a source build will fail until one is installed.
3. **No `.gguf` weights on disk** anywhere under the project.

The design consequence to settle before writing the adapter: `llama.cpp`'s
`Llama` object is not thread-safe and one instance serves one generation at a
time, so in-process inference makes the model a serialized resource and each
worker holds a full copy of the weights in RAM. That is fine for a self-hosted
or single-tenant deployment; for the multi-tenant target it caps throughput in a
way the current Ollama-as-a-service split does not. The adapter will therefore
own a dedicated worker thread with bounded admission and honest backpressure.

Then, in order:

| # | Module | Depends on |
| --- | --- | --- |
| 1 | Tenancy, settings, sessions, API, SSE | — (**done**) |
| 2 | `LlamaCppClient` — native inference behind `LLMClient` | 1 |
| 3 | `PostgresSessionStore`, Redis-backed limits | 1 |
| 4 | RAG pipeline + `ChromaVectorStore` + local embeddings | 1, 2 |
| 5 | Tool registry, executor, sandbox runner | 1, 2 |
| 6 | Real agent loop replacing `ReferenceAgentRunner` | 2, 4, 5 |
| 7 | Billing, usage metering, admin API | 1–6 |

Each lands behind a port that already exists, so nothing above it changes when
it does.

---

## Legacy code

`main.py`, `core/`, `memory/` and `tools/` are the original single-user desktop
assistant. They still run, and they are the reference for what the platform must
eventually do — but they are not part of the `bat/` package and are being
retired module by module as the equivalents land.
