# BAT

A multi-tenant agentic AI platform: FastAPI backend, **native in-process
inference** over local `.gguf` weights via `llama-cpp-python`, RAG memory over
ChromaDB, and a policy-gated agent loop that runs tools under an explicit
capability model.

No model server. No Ollama. No network hop for inference or embeddings.

Full design rationale lives in **[ARCHITECTURE.md](ARCHITECTURE.md)**.

---

## Status

| Area | State |
| --- | --- |
| Tenancy, auth, scopes | **implemented** |
| Session + transcript store | **implemented** (in-memory adapter; Postgres pending) |
| HTTP API, SSE streaming, rate limits | **implemented** |
| Native `llama-cpp-python` inference | **implemented** — untested against real weights, see below |
| Local `.gguf` embeddings | **implemented** |
| RAG pipeline (chunk → embed → retrieve → budgeted context) | **implemented** |
| Vector stores (in-memory + Chroma) | **implemented** |
| Agent loop + policy-gated tool executor | **implemented** |
| Postgres sessions, Redis limits | pending |
| Sandboxed tool runtime | pending — no tool ships enabled |

72 tests pass, covering tenant isolation, auth, scopes, limits, streaming,
chunking, retrieval, tool policy and the agent loop.

**The inference path has not been run against a real model.** It is verified
against a fake that reproduces llama.cpp's awkward properties — blocking calls
and a non-thread-safe handle — so the adapter's threading, admission, streaming
and timeout logic is genuinely exercised. What is unverified is llama.cpp
itself: real weights, real tokenizer, real chat template. See
[Running real inference](#running-real-inference).

---

## Quickstart

```bash
pip install -r requirements.txt
```

The API starts without weights — the agent falls back to a reference runner and
retrieval uses a non-semantic hash embedder, so you can exercise every endpoint
before downloading a model. Both fallbacks are refused in production.

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
| `POST` | `/v1/memory/documents` | `memory:write` |
| `GET` | `/v1/memory/search` | `memory:read` |
| `GET` | `/v1/memory/context` | `memory:read` |
| `DELETE` | `/v1/memory` | `memory:write` |

`/v1/memory/context` returns the exact block the agent would prepend for a
query, which makes retrieval debuggable rather than a black box.

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
  adapters/     llama.cpp client + embedder, Chroma / in-memory vector stores,
                session store
  api/          routers, dependencies, auth, middleware, error handlers
  settings.py   typed env-driven config with production hardening
  ratelimit.py  per-tenant token bucket + run concurrency cap
  cli.py        issue-key / check-config / serve
  services/
    rag/        chunking + the tenant-facing memory pipeline
    agent/      the loop and the policy-gated tool executor
tests/          72 stdlib unittest cases
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

## Running real inference

Point the config at a `.gguf` and restart — no code change:

```bash
pip install llama-cpp-python
```

```
BAT_MODEL__MODEL_PATH=./models/your-model.Q4_K_M.gguf
BAT_EMBEDDING__MODEL_PATH=./models/nomic-embed-text-v1.5.Q4_K_M.gguf
```

Three environment blockers on this machine, recorded so they don't need
rediscovering:

1. **No `llama-cpp-python` wheel for Python 3.14.** It resolves only as an
   sdist; `pip install --only-binary=:all:` finds no distribution. **Python 3.12
   has prebuilt wheels and is the supported path.**
2. **No build toolchain** — neither `cmake` nor MSVC `cl.exe` is on PATH, so a
   source build fails until one is installed.
3. **No `.gguf` weights** anywhere under the project.

Two things to check first with real weights, because they are model-specific and
a fake cannot cover them:

- **Chat template.** `BAT_MODEL__CHAT_FORMAT` defaults to the template embedded
  in the `.gguf`. If replies come back with role markers leaking into the text,
  set it explicitly.
- **Tool calling.** Native tool calls need a chat format that supports them
  (`chatml-function-calling`, functionary, …). With a format that doesn't, the
  model emits tool calls as prose and `wants_tools` stays false. Tools ship
  disabled, so this affects nothing until you enable one — the durable fix is
  GBNF grammar-constrained decoding.

### The capacity trade-off

In-process inference makes the model a **serialised** resource: `Llama` is not
thread-safe, so one instance serves one generation at a time, and every worker
holds its own full copy of the weights in RAM. The adapter handles this honestly
— a single-worker executor for thread safety, a bounded queue, and 429 with
`Retry-After` when the queue is full rather than silent queueing.

But it is a real ceiling, and it is the one thing the Ollama split gave you for
free: a separate model server can be scaled and pooled independently of the API.
For a self-hosted or single-tenant deployment this is the right trade. For the
multi-tenant target at scale, expect to reintroduce a dedicated inference tier —
behind `LLMClient`, so it is an adapter swap and nothing above it changes.

---

## Roadmap

| # | Module | Status |
| --- | --- | --- |
| 1 | Tenancy, settings, sessions, API, SSE | **done** |
| 2 | `LlamaCppClient` — native inference | **done** (unverified on real weights) |
| 3 | RAG pipeline, local embeddings, vector stores | **done** |
| 4 | Agent loop + tool executor | **done** |
| 5 | `PostgresSessionStore`, Redis-backed limits | next |
| 6 | Sandboxed tool runtime, then re-register tools | next |
| 7 | Billing, usage metering, admin API | later |

Each lands behind a port that already exists, so nothing above it changes when
it does.

---

## Legacy code

`main.py` and `core/brain.py` are the desktop CLI. They were rewritten for
native inference and are now thin wrappers over `bat/` — same loop, same
retrieval, same tool policy, just one fixed local tenant and no auth. Run it
with `python main.py`.

`engine.py` was **deleted**: it was the Ollama-based duplicate brain, superseded
by `bat/services/agent/loop.py`. It remains in git history.

`memory/chroma_cloud.py` and `tools/` are the original single-user modules.
Nothing imports them any more — `memory/` is superseded by
`bat/services/rag/`, and `tools/` waits on the sandboxed runtime. They are left
in place as the reference for what to port, not as live code.
