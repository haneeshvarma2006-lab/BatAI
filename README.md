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
| Session + transcript store | **implemented** (in-memory + Postgres) |
| HTTP API, SSE streaming, rate limits | **implemented** |
| Native `llama-cpp-python` inference | **implemented** — untested against real weights, see below |
| Local `.gguf` embeddings | **implemented** |
| RAG pipeline (chunk → embed → retrieve → budgeted context) | **implemented** |
| Vector stores (in-memory + Chroma) | **implemented** |
| Agent loop + policy-gated tool executor | **implemented** |
| Built-in tools + subprocess sandbox | **implemented** |
| Global rate limits + run leases (Redis) | **implemented** |
| Container-isolated code execution | pending — `python_exec` is desktop-only |

131 tests pass, covering tenant isolation, auth, scopes, limits, streaming,
chunking, retrieval, sandbox containment, tool policy, the agent loop and the
Redis limiters (against real Lua execution).

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

The tool boundary is **default-deny**: a tool is unavailable unless its name is
in the tenant's allowlist. Registration is not authorization.

Safety is **two independent axes**, because one ordered ladder cannot express
both. (An earlier version tried, and the floor could be set safely or usefully
but never both — a bar strict enough to exclude host access also excluded a
tenant-scoped memory lookup.)

**Authority — what the tool can reach:**

| | Reach | Allowed on a shared deployment |
| --- | --- | --- |
| `PURE` | nothing; pure computation | yes |
| `TENANT` | the caller's own tenant data only | yes |
| `NETWORK` | vetted outbound egress | yes |
| `HOST` | filesystem, shell, desktop | **never** |

**Isolation — where its code runs:** `IN_PROCESS` → `SUBPROCESS` → `CONTAINER`.
This floor applies only to tools that run *caller-supplied code*; a
fixed-function tool with a validated schema is bounded by its own
implementation.

`Settings._harden` refuses to boot production with `max_tool_authority = HOST`
or `min_code_isolation < SUBPROCESS`.

### The built-in tools

| Tool | Authority | Isolation | Shippable |
| --- | --- | --- | --- |
| `calculator` | `PURE` | in-process | yes |
| `memory_search` | `TENANT` | in-process | yes |
| `python_exec` | `HOST` | subprocess | **no — desktop only** |

`calculator` evaluates arithmetic by walking a parsed AST against an allowlist of
*node types* — not a filtered `eval`. Attribute access, subscripts,
comprehensions and unknown names have no branch, so there is no reachable path
to `__class__` or `__import__`. It exists because the legacy prompt told the
model to call `execute_python_code` for every calculation — arbitrary code
execution for the sake of long multiplication.

`memory_search` takes the tenant from the invocation context, never from an
argument, so no injected instruction can steer it at another tenant's data.

`python_exec` replaces `tools/code_exec.py`, which called `exec()` with full
`__builtins__` in the API process. It now runs in a child process with a
scrubbed environment (the parent's API keys and DSNs are simply absent), `-I`
isolated mode, an empty temporary cwd, a hard timeout and capped output.

**It still declares `Authority.HOST`, and that is deliberate.** On this platform
the child runs as the same OS user, so it can reach the filesystem and open
sockets — and Windows has no `resource` module, so CPU and memory limits are
*not* enforced, only the wall clock. The declaration states what the tool can
actually do, so every server policy refuses it. It becomes shippable when it
runs in a container with no network and a read-only rootfs, at which point its
authority genuinely drops to `PURE`. **Never widen the policy to fit a tool;
change the tool and let its declaration follow.**

The remaining legacy tools in `tools/` are still unported: `automate_typing`
drives the host's real keyboard and has no server-side meaning at all;
`open_application` and `search_and_open_file` act on the server's desktop and
shell. `find_file_location` is superseded by `memory_search`.

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

## Durable backends

`BAT_SESSION__BACKEND=postgres` and `BAT_RATE_LIMIT__BACKEND=redis` replace the
per-process defaults. Both are required in production, and `Settings._harden`
says why: in-memory sessions vanish on restart and differ per replica, and an
in-process rate limit becomes `replicas x configured` — a limit that moves
whenever the cluster autoscales.

**Postgres.** Composite `(tenant_id, id)` primary keys, so a query that forgets
the tenant misses the index rather than returning another tenant's rows.
`message_count` is incremented in SQL, never read-modify-written. The session
quota is checked *inside* the `INSERT ... SELECT`, so two concurrent creates
cannot both pass a separate check. Deleting a session takes its transcript with
it via `ON DELETE CASCADE`. The schema is applied on connect, and `SCHEMA_RLS`
documents how to add row-level security on top.

**Redis.** Both operations are single Lua scripts, because a client-side
read-modify-write is a lost-update race — and the race favours the tenant, so
the limit leaks precisely when it is under most pressure.

Run slots are **leases in a sorted set, not a counter.** A counter is released
by a `finally` block, which never runs if the holder is OOM-killed or evicted;
that would permanently shrink a tenant's capacity with no recovery short of
manual intervention. A lease expires on its own, so a crashed worker's slot
comes back. Keep `lease_ttl_s` above `agent.deadline_s` — the config guard
enforces it.

`fail_open` decides what a Redis outage costs: availability or limiting. It
defaults to allowing requests, which is right when the limiter protects
capacity, and should be `false` where it is an abuse or billing control.
Independently of that, both limiters run their script once at startup —
without it, a Redis lacking scripting fails every call, `fail_open` swallows it,
and the limiter silently stops limiting while every health check stays green.

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
| 5 | Built-in tools + subprocess sandbox | **done** |
| 6 | `PostgresSessionStore`, Redis limits and run leases | **done** (Postgres SQL unrun) |
| 7 | Container runtime, so `python_exec` can ship | next |
| 8 | Network tools (HTTP fetch, web search) with an egress allowlist | next |
| 9 | Billing, usage metering, admin API | later |

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
