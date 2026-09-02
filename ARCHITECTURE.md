# BAT Platform Architecture

Target: a commercial, multi-tenant agentic AI platform on FastAPI.
Status: the API/tenancy/session module is implemented; the RAG pipeline and the
agent loop are specified here and have ports waiting for them in `bat/ports/`.

---

## 1. Where we start from

The existing code is a competent single-user desktop assistant. That is a
different product from a multi-tenant SaaS, and three of its properties are
actively hostile to the target:

| Existing | Why it blocks multi-tenancy |
| --- | --- |
| `tools/code_exec.py` — `exec(code, {"__builtins__": __builtins__})` | Arbitrary RCE on the server for any caller. `safe_env` is not a sandbox: `__builtins__` includes `__import__`, `open`, `eval`. |
| `tools/system_tools.py` — `pyautogui.write`, `os.startfile`, `subprocess(shell=True)` | Drives the *server's* desktop and shell. There is no per-tenant version of "move the mouse". |
| `find_file_location` walking `D:\` | Reads the host filesystem, and hardcodes one machine's layout. |
| `chromadb.PersistentClient(path=...)` | Process-local file lock. Breaks with >1 worker; no tenant partition. |
| `engine.py` vs `core/brain.py` | Two divergent brains — an intent-router and a tool-caller. Only one can be the product. |
| In-process `working_memory` list | Lost on restart, invisible to other replicas. |
| Blocking `ollama.chat` in a request path | Stalls the event loop; one slow generation blocks every other request on the worker. |
| `server.py` (untracked) — `f'ollama run bat-engine "{prompt}"'` with `shell=True` | Shell injection. A prompt containing `"` and `;` runs commands. |

The through-line: **on your laptop, ambient authority is the feature. Behind a
shared API, it is the vulnerability.** A model steered by untrusted text —
a retrieved document, an uploaded PDF, a web result — becomes the attacker's
proxy. Prompt injection promotes to RCE unless the tool boundary refuses by
default.

Nothing above needs to be thrown away. `core/brain.py`'s tool-calling loop is
the right shape; it needs a tenancy boundary around it and a policy gate under
it. `engine.py` should be retired once its live-API tools are re-registered as
policy-gated tools.

---

## 2. Layering

Dependencies point strictly inward. The domain knows nothing about FastAPI;
the API knows nothing about ChromaDB.

```
┌─────────────────────────────────────────────────────────┐
│  api/          routers · deps · security · middleware   │  HTTP, SSE
├─────────────────────────────────────────────────────────┤
│  services/     agent loop · RAG pipeline · tool exec    │  orchestration
├─────────────────────────────────────────────────────────┤
│  ports/        SessionStore · MemoryPipeline · LLMClient│  protocols
│                Tool · ToolRegistry · AgentRunner        │
├─────────────────────────────────────────────────────────┤
│  domain/       Tenant · Session · Message · errors      │  pure types
└─────────────────────────────────────────────────────────┘
        ▲
        └── adapters/  Chroma · Redis · Postgres · Ollama  (implement ports)
```

Ports are `typing.Protocol`, so adapters need no base class and tests can pass a
plain object. `isinstance(store, SessionStore)` works at runtime for wiring
assertions.

---

## 3. Tenancy — the spine

Every operation belongs to exactly one tenant. The rules:

1. **One place mints identity.** `ApiKeyAuthenticator.authenticate` is the only
   code that turns a credential into a `TenantContext`. Nothing downstream
   re-parses a header.
2. **Tenancy is a parameter, never ambient.** `TenantContext` is passed
   explicitly. It is deliberately *not* read from a contextvar for
   authorization — a function that can touch tenant data has the tenant in its
   signature, so a missing check is visible in review. (Contextvars carry the
   ids into *log records* only.)
3. **Clients cannot assert tenancy.** `X-Tenant-Id`, if sent, is checked
   against the credential and rejected on mismatch; it is never a source.
4. **Cross-tenant access is 404, not 403.** A 403 confirms the resource exists.
   Every store method takes `tenant_id` first and treats foreign ids as absent.
5. **Isolation is structural where possible.** The session store is partitioned
   `dict[tenant_id][session_id]`, so forgetting to filter is a `KeyError`, not
   a leak. Vector storage uses one collection per tenant
   (`tenant_namespace()`), so a missing filter cannot cross tenants either.

Scopes (`sessions:read`, `agent:invoke`, `tools:execute`, …) are checked at the
route boundary via `deps.scoped(...)` and **again** at the tool boundary, so a
routing mistake cannot widen a principal's reach.

### Credentials

Only SHA-256 digests are configured or stored, compared with
`hmac.compare_digest` across the whole registry so lookup time is independent of
position. Keys carry the public prefix `bat_sk_` so leaks are greppable.
Production replaces the in-config registry with a DB table (revocation,
rotation, last-used) — the lookup surface stays `digest -> ApiKeyRecord`, so only
`ApiKeyAuthenticator` changes.

---

## 4. Request layer (implemented)

```
POST   /v1/sessions                          open a session
GET    /v1/sessions                          keyset-paginated list
GET    /v1/sessions/{id}                     fetch
DELETE /v1/sessions/{id}                     delete + transcript
GET    /v1/sessions/{id}/messages            transcript
POST   /v1/sessions/{id}/messages            one turn, buffered
POST   /v1/sessions/{id}/messages/stream     one turn, SSE
GET    /v1/whoami                            resolve the calling credential
GET    /healthz  /readyz                     liveness / readiness
```

**Async model.** Handlers are `async`. The blocking dependencies do not belong
on the event loop, and each has a specific answer:

- *Model server* — `ollama.AsyncClient` over httpx. A process-wide semaphore
  (`model.max_concurrency`) protects the model server from being swamped.
- *ChromaDB* — the client is sync; calls go through
  `anyio.to_thread.run_sync` in the adapter, never inline in a handler.
- *Long turns* — SSE, not a blocking POST. The connection streams tokens as
  they are produced and a keepalive comment every 15s so proxies don't reap an
  idle connection during a tool call.

**One code path for both endpoints.** `AgentRunner.run()` returns an
`AsyncIterator[AgentEvent]`. The streaming route forwards each event as an SSE
frame; the buffered route folds the same stream with `agent.collect()`. There is
no second implementation to drift.

**Cancellation is structural.** A client disconnect closes the async generator;
in-flight work unwinds through ordinary `finally` blocks and the concurrency
slot is released. (Two bugs worth remembering, both fixed in
`api/routers/messages.py`: the SSE `done` frame must *not* be emitted from a
`finally`, because yielding while the generator is being closed raises
`RuntimeError`; and a keepalive must hold one shielded pending pull across
ticks, because starting a second `__anext__` raises "already running".)

**Errors.** One shape everywhere: RFC 9457 `application/problem+json` carrying
`code`, `detail` and `request_id`. A message crosses the boundary only if its
error is marked `public`; everything else becomes a generic 500 with the detail
in the logs, keyed by request id.

**Limits.** Per-tenant token bucket on request rate; a separate per-tenant
concurrency cap on agent runs, because one run can hold a model slot for a
minute and fifty of them starve the box. Both are per-process today — for a
hard global limit they move behind Redis with no change to call sites.

---

## 5. RAG memory pipeline (specified; ports in `bat/ports/retrieval.py`)

Three replaceable stages so Chroma stays an implementation detail:

```
ingest:    text ─► Chunker ─► Embedder ─► VectorStore.upsert(namespace)
retrieve:  query ─► Embedder ─► VectorStore.search(namespace) ─► rerank ─► budget ─► context
```

**Isolation.** One Chroma collection per tenant, named by `tenant_namespace()`.
Memory *kind* (`knowledge` / `user_profile` / `document` / `conversation`) is
metadata on that collection rather than a separate collection, so one query
spans banks and stays tenant-filtered. `drop_namespace()` exists because GDPR
erasure is a hard requirement, not a nice-to-have.

**Deployment.** `PersistentClient` holds a file lock and is single-process only;
anything with more than one worker must run Chroma in server mode
(`vector.mode="http"`). `Settings._harden` refuses to boot production on
`mode="memory"`.

Improvements over the current `search_all`, which fires three fixed queries and
concatenates whatever comes back:

- **Score thresholds.** Chroma always returns `n_results` rows. Today an
  unrelated stored fact gets injected as "USER PROFILE MEMORY" with equal
  authority. Drop anything under `min_score` rather than padding.
- **A token budget, not a row count.** `build_context(token_budget=...)` fills
  to a budget, highest score first.
- **Chunk with overlap and respect boundaries.** `text[i:i+1000]` splits
  mid-sentence and mid-word; a 150-token overlap keeps split facts retrievable.
- **Retrieved text is data, not instruction.** Fence it in the prompt and label
  it untrusted. A document that says "ignore your instructions" is exactly the
  injection vector the tool policy exists to contain.

---

## 6. Agent loop (specified; ports in `bat/ports/agent.py`, `tools.py`)

```
build prompt (system + RAG context + history + user)
  └─► LLMClient.complete(tools=registry.specs_for(policy))
        ├── no tool calls  ──► FinalEvent
        └── tool calls     ──► for each: ToolExecutor.execute(invocation, policy)
                                 ├── policy check (re-checked, not trusted from the advert)
                                 ├── JSON-schema validation of arguments
                                 ├── timeout + output cap
                                 ├── audit record
                                 └── ToolResultEvent ──► append observation, loop
```

Bounded by `max_steps`, a wall-clock `deadline_s`, and
`policy.max_calls_per_run`. Exactly one terminal event is always emitted, even
on an internal failure — the contract the API layer relies on.

### The tool security model

This is the part that must not be compromised for convenience.

**Default deny.** A tool is unavailable unless its name is in the tenant's
allowlist. Registration is not authorization.

**Declared isolation, enforced by deployment.** Every tool states an
`Isolation` level; the deployment sets a floor and anything below it is refused:

| Level | Meaning | Allowed in SaaS |
| --- | --- | --- |
| `NONE` | in-process, ambient authority | **no** — desktop build only |
| `PURE` | in-process, no I/O at all | dev only |
| `NETWORK` | egress to an allowlist, no local effects | yes |
| `SUBPROCESS` | separate process, dropped privileges, rlimits | yes |
| `SANDBOX` | container/microVM per call, no host mounts, no creds | yes |

`Settings._harden` **refuses to boot production** below `SUBPROCESS`. That is
why the existing tools cannot simply be registered: `execute_python_code`,
`automate_typing`, `search_and_open_file` and `open_application` are all
`Isolation.NONE`.

Their migration path:

- `execute_python_code` → a real sandbox: container per invocation, no network,
  read-only rootfs, memory/CPU/wall limits, non-root, dropped capabilities.
  Nothing less is worth shipping — `exec` with a filtered globals dict is
  escapable in one line.
- `automate_typing` → **delete**. There is no server-side meaning for it.
- `open_application` / `search_and_open_file` → **delete** the host versions.
  The tenant-scoped analogue is "open a document from *this tenant's* object
  storage", which is a different tool.
- `find_file_location` → a tenant-scoped document search over the RAG index.
- `get_weather` / `get_news` / `get_f1_standings` / `search_web` → keep, as
  `Isolation.NETWORK` with an egress allowlist, per-tenant quota and timeouts.

**Also enforced:** arguments validated against the declared JSON schema before
the tool sees them; per-call timeout and output cap; and an audit record
(tenant, principal, session, tool, arguments, allow/deny, duration) for every
invocation. `requires_confirmation` marks tools that need a human ack before the
loop may run them.

---

## 7. Deployment

```
client ─► edge (TLS, WAF, unauthenticated flood control)
           └─► uvicorn workers (stateless)
                 ├─► Postgres   sessions, transcripts, tenants, keys, tool audit
                 ├─► Redis      rate limits, run leases, idempotency keys
                 ├─► Chroma     server mode, one collection per tenant
                 └─► model      Ollama/vLLM pool, semaphore-guarded
```

Workers hold no state, so they scale horizontally and restart freely.
`Settings._harden` fails the boot — loudly, before serving a request — on any
config that only makes sense in development: no API keys, wildcard CORS,
in-memory sessions or vectors, or tool isolation below `SUBPROCESS`.

---

## 8. Build order

| # | Module | Depends on | Status |
| --- | --- | --- | --- |
| 1 | Tenancy, settings, session store, API, SSE | — | **done** |
| 2 | `PostgresSessionStore`, `RedisRateLimiter` | 1 | next |
| 3 | `OllamaClient` (async, semaphore, retries) | 1 | next |
| 4 | RAG pipeline + `ChromaVectorStore` | 1, 3 | then |
| 5 | Tool registry, executor, sandbox runner | 1, 3 | then |
| 6 | Agent loop replacing `ReferenceAgentRunner` | 3, 4, 5 | then |
| 7 | Billing, usage metering, admin API | 1–6 | later |

Each row lands behind a port that already exists, so nothing above it changes
when it does.
