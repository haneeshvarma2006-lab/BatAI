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
| Native `llama-cpp-python` inference | **implemented and running** on real weights |
| Local `.gguf` embeddings | **implemented** |
| RAG pipeline (chunk → embed → retrieve → budgeted context) | **implemented and verified on real Chroma** |
| Vector stores (in-memory + Chroma) | **implemented** |
| Agent loop + policy-gated tool executor | **implemented** |
| Built-in tools + subprocess sandbox | **implemented** |
| Global rate limits + run leases (Redis) | **implemented** |
| Container-isolated code execution | pending — `python_exec` is desktop-only |

140 tests pass, covering tenant isolation, auth, scopes, limits, streaming,
chunking, retrieval, sandbox containment, tool policy, the agent loop and the
Redis limiters (against real Lua execution).

### Verified end to end

The whole product has been driven over real HTTP against real weights and a real
ChromaDB — not through direct function calls:

```
readiness    {'session_store': 'ok', 'vector_store': 'ok',
              'embeddings': 'ok', 'model': 'loaded'}
ingest       201  1 chunk indexed
SSE turn     "Alice Trent owns the billing service and is on call on
              Thursdays."     <- from the document ingested seconds earlier
tool turn    "The result of 27 * 43 + 119 is 1280."
isolation    globex memory search -> []   globex reads acme session -> 404
```

Everything below the "Running the model" heading is what it took to get there,
including the parts that did not work.

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

## Running the model

Inference needs **Python 3.12**, not 3.14 — `llama-cpp-python` has no wheel for
3.14 on any platform yet.

**On Windows, PyPI has no wheel at all**, for any Python version; `pip install
llama-cpp-python` tries to build from source and fails without MSVC. Use the
maintainer's index instead:

```bash
py -3.12 -m venv .venv312
```

CPU build:

```bash
.venv312\Scripts\python -m pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

CUDA build (NVIDIA GPU). The wheel links against the CUDA runtime, which the
`nvidia-*-cu12` packages supply — no CUDA Toolkit install needed:

```bash
.venv312\Scripts\python -m pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

```bash
.venv312\Scripts\python -m pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12
```

pip puts those DLLs under `site-packages/nvidia/*/bin`, which is not on Windows'
DLL search path, so the import fails with a misleading "Could not find module
llama.dll". `bat.adapters.llama_cpp_client` registers those directories before
importing, so this works with no `PATH` fiddling — and if the runtime really is
missing, the error names the actual cause.

Then point the config at your weights:

```
BAT_MODEL__MODEL_PATH=D:/BatAI/models/bat-engine-3b-q4_k_m.gguf
BAT_EMBEDDING__MODEL_PATH=D:/BatAI/models/bat-embed-nomic-v1.5-q4_k_m.gguf
BAT_MODEL__CHAT_FORMAT=chatml-function-calling
BAT_MODEL__N_GPU_LAYERS=0
```

`n_gpu_layers=-1` offloads every layer that fits; the CPU build ignores it.

### Tool calling needs two chat templates, not one

This took three wrong turns to pin down, so the reasoning is worth keeping.

**Under the `.gguf`'s own template**, Qwen2.5 emits its tool call as plain text:

```
<tool_call>{"name": "calculator", "arguments": {"expression": "27 * 43 + 119"}}</tool_call>
```

The loop never sees a call, `wants_tools` stays false, and the agent answers
without the tool — silently, no error anywhere.

**Under `chatml-function-calling`**, calls parse correctly — but the answer that
follows is wrong. Measured, repeatedly: the calculator returned **1280** and the
model replied **1160**, then 1220, then 3381. It was inventing a number while
the correct one sat in its context. That is the worst failure mode available:
a confident wrong answer built on top of correct data.

Prompting does not fix it — a strict "report the tool's value verbatim" system
prompt made it *worse*. Nor is it the model: given the identical conversation
under the `.gguf`'s own template, it answers 1280 every time. The handler
rewrites the conversation when it renders the prompt, and a history that already
contains tool results comes back corrupted.

So the two halves of a tool turn use different templates:

| Turn | Template | Why |
| --- | --- | --- |
| may emit a tool call | `tool_chat_format` | parses `<tool_call>` blocks |
| writes the answer | `chat_format` | keeps the tool result intact |

The loop switches by dropping the tool advertisement after
`agent.tool_rounds_per_turn` rounds (default **1**), which is what selects the
plain template. Raise it only for a model and handler you have actually verified
keep tool results intact across rounds.

Verified end to end after the fix: one call to `calculator`, and the answer
reads "The calculation result is 1280."

Repeated identical calls within a turn are also served from a per-turn cache
rather than re-run — small models re-request a tool instead of using the answer
already in front of them. Tools declaring `deterministic=False` (a clock, a live
feed) are never cached.

### Three bugs only a live run could find

Every one of these passed a full unit suite first. They are recorded because
each names a class of thing fakes cannot catch.

**1. Chroma collection creation was wrong.** The adapter passed
`{"hnsw:space": "cosine"}` as the second positional argument to
`get_or_create_collection`. In chromadb 1.5.9 that parameter is `configuration`,
not `metadata`, and the failure surfaces as
`'dict' object has no attribute 'serialize_to_json'` — which names neither the
argument nor the call. Now passed by keyword, with `embedding_function=None` so
Chroma cannot quietly install its default and download an ONNX model into a
second, inconsistent vector space.

**2. Merely advertising tools corrupted ordinary answers.** Asked a plain
RAG question with tools enabled, the model returned `functions.memory_search:`
— the handler's own internal prefix, as the message body. It is not a tool call,
so the loop handed it to the user as the final reply. The loop now detects that
shape and retries with tools withdrawn, which puts the model back on its own
chat template. Same question, after the fix: *"Alice Trent owns the billing
service and is on call on Thursdays."*

**3. Tools were advertised to principals who could not run them.** The policy
allowlist and the principal's scopes are separate gates, and only the policy was
consulted when deciding what to show the model. A key without `tools:execute`
was offered the calculator, called it, was denied by the executor, and told the
user *"I need to use the calculator tool, but it seems I don't have
permission."* — a wasted turn to report a misconfiguration. Advertisement now
respects both gates. This is presentation only: the executor still re-checks
scopes on every invocation, so nothing about enforcement changed.

### Multi-round tool calling does not work with this model

Tested, not assumed. A two-step task (compute, then double the result) under
`chatml-function-calling`:

| Round-two history | Result |
| --- | --- |
| tool-protocol messages | rebuilt the expression from scratch, wrongly |
| flattened to plain turns | emitted `functions.calculator:`, no call |

Flattening *does* work for writing the final answer — but only when tools are
not advertised on that turn, which is exactly what `tool_rounds_per_turn=1`
arranges. So the agent gets one round of tools per turn: it cannot call a tool,
read the result, and then call a different one. That is a real capability limit
of this model and handler, not a design preference.

---

### CUDA on this hardware

The `cu124` wheel installs and initialises CUDA correctly (it detects the GPU
and reports `supports_gpu_offload: True`), but **loading a model crashes with
`0xc000001d`, STATUS_ILLEGAL_INSTRUCTION**. That is the wheel's CPU baseline,
not CUDA: its ggml layer is built for an instruction set this CPU
(i5-12450H, Alder Lake — AVX2, no AVX-512) does not implement. Nothing in the
config can work around it.

The CPU wheel runs fine, so that is what is configured. Getting GPU offload
needs a llama-cpp-python built for this CPU, i.e. compiling from source with
`-DGGML_CUDA=on` after installing MSVC Build Tools and CMake.

### Two things to know about the weights

**Measured on this machine** (i5-12450H, CPU only, Qwen2.5-3B Q4_K_M): loads in
~4.5s, generates at ~5-7 tok/s. Slow but entirely usable for development.

**A GGUF cannot be fine-tuned.** It is a quantised inference format — the
weights are compressed for fast CPU/GPU decode, and the gradients needed for
training are gone. Renaming the file changes nothing about that. To actually
train on your own data you fine-tune the *original* checkpoint (safetensors,
usually with LoRA), then convert and quantise the result back to GGUF with
`llama.cpp/convert_hf_to_gguf.py`. The path is real, it just doesn't start from
the file the server loads.

**Check the licence before launch.** The bundled default is Qwen2.5-3B-Instruct,
which ships under the **Qwen Research License — non-commercial**. That is fine
for development and evaluation, and not fine for a commercial SaaS. Swap it
before you charge anyone; the config is one line and nothing in the code
changes. Commercially usable models of similar size:

| Model | Licence |
| --- | --- |
| Llama-3.2-3B-Instruct | Llama 3.2 Community (commercial under 700M MAU, requires "Built with Llama" attribution) |
| Qwen2.5-1.5B-Instruct | Apache 2.0 |
| Phi-3.5-mini-instruct (3.8B) | MIT |

---

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
