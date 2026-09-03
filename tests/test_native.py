"""Tests for native inference, the RAG pipeline and the agent loop.

    python -m unittest discover -s tests -t . -v

Runs without ``llama-cpp-python`` or any ``.gguf``: a fake module is injected
into ``sys.modules`` so the adapter's own threading, admission and streaming
code is what executes.
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from bat.adapters.llama_cpp_embedder import HashingEmbedder
from bat.adapters.vector_memory import InMemoryVectorStore
from bat.domain.conversation import Message, Role, Session, utcnow
from bat.domain.errors import RateLimitError
from bat.domain.tenancy import Principal, Scope, TenantContext
from bat.ports.agent import ErrorEvent, FinalEvent, StopReason, TokenEvent, ToolCallEvent, ToolResultEvent
from bat.ports.llm import Completion, ToolSpec
from bat.ports.retrieval import MemoryKind, RetrievalQuery, tenant_namespace
from bat.ports.tools import (
    Authority,
    Isolation,
    SideEffect,
    ToolDefinition,
    ToolInvocation,
    ToolPolicy,
)
from bat.services.agent.loop import REPEAT_NOTICE, NativeAgentRunner, _is_unusable
from bat.services.agent.tools import (
    InMemoryToolRegistry,
    PolicyToolExecutor,
    validate_arguments,
)
from bat.services.rag.chunking import RecursiveChunker
from bat.services.rag.pipeline import CONTEXT_PREAMBLE, LocalMemoryPipeline
from bat.settings import ModelSettings, VectorSettings
from tests.fakes import FakeLlamaModule, ScriptedLLM


def run(coro):
    return asyncio.run(coro)


def make_context(tenant: str = "acme", scopes: frozenset[Scope] | None = None) -> TenantContext:
    return TenantContext(
        tenant_id=tenant,
        principal=Principal(
            id="u1", tenant_id=tenant, scopes=scopes or frozenset(Scope)
        ),
        request_id="req",
        started_at=utcnow(),
    )


# ---------------------------------------------------------------------------
# Native inference client
# ---------------------------------------------------------------------------


class TestLlamaCppClient(unittest.TestCase):
    def setUp(self) -> None:
        self.module = FakeLlamaModule()
        sys.modules["llama_cpp"] = self.module
        self.addCleanup(sys.modules.pop, "llama_cpp", None)

        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.weights = Path(self.tmp.name) / "model.gguf"
        self.weights.write_bytes(b"not real weights")

    def build(self, **overrides):
        from bat.adapters.llama_cpp_client import LlamaCppClient

        return LlamaCppClient(
            ModelSettings(model_path=self.weights, **overrides)
        )

    def test_loads_weights_with_configured_parameters(self) -> None:
        client = self.build(n_ctx=4096, n_gpu_layers=12)
        run(client.load())
        self.assertTrue(client.is_loaded)
        kwargs = self.module.instances[0].kwargs
        self.assertEqual(kwargs["model_path"], str(self.weights))
        self.assertEqual(kwargs["n_ctx"], 4096)
        self.assertEqual(kwargs["n_gpu_layers"], 12)

    def test_missing_weights_file_is_reported_clearly(self) -> None:
        from bat.adapters.llama_cpp_client import ModelUnavailableError

        client = self.build()
        client._settings = ModelSettings(model_path=Path("/nope/missing.gguf"))
        with self.assertRaises(ModelUnavailableError) as caught:
            run(client.load())
        self.assertIn("not found", caught.exception.message)

    def test_completion_is_parsed(self) -> None:
        client = self.build()
        run(client.load())
        result = run(client.complete(messages=[]))
        self.assertEqual(result.content, "Hello from the fake model.")
        self.assertEqual(result.prompt_tokens, 11)
        self.assertFalse(result.wants_tools)

    def test_tool_calls_are_parsed_from_json_arguments(self) -> None:
        client = self.build()
        run(client.load())
        self.module.instances[0].tool_calls = [
            {
                "id": "call_1",
                "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'},
            }
        ]
        result = run(client.complete(messages=[], tools=[ToolSpec("add", "adds")]))
        self.assertTrue(result.wants_tools)
        self.assertEqual(result.tool_calls[0].name, "add")
        self.assertEqual(result.tool_calls[0].arguments, {"a": 1, "b": 2})

    def test_malformed_tool_arguments_do_not_raise(self) -> None:
        """Small quantised models emit broken JSON; that is data, not a crash."""
        client = self.build()
        run(client.load())
        self.module.instances[0].tool_calls = [
            {"id": "c", "function": {"name": "add", "arguments": "{not json"}}
        ]
        result = run(client.complete(messages=[]))
        self.assertIn("__unparsed__", result.tool_calls[0].arguments)

    def test_streaming_yields_deltas(self) -> None:
        client = self.build()
        run(client.load())

        async def collect():
            return [d async for d in client.stream(messages=[])]

        deltas = run(collect())
        self.assertGreater(len(deltas), 1)
        self.assertEqual("".join(deltas).strip(), "Hello from the fake model.")

    def test_concurrent_calls_are_serialised(self) -> None:
        """The whole point of the single-worker executor.

        `Llama` is not thread-safe; if two generations ever overlap, the fake
        records it and this fails.
        """
        client = self.build()
        run(client.load())
        self.module.instances[0].delay_s = 0.01

        async def hammer():
            await asyncio.gather(*(client.complete(messages=[]) for _ in range(6)))

        run(hammer())
        self.assertEqual(
            self.module.instances[0].concurrent_calls,
            0,
            "concurrent access to a non-thread-safe Llama instance",
        )

    def test_queue_depth_is_bounded(self) -> None:
        client = self.build(max_queue_depth=2)
        run(client.load())
        self.module.instances[0].delay_s = 0.05

        async def flood():
            tasks = [
                asyncio.create_task(client.complete(messages=[])) for _ in range(6)
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)

        results = run(flood())
        rejected = [r for r in results if isinstance(r, RateLimitError)]
        self.assertTrue(rejected, "over-depth callers should be rejected, not queued")
        self.assertIn("max_queue_depth", rejected[0].details)

    def test_timeout_surfaces_as_upstream_timeout(self) -> None:
        from bat.domain.errors import UpstreamTimeoutError

        client = self.build()
        run(client.load())
        self.module.instances[0].delay_s = 0.3
        with self.assertRaises(UpstreamTimeoutError):
            run(client.complete(messages=[], timeout_s=0.05))


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------


class TestChunking(unittest.TestCase):
    def test_respects_size_and_keeps_overlap(self) -> None:
        chunker = RecursiveChunker(chunk_size=120, chunk_overlap=30)
        text = " ".join(f"Sentence number {i} about the system." for i in range(40))
        chunks = chunker.split(text)
        self.assertTrue(all(len(c) <= 120 for c in chunks))
        shared = sum(
            1
            for a, b in zip(chunks, chunks[1:])
            if any(w in b for w in a.split()[-4:] if len(w) > 3)
        )
        self.assertEqual(shared, len(chunks) - 1, "neighbours must share context")

    def test_unbroken_run_is_hard_split(self) -> None:
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.split("x" * 220)
        self.assertTrue(chunks)
        self.assertTrue(all(len(c) <= 50 for c in chunks))

    def test_blank_input_yields_nothing(self) -> None:
        self.assertEqual(list(RecursiveChunker().split("   \n\n  ")), [])

    def test_overlap_must_be_smaller_than_size(self) -> None:
        with self.assertRaises(ValueError):
            RecursiveChunker(chunk_size=100, chunk_overlap=100)


class TestMemoryPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = VectorSettings(
            chunk_size=200, chunk_overlap=40, default_top_k=4, min_score=0.15
        )
        self.store = InMemoryVectorStore(HashingEmbedder(dimensions=512))
        self.rag = LocalMemoryPipeline(store=self.store, settings=self.settings)
        self.doc = (
            "The project codename is Midnight. The weekly sync is at 5 PM Thursdays. "
            "Alice owns the billing service. Bob owns the retrieval pipeline."
        )

    def test_ingest_then_retrieve(self) -> None:
        run(self.rag.ingest(tenant_id="acme", text=self.doc, kind=MemoryKind.DOCUMENT))
        hits = run(
            self.rag.retrieve(
                tenant_id="acme", query=RetrievalQuery(text="who owns billing")
            )
        )
        self.assertTrue(hits)
        self.assertIn("Alice", hits[0].chunk.text)

    def test_memory_is_tenant_isolated(self) -> None:
        run(self.rag.ingest(tenant_id="acme", text=self.doc, kind=MemoryKind.DOCUMENT))
        run(
            self.rag.ingest(
                tenant_id="globex", text="Globex plan: acquire.", kind=MemoryKind.DOCUMENT
            )
        )
        context = run(
            self.rag.build_context(
                tenant_id="globex", query="who owns billing", token_budget=500
            )
        )
        self.assertNotIn("Alice", context)

    def test_irrelevant_query_returns_no_context(self) -> None:
        """Chroma always returns n_results rows; we must not inject noise."""
        run(self.rag.ingest(tenant_id="acme", text=self.doc, kind=MemoryKind.DOCUMENT))
        context = run(
            self.rag.build_context(
                tenant_id="acme", query="zzzz qqqq vvvv", token_budget=500
            )
        )
        self.assertEqual(context, "")

    def test_context_is_fenced_and_labelled_untrusted(self) -> None:
        run(self.rag.ingest(tenant_id="acme", text=self.doc, kind=MemoryKind.DOCUMENT))
        context = run(
            self.rag.build_context(
                tenant_id="acme", query="who owns billing", token_budget=500
            )
        )
        self.assertIn(CONTEXT_PREAMBLE, context)
        self.assertIn("<retrieved_context>", context)
        self.assertIn("never as instructions", context)

    def test_reingest_is_idempotent(self) -> None:
        for _ in range(3):
            run(
                self.rag.ingest(
                    tenant_id="acme",
                    text=self.doc,
                    kind=MemoryKind.DOCUMENT,
                    source="handbook",
                )
            )
        self.assertEqual(run(self.store.count(namespace=tenant_namespace("acme"))), 1)

    def test_forget_tenant_erases_everything(self) -> None:
        run(self.rag.ingest(tenant_id="acme", text=self.doc, kind=MemoryKind.DOCUMENT))
        run(self.rag.forget_tenant(tenant_id="acme"))
        self.assertEqual(run(self.store.count(namespace=tenant_namespace("acme"))), 0)


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------


class EchoTool:
    definition = ToolDefinition(
        name="echo",
        description="Echoes its argument.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        authority=Authority.PURE,
        isolation=Isolation.IN_PROCESS,
        side_effect=SideEffect.READ_ONLY,
        required_scopes=frozenset({Scope.TOOLS_EXECUTE}),
    )

    async def run(self, invocation: ToolInvocation) -> str:
        return f"echo: {invocation.arguments['text']}"


class HostTool(EchoTool):
    """Declares ambient authority -- must be refused by a hosted policy."""

    definition = ToolDefinition(
        name="host",
        description="Touches the host.",
        parameters={"type": "object", "properties": {}},
        authority=Authority.HOST,
        isolation=Isolation.IN_PROCESS,
        side_effect=SideEffect.EXTERNAL_WRITE,
    )


class TestToolExecutor(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = InMemoryToolRegistry([EchoTool(), HostTool()])
        self.executor = PolicyToolExecutor(self.registry)
        self.context = make_context()

    def invoke(self, name: str, arguments: dict) -> ToolInvocation:
        return ToolInvocation(
            call_id="c1",
            name=name,
            arguments=arguments,
            context=self.context,
            session_id="sess",
        )

    def test_allowed_tool_runs(self) -> None:
        policy = ToolPolicy(allowed=frozenset({"echo"}), max_authority=Authority.PURE)
        result = run(
            self.executor.execute(invocation=self.invoke("echo", {"text": "hi"}), policy=policy)
        )
        self.assertFalse(result.is_error)
        self.assertEqual(result.content, "echo: hi")

    def test_unlisted_tool_is_denied(self) -> None:
        """Default deny: registration is not authorization."""
        policy = ToolPolicy(allowed=frozenset(), max_authority=Authority.PURE)
        result = run(
            self.executor.execute(invocation=self.invoke("echo", {"text": "hi"}), policy=policy)
        )
        self.assertTrue(result.is_error)
        self.assertIn("not enabled", result.content)

    def test_authority_ceiling_refuses_host_tools(self) -> None:
        policy = ToolPolicy(
            allowed=frozenset({"host"}),
            max_authority=Authority.NETWORK,
            allowed_side_effects=frozenset(SideEffect),
        )
        result = run(self.executor.execute(invocation=self.invoke("host", {}), policy=policy))
        self.assertTrue(result.is_error)
        self.assertIn("HOST authority", result.content)

    def test_hallucinated_tool_name_is_an_observation(self) -> None:
        policy = ToolPolicy(allowed=frozenset({"echo"}), max_authority=Authority.PURE)
        result = run(self.executor.execute(invocation=self.invoke("nope", {}), policy=policy))
        self.assertTrue(result.is_error)
        self.assertIn("not registered", result.content)

    def test_bad_arguments_are_rejected_before_the_tool_runs(self) -> None:
        policy = ToolPolicy(allowed=frozenset({"echo"}), max_authority=Authority.PURE)
        result = run(
            self.executor.execute(invocation=self.invoke("echo", {"text": 42}), policy=policy)
        )
        self.assertTrue(result.is_error)
        self.assertIn("must be of type string", result.content)

    def test_missing_scope_is_refused(self) -> None:
        self.context = make_context(scopes=frozenset({Scope.SESSIONS_READ}))
        policy = ToolPolicy(allowed=frozenset({"echo"}), max_authority=Authority.PURE)
        result = run(
            self.executor.execute(invocation=self.invoke("echo", {"text": "x"}), policy=policy)
        )
        self.assertTrue(result.is_error)
        self.assertIn("requires scope", result.content)

    def test_registry_advertises_only_permitted_tools(self) -> None:
        policy = ToolPolicy(allowed=frozenset({"echo"}), max_authority=Authority.PURE)
        names = [s.name for s in self.registry.specs_for(policy)]
        self.assertEqual(names, ["echo"])

    def test_argument_validation_rejects_bool_as_integer(self) -> None:
        schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
        self.assertTrue(validate_arguments(schema, {"n": True}))
        self.assertFalse(validate_arguments(schema, {"n": 3}))


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


class TestNativeAgentRunner(unittest.TestCase):
    def setUp(self) -> None:
        self.context = make_context()
        self.session = Session.create(tenant_id="acme", principal_id="u1")
        self.store = InMemoryVectorStore(HashingEmbedder(dimensions=512))
        self.rag = LocalMemoryPipeline(
            store=self.store,
            settings=VectorSettings(chunk_size=200, chunk_overlap=40, min_score=0.15),
        )

    def request(self, text: str = "hello", **overrides):
        from bat.ports.agent import RunRequest

        defaults = dict(
            context=self.context,
            session=self.session,
            user_input=text,
            history=(),
            policy=ToolPolicy(),
            max_steps=4,
            deadline_s=10.0,
            stream_tokens=True,
        )
        defaults.update(overrides)
        return RunRequest(**defaults)

    async def drain(self, runner, request) -> list:
        return [event async for event in runner.run(request)]

    def test_streams_tokens_then_exactly_one_final(self) -> None:
        runner = NativeAgentRunner(llm=ScriptedLLM([], "the answer is four"))
        events = run(self.drain(runner, self.request()))
        self.assertTrue(any(isinstance(e, TokenEvent) for e in events))
        finals = [e for e in events if isinstance(e, (FinalEvent, ErrorEvent))]
        self.assertEqual(len(finals), 1)
        self.assertEqual(finals[0].content.strip(), "the answer is four")

    def test_retrieved_context_reaches_the_system_prompt(self) -> None:
        run(
            self.rag.ingest(
                tenant_id="acme",
                text="Alice owns the billing service and is on call Thursdays.",
                kind=MemoryKind.DOCUMENT,
            )
        )
        llm = ScriptedLLM([], "Alice does.")
        runner = NativeAgentRunner(llm=llm, memory=self.rag)
        run(self.drain(runner, self.request("who owns billing")))
        self.assertIn("Alice owns the billing service", llm.last_system_prompt)
        self.assertIn(CONTEXT_PREAMBLE, llm.last_system_prompt)

    def test_no_context_when_nothing_is_relevant(self) -> None:
        run(self.rag.ingest(tenant_id="acme", text="Unrelated note.", kind=MemoryKind.DOCUMENT))
        llm = ScriptedLLM([], "I do not know.")
        runner = NativeAgentRunner(llm=llm, memory=self.rag)
        run(self.drain(runner, self.request("zzzz qqqq")))
        self.assertNotIn("<retrieved_context>", llm.last_system_prompt)

    def test_retrieval_failure_degrades_instead_of_failing_the_turn(self) -> None:
        class BrokenMemory:
            async def build_context(self, **kwargs):
                raise RuntimeError("vector store down")

        runner = NativeAgentRunner(llm=ScriptedLLM([], "still works"), memory=BrokenMemory())
        events = run(self.drain(runner, self.request()))
        finals = [e for e in events if isinstance(e, FinalEvent)]
        self.assertEqual(len(finals), 1)
        self.assertEqual(finals[0].content.strip(), "still works")

    def test_tool_call_round_trip(self) -> None:
        from bat.domain.conversation import ToolCall

        registry = InMemoryToolRegistry([EchoTool()])
        llm = ScriptedLLM(
            [
                Completion(
                    content="",
                    tool_calls=(ToolCall(id="c1", name="echo", arguments={"text": "hi"}),),
                )
            ],
            stream_text="The tool said hi.",
        )
        runner = NativeAgentRunner(
            llm=llm, registry=registry, executor=PolicyToolExecutor(registry)
        )
        events = run(
            self.drain(
                runner,
                self.request(
                    policy=ToolPolicy(
                        allowed=frozenset({"echo"}), max_authority=Authority.PURE
                    )
                ),
            )
        )
        self.assertTrue(any(isinstance(e, ToolCallEvent) for e in events))
        results = [e for e in events if isinstance(e, ToolResultEvent)]
        self.assertEqual(results[0].content, "echo: hi")
        finals = [e for e in events if isinstance(e, FinalEvent)]
        self.assertIn("The tool said hi.", finals[0].content)

    def test_synthesis_turn_drops_the_tools(self) -> None:
        """The whole point of tool_rounds_per_turn.

        llama.cpp's tool-calling handlers corrupt a history that already holds
        tool results, so the turn that writes the answer must not advertise
        tools -- that is what selects the model's own chat template.
        """
        from bat.domain.conversation import ToolCall

        registry = InMemoryToolRegistry([EchoTool()])
        llm = ScriptedLLM(
            [
                Completion(
                    content="",
                    tool_calls=(ToolCall(id="c1", name="echo", arguments={"text": "hi"}),),
                )
            ],
            stream_text="done",
        )
        runner = NativeAgentRunner(
            llm=llm, registry=registry, executor=PolicyToolExecutor(registry)
        )
        run(
            self.drain(
                runner,
                self.request(
                    policy=ToolPolicy(
                        allowed=frozenset({"echo"}), max_authority=Authority.PURE
                    )
                ),
            )
        )
        self.assertEqual(len(llm.calls), 2, "one tool turn, then one synthesis turn")
        self.assertTrue(llm.calls[0]["tools"], "the first turn should offer tools")
        self.assertEqual(
            llm.calls[1]["tools"], [], "the synthesis turn must offer none"
        )

    def test_no_tools_are_offered_when_the_policy_is_empty(self) -> None:
        registry = InMemoryToolRegistry([EchoTool()])
        llm = ScriptedLLM([], "no tools needed")
        runner = NativeAgentRunner(
            llm=llm, registry=registry, executor=PolicyToolExecutor(registry)
        )
        run(self.drain(runner, self.request()))
        self.assertEqual(llm.calls[-1]["tools"], [])


    def test_repeated_identical_tool_call_is_served_from_cache(self) -> None:
        """Small models re-request a tool instead of using the answer.

        Re-running wastes a step and teaches nothing -- the same result comes
        back and the model repeats again. The cached reply says so explicitly.
        """
        from bat.domain.conversation import ToolCall

        class CountingEcho(EchoTool):
            runs = 0

            async def run(self, invocation):
                type(self).runs += 1
                return f"echo: {invocation.arguments['text']}"

        tool = CountingEcho()
        registry = InMemoryToolRegistry([tool])
        repeat = ToolCall(id="c1", name="echo", arguments={"text": "hi"})
        llm = ScriptedLLM(
            [
                Completion(content="", tool_calls=(repeat,)),
                Completion(
                    content="",
                    tool_calls=(ToolCall(id="c2", name="echo", arguments={"text": "hi"}),),
                ),
                Completion(content="Done."),
            ]
        )
        runner = NativeAgentRunner(
            llm=llm, registry=registry, executor=PolicyToolExecutor(registry)
        )
        events = run(
            self.drain(
                runner,
                self.request(
                    policy=ToolPolicy(
                        allowed=frozenset({"echo"}),
                        max_authority=Authority.PURE,
                        max_calls_per_run=4,
                    ),
                    tool_rounds_per_turn=3,
                ),
            )
        )
        results = [e for e in events if isinstance(e, ToolResultEvent)]
        self.assertEqual(len(results), 2, "both calls should produce an observation")
        self.assertEqual(CountingEcho.runs, 1, "the tool should execute only once")
        self.assertNotIn(REPEAT_NOTICE, results[0].content)
        self.assertIn(REPEAT_NOTICE, results[1].content)
        self.assertIn("echo: hi", results[1].content)

    def test_different_arguments_are_not_deduplicated(self) -> None:
        from bat.domain.conversation import ToolCall

        class CountingEcho(EchoTool):
            runs = 0

            async def run(self, invocation):
                type(self).runs += 1
                return f"echo: {invocation.arguments['text']}"

        registry = InMemoryToolRegistry([CountingEcho()])
        llm = ScriptedLLM(
            [
                Completion(
                    content="",
                    tool_calls=(
                        ToolCall(id="a", name="echo", arguments={"text": "one"}),
                        ToolCall(id="b", name="echo", arguments={"text": "two"}),
                    ),
                ),
                Completion(content="Done."),
            ]
        )
        runner = NativeAgentRunner(
            llm=llm, registry=registry, executor=PolicyToolExecutor(registry)
        )
        run(
            self.drain(
                runner,
                self.request(
                    policy=ToolPolicy(
                        allowed=frozenset({"echo"}),
                        max_authority=Authority.PURE,
                        max_calls_per_run=4,
                    ),
                    tool_rounds_per_turn=3,
                ),
            )
        )
        self.assertEqual(CountingEcho.runs, 2, "distinct arguments must both run")

    def test_argument_order_does_not_defeat_the_cache(self) -> None:
        from bat.services.agent.loop import _call_key
        from bat.domain.conversation import ToolCall

        first = ToolCall(id="a", name="t", arguments={"x": 1, "y": 2})
        second = ToolCall(id="b", name="t", arguments={"y": 2, "x": 1})
        self.assertEqual(_call_key(first), _call_key(second))

    def test_non_deterministic_tools_are_never_cached(self) -> None:
        """A clock or live feed must actually run again."""
        from bat.ports.tools import ToolDefinition
        from bat.domain.conversation import ToolCall

        class Clock:
            runs = 0
            definition = ToolDefinition(
                name="clock",
                description="Current time.",
                parameters={"type": "object", "properties": {}},
                authority=Authority.PURE,
                isolation=Isolation.IN_PROCESS,
                deterministic=False,
            )

            async def run(self, invocation):
                type(self).runs += 1
                return f"tick {type(self).runs}"

        registry = InMemoryToolRegistry([Clock()])
        llm = ScriptedLLM(
            [
                Completion(content="", tool_calls=(ToolCall(id="a", name="clock", arguments={}),)),
                Completion(content="", tool_calls=(ToolCall(id="b", name="clock", arguments={}),)),
                Completion(content="Done."),
            ]
        )
        runner = NativeAgentRunner(
            llm=llm, registry=registry, executor=PolicyToolExecutor(registry)
        )
        run(
            self.drain(
                runner,
                self.request(
                    policy=ToolPolicy(
                        allowed=frozenset({"clock"}),
                        max_authority=Authority.PURE,
                        max_calls_per_run=4,
                    ),
                    tool_rounds_per_turn=3,
                ),
            )
        )
        self.assertEqual(Clock.runs, 2, "a non-deterministic tool must re-run")


    def test_handler_noise_is_retried_without_tools(self) -> None:
        """Advertising tools must not corrupt an ordinary answer.

        llama.cpp's tool handler sometimes returns its own prefix as the
        message. It is not a tool call, so without this the loop hands the user
        "functions.memory_search:" as the final reply -- observed on a plain
        RAG question through the real API.
        """
        registry = InMemoryToolRegistry([EchoTool()])
        llm = ScriptedLLM(
            [Completion(content="functions.memory_search:")],
            stream_text="Alice owns the billing service.",
        )
        runner = NativeAgentRunner(
            llm=llm, registry=registry, executor=PolicyToolExecutor(registry)
        )
        events = run(
            self.drain(
                runner,
                self.request(
                    policy=ToolPolicy(
                        allowed=frozenset({"echo"}), max_authority=Authority.PURE
                    )
                ),
            )
        )
        finals = [e for e in events if isinstance(e, FinalEvent)]
        self.assertEqual(len(finals), 1)
        self.assertIn("Alice owns the billing service.", finals[0].content)
        self.assertNotIn("functions.", finals[0].content)
        self.assertEqual(llm.calls[1]["tools"], [], "the retry must withdraw tools")

    def test_leak_detection_does_not_flag_ordinary_prose(self) -> None:
        for good in (
            "The calculation result is 1280.",
            "Functions like these are useful.",
            "Alice owns billing.",
        ):
            self.assertFalse(_is_unusable(good), good)
        for bad in ("functions.calculator:", "  functions.x: ", "<tool_call>", ""):
            self.assertTrue(_is_unusable(bad), repr(bad))


    def test_tools_the_principal_cannot_run_are_not_advertised(self) -> None:
        """Otherwise the model is offered a tool, tries it, is denied, and
        tells the user it lacks permission -- burning a turn to say so."""
        registry = InMemoryToolRegistry([EchoTool()])
        llm = ScriptedLLM([], "no tools offered")
        runner = NativeAgentRunner(
            llm=llm, registry=registry, executor=PolicyToolExecutor(registry)
        )
        # Policy allows it; the principal lacks Scope.TOOLS_EXECUTE.
        self.context = make_context(scopes=frozenset({Scope.AGENT_INVOKE}))
        run(
            self.drain(
                runner,
                self.request(
                    policy=ToolPolicy(
                        allowed=frozenset({"echo"}), max_authority=Authority.PURE
                    )
                ),
            )
        )
        self.assertEqual(llm.calls[0]["tools"], [])

    def test_tools_are_advertised_when_the_principal_has_the_scope(self) -> None:
        from bat.domain.conversation import ToolCall

        registry = InMemoryToolRegistry([EchoTool()])
        llm = ScriptedLLM(
            [
                Completion(
                    content="",
                    tool_calls=(ToolCall(id="c", name="echo", arguments={"text": "hi"}),),
                )
            ],
            stream_text="done",
        )
        runner = NativeAgentRunner(
            llm=llm, registry=registry, executor=PolicyToolExecutor(registry)
        )
        run(
            self.drain(
                runner,
                self.request(
                    policy=ToolPolicy(
                        allowed=frozenset({"echo"}), max_authority=Authority.PURE
                    )
                ),
            )
        )
        self.assertEqual([s.name for s in llm.calls[0]["tools"]], ["echo"])

    def test_llm_failure_produces_a_single_error_event(self) -> None:
        class BrokenLLM:
            async def complete(self, **kwargs):
                raise RuntimeError("boom")

            async def stream(self, **kwargs):
                raise RuntimeError("boom")
                yield ""  # pragma: no cover

        events = run(self.drain(NativeAgentRunner(llm=BrokenLLM()), self.request()))
        terminals = [e for e in events if isinstance(e, (FinalEvent, ErrorEvent))]
        self.assertEqual(len(terminals), 1)
        self.assertIsInstance(terminals[0], ErrorEvent)

    def test_expired_deadline_terminates_cleanly(self) -> None:
        runner = NativeAgentRunner(llm=ScriptedLLM([], "never reached"))
        events = run(self.drain(runner, self.request(deadline_s=0.0)))
        finals = [e for e in events if isinstance(e, FinalEvent)]
        self.assertEqual(len(finals), 1)
        self.assertEqual(finals[0].stop_reason, StopReason.MAX_STEPS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
