"""Desktop brain: single-user CLI over the native llama.cpp engine.

This is the local, single-tenant face of BAT. It loads the ``.gguf`` weights
directly through :class:`~bat.adapters.llama_cpp_client.LlamaCppClient` -- there
is no Ollama, and no model server of any kind.

It is a *thin* wrapper: prompt assembly, retrieval and the tool loop all live in
``bat/`` and are shared with the API. The desktop build differs from the hosted
one in exactly two ways, both explicit here:

* it runs as one fixed local tenant, so there is no authentication; and
* it may use :data:`DESKTOP_POLICY`, which permits host-touching tools, because
  on your own machine ambient authority is the point. That policy must never be
  selected from a request-scoped value on the server -- see ``bat/ports/tools.py``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from bat.adapters.llama_cpp_client import LlamaCppClient
from bat.adapters.llama_cpp_embedder import HashingEmbedder, LlamaCppEmbedder
from bat.adapters.vector_chroma import ChromaVectorStore
from bat.adapters.vector_memory import InMemoryVectorStore
from bat.domain.conversation import Message, Role, Session, utcnow
from bat.domain.tenancy import Principal, Scope, TenantContext
from bat.ports.agent import ErrorEvent, FinalEvent, RunRequest, TokenEvent
from bat.ports.retrieval import MemoryKind
from bat.ports.tools import DESKTOP_POLICY, ToolPolicy
from bat.services.agent.loop import NativeAgentRunner
from bat.services.agent.tools import InMemoryToolRegistry, PolicyToolExecutor
from bat.services.rag.pipeline import LocalMemoryPipeline
from bat.tools.builtin import build_default_tools
from bat.settings import Settings

#: The desktop build is one tenant with one user.
LOCAL_TENANT = "local"
LOCAL_PRINCIPAL = "owner"


def _local_context() -> TenantContext:
    """Full scopes: on a personal machine the owner is the administrator."""
    return TenantContext(
        tenant_id=LOCAL_TENANT,
        principal=Principal(
            id=LOCAL_PRINCIPAL,
            tenant_id=LOCAL_TENANT,
            scopes=frozenset(Scope),
            display_name="Local user",
        ),
        request_id="cli",
        started_at=utcnow(),
    )


class CognitiveBrain:
    """Loads local weights and answers turns, with memory across the session."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        tool_policy: ToolPolicy | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.context = _local_context()
        self.session = Session.create(
            tenant_id=LOCAL_TENANT, principal_id=LOCAL_PRINCIPAL, title="cli"
        )
        self.history: list[Message] = []
        # Default-deny even here: nothing is enabled until a tool is registered
        # *and* named in the policy allowlist.
        self.policy = tool_policy or ToolPolicy(
            allowed=frozenset(self.settings.agent.enabled_tools),
            max_authority=DESKTOP_POLICY.max_authority,
            min_code_isolation=DESKTOP_POLICY.min_code_isolation,
            allowed_side_effects=DESKTOP_POLICY.allowed_side_effects,
            max_calls_per_run=self.settings.agent.max_tool_calls_per_run,
        )

        self.llm = LlamaCppClient(self.settings.model)
        self.embedder = (
            LlamaCppEmbedder(self.settings.embedding)
            if self.settings.embedding.is_configured
            else HashingEmbedder()
        )
        self.vector_store: Any = (
            InMemoryVectorStore(self.embedder)
            if self.settings.vector.mode == "memory"
            else ChromaVectorStore(self.settings.vector, self.embedder)
        )
        self.memory = LocalMemoryPipeline(
            store=self.vector_store, settings=self.settings.vector
        )
        registry = InMemoryToolRegistry(
            build_default_tools(
                memory=self.memory, vector_settings=self.settings.vector
            )
        )
        self.registry = registry
        self.runner = NativeAgentRunner(
            llm=self.llm,
            memory=self.memory,
            registry=registry,
            executor=PolicyToolExecutor(registry),
            vector_settings=self.settings.vector,
            model_name=self.settings.model.name,
        )

    # -- lifecycle ---------------------------------------------------------

    async def load(self) -> None:
        """Load the weights up front so the first question is not slow."""
        await self.llm.load()

    async def aclose(self) -> None:
        for resource in (self.llm, self.embedder):
            closer = getattr(resource, "close", None)
            if closer is not None:
                await closer()

    # -- memory ------------------------------------------------------------

    async def remember(self, fact: str) -> int:
        return await self.memory.remember_fact(tenant_id=LOCAL_TENANT, fact=fact)

    async def read_document(self, path: str | Path) -> str:
        """Index a local file so later questions can draw on it."""
        target = Path(str(path).strip().strip('"').strip("'"))
        if not target.is_file():
            return f"Cannot find a file at {target}."

        try:
            text = _extract_text(target)
        except Exception as exc:
            return f"Failed to read {target.name}: {exc}"
        if not text.strip():
            return f"{target.name} is empty or has no extractable text."

        chunks = await self.memory.ingest(
            tenant_id=LOCAL_TENANT,
            text=text,
            kind=MemoryKind.DOCUMENT,
            source=target.name,
        )
        return f"Indexed {chunks} chunk(s) from {target.name}."

    # -- turns -------------------------------------------------------------

    async def ask(self, user_input: str) -> str:
        """Run one turn and return the answer, recording it in history."""
        request = RunRequest(
            context=self.context,
            session=self.session,
            user_input=user_input,
            history=tuple(self.history[-self.settings.session.history_window :]),
            policy=self.policy,
            max_steps=self.settings.agent.max_steps,
            deadline_s=self.settings.agent.deadline_s,
            tool_rounds_per_turn=self.settings.agent.tool_rounds_per_turn,
            stream_tokens=False,
        )

        answer = ""
        async for event in self.runner.run(request):
            if isinstance(event, FinalEvent):
                answer = event.content
            elif isinstance(event, ErrorEvent):
                return f"[{event.code}] {event.message}"

        self._record(Role.USER, user_input)
        self._record(Role.ASSISTANT, answer)
        return answer

    async def stream(self, user_input: str):
        """Yield answer tokens as they are generated."""
        request = RunRequest(
            context=self.context,
            session=self.session,
            user_input=user_input,
            history=tuple(self.history[-self.settings.session.history_window :]),
            policy=self.policy,
            max_steps=self.settings.agent.max_steps,
            deadline_s=self.settings.agent.deadline_s,
            tool_rounds_per_turn=self.settings.agent.tool_rounds_per_turn,
            stream_tokens=True,
        )
        collected: list[str] = []
        async for event in self.runner.run(request):
            if isinstance(event, TokenEvent):
                collected.append(event.text)
                yield event.text
            elif isinstance(event, ErrorEvent):
                yield f"\n[{event.code}] {event.message}"

        self._record(Role.USER, user_input)
        self._record(Role.ASSISTANT, "".join(collected))

    def think_and_act(self, user_input: str) -> str:
        """Blocking entry point, kept for the existing CLI."""
        return asyncio.run(self.ask(user_input))

    def _record(self, role: Role, content: str) -> None:
        self.history.append(
            Message.create(
                session_id=self.session.id,
                tenant_id=LOCAL_TENANT,
                role=role,
                content=content,
            )
        )


def _extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n".join((page.extract_text() or "") for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8", errors="replace")
