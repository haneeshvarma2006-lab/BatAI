"""Tool port and capability policy.

The threat model
----------------
In a single-user CLI, a tool that runs code or opens a file acts on the machine
its owner already controls. Behind a multi-tenant API the same tool acts on
*shared infrastructure* on behalf of an untrusted caller, driven by a model that
is itself steerable by untrusted text (a retrieved document, a web page, a file
the tenant uploaded). Prompt injection therefore promotes to remote code
execution unless the tool boundary refuses by default.

Two orthogonal axes
-------------------
An earlier version of this module used one ordered ``Isolation`` ladder as the
whole safety model, and it was wrong: it conflated *where a tool's code runs*
with *what that code can reach*. On a single ladder, pure computation ranked
below network access, so a deployment floor strict enough to exclude ambient
host authority also excluded a tenant-scoped memory lookup -- which is
completely safe. The floor could be set safely or usefully, never both.

They are now separate:

* :class:`Authority` -- what the tool can reach: nothing, the caller's own
  tenant data, vetted network egress, or the host itself.
* :class:`Isolation` -- where its code runs: this process, a child process, or
  a container.

That makes the rules statable without contradiction:

1. **Default deny.** A tool is unavailable unless its name is in the tenant's
   allowlist. Registration is not authorization.
2. **No host authority on a server, ever.** ``Authority.HOST`` is refused
   whenever ``max_authority`` is below it; it exists for the desktop build,
   where ambient authority is the entire point.
3. **Arbitrary code needs real containment.** A tool that runs caller-supplied
   code must declare at least ``min_code_isolation`` (SUBPROCESS in production).
   A fixed-function tool with a validated schema does not -- its behaviour is
   bounded by its own implementation, not by the caller.
4. **Arguments are validated before the tool sees them.** Tools declare a JSON
   schema; the executor validates against it. A tool never parses raw model
   output.
5. **Every call is bounded and audited** -- timeout, output cap, and an audit
   record carrying tenant, principal and arguments.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum, StrEnum
from typing import Any, Protocol, runtime_checkable

from bat.domain.conversation import ToolResult
from bat.domain.tenancy import Scope, TenantContext
from bat.ports.llm import ToolSpec


class Authority(IntEnum):
    """What a tool can reach. Ordered by increasing reach."""

    #: Pure computation. No I/O of any kind.
    PURE = 0
    #: Only the calling tenant's own platform data (its memory, its sessions).
    #: Cannot name another tenant: the scope comes from the invocation context,
    #: never from a model-supplied argument.
    TENANT = 1
    #: Outbound network to a vetted allowlist. No local side effects.
    NETWORK = 2
    #: The host itself -- filesystem, shell, desktop, arbitrary processes.
    #: Never permissible on shared infrastructure.
    HOST = 3


class Isolation(IntEnum):
    """Where a tool's code runs. Ordered by increasing containment."""

    #: In this process, sharing its memory and credentials.
    IN_PROCESS = 0
    #: A separate OS process: own memory, scratch cwd, scrubbed environment,
    #: killed on timeout. On POSIX this also carries rlimits; see the note in
    #: bat.services.agent.sandbox about what Windows cannot enforce.
    SUBPROCESS = 1
    #: Container or microVM per invocation: no host mounts, no ambient creds,
    #: enforced CPU/memory limits, no network unless granted.
    CONTAINER = 2


class SideEffect(StrEnum):
    """What a tool changes, for audit and for confirm-before-run policies."""

    READ_ONLY = "read_only"
    TENANT_WRITE = "tenant_write"
    EXTERNAL_WRITE = "external_write"


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Static description of a tool: identity, schema and safety posture."""

    name: str
    description: str
    parameters: dict[str, Any]
    #: What this tool can reach.
    authority: Authority
    #: Where its code runs.
    isolation: Isolation = Isolation.IN_PROCESS
    #: True when the tool runs caller-supplied code or commands, so its
    #: behaviour is not bounded by its own implementation. These are the only
    #: tools that need a containment floor.
    executes_arbitrary_code: bool = False
    side_effect: SideEffect = SideEffect.READ_ONLY
    required_scopes: frozenset[Scope] = frozenset({Scope.TOOLS_EXECUTE})
    timeout_s: float = 20.0
    max_output_chars: int = 8_000
    #: Requires an explicit human confirmation before the loop may run it.
    requires_confirmation: bool = False

    def to_spec(self) -> ToolSpec:
        """Render the model-facing advertisement of this tool."""
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


@dataclass(frozen=True, slots=True)
class ToolInvocation:
    """A validated, authorized request to run a tool."""

    call_id: str
    name: str
    arguments: Mapping[str, Any]
    context: TenantContext
    session_id: str


@runtime_checkable
class Tool(Protocol):
    """An executable capability."""

    @property
    def definition(self) -> ToolDefinition: ...

    async def run(self, invocation: ToolInvocation) -> str:
        """Execute and return the observation text.

        Raise :class:`~bat.domain.errors.ToolError` for expected failures; the
        runner converts those into observations instead of aborting the turn.
        """
        ...


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    """The per-tenant capability grant, evaluated on every invocation."""

    #: Tool names this tenant may call. Empty means no tools at all.
    allowed: frozenset[str] = frozenset()
    #: Ceiling on what a tool may reach. Below HOST on any shared deployment.
    max_authority: Authority = Authority.TENANT
    #: Containment required of tools that run caller-supplied code.
    min_code_isolation: Isolation = Isolation.SUBPROCESS
    #: Side effects the tenant has consented to.
    allowed_side_effects: frozenset[SideEffect] = frozenset({SideEffect.READ_ONLY})
    #: Ceiling on tool calls within a single agent run.
    max_calls_per_run: int = 8

    def permits(self, definition: ToolDefinition) -> tuple[bool, str]:
        """Return ``(allowed, reason)``. Reason is empty when allowed."""
        if definition.name not in self.allowed:
            return False, "tool is not enabled for this tenant"
        if definition.authority > self.max_authority:
            return False, (
                f"tool requires {definition.authority.name} authority but this "
                f"deployment permits at most {self.max_authority.name}"
            )
        if (
            definition.executes_arbitrary_code
            and definition.isolation < self.min_code_isolation
        ):
            return False, (
                f"tool runs caller-supplied code at {definition.isolation.name} "
                f"containment; {self.min_code_isolation.name} or stronger is "
                "required"
            )
        if definition.side_effect not in self.allowed_side_effects:
            return False, f"side effect {definition.side_effect} not permitted"
        return True, ""


@dataclass(frozen=True, slots=True)
class ToolAuditRecord:
    """One row of the tool audit trail."""

    call_id: str
    tenant_id: str
    principal_id: str
    session_id: str
    tool_name: str
    arguments: Mapping[str, Any]
    allowed: bool
    denial_reason: str
    duration_ms: float
    is_error: bool
    occurred_at: datetime


@runtime_checkable
class ToolRegistry(Protocol):
    """Holds tool implementations and resolves what a tenant may see."""

    def register(self, tool: Tool) -> None: ...

    def get(self, name: str) -> Tool:
        """Raise ``NotFoundError`` when unknown."""
        ...

    def specs_for(self, policy: ToolPolicy) -> Sequence[ToolSpec]:
        """Only tools the policy permits are advertised to the model.

        Withholding the advertisement is not a security control on its own —
        :meth:`ToolExecutor.execute` re-checks — but it stops the model wasting
        turns on calls that will be denied.
        """
        ...


@runtime_checkable
class ToolExecutor(Protocol):
    """Authorizes, validates, runs and audits a single tool call."""

    async def execute(
        self,
        *,
        invocation: ToolInvocation,
        policy: ToolPolicy,
    ) -> ToolResult:
        """Never raises for tool-level failure; failures come back as results.

        The executor re-checks ``policy`` even though the registry already
        filtered the advertisement, because the model can hallucinate a name.
        """
        ...


@runtime_checkable
class ToolAuditSink(Protocol):
    """Where audit records go. Must be durable in production."""

    async def record(self, entry: ToolAuditRecord) -> None: ...


#: Policy for the hosted, multi-tenant deployment. Host authority is refused
#: outright, and anything running caller-supplied code needs a subprocess at
#: minimum. The allowlist stays empty: a deployment opts tools in by name.
SAAS_BASELINE_POLICY = ToolPolicy(
    allowed=frozenset(),
    max_authority=Authority.NETWORK,
    min_code_isolation=Isolation.SUBPROCESS,
    allowed_side_effects=frozenset({SideEffect.READ_ONLY, SideEffect.TENANT_WRITE}),
)

#: Policy for the single-tenant desktop build, where ambient authority is the
#: point. Never select this from a request-scoped value.
DESKTOP_POLICY = ToolPolicy(
    allowed=frozenset(),
    max_authority=Authority.HOST,
    min_code_isolation=Isolation.IN_PROCESS,
    allowed_side_effects=frozenset(SideEffect),
    max_calls_per_run=12,
)
