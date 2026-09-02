"""Tool port and capability policy.

The threat model
----------------
In a single-user CLI, a tool that runs code or opens a file acts on the machine
its owner already controls. Behind a multi-tenant API the same tool acts on
*shared infrastructure* on behalf of an untrusted caller, driven by a model that
is itself steerable by untrusted text (a retrieved document, a web page, a file
the tenant uploaded). Prompt injection therefore promotes to remote code
execution unless the tool boundary refuses by default.

The rules this module enforces:

1. **Default deny.** A tool is unavailable to a tenant unless its id appears in
   that tenant's allowlist. Registration is not authorization.
2. **Isolation is declared, not assumed.** Every tool states an
   :class:`Isolation` level. Tools that touch the host process
   (``Isolation.NONE``) are refused outright when the platform runs in server
   mode; they exist only for the single-tenant desktop build.
3. **Arguments are validated before the tool sees them.** Tools declare a JSON
   schema; the executor validates against it. A tool never parses raw model
   output.
4. **Every call is bounded and audited** — timeout, output cap, and an audit
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


class Isolation(IntEnum):
    """How strongly a tool is separated from the host process.

    Ordered: a deployment sets a minimum, and anything below it is refused.
    """

    #: Runs in-process with ambient authority. Desktop build only.
    NONE = 0
    #: Pure computation in-process: no I/O, no filesystem, no network.
    PURE = 1
    #: Network egress to a vetted allowlist of hosts, no local side effects.
    NETWORK = 2
    #: Separate OS process, dropped privileges, rlimits, scratch filesystem.
    SUBPROCESS = 3
    #: Container or microVM per invocation, no host mounts, no ambient creds.
    SANDBOX = 4


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
    isolation: Isolation
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
    #: Lowest acceptable isolation level for this deployment.
    min_isolation: Isolation = Isolation.PURE
    #: Side effects the tenant has consented to.
    allowed_side_effects: frozenset[SideEffect] = frozenset({SideEffect.READ_ONLY})
    #: Ceiling on tool calls within a single agent run.
    max_calls_per_run: int = 8

    def permits(self, definition: ToolDefinition) -> tuple[bool, str]:
        """Return ``(allowed, reason)``. Reason is empty when allowed."""
        if definition.name not in self.allowed:
            return False, "tool is not enabled for this tenant"
        if definition.isolation < self.min_isolation:
            return False, (
                f"tool isolation {definition.isolation.name} is below the required "
                f"{self.min_isolation.name} for this deployment"
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


#: Policy for the hosted, multi-tenant deployment: nothing that can touch the
#: host process is acceptable, whatever the tenant asks for.
SAAS_BASELINE_POLICY = ToolPolicy(
    allowed=frozenset(),
    min_isolation=Isolation.NETWORK,
    allowed_side_effects=frozenset({SideEffect.READ_ONLY}),
)

#: Policy for the single-tenant desktop build, where ambient authority is the
#: point. Never select this from a request-scoped value.
DESKTOP_POLICY = ToolPolicy(
    allowed=frozenset(),
    min_isolation=Isolation.NONE,
    allowed_side_effects=frozenset(SideEffect),
    max_calls_per_run=12,
)
