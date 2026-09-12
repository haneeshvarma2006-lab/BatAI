"""Tool registry and policy-enforcing executor.

The executor is the choke point every tool call passes through. It re-checks the
policy even though the registry already filtered what the model was told about,
because the model can invent a name it was never offered -- and with a
quantised local model that is common, not exotic.

Order of checks, deliberately cheapest-and-most-decisive first:

1. does the tool exist?
2. does the tenant's policy permit it (allowlist, isolation floor, side effect)?
3. does the principal hold the tool's required scopes?
4. do the arguments satisfy the declared schema?
5. run it, under a timeout, with the output truncated.

Every outcome -- allowed or denied -- produces an audit record. Failures come
back as :class:`ToolResult` observations rather than exceptions, because a
failed tool call is information the loop should reason about, not a reason to
abandon the turn.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from typing import Any

from bat.domain.conversation import ToolResult, utcnow
from bat.domain.errors import NotFoundError, ToolError
from bat.ports.llm import ToolSpec
from bat.ports.tools import (
    Tool,
    ToolAuditRecord,
    ToolDefinition,
    ToolInvocation,
    ToolPolicy,
)

logger = logging.getLogger("bat.agent.tools")


class InMemoryToolRegistry:
    """Holds tool implementations and advertises only what a policy permits."""

    __slots__ = ("_tools",)

    def __init__(self, tools: Sequence[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        name = tool.definition.name
        if name in self._tools:
            raise ValueError(f"tool {name!r} is already registered")
        self._tools[name] = tool

    def get(self, name: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None:
            raise NotFoundError(f"tool {name!r} is not registered")
        return tool

    def specs_for(self, policy: ToolPolicy) -> Sequence[ToolSpec]:
        """Advertise only permitted tools.

        Not a security control on its own -- the executor re-checks -- but it
        stops the model burning steps on calls that would be denied.
        """
        return [
            tool.definition.to_spec()
            for tool in self._tools.values()
            if policy.permits(tool.definition)[0]
        ]

    def __len__(self) -> int:
        return len(self._tools)


class NullAuditSink:
    """Logs audit records. Production swaps in a durable sink."""

    async def record(self, entry: ToolAuditRecord) -> None:
        logger.info(
            "tool invocation",
            extra={
                "tenant_id": entry.tenant_id,
                "principal_id": entry.principal_id,
                "session_id": entry.session_id,
                "tool": entry.tool_name,
                "allowed": entry.allowed,
                "denial_reason": entry.denial_reason,
                "duration_ms": entry.duration_ms,
                "is_error": entry.is_error,
            },
        )


class PolicyToolExecutor:
    """Authorises, validates, runs and audits one tool call."""

    __slots__ = ("_audit", "_registry")

    def __init__(self, registry: InMemoryToolRegistry, audit: Any | None = None) -> None:
        self._registry = registry
        self._audit = audit or NullAuditSink()

    async def execute(
        self, *, invocation: ToolInvocation, policy: ToolPolicy
    ) -> ToolResult:
        started = time.perf_counter()
        denial = ""
        definition: ToolDefinition | None = None

        try:
            tool = self._registry.get(invocation.name)
            definition = tool.definition

            permitted, reason = policy.permits(definition)
            if not permitted:
                denial = reason
                raise ToolError(f"tool {invocation.name!r} denied: {reason}")

            missing = [
                s for s in definition.required_scopes
                if not invocation.context.principal.has(s)
            ]
            if missing:
                denial = "missing scope"
                raise ToolError(
                    f"tool {invocation.name!r} requires scope(s): "
                    + ", ".join(sorted(str(s) for s in missing))
                )

            problems = validate_arguments(definition.parameters, invocation.arguments)
            if problems:
                denial = "invalid arguments"
                raise ToolError(
                    f"invalid arguments for {invocation.name!r}: " + "; ".join(problems)
                )

            output = await asyncio.wait_for(
                tool.run(invocation), timeout=definition.timeout_s
            )
            content = _truncate(str(output), definition.max_output_chars)
            result = ToolResult(
                call_id=invocation.call_id,
                name=invocation.name,
                content=content,
                is_error=False,
                duration_ms=_ms(started),
            )
        except TimeoutError:
            limit = definition.timeout_s if definition else 0.0
            result = ToolResult(
                call_id=invocation.call_id,
                name=invocation.name,
                content=f"Tool timed out after {limit}s.",
                is_error=True,
                duration_ms=_ms(started),
            )
        except (ToolError, NotFoundError) as exc:
            result = ToolResult(
                call_id=invocation.call_id,
                name=invocation.name,
                content=exc.message,
                is_error=True,
                duration_ms=_ms(started),
            )
        except Exception as exc:
            # An unexpected tool bug must not kill the turn, but its detail must
            # not reach the model either -- it can carry internal state.
            logger.exception("tool raised", extra={"tool": invocation.name})
            result = ToolResult(
                call_id=invocation.call_id,
                name=invocation.name,
                content=f"Tool {invocation.name!r} failed unexpectedly.",
                is_error=True,
                duration_ms=_ms(started),
            )
            del exc

        await self._audit.record(
            ToolAuditRecord(
                call_id=invocation.call_id,
                tenant_id=invocation.context.tenant_id,
                principal_id=invocation.context.principal_id,
                session_id=invocation.session_id,
                tool_name=invocation.name,
                arguments=dict(invocation.arguments),
                allowed=not result.is_error or not denial,
                denial_reason=denial,
                duration_ms=result.duration_ms,
                is_error=result.is_error,
                occurred_at=utcnow(),
            )
        )
        return result


def validate_arguments(schema: dict[str, Any], arguments: Any) -> list[str]:
    """Check arguments against a JSON-Schema subset.

    Covers the shapes tools actually declare -- object with typed properties,
    required keys, no extras. Not a full validator; if tools ever need `anyOf`
    or nested schemas, swap in `jsonschema` behind this function.
    """
    problems: list[str] = []
    if not isinstance(arguments, dict):
        return ["arguments must be a JSON object"]

    properties: dict[str, Any] = schema.get("properties") or {}
    required: Sequence[str] = schema.get("required") or ()

    for key in required:
        if key not in arguments:
            problems.append(f"missing required argument {key!r}")

    for key, value in arguments.items():
        spec = properties.get(key)
        if spec is None:
            if schema.get("additionalProperties") is False or properties:
                problems.append(f"unexpected argument {key!r}")
            continue
        expected = spec.get("type")
        if expected and not _type_ok(expected, value):
            problems.append(
                f"argument {key!r} must be of type {expected}, got "
                f"{type(value).__name__}"
            )
        choices = spec.get("enum")
        if choices and value not in choices:
            problems.append(f"argument {key!r} must be one of {list(choices)}")
    return problems


_JSON_TYPES: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": (list, tuple),
    "object": dict,
}


def _type_ok(expected: str, value: Any) -> bool:
    python_type = _JSON_TYPES.get(expected)
    if python_type is None:
        return True
    # bool is a subclass of int in Python; JSON Schema treats them as distinct.
    if expected in {"integer", "number"} and isinstance(value, bool):
        return False
    return isinstance(value, python_type)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} chars omitted]"


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
