"""The built-in tool set.

Each tool declares what it can reach (:class:`Authority`) and where it runs
(:class:`Isolation`), and the executor enforces both. The declarations here are
deliberately pessimistic: a tool claims the authority it *actually* has, not the
authority we wish it had.

The clearest case is :class:`PythonExecTool`. It runs in a scrubbed child
process, which is a real improvement on ``exec()`` in the API process -- but on
this platform that child still runs as the same OS user, with the filesystem and
network reachable. So it declares ``Authority.HOST`` and is consequently refused
by any server-side policy. It becomes shippable when it runs in a container with
no network and a read-only rootfs, at which point its authority genuinely drops
to ``PURE`` and the declaration changes with the implementation.
"""

from __future__ import annotations

import ast
import logging
import math
import operator
from typing import Any, Final

from bat.domain.errors import ToolError
from bat.ports.retrieval import RetrievalQuery
from bat.ports.tools import (
    Authority,
    Isolation,
    SideEffect,
    ToolDefinition,
    ToolInvocation,
)
from bat.domain.tenancy import Scope
from bat.services.agent.sandbox import SandboxLimits, SubprocessSandbox
from bat.settings import VectorSettings

logger = logging.getLogger("bat.tools")


# ---------------------------------------------------------------------------
# calculator -- PURE authority, safe to enable anywhere
# ---------------------------------------------------------------------------

_BIN_OPS: Final[dict[type[ast.operator], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: Final[dict[type[ast.unaryop], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS: Final[dict[str, Any]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "factorial": math.factorial,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
    "gcd": math.gcd,
}
_CONSTANTS: Final[dict[str, float]] = {"pi": math.pi, "e": math.e, "tau": math.tau}

#: Guards against `2**10**10`, which is a memory bomb, not a calculation.
_MAX_EXPONENT = 10_000
_MAX_FACTORIAL = 2_000


def evaluate_expression(expression: str) -> float | int:
    """Evaluate arithmetic by walking a parsed AST.

    An allowlist of *node types*, not a filtered ``eval``. Anything not
    explicitly permitted -- attribute access, subscripts, comprehensions, names
    outside the constant table -- has no branch here and is rejected, so there
    is no reachable path to ``__class__``, imports or arbitrary calls.
    """
    if len(expression) > 500:
        raise ToolError("expression is too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"could not parse expression: {exc.msg}") from exc
    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> Any:
    match node:
        case ast.Constant(value=bool()):
            raise ToolError("booleans are not valid in an arithmetic expression")
        case ast.Constant(value=int() | float() as value):
            return value
        case ast.Name(id=name) if name in _CONSTANTS:
            return _CONSTANTS[name]
        case ast.BinOp(left=left, op=op, right=right) if type(op) in _BIN_OPS:
            lhs, rhs = _eval_node(left), _eval_node(right)
            if isinstance(op, ast.Pow) and _too_big(rhs):
                raise ToolError(f"exponent above {_MAX_EXPONENT} is not permitted")
            try:
                return _BIN_OPS[type(op)](lhs, rhs)
            except ZeroDivisionError as exc:
                raise ToolError("division by zero") from exc
        case ast.UnaryOp(op=op, operand=operand) if type(op) in _UNARY_OPS:
            return _UNARY_OPS[type(op)](_eval_node(operand))
        case ast.Call(func=ast.Name(id=name), args=args, keywords=[]) if (
            name in _FUNCTIONS
        ):
            values = [_eval_node(a) for a in args]
            if name == "factorial" and values and _too_big(values[0], _MAX_FACTORIAL):
                raise ToolError(f"factorial above {_MAX_FACTORIAL} is not permitted")
            try:
                return _FUNCTIONS[name](*values)
            except (ValueError, OverflowError, TypeError) as exc:
                raise ToolError(f"{name}: {exc}") from exc
        case ast.Tuple(elts=elts) | ast.List(elts=elts):
            return [_eval_node(e) for e in elts]
    raise ToolError(
        f"{type(node).__name__} is not permitted in an arithmetic expression"
    )


def _too_big(value: Any, limit: int = _MAX_EXPONENT) -> bool:
    return isinstance(value, (int, float)) and abs(value) > limit


class CalculatorTool:
    """Exact arithmetic without handing the model an interpreter.

    The legacy prompt told the model to call ``execute_python_code`` for every
    calculation, which meant arbitrary code execution for the sake of long
    multiplication. This covers that need at ``Authority.PURE``.
    """

    definition = ToolDefinition(
        name="calculator",
        description=(
            "Evaluate an arithmetic expression exactly. Use this for any "
            "calculation instead of computing mentally. Supports + - * / // % "
            "**, and abs, round, min, max, sum, sqrt, factorial, log, log2, "
            "log10, exp, sin, cos, tan, floor, ceil, gcd, plus the constants "
            "pi, e and tau."
        ),
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "e.g. '(2 + 3) * 7' or 'factorial(20)'",
                }
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
        authority=Authority.PURE,
        isolation=Isolation.IN_PROCESS,
        side_effect=SideEffect.READ_ONLY,
        required_scopes=frozenset({Scope.TOOLS_EXECUTE}),
        timeout_s=5.0,
    )

    async def run(self, invocation: ToolInvocation) -> str:
        expression = str(invocation.arguments["expression"])
        return f"{expression} = {evaluate_expression(expression)}"


# ---------------------------------------------------------------------------
# memory_search -- TENANT authority, the tool that makes RAG agentic
# ---------------------------------------------------------------------------


class MemorySearchTool:
    """Lets the agent query its own tenant's memory mid-turn.

    Retrieval already happens once per turn against the user's opening message.
    This is for when the model realises it needs something else -- a follow-up
    the first query would not have matched.

    The tenant comes from ``invocation.context``, never from an argument, so
    there is no parameter through which a model could reach another tenant's
    memory even if an injected instruction told it to.
    """

    __slots__ = ("_min_score", "_pipeline", "_top_k")

    def __init__(
        self, pipeline: Any, top_k: int = 5, min_score: float = 0.25
    ) -> None:
        self._pipeline = pipeline
        self._top_k = top_k
        # Must be passed through explicitly. RetrievalQuery defaults to 0.0, and
        # a vector search always returns its nearest rows, so omitting this
        # hands the model the closest chunk however irrelevant it is -- the
        # exact failure the pipeline's own threshold exists to prevent.
        self._min_score = min_score

    definition = ToolDefinition(
        name="memory_search",
        description=(
            "Search the user's stored documents and remembered facts. Use when "
            "the answer may be in material the user saved earlier."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look for, in natural language.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        authority=Authority.TENANT,
        isolation=Isolation.IN_PROCESS,
        side_effect=SideEffect.READ_ONLY,
        required_scopes=frozenset({Scope.TOOLS_EXECUTE, Scope.MEMORY_READ}),
        timeout_s=20.0,
    )

    async def run(self, invocation: ToolInvocation) -> str:
        query = str(invocation.arguments["query"])
        results = await self._pipeline.retrieve(
            tenant_id=invocation.context.tenant_id,
            query=RetrievalQuery(
                text=query, top_k=self._top_k, min_score=self._min_score
            ),
        )
        if not results:
            return f"Nothing in memory matches {query!r}."

        lines = [f"{len(results)} result(s) for {query!r}:"]
        for hit in results:
            origin = f" ({hit.chunk.source})" if hit.chunk.source else ""
            lines.append(f"- [{hit.score:.2f}{origin}] {hit.chunk.text}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# python_exec -- HOST authority. Desktop only until it runs in a container.
# ---------------------------------------------------------------------------


class PythonExecTool:
    """Runs caller-supplied Python in a scrubbed child process.

    Declares ``Authority.HOST`` because that is the truth on this platform: the
    child runs as the same OS user, so it can reach the filesystem and open
    sockets. Every server-side policy therefore refuses it, and it is not in any
    default allowlist.

    To make it shippable, run it in a container with no network and a read-only
    rootfs and change the declaration to ``authority=PURE,
    isolation=CONTAINER``. The declaration is what the policy trusts, so it must
    track the implementation -- never widen the policy to fit the tool.
    """

    __slots__ = ("_sandbox",)

    def __init__(self, sandbox: SubprocessSandbox | None = None) -> None:
        self._sandbox = sandbox or SubprocessSandbox(SandboxLimits())

    definition = ToolDefinition(
        name="python_exec",
        description=(
            "Run a short Python program and return its stdout. The program runs "
            "with no network credentials, an empty working directory and a "
            "short time limit. Print the result you want to see."
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source. Use print() to emit results.",
                }
            },
            "required": ["code"],
            "additionalProperties": False,
        },
        authority=Authority.HOST,
        isolation=Isolation.SUBPROCESS,
        executes_arbitrary_code=True,
        side_effect=SideEffect.READ_ONLY,
        required_scopes=frozenset({Scope.TOOLS_EXECUTE}),
        timeout_s=30.0,
        requires_confirmation=True,
    )

    async def run(self, invocation: ToolInvocation) -> str:
        code = str(invocation.arguments["code"])
        if not code.strip():
            raise ToolError("code must not be empty")
        result = await self._sandbox.run_python(code)
        logger.info(
            "sandboxed execution",
            extra={
                "tenant_id": invocation.context.tenant_id,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "duration_ms": result.duration_ms,
            },
        )
        return result.render()


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


def build_default_tools(
    *, memory: Any | None = None, vector_settings: Any | None = None
) -> list[Any]:
    """Every built-in tool. Registration is not authorization.

    A tool appearing here is still unusable until its name is in the tenant's
    allowlist *and* its declared authority fits the deployment's ceiling.
    """
    tools: list[Any] = [CalculatorTool(), PythonExecTool()]
    if memory is not None:
        settings = vector_settings or VectorSettings()
        tools.append(
            MemorySearchTool(
                memory,
                top_k=settings.default_top_k,
                min_score=settings.min_score,
            )
        )
    return tools


#: Safe to enable on a shared, multi-tenant deployment: no host reach, no
#: caller-supplied code.
SAAS_SAFE_TOOLS: Final[frozenset[str]] = frozenset({"calculator", "memory_search"})
