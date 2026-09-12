"""Tests for the sandbox and the built-in tool set.

The security assertions here are the point of the module: a change that makes
`python_exec` reachable from a server policy, or lets `calculator` escape its
AST allowlist, should fail loudly.
"""

from __future__ import annotations

import asyncio
import os
import unittest

from bat.adapters.llama_cpp_embedder import HashingEmbedder
from bat.adapters.vector_memory import InMemoryVectorStore
from bat.domain.errors import ToolError
from bat.domain.tenancy import Principal, Scope, TenantContext
from bat.domain.conversation import utcnow
from bat.ports.retrieval import MemoryKind
from bat.ports.tools import (
    SAAS_BASELINE_POLICY,
    Authority,
    DESKTOP_POLICY,
    Isolation,
    ToolInvocation,
    ToolPolicy,
)
from bat.services.agent.sandbox import SandboxLimits, SubprocessSandbox
from bat.services.agent.tools import InMemoryToolRegistry, PolicyToolExecutor
from bat.services.rag.pipeline import LocalMemoryPipeline
from bat.settings import VectorSettings
from bat.tools.builtin import (
    SAAS_SAFE_TOOLS,
    CalculatorTool,
    MemorySearchTool,
    PythonExecTool,
    build_default_tools,
    evaluate_expression,
)


def run(coro):
    return asyncio.run(coro)


def context(tenant: str = "acme") -> TenantContext:
    return TenantContext(
        tenant_id=tenant,
        principal=Principal(id="u1", tenant_id=tenant, scopes=frozenset(Scope)),
        request_id="req",
        started_at=utcnow(),
    )


def invocation(name: str, arguments: dict, tenant: str = "acme") -> ToolInvocation:
    return ToolInvocation(
        call_id="c1",
        name=name,
        arguments=arguments,
        context=context(tenant),
        session_id="sess",
    )


class TestCalculator(unittest.TestCase):
    def test_arithmetic(self) -> None:
        self.assertEqual(evaluate_expression("(2 + 3) * 7"), 35)
        self.assertEqual(evaluate_expression("factorial(20)"), 2432902008176640000)
        self.assertEqual(evaluate_expression("2 ** 10"), 1024)
        self.assertAlmostEqual(evaluate_expression("sqrt(2)"), 1.4142135623730951)

    def test_rejects_anything_outside_the_allowlist(self) -> None:
        """No reachable path to attributes, imports or arbitrary calls."""
        for hostile in (
            "__import__('os').system('echo pwned')",
            "().__class__.__bases__[0].__subclasses__()",
            "open('/etc/passwd').read()",
            "[x for x in range(10)]",
            "lambda: 1",
            "print('hi')",
            "eval('1+1')",
            "os.getcwd()",
            "globals()",
        ):
            with self.subTest(expression=hostile), self.assertRaises(ToolError):
                evaluate_expression(hostile)

    def test_guards_against_resource_bombs(self) -> None:
        with self.assertRaises(ToolError):
            evaluate_expression("2 ** 10 ** 10")
        with self.assertRaises(ToolError):
            evaluate_expression("factorial(999999)")

    def test_division_by_zero_is_a_tool_error(self) -> None:
        with self.assertRaises(ToolError):
            evaluate_expression("1 / 0")

    def test_tool_runs(self) -> None:
        result = run(CalculatorTool().run(invocation("calculator", {"expression": "6*7"})))
        self.assertIn("42", result)


class TestSandbox(unittest.TestCase):
    def setUp(self) -> None:
        self.sandbox = SubprocessSandbox(SandboxLimits(timeout_s=8.0))

    def test_runs_code_and_captures_stdout(self) -> None:
        result = run(self.sandbox.run_python("print(6 * 7)"))
        self.assertTrue(result.ok)
        self.assertIn("42", result.stdout)

    def test_parent_environment_is_not_inherited(self) -> None:
        """The parent holds API keys and DSNs; the child must not see them."""
        os.environ["BAT_SANDBOX_CANARY"] = "leaked-secret"
        self.addCleanup(os.environ.pop, "BAT_SANDBOX_CANARY", None)
        result = run(
            self.sandbox.run_python(
                "import os; print(os.environ.get('BAT_SANDBOX_CANARY', 'ABSENT'))"
            )
        )
        self.assertIn("ABSENT", result.stdout)
        self.assertNotIn("leaked-secret", result.stdout)

    def test_platform_packages_are_not_importable(self) -> None:
        result = run(
            self.sandbox.run_python(
                "import sys; print(sum('site-packages' in p for p in sys.path))"
            )
        )
        self.assertIn("0", result.stdout.strip())

    def test_working_directory_is_empty_and_temporary(self) -> None:
        result = run(self.sandbox.run_python("import os; print(os.listdir('.'))"))
        self.assertIn("[]", result.stdout)

    def test_infinite_loop_is_killed(self) -> None:
        sandbox = SubprocessSandbox(SandboxLimits(timeout_s=1.0))
        result = run(sandbox.run_python("while True: pass"))
        self.assertTrue(result.timed_out)
        self.assertFalse(result.ok)
        self.assertIn("timed out", result.render())

    def test_output_is_capped(self) -> None:
        sandbox = SubprocessSandbox(SandboxLimits(timeout_s=8.0, max_output_chars=500))
        result = run(sandbox.run_python("print('x' * 100000)"))
        self.assertLess(len(result.stdout), 700)
        self.assertIn("truncated", result.stdout)

    def test_failure_is_reported_not_raised(self) -> None:
        result = run(self.sandbox.run_python("raise ValueError('boom')"))
        self.assertFalse(result.ok)
        self.assertIn("boom", result.render())

    def test_resource_limit_support_is_reported_honestly(self) -> None:
        """Windows has no rlimits; the sandbox must not claim otherwise."""
        self.assertEqual(self.sandbox.enforces_resource_limits, os.name == "posix")


class TestToolPolicyBoundaries(unittest.TestCase):
    """The declarations that decide what can ship."""

    def test_python_exec_is_refused_by_the_saas_baseline(self) -> None:
        policy = ToolPolicy(
            allowed=frozenset({"python_exec"}),
            max_authority=SAAS_BASELINE_POLICY.max_authority,
            min_code_isolation=SAAS_BASELINE_POLICY.min_code_isolation,
        )
        allowed, reason = policy.permits(PythonExecTool.definition)
        self.assertFalse(allowed, "python_exec must never be reachable on a server")
        self.assertIn("HOST authority", reason)

    def test_python_exec_is_available_to_the_desktop_build(self) -> None:
        policy = ToolPolicy(
            allowed=frozenset({"python_exec"}),
            max_authority=DESKTOP_POLICY.max_authority,
            min_code_isolation=DESKTOP_POLICY.min_code_isolation,
        )
        allowed, reason = policy.permits(PythonExecTool.definition)
        self.assertTrue(allowed, reason)

    def test_saas_safe_tools_pass_the_baseline(self) -> None:
        policy = ToolPolicy(
            allowed=SAAS_SAFE_TOOLS,
            max_authority=SAAS_BASELINE_POLICY.max_authority,
            min_code_isolation=SAAS_BASELINE_POLICY.min_code_isolation,
        )
        for definition in (CalculatorTool.definition, MemorySearchTool.definition):
            allowed, reason = policy.permits(definition)
            self.assertTrue(allowed, f"{definition.name}: {reason}")

    def test_python_exec_is_not_in_the_saas_safe_set(self) -> None:
        self.assertNotIn("python_exec", SAAS_SAFE_TOOLS)

    def test_code_executing_tool_needs_containment(self) -> None:
        """Even with HOST authority allowed, in-process code execution is out."""
        policy = ToolPolicy(
            allowed=frozenset({"python_exec"}),
            max_authority=Authority.HOST,
            min_code_isolation=Isolation.CONTAINER,
        )
        allowed, reason = policy.permits(PythonExecTool.definition)
        self.assertFalse(allowed)
        self.assertIn("containment", reason)

    def test_empty_allowlist_denies_everything(self) -> None:
        policy = ToolPolicy(allowed=frozenset(), max_authority=Authority.HOST)
        for definition in (
            CalculatorTool.definition,
            MemorySearchTool.definition,
            PythonExecTool.definition,
        ):
            allowed, _ = policy.permits(definition)
            self.assertFalse(allowed, f"{definition.name} escaped default-deny")


class TestMemorySearchTool(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InMemoryVectorStore(HashingEmbedder(dimensions=512))
        self.rag = LocalMemoryPipeline(
            store=self.store,
            settings=VectorSettings(chunk_size=200, chunk_overlap=40, min_score=0.15),
        )
        run(
            self.rag.ingest(
                tenant_id="acme",
                text="Alice owns the billing service and is on call Thursdays.",
                kind=MemoryKind.DOCUMENT,
            )
        )
        run(
            self.rag.ingest(
                tenant_id="globex",
                text="Globex confidential: the acquisition target is Initech.",
                kind=MemoryKind.DOCUMENT,
            )
        )
        self.tool = MemorySearchTool(self.rag)

    def test_finds_the_tenants_own_material(self) -> None:
        out = run(self.tool.run(invocation("memory_search", {"query": "who owns billing"})))
        self.assertIn("Alice", out)

    def test_cannot_reach_another_tenant(self) -> None:
        """Tenant comes from the context, so there is no argument to abuse."""
        out = run(
            self.tool.run(
                invocation("memory_search", {"query": "acquisition target"}, tenant="acme")
            )
        )
        self.assertNotIn("Initech", out)

    def test_empty_result_is_stated_plainly(self) -> None:
        out = run(self.tool.run(invocation("memory_search", {"query": "zzz qqq vvv"})))
        self.assertIn("Nothing in memory", out)


class TestExecutorWithBuiltins(unittest.TestCase):
    def setUp(self) -> None:
        store = InMemoryVectorStore(HashingEmbedder(dimensions=256))
        rag = LocalMemoryPipeline(store=store, settings=VectorSettings())
        self.registry = InMemoryToolRegistry(build_default_tools(memory=rag))
        self.executor = PolicyToolExecutor(self.registry)

    def test_saas_policy_advertises_only_safe_tools(self) -> None:
        policy = ToolPolicy(
            allowed=SAAS_SAFE_TOOLS,
            max_authority=SAAS_BASELINE_POLICY.max_authority,
            min_code_isolation=SAAS_BASELINE_POLICY.min_code_isolation,
        )
        names = sorted(s.name for s in self.registry.specs_for(policy))
        self.assertEqual(names, ["calculator", "memory_search"])

    def test_denied_tool_yields_an_observation_not_an_exception(self) -> None:
        policy = ToolPolicy(
            allowed=frozenset({"python_exec"}),
            max_authority=Authority.NETWORK,
            min_code_isolation=Isolation.SUBPROCESS,
        )
        result = run(
            self.executor.execute(
                invocation=invocation("python_exec", {"code": "print(1)"}), policy=policy
            )
        )
        self.assertTrue(result.is_error)
        self.assertIn("denied", result.content)

    def test_calculator_runs_through_the_executor(self) -> None:
        policy = ToolPolicy(
            allowed=frozenset({"calculator"}), max_authority=Authority.PURE
        )
        result = run(
            self.executor.execute(
                invocation=invocation("calculator", {"expression": "12 * 12"}),
                policy=policy,
            )
        )
        self.assertFalse(result.is_error)
        self.assertIn("144", result.content)

    def test_hostile_expression_is_a_clean_tool_error(self) -> None:
        policy = ToolPolicy(
            allowed=frozenset({"calculator"}), max_authority=Authority.PURE
        )
        result = run(
            self.executor.execute(
                invocation=invocation(
                    "calculator", {"expression": "__import__('os').system('x')"}
                ),
                policy=policy,
            )
        )
        self.assertTrue(result.is_error)
        self.assertIn("not permitted", result.content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
