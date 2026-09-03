"""Subprocess sandbox for running caller-supplied Python.

Replaces ``tools/code_exec.py``, which did::

    exec(code, {"math": math, "os": os, "__builtins__": __builtins__})

That is not a sandbox. ``__builtins__`` carries ``__import__``, ``open`` and
``eval``, so the first line of any escape is ``__import__('os').system(...)``;
``os`` was handed over directly anyway. Nothing running in the API process can
be made safe by filtering a globals dict -- the interpreter has too many paths
back to the host, and the process holds the platform's own credentials.

What this does instead
----------------------
Runs the code in a **fresh child process** with:

* ``-I`` (isolated mode): ignores ``PYTHONPATH``, ``PYTHONSTARTUP`` and the
  user site-packages directory, so the child cannot be steered by the parent's
  environment or import the platform's own modules;
* a scrubbed environment -- nothing inherited, so API keys, DSNs and model
  paths in the parent's env are simply not present to be read;
* an empty temporary working directory, deleted afterwards;
* a hard wall-clock timeout, after which the process tree is killed;
* capped stdout/stderr, so a print loop cannot exhaust memory in the parent;
* no stdin.

On POSIX it additionally applies ``RLIMIT_CPU``, ``RLIMIT_AS`` and
``RLIMIT_NPROC`` via ``preexec_fn``, and starts a new session so the whole group
can be killed.

What this does NOT do -- read this before enabling it
-----------------------------------------------------
It is **process isolation, not a security boundary against a determined
attacker**, and on Windows it is materially weaker:

* Windows has no ``resource`` module, so **CPU and memory limits are not
  enforced there** -- only the wall-clock timeout is. A memory bomb can still
  take down the host. Proper limits need a Job Object, or a container.
* The child still runs as the **same OS user** with the same filesystem
  permissions. It can read anything that user can read, outside the scrubbed
  env and cwd.
* There is **no network namespace**. The child can open sockets.

So this clears ``Isolation.SUBPROCESS`` and no more. For untrusted
multi-tenant code execution the honest bar is ``Isolation.CONTAINER``:
container-per-invocation, read-only rootfs, no network, dropped capabilities,
enforced CPU/memory. This module is the stepping stone to that, and the reason
`python_exec` is not in any default allowlist.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass

logger = logging.getLogger("bat.agent.sandbox")

IS_POSIX = os.name == "posix"

#: Environment the child gets. Deliberately minimal -- everything else the
#: parent holds (credentials, DSNs, model paths) is withheld by omission.
_BASE_ENV: dict[str, str] = {
    "PATH": "",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "LANG": "C.UTF-8",
}


@dataclass(frozen=True, slots=True)
class SandboxResult:
    """Outcome of one sandboxed execution."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    duration_ms: float

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def render(self) -> str:
        """Format for the model as an observation."""
        if self.timed_out:
            return f"Execution timed out.\n{self.stdout}".strip()
        if self.ok:
            return self.stdout.strip() or "Executed successfully with no output."
        detail = (self.stderr or self.stdout).strip()
        return f"Execution failed (exit {self.exit_code}):\n{detail}"


@dataclass(frozen=True, slots=True)
class SandboxLimits:
    timeout_s: float = 10.0
    max_output_chars: int = 8_000
    #: POSIX only. Ignored on Windows, which has no rlimits.
    max_cpu_seconds: int = 5
    max_memory_bytes: int = 256 * 1024 * 1024


class SubprocessSandbox:
    """Runs Python source in a scrubbed, time-bounded child process."""

    __slots__ = ("_limits", "_python")

    def __init__(
        self, limits: SandboxLimits | None = None, python: str | None = None
    ) -> None:
        self._limits = limits or SandboxLimits()
        self._python = python or sys.executable

    @property
    def limits(self) -> SandboxLimits:
        return self._limits

    @property
    def enforces_resource_limits(self) -> bool:
        """False on Windows. Callers should surface this, not assume it."""
        return IS_POSIX

    async def run_python(self, code: str) -> SandboxResult:
        loop = asyncio.get_running_loop()
        started = loop.time()
        workdir = tempfile.mkdtemp(prefix="bat-sandbox-")

        try:
            process = await asyncio.create_subprocess_exec(
                self._python,
                # -I: isolated. -S: no site processing. -B: no .pyc writes.
                "-I",
                "-S",
                "-B",
                "-c",
                code,
                cwd=workdir,
                env=dict(_BASE_ENV),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **self._platform_kwargs(),
            )
        except Exception as exc:
            logger.exception("failed to start sandbox process")
            shutil.rmtree(workdir, ignore_errors=True)
            return SandboxResult(
                stdout="",
                stderr=f"could not start sandbox: {exc}",
                exit_code=-1,
                timed_out=False,
                duration_ms=round((loop.time() - started) * 1000, 2),
            )

        timed_out = False
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self._limits.timeout_s
            )
        except TimeoutError:
            timed_out = True
            await self._terminate(process)
            stdout, stderr = b"", b""
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        return SandboxResult(
            stdout=_decode(stdout, self._limits.max_output_chars),
            stderr=_decode(stderr, self._limits.max_output_chars),
            exit_code=process.returncode if process.returncode is not None else -1,
            timed_out=timed_out,
            duration_ms=round((loop.time() - started) * 1000, 2),
        )

    def _platform_kwargs(self) -> dict[str, object]:
        if IS_POSIX:
            return {"preexec_fn": self._apply_rlimits, "start_new_session": True}
        # Windows: no rlimits and no process groups here. Only the wall-clock
        # timeout applies; see the module docstring.
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return {"creationflags": flags}

    def _apply_rlimits(self) -> None:  # pragma: no cover - POSIX only
        import resource

        cpu = self._limits.max_cpu_seconds
        memory = self._limits.max_memory_bytes
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    async def _terminate(self, process: asyncio.subprocess.Process) -> None:
        """Kill hard. A timed-out child has already ignored its budget."""
        if process.returncode is not None:
            return
        try:
            process.kill()
        except ProcessLookupError:  # pragma: no cover - already gone
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except TimeoutError:  # pragma: no cover - unkillable child
            logger.error("sandbox process survived kill", extra={"pid": process.pid})


def _decode(raw: bytes, limit: int) -> str:
    text = raw.decode("utf-8", errors="replace")
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} chars omitted]"
