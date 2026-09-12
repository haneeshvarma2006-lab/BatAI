"""BAT desktop CLI.

Runs the native llama.cpp engine locally. Requires ``BAT_MODEL__MODEL_PATH`` to
point at a ``.gguf`` file (put it in ``.env``; see ``.env.example``).

    python main.py

Commands:
    :read <path>     index a local file into memory
    :remember <fact> store a durable fact about yourself
    :forget          erase this machine's memory
    exit             quit
"""

from __future__ import annotations

import asyncio
import sys

from bat.adapters.llama_cpp_client import ModelUnavailableError
from bat.settings import Settings
from core.brain import CognitiveBrain


async def run() -> int:
    settings = Settings()
    name = settings.model.name

    if not settings.model.is_configured:
        print(
            "No model weights configured.\n"
            "Set BAT_MODEL__MODEL_PATH to a .gguf file (see .env.example), then "
            "run again.",
            file=sys.stderr,
        )
        return 2

    brain = CognitiveBrain(settings)
    print(f"=== {name} (native llama.cpp) ===")
    print(f"Loading {settings.model.model_path} ...", flush=True)
    try:
        await brain.load()
    except ModelUnavailableError as exc:
        print(f"\n{exc.message}", file=sys.stderr)
        return 2
    print("Ready. Type a question, or 'exit' to quit.\n")

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                break

            try:
                handled = await _command(brain, user_input)
                if handled:
                    continue

                print(f"\n{name}: ", end="", flush=True)
                async for token in brain.stream(user_input):
                    print(token, end="", flush=True)
                print("\n")
            except Exception as exc:
                print(f"\n[error] {exc}\n", file=sys.stderr)
    finally:
        await brain.aclose()

    print(f"Shutting down {name}. Goodbye.")
    return 0


async def _command(brain: CognitiveBrain, line: str) -> bool:
    """Handle a ``:`` command. Returns True if the line was a command."""
    if not line.startswith(":"):
        return False

    verb, _, argument = line[1:].partition(" ")
    argument = argument.strip()

    match verb.lower():
        case "read" if argument:
            print(await brain.read_document(argument))
        case "remember" if argument:
            print(f"Stored ({await brain.remember(argument)} chunk(s)).")
        case "forget":
            await brain.memory.forget_tenant(tenant_id="local")
            print("Memory erased.")
        case _:
            print("Commands: :read <path> | :remember <fact> | :forget")
    return True


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
