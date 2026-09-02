"""Operator CLI.

    python -m bat.cli issue-key --tenant acme --principal alice
    python -m bat.cli check-config
    python -m bat.cli serve --reload
"""

from __future__ import annotations

import argparse
import json
import sys

from bat.api.security import generate_api_key
from bat.domain.tenancy import DEFAULT_SCOPES, Scope
from bat.settings import Settings


def cmd_issue_key(args: argparse.Namespace) -> int:
    """Mint a key, print the plaintext once, and emit the config entry."""
    plaintext, digest = generate_api_key()
    scopes = (
        [Scope(s) for s in args.scope]
        if args.scope
        else sorted(str(s) for s in DEFAULT_SCOPES)
    )
    record = {
        "key_sha256": digest,
        "tenant_id": args.tenant,
        "principal_id": args.principal,
        "scopes": [str(s) for s in scopes],
    }
    if args.label:
        record["label"] = args.label

    print("API key (shown once, store it in your secret manager now):")
    print(f"  {plaintext}\n")
    print("Add this record to BAT_API_KEYS:")
    print(f"  {json.dumps(record)}")
    return 0


def cmd_check_config(args: argparse.Namespace) -> int:
    """Validate the current environment, including production hardening."""
    try:
        settings = Settings()
    except Exception as exc:
        print("configuration is INVALID:\n", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1

    print(f"environment       {settings.environment}")
    print(f"session backend   {settings.session.backend}")
    print(f"vector mode       {settings.vector.mode}")
    print(f"model             {settings.model.name} @ {settings.model.host}")
    print(f"credentials       {len(settings.api_keys)} key(s), "
          f"{len({k.tenant_id for k in settings.api_keys})} tenant(s)")
    print(f"tool isolation    >= {settings.agent.min_tool_isolation.name}")
    print(f"enabled tools     {sorted(settings.agent.enabled_tools) or 'none'}")

    if not settings.api_keys:
        print("\nwarning: no API keys configured; every request returns 401")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "bat.api.app:app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        # Extra workers each get their own in-memory store, so sessions would
        # appear and vanish depending on which one answers.
        workers=1,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bat", description="BAT platform operator CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    issue = sub.add_parser("issue-key", help="mint an API key for a tenant")
    issue.add_argument("--tenant", required=True)
    issue.add_argument("--principal", required=True)
    issue.add_argument("--label")
    issue.add_argument(
        "--scope",
        action="append",
        choices=[str(s) for s in Scope],
        help="repeatable; defaults to the standard end-user scope set",
    )
    issue.set_defaults(func=cmd_issue_key)

    check = sub.add_parser("check-config", help="validate the environment")
    check.set_defaults(func=cmd_check_config)

    serve = sub.add_parser("serve", help="run the API with uvicorn")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
