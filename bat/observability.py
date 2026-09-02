"""Structured logging and request-scoped correlation.

Log records are JSON so a hosted platform can index them. Request id, tenant id
and principal id ride in contextvars purely so that *log records* carry them
without every call site passing a logger — authorization never reads them.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Iterator
from typing import Any

request_id_var: ContextVar[str | None] = ContextVar("bat_request_id", default=None)
tenant_id_var: ContextVar[str | None] = ContextVar("bat_tenant_id", default=None)
principal_id_var: ContextVar[str | None] = ContextVar("bat_principal_id", default=None)

_RESERVED = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"asctime", "message", "taskName"}


def new_request_id() -> str:
    return uuid.uuid4().hex


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per record, with correlation fields merged in."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for var, key in (
            (request_id_var, "request_id"),
            (tenant_id_var, "tenant_id"),
            (principal_id_var, "principal_id"),
        ):
            value = var.get()
            if value:
                payload[key] = value

        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, separators=(",", ":"))


def configure_logging(*, level: str = "INFO", json_output: bool = True) -> None:
    """Install a single stdout handler. Safe to call more than once."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter()
        if json_output
        else logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    )
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)
    # uvicorn installs its own noisy access log; ours carries more context.
    logging.getLogger("uvicorn.access").disabled = True


@contextmanager
def correlation(
    *,
    request_id: str,
    tenant_id: str | None = None,
    principal_id: str | None = None,
) -> Iterator[None]:
    """Bind correlation fields for the duration of the block."""
    tokens = (
        request_id_var.set(request_id),
        tenant_id_var.set(tenant_id),
        principal_id_var.set(principal_id),
    )
    try:
        yield
    finally:
        request_id_var.reset(tokens[0])
        tenant_id_var.reset(tokens[1])
        principal_id_var.reset(tokens[2])


def bind_tenant(tenant_id: str, principal_id: str) -> None:
    """Attach tenancy to the current correlation scope after authentication."""
    tenant_id_var.set(tenant_id)
    principal_id_var.set(principal_id)
