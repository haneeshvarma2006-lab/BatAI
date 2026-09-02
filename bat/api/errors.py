"""Exception handlers producing RFC 9457 problem documents.

Two rules hold across every handler:

* the response shape is identical for every failure, so clients parse one thing;
* a message crosses the boundary only if its error is marked ``public``.
  Anything else becomes a generic 500 with the detail in the logs, keyed by
  request id, so an internal exception cannot leak a DSN or a stack frame.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from bat.domain.errors import BatError, RateLimitError
from bat.observability import request_id_var

logger = logging.getLogger("bat.api.errors")

PROBLEM_CONTENT_TYPE = "application/problem+json"


def problem(
    *,
    status: int,
    code: str,
    detail: str,
    request_id: str | None = None,
    extra: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body: dict[str, Any] = {
        "type": f"https://docs.bat.ai/errors/{code}",
        "title": code.replace("_", " "),
        "status": status,
        "code": code,
        "detail": detail,
    }
    if request_id:
        body["request_id"] = request_id
    if extra:
        body["errors"] = extra
    return JSONResponse(
        status_code=status,
        content=body,
        media_type=PROBLEM_CONTENT_TYPE,
        headers=headers,
    )


async def bat_error_handler(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, BatError):  # pragma: no cover - wiring guard
        return await unhandled_exception_handler(request, exc)
    request_id = request_id_var.get()
    headers: dict[str, str] = {}
    if isinstance(exc, RateLimitError):
        headers["Retry-After"] = str(max(1, int(exc.retry_after_seconds + 0.999)))

    if exc.public:
        logger.info(
            "request rejected",
            extra={"error_code": exc.code, "status": exc.status, "path": request.url.path},
        )
        return problem(
            status=exc.status,
            code=exc.code,
            detail=exc.message,
            request_id=request_id,
            extra=exc.details or None,
            headers=headers or None,
        )

    logger.error(
        "internal error",
        exc_info=exc,
        extra={"error_code": exc.code, "path": request.url.path},
    )
    return problem(
        status=exc.status,
        code=exc.code,
        detail="an internal error occurred; quote the request_id when reporting it",
        request_id=request_id,
    )


async def validation_error_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    if not isinstance(exc, RequestValidationError):  # pragma: no cover
        return await unhandled_exception_handler(request, exc)
    return problem(
        status=422,
        code="invalid_request",
        detail="request body or parameters failed validation",
        request_id=request_id_var.get(),
        extra={"fields": _summarize(exc.errors())},
    )


async def http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, StarletteHTTPException):  # pragma: no cover
        return await unhandled_exception_handler(request, exc)
    detail = exc.detail if isinstance(exc.detail, str) else "request failed"
    return problem(
        status=exc.status_code,
        code=_CODES.get(exc.status_code, "http_error"),
        detail=detail,
        request_id=request_id_var.get(),
        headers=dict(exc.headers) if exc.headers else None,
    )


async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Last resort. Never returns the exception text to the caller."""
    logger.exception("unhandled exception", extra={"path": request.url.path})
    return problem(
        status=500,
        code="internal_error",
        detail="an internal error occurred; quote the request_id when reporting it",
        request_id=request_id_var.get(),
    )


_CODES: dict[int, str] = {
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    429: "rate_limited",
}


def _summarize(errors: list[Any]) -> list[dict[str, str]]:
    """Flatten pydantic errors into a stable, client-friendly shape."""
    summary: list[dict[str, str]] = []
    for err in errors[:20]:
        location = ".".join(str(part) for part in err.get("loc", ()) if part != "body")
        summary.append({"field": location or "body", "message": str(err.get("msg", ""))})
    return summary


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(BatError, bat_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
