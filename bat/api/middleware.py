"""Request-scoped middleware: correlation id, access logging, body limits."""

from __future__ import annotations

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from bat.api.errors import problem
from bat.observability import correlation, new_request_id

logger = logging.getLogger("bat.api.access")

REQUEST_ID_HEADER = "X-Request-Id"


class CorrelationMiddleware(BaseHTTPMiddleware):
    """Assigns a request id, binds it for logging, and echoes it back.

    An inbound ``X-Request-Id`` is honoured so a trace survives across services,
    but it is length-capped and sanitised: it reaches log indexes, and an
    unbounded caller-controlled string there is a log-injection vector.
    """

    def __init__(self, app: ASGIApp, *, max_id_length: int = 64) -> None:
        super().__init__(app)
        self._max_id_length = max_id_length

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = self._resolve_id(request)
        started = time.perf_counter()

        with correlation(request_id=request_id):
            # Routes read this to stamp the id onto their own responses.
            request.state.request_id = request_id
            try:
                response = await call_next(request)
            except Exception:
                # The handler chain converts this into a 500; log the timing
                # here so failed requests still produce an access record.
                logger.warning(
                    "request failed",
                    extra={
                        "method": request.method,
                        "path": request.url.path,
                        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    },
                )
                raise

            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            response.headers[REQUEST_ID_HEADER] = request_id
            response.headers["Server-Timing"] = f"app;dur={duration_ms}"
            logger.info(
                "request completed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": duration_ms,
                },
            )
            return response

    def _resolve_id(self, request: Request) -> str:
        inbound = request.headers.get(REQUEST_ID_HEADER, "")
        cleaned = "".join(
            c for c in inbound if c.isalnum() or c in "-_"
        )[: self._max_id_length]
        return cleaned or new_request_id()


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Rejects oversized bodies before they are buffered into memory.

    Content-Length is advisory, so this is a cheap first gate; the per-field
    limits in the request schemas are the authoritative check.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int = 1_048_576) -> None:
        super().__init__(app)
        self._max_bytes = max_bytes

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > self._max_bytes:
            return problem(
                status=413,
                code="payload_too_large",
                detail=f"request body exceeds {self._max_bytes} bytes",
                request_id=getattr(request.state, "request_id", None),
            )
        return await call_next(request)
