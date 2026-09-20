"""Serves the single-file chat UI at ``/``.

One static HTML file, vanilla JS, no build step. It talks to the same API a
client would, with the API key pasted in by the user and kept in the browser's
localStorage -- so it exercises the real auth path rather than bypassing it.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["ui"])

_INDEX = Path(__file__).resolve().parent.parent.parent / "ui" / "index.html"


@router.get("/", include_in_schema=False, response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(_INDEX.read_text(encoding="utf-8"))
