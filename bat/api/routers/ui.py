"""Serves the chat UI: ``/`` for the page, ``/static`` for its CSS and JS.

Plain files, no build step. The page talks to the same API any client would,
with the user's API key held in the browser, so it exercises the real auth path
rather than bypassing it.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

UI_DIR = Path(__file__).resolve().parent.parent.parent / "ui"

router = APIRouter(tags=["ui"])


@router.get("/", include_in_schema=False)
async def index() -> FileResponse:
    # no-cache so a UI update is picked up without a hard refresh.
    return FileResponse(UI_DIR / "index.html", headers={"Cache-Control": "no-cache"})


def mount(app: FastAPI) -> None:
    app.include_router(router)
    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")
