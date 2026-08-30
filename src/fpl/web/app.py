"""App factory: wires routers + static frontend. Contains no logic."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from fpl.web.queries import Store

STATIC_DIR = Path(__file__).parent / "static"


def create_app(store: Store | None = None) -> FastAPI:
    app = FastAPI(
        title="FPL web UI (PoC)", version="0.1",
        description="Read-only window over the repo's pipeline outputs; "
                    "the frontend is programmed against this OpenAPI schema.",
    )
    app.state.store = store or Store()

    from fpl.web import api
    for router in api.API_ROUTERS:
        app.include_router(router, prefix="/api", tags=[router.prefix or "api"])

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
