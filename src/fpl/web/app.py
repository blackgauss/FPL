"""App factory: wires routers + static frontend. Contains no logic."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from fpl.web.queries import Store

STATIC_DIR = Path(__file__).parent / "static"


class RevalidateStatic(StaticFiles):
    """Always revalidate (ETag/Last-Modified -> cheap 304s).

    Without Cache-Control, browsers apply heuristic freshness and can serve
    days-old JS after a deploy — a shipped-but-inert fix looks like a
    still-broken bug (exactly what happened with explorer.js).
    """

    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


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

    app.mount("/", RevalidateStatic(directory=STATIC_DIR, html=True), name="static")
    return app
