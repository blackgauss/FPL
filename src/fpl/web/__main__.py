"""Serve the web UI: python -m fpl.web [--port 8000] [--root .]"""

from __future__ import annotations

import argparse


def main() -> None:
    import uvicorn

    from fpl.web.app import create_app
    from fpl.web.queries import Store

    ap = argparse.ArgumentParser(prog="python -m fpl.web")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--root", default=".", help="repo root (data/ paths)")
    ap.add_argument("--season", default="2026-2027")
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()

    store = Store(root=args.root, season=args.season)
    app = create_app(store)
    if args.reload:  # import-string needed for reloader
        app = "fpl.web.app:create_app"
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
