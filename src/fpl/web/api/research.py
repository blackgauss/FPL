"""GET /api/research/metrics — parsed research-run metrics (+ config)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from fpl.web.queries import Store

router = APIRouter(prefix="/research")


def get_store(request: Request) -> Store:
    return request.app.state.store


def _payload(store: Store, run: str, metrics: dict) -> dict:
    config = None
    if run.endswith(".metrics.json"):
        try:
            config = store.artifact_json(run.removesuffix(".metrics.json") + ".json")
        except ValueError:
            config = None
    return {"available": True, "run": run, "metrics": metrics, "config": config}


@router.get("/metrics")
def get_metrics(request: Request, run: str | None = None) -> dict:
    store = get_store(request)
    try:
        if run is None:
            metrics = store.artifact_json("ranking.metrics.json")
            if metrics is not None:
                return _payload(store, "ranking.metrics.json", metrics)
            candidates = [a for a in store.artifacts()
                          if a["name"].endswith(".metrics.json")]
            if not candidates:
                return {"available": False,
                        "reason": "no *.metrics.json artifacts found"}
            newest = max(candidates, key=lambda a: a["mtime"])["name"]
            metrics = store.artifact_json(newest)
            if metrics is None:
                return {"available": False, "run": newest,
                        "reason": "artifact vanished mid-request"}
            return _payload(store, newest, metrics)
        metrics = store.artifact_json(run)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    if metrics is None:
        return {"available": False, "run": run,
                "reason": f"artifact {run!r} not found"}
    return _payload(store, run, metrics)
