"""GET /api/team/flags — live health flags for the collected squad picks."""

from __future__ import annotations

import polars as pl
from fastapi import APIRouter, HTTPException, Request

from fpl.live.filters import flag_squad_player
from fpl.web.queries import Store

router = APIRouter(prefix="/team")


def get_store(request: Request) -> Store:
    return request.app.state.store



def _comparison(store: Store, gw: int) -> dict | None:
    wanted = f"gw{gw}_comparison.json"
    for doc in store.account_json(r"gw\d+_comparison"):
        if doc.get("file") == wanted:
            summary = doc.get("summary")
            if not isinstance(summary, dict):
                return None
            players = doc.get("players")
            players = players if isinstance(players, list) else []
            players = sorted(players, key=lambda p: p.get("actual_points") or 0,
                             reverse=True)
            return {**summary, "players": players}
    return None


def _latest_squad_picks(store, frame, entry_id: int | None,
                        gw: int | None) -> tuple[pl.DataFrame, int | None, int]:
    """Filter picks to one entry (default: the collection's own) and one GW
    (default: latest). Shared by flags — single 'which 15 players' reader."""
    if entry_id is None and "entry_id" in frame.columns:
        ids = frame.get_column("entry_id").drop_nulls().unique().to_list()
        entry_id = ids[0] if len(ids) == 1 else store.entry_id()
    if entry_id is not None and "entry_id" in frame.columns:
        frame = frame.filter(pl.col("entry_id") == entry_id)
    if gw is None:
        gws = frame.get_column("gw").drop_nulls().to_list()
        gw = max(gws) if gws else 0
    return frame.filter(pl.col("gw") == gw), entry_id, gw


@router.get("/flags")
def get_flags(request: Request, gw: int | None = None,
              entry_id: int | None = None) -> dict:
    store = get_store(request)
    picks = store.account("team_picks")
    if picks is None:
        return {"available": False, "reason": "no collected team_picks"}

    frame, entry_id, gw = _latest_squad_picks(store, picks, entry_id, gw)
    if frame.height == 0:
        return {"available": False, "gw": gw, "entry_id": entry_id,
                "reason": f"no collected picks for gw {gw}"}

    # Row-wise flags via fpl.live.filters.flag_squad_player over picks x live.
    # Picks carry no price/club snapshot of the picked GW, so price fields are
    # live-current (a Squad from a GW-frozen frame is not buildable from disk
    # alone); status flags are exact.
    frame = store.with_live(frame.sort("position"), on="element")
    have_live = store.live() is not None
    expected = store.expected_next()

    out_rows: list[dict] = []
    captain: dict | None = None
    vice_captain: dict | None = None
    for r in frame.to_dicts():
        flag = ("no live snapshot" if not have_live
                else flag_squad_player({"status": r.get("status")}))
        row = {
            "player_id": r["element"],
            "web_name": r.get("web_name"),
            "player_code": r.get("player_code"),
            "slot": r.get("position"),
            "element_type": r.get("element_type"),
            "multiplier": r.get("multiplier"),
            "is_captain": bool(r.get("is_captain")),
            "is_vice_captain": bool(r.get("is_vice_captain")),
            "now_cost": r.get("now_cost"),
            "ep_next": r.get("ep_next"),
            "selected_by_percent": r.get("selected_by_percent"),
            "status": r.get("status"),
            "news": r.get("news"),
            "expected": expected.get(r.get("player_code"), (None, "none"))[0],
            "expected_source": expected.get(r.get("player_code"),
                                            (None, "none"))[1],
            "flag": flag,
        }
        if r.get("is_captain"):
            captain = row
        if r.get("is_vice_captain"):
            vice_captain = row
        out_rows.append(row)

    return {
        "available": True,
        "gw": gw,
        "entry_id": entry_id,
        "captain": captain,
        "vice_captain": vice_captain,
        "rows": out_rows,
        "comparison": _comparison(store, gw),
    }


@router.get("/performance")
def get_performance(request: Request, gw: int | None = None) -> dict:
    """How surprising was each GW? For every settled player row, place the
    actual score inside the player's own forecast CDF (t-digest quantiles):
    percentile of the actual plus tail probability — CDF-based credible
    interval reading, not just a point error."""
    from fpl.dist import QS, probability_below

    store = get_store(request)
    try:
        current = gw if gw is not None else store.clock()["current"]
    except Exception:
        return {"available": False, "reason": "no gameweek state"}
    wanted = f"gw{current}_comparison.json"
    doc = next((d for d in store.account_json(r"gw\d+_comparison")
                if d.get("file") == wanted), None)
    players = (doc or {}).get("players")
    if not isinstance(players, list) or not players:
        return {"available": False, "gw": current,
                "reason": f"no collected {wanted} (run: fpl compare --gw {current})"}
    try:
        frame = store.forecast(current, current)
    except Exception as exc:
        raise HTTPException(503, f"forecast build failed: {exc}") from exc
    by_code = {r["player_code"]: r for r in frame.to_dicts()}

    rows, beat_95, under_5 = [], [], []
    for p in players:
        fc = by_code.get(p.get("player_code"))
        row = {**{k: p.get(k) for k in
                  ("player_code", "web_name", "position", "minutes",
                   "actual_points", "expected_points", "is_captain",
                   "is_vice_captain", "multiplier")},
               "gw": current, "model_pred": None, "q05": None, "q50": None,
               "q95": None, "percentile": None, "p_exceed": None}
        if fc is not None:
            vals = [fc[f"q{int(q * 100)}"] for q in QS]
            actual = float(p.get("actual_points") or 0.0)
            below = probability_below(vals, actual)
            row.update(
                model_pred=fc.get("pred"),
                q05=vals[1], q50=vals[4], q95=vals[7],
                percentile=round(below * 100, 1),
                p_exceed=round(1.0 - below, 4),
            )
            if below >= 0.95 and actual >= 3:
                beat_95.append(row["web_name"])
            elif below <= 0.05:
                under_5.append(row["web_name"])
        rows.append(row)
    rows.sort(key=lambda r: -(r["percentile"] if r["percentile"] is not None
                              else -1))
    return {"available": True, "gw": current, "rows": rows,
            "summary": {"beat_95th": beat_95, "below_5th": under_5,
                        "n": len(rows)},
            "note": "percentile = P(X <= actual) from the player's forecast "
                    "digest; 95+ = overperformed, <5 = underperformed"}


@router.get("/history")
def get_history(request: Request) -> dict:
    """Points per GW from collected team history (own-form curve, bench
    points, transfers), with GW-comparison xScores where collected."""
    store = get_store(request)
    history = store.account("team_history")
    if history is None or not history.height:
        return {"available": False, "reason": "no collected team_history"}
    entry_id = store.entry_id()
    rows = history.to_dicts()
    if entry_id is not None:
        mine = [r for r in rows if r.get("entry_id") == entry_id]
        rows = mine or rows
    rows.sort(key=lambda r: r.get("event") or 0)
    xs: dict[int, float] = {}
    for doc in store.account_json(r"gw\d+_comparison"):
        summary = doc.get("summary")
        if isinstance(summary, dict) and summary.get("xscore") is not None:
            xs[int(summary.get("gw", -1))] = float(summary["xscore"])
    return {
        "available": True,
        "season": store.season,
        "rows": [{"gw": r.get("event"), "points": r.get("points"),
                  "total_points": r.get("total_points"),
                  "bench_points": r.get("points_on_bench"),
                  "transfers": r.get("event_transfers"),
                  "xscore": xs.get(r.get("event")), "rank": r.get("rank")}
                 for r in rows],
    }
