"""GET /api/transfers/suggestions — newest collected weekly transfer plan."""

from __future__ import annotations

from fastapi import APIRouter, Request

from fpl.web.queries import Store

router = APIRouter(prefix="/transfers")


def get_store(request: Request) -> Store:
    return request.app.state.store




@router.get("/suggestions")
def get_suggestions(request: Request, gw: int | None = None,
                    entry_id: int | None = None) -> dict:
    store = get_store(request)
    newest = store.latest_account("plan")
    if newest is None:
        return {"available": False,
                "reason": "no collected gw{N}_plan.json (run: fpl plan)"}
    if gw is not None and newest.get("gw") != gw:
        return {"available": False, "plan_gw": newest.get("gw"),
                "source": newest.get("file"), "requested_gw": gw,
                "reason": f"only the GW {newest.get('gw')} plan is collected"}

    suggestions = []
    for o in newest.get("options", []):
        suggestions.append({
            "transfer_in": o.get("transfer_in"),
            "transfer_out": o.get("transfer_out"),
            "transfer_in_code": o.get("transfer_in_code"),
            "transfer_out_code": o.get("transfer_out_code"),
            "expected_gain": o.get("expected_delta"),
            "expected_score": o.get("expected_score"),
            "penalty": o.get("penalty"),  # absent in current planner payloads
            "ownership_in": o.get("ownership_in"),
            "ownership_out": o.get("ownership_out"),
            "captain": o.get("captain"),
            "expected_source": o.get("expected_source"),
            "model_expected_in": o.get("model_expected_in"),
            "model_expected_out": o.get("model_expected_out"),
            "official_ep_next_in": o.get("official_ep_next_in"),
            "official_ep_next_out": o.get("official_ep_next_out"),
            "xi_q10": o.get("xi_q10"),
            "xi_q50": o.get("xi_q50"),
            "xi_q90": o.get("xi_q90"),
            "prob_beat_hold": o.get("prob_beat_hold"),
            "xi_quantiles": o.get("xi_quantiles"),
            "vice_captain": o.get("vice_captain"),
        })
    return {
        "available": True,
        "source": newest.get("file"),
        "gw": newest.get("gw"),
        "entry_id": entry_id,
        "bank_tenths": newest.get("bank_tenths"),
        "current_squad": newest.get("current_squad"),
        "ownership_basis": newest.get("ownership_basis"),
        "sell_bottom_pct": (None if newest.get("sell_bottom_q") is None
                            else round(newest["sell_bottom_q"] * 100)),
        "sell_form_weight": newest.get("sell_form_weight"),
        "expected_source": newest.get("expected_source"),
        "suggestions": suggestions,
    }
