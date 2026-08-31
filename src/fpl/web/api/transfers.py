"""GET /api/transfers/suggestions — newest collected weekly transfer plan."""

from __future__ import annotations

import re

from fastapi import APIRouter, Request

from fpl.web.queries import Store

router = APIRouter(prefix="/transfers")


def get_store(request: Request) -> Store:
    return request.app.state.store


def _plan_gw(doc: dict) -> int:
    m = re.match(r"gw(\d+)_plan", str(doc.get("file", "")))
    return int(m.group(1)) if m else -1


@router.get("/suggestions")
def get_suggestions(request: Request, gw: int | None = None,
                    entry_id: int | None = None) -> dict:
    store = get_store(request)
    plans = store.account_json(r"gw\d+_plan")
    if not plans:
        return {"available": False,
                "reason": "no collected gw{N}_plan.json (run the weekly planner)"}

    newest = max(plans, key=_plan_gw)
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
        "suggestions": suggestions,
    }
