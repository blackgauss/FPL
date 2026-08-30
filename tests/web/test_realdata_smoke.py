"""Real-data smoke: the traps synthetic fixtures cannot know about.

Skips when the repo's data/ tree is absent (CI); runs against actual
collected snapshots + processed store locally. These codify failures that
reached the browser: single-GW forecast windows, players absent from the
feature store, H2H league tables that only read right resolved.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from itertools import pairwise
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fpl.web import app as web_app
from fpl.web.app import create_app
from fpl.web.queries import Store

REPO_DATA = Path(__file__).resolve().parents[2] / "data"
requires_data = pytest.mark.skipif(
    not (REPO_DATA / "raw" / "fpl_api" / "account" / "collection.json").is_file(),
    reason="repo data/ not present",
)


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app(Store(root=REPO_DATA.parent)))


@requires_data
def test_forecast_quantiles_are_valid_and_clamped(client: TestClient) -> None:
    meta = client.get("/api/meta").json()
    body = client.get("/api/forecast", params={
        "gw_start": meta["current_gw"] + 1, "horizon": 5}).json()
    rows = body["rows"]
    assert rows, "forecast window returned no rows for the current season"
    qkeys = [f"q{q}" for q in
             (1, 5, 10, 25, 50, 75, 90, 95, 99)]
    players_seen: set[int] = set()
    for row in rows:
        players_seen.add(row["player_code"])
        qs = [row["quantiles"][k] for k in qkeys]
        assert all(v >= 0.0 for v in qs), f"negative points q: {row}"
        assert qs == sorted(qs), f"quantiles crossed: {row}"
    assert len(players_seen) > 100, "feature store covers too few players"
    codes = sorted(players_seen)[:5]
    per_player = client.get("/api/forecast", params={
        "player_codes": ",".join(map(str, codes)),
        "gw_start": meta["current_gw"] + 1, "horizon": 5}).json()
    assert {r["player_code"] for r in per_player["rows"]} == set(codes)


@requires_data
def test_forecast_cdf_real_data(client: TestClient) -> None:
    meta = client.get("/api/meta").json()
    rows = client.get("/api/forecast", params={
        "gw_start": meta["current_gw"] + 1, "horizon": 1}).json()["rows"]
    code = sorted({r["player_code"] for r in rows})[3]
    body = client.get("/api/forecast/cdf", params={
        "player_code": code, "gw": meta["current_gw"] + 1}).json()
    assert all(-1e-9 <= p <= 1 + 1e-9 for p in body["cdf"])
    assert body["cdf"][-1] >= 0.99 and body["cdf"][0] <= 0.01
    for a, b in pairwise(body["cdf"]):
        # near-ties in zero-inflated quantiles allow tiny non-monotone blips
        assert b >= a - 0.11


@requires_data
def test_league_standings_resolved_when_h2h(client: TestClient) -> None:
    resolved_path = (REPO_DATA / "raw" / "fpl_api" / "account"
                     / "resolved_standings.parquet")
    body = client.get("/api/league/standings").json()
    assert body["available"] is True
    assert body["rows"], "standings empty despite collection"
    if resolved_path.is_file():
        # resolved H2H table: every row has points + rank, ranks unique
        assert body["kind"] == "h2h_resolved"
        ranks = [r["rank"] for r in body["rows"]]
        assert len(set(ranks)) == len(ranks)
        assert any(r["resolved_score"] for r in body["rows"])


@requires_data
def test_players_cover_live_status_official_ep(client: TestClient) -> None:
    body = client.get("/api/players", params={"limit": 100}).json()
    assert body["total"] > 500
    rows = body["rows"]
    assert all(r["status"] in {"a", "d", "i", "s", "u"} for r in rows)
    # model pred absent early season => official ep_next fallback present
    assert any(r.get("pred_next") is not None or r.get("ep_next") is not None
               for r in rows)


@requires_data
def test_headless_render_smoke_real_data(client: TestClient,
                                         tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    from tests.web.test_web_api import _capture_payloads

    payloads = tmp_path / "payloads.json"
    payloads.write_text(json.dumps(_capture_payloads(client)), encoding="utf-8")
    js_dir = Path(web_app.__file__).parent / "static" / "js"
    r = subprocess.run([node, str(js_dir / "render_smoke.mjs"), str(payloads)],
                       capture_output=True, text=True, timeout=240, cwd=js_dir)
    assert r.returncode == 0, f"render smoke failed:\n{r.stdout}\n{r.stderr}"
