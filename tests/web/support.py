"""shared support for the synthetic web data world: root builder, payload
capture and the headless harness runner (used by the contract suite and the
real-data smoke suite)."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

from fpl.dist import QS
from fpl.web import app as web_app
from fpl.web.app import create_app
from fpl.web.queries import Store

SEASON = "2026-2027"
ENTRY = 4242
N_PLAYERS = 20
POS = ["GKP", "DEF", "MID", "FWD"]
ETYPE = {"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}


def _player_record(i: int) -> dict:
    pos = POS[i % 4]
    return {
        "player_id": i, "player_code": 1000 + i, "web_name": "Salah" if i == 10 else f"P{i:02d}",
        "position": pos, "team_code": 1 + (i - 1) % 20, "element_type": ETYPE[pos],
    }


PLAYERS = [_player_record(i) for i in range(1, N_PLAYERS + 1)]


def _write_root(root: Path) -> Path:
    processed = root / "data/processed"
    account = root / "data/raw/fpl_api/account"
    artifacts = root / "experiments/artifacts"
    for d in (processed, account, root / "data/raw/fpl_api",
              root / "data/webcache", artifacts):
        d.mkdir(parents=True, exist_ok=True)

    pl.DataFrame({
        "player_code": [p["player_code"] for p in PLAYERS],
        "player_id": [p["player_id"] for p in PLAYERS],
        "first_name": [f"First{p['player_id']}" for p in PLAYERS],
        "second_name": ["Faker" if p["player_id"] != 10 else "Salah" for p in PLAYERS],
        "web_name": [p["web_name"] for p in PLAYERS],
        "team_code": [p["team_code"] for p in PLAYERS],
        "position": [p["position"] for p in PLAYERS],
        "season": [SEASON] * N_PLAYERS,
    }).write_parquet(processed / f"players_{SEASON}.parquet")

    pl.DataFrame({
        "player_id": list(range(1, N_PLAYERS + 1)),
        "gw": [1] * N_PLAYERS,
        "now_cost": [60 + i for i in range(1, N_PLAYERS + 1)],
        "total_points": [5] * N_PLAYERS,
    }).write_parquet(processed / f"gw_stats_{SEASON}.parquet")

    pl.DataFrame({
        "player_id": list(range(1, N_PLAYERS + 1)),
        "player_code": [1000 + i for i in range(1, N_PLAYERS + 1)],
        "gw": [1] * N_PLAYERS,
    }).write_parquet(processed / f"features_{SEASON}.parquet")

    pl.DataFrame({
        "gw": [1] * N_PLAYERS,
        "player_id": list(range(1, N_PLAYERS + 1)),
        "minutes": [90] * N_PLAYERS,
        "total_points": [5] * N_PLAYERS,
    }).write_parquet(account / "event_live.parquet")

    picks_frame = pl.DataFrame({
        "entry_id": [ENTRY] * 15,
        "gw": [1] * 15,
        "element": list(range(1, 16)),
        "position": list(range(1, 16)),
        "multiplier": [2 if e == 5 else 1 for e in range(1, 16)],
        "is_captain": [e == 5 for e in range(1, 16)],
        "is_vice_captain": [e == 9 for e in range(1, 16)],
        "element_type": [p["element_type"] for p in PLAYERS[:15]],
    }, schema={"entry_id": pl.Int64, "gw": pl.Int64, "element": pl.Int64,
               "position": pl.Int64, "multiplier": pl.Int64,
               "is_captain": pl.Boolean, "is_vice_captain": pl.Boolean,
               "element_type": pl.Int64})
    # two rival managers' picks make league ownership meaningful (3 entries)
    rivals = pl.concat([
        pl.DataFrame({"entry_id": [111] * 10, "gw": [1] * 10,
                      "element": list(range(1, 11)),
                      "position": list(range(1, 11))}),
        pl.DataFrame({"entry_id": [333] * 10, "gw": [1] * 10,
                      "element": list(range(11, 21)),
                      "position": list(range(1, 11))}),
    ])
    pl.concat([picks_frame, rivals.select(
        "entry_id", "gw", "element", "position",
        pl.lit(1, dtype=pl.Int64).alias("multiplier"),
        pl.lit(False).alias("is_captain"),
        pl.lit(False).alias("is_vice_captain"),
        pl.lit(1, dtype=pl.Int64).alias("element_type"))],
        how="vertical_relaxed").write_parquet(account / "team_picks.parquet")

    pl.DataFrame({
        "entry_id": [ENTRY] * 2 + [111, 333],
        "event": [1, 2, 1, 2],
        "points": [55, 61, 52, 48],
        "total_points": [55, 116, 52, 100],
        "rank": [100, 90, 40, 55],
        "points_on_bench": [2, 6, 0, 1],
        "event_transfers": [0, 1, 0, 0],
    }).write_parquet(account / "team_history.parquet")

    # fixtures: GW1 settled with ELO, GW2 upcoming (difficulty fallback uses it)
    gw_home = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] + [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    gw_away = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20] + [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    pl.DataFrame({
        "match_id": list(range(1, 21)),
        "gw": [1] * 10 + [2] * 10,
        "kickoff_time": ["x"] * 20,
        "home_team": gw_home,
        "away_team": gw_away,
        "home_score": [2] * 20, "away_score": [1] * 20,
        "home_team_elo": [1600 + 20 * i for i in range(1, 11)] + [None] * 10,
        "away_team_elo": [1500 + 20 * i for i in range(1, 11)] + [None] * 10,
        "tournament": ["epl"] * 20, "finished": [True] * 10 + [False] * 10,
        "season": [SEASON] * 20,
    }).write_parquet(processed / f"matches_{SEASON}.parquet")

    pl.DataFrame({
        "entry_1_entry": [111, ENTRY], "entry_1_points": [52, 61],
        "entry_2_entry": [ENTRY, 333], "entry_2_points": [55, 48],
        "entry_1_player_name": ["Ann Other", "Erik IJ"],
        "entry_2_player_name": ["Erik IJ", "Sam Other"],
        "event": [1, 2], "is_bye": [False, False],
    }).write_parquet(account / "league_matches.parquet")

    pl.DataFrame({
        "league_id": [1005115] * 3,
        "entry_id": [111, ENTRY, 333],
        "rank": [1, 2, 3],
        "player_name": ["Ann Other", "Erik IJ", "Sam Other"],
        "entry_name": ["Alpha", "Mine", "Gamma"],
        "total": [60, 55, 40],
        "event_total": [60, 55, 40],
        "last_rank": [0, 0, 0],
    }).write_parquet(account / "league_standings.parquet")

    (account / "collection.json").write_text(json.dumps(
        {"entry": {"id": ENTRY}, "entry_id": ENTRY}))
    (account / "gw1_comparison.json").write_text(json.dumps({
        "summary": {"gw": 1, "xscore": 40.2, "actual_score": 55.0,
                    "history_score": 55, "score_source": "event_live_settlement",
                    "error": -14.8, "player_count": 15},
        "players": [
            {"gw": 1, "position": 1, "player_id": 0, "player_code": 1001,
             "web_name": "P01", "multiplier": 1, "is_captain": False,
             "is_vice_captain": False, "minutes": 90, "actual_points": 8,
             "expected_points": 5.4, "weighted_expected": 5.4},
            {"gw": 1, "position": 12, "player_id": 11, "player_code": 1012,
             "web_name": "P12", "multiplier": 1, "is_captain": False,
             "is_vice_captain": False, "minutes": 0, "actual_points": 1,
             "expected_points": 2.1, "weighted_expected": 2.1},
        ],
    }))
    (account / "gw2_plan.json").write_text(json.dumps({
        "gw": 2, "bank_tenths": 1,
        "current_squad": [1001 + i for i in range(1, 15)],
        "ownership_basis": "unique league entries selecting the player",
        "options": [{
            "transfer_out": "P03", "transfer_in": "New Guy",
            "transfer_out_code": 1013, "transfer_in_code": 999999,
            "expected_score": 60.1, "expected_delta": 2.4,
            "ownership_in": 0.1, "ownership_out": 0.68,
            "captain": 1005, "vice_captain": 999999,
        }],
    }))

    elements = [{
        "id": p["player_id"], "code": p["player_code"], "web_name": p["web_name"],
        "team": 1, "team_code": p["team_code"], "element_type": p["element_type"],
        "now_cost": 60 + p["player_id"],
        "status": "i" if p["player_id"] == 3 else "a",
        "news": "", "news_added": None,
        "chance_of_playing_this_round": None,
        "chance_of_playing_next_round": None,
        "selected_by_percent": "3.0", "minutes": 90,
        "ep_next": "4.5", "ep_this": "4.1",
        "removed": False, "can_select": True, "can_transact": True,
    } for p in PLAYERS]
    (root / "data/raw/fpl_api/live.json").write_text(json.dumps({
        "fetched_at": datetime.now(UTC).isoformat(),
        "fetched_epoch": time.time(),
        "payload": {"elements": elements, "teams": [
            {"code": c, "short_name": f"T{c}", "name": f"Team {c}"}
            for c in range(1, 21)]},
    }))

    (artifacts / "ranking.metrics.json").write_text(json.dumps(
        {"lgbm_all": {"spearman_rho": 0.67}}))
    (artifacts / "ranking.json").write_text(json.dumps({"models": ["lgbm"]}))
    (artifacts / "search.metrics.json").write_text(json.dumps({"best": 61.0}))
    return root


def _client(root: Path) -> TestClient:
    return TestClient(create_app(Store(root=root, season=SEASON)))


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    return _client(_write_root(tmp_path / "full"))


@pytest.fixture()
def empty_client(tmp_path: Path) -> TestClient:
    root = tmp_path / "empty"
    root.mkdir()
    return _client(root)


def fake_forecast(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the model cold-build with a deterministic frame over our players."""
    def fake(processed: str, season: str, gw_start: int, gw_end: int, **kw):
        qcols = {f"q{int(q * 100)}": [] for q in QS}
        rows: dict[str, list] = {"player_code": [], "web_name": [],
                                 "position": [], "gw": [], "pred": []}
        for p in PLAYERS:
            for gw in range(gw_start, gw_end + 1):
                rows["player_code"].append(p["player_code"])
                rows["web_name"].append(p["web_name"])
                rows["position"].append(p["position"])
                rows["gw"].append(gw)
                rows["pred"].append(4.0 + gw)
                for q in QS:
                    qcols[f"q{int(q * 100)}"].append(gw * 0.5 + q * p["player_code"] / 100)
        return (pl.DataFrame(rows)
                .with_columns(pl.DataFrame(qcols).to_struct("quantiles_struct")))
    monkeypatch.setattr("fpl.team.distribution.distributional_forecast", fake)
    monkeypatch.setattr("fpl.web.queries.Store.max_forecast_gw", lambda self: 99)


# -- meta ------------------------------------------------------------------
def _capture_payloads(client: TestClient) -> dict:
    meta = client.get("/api/meta").json()
    gw = meta.get("current_gw") or 1
    return {
        "/api/meta": meta,
        "/api/players": client.get("/api/players",
                                   params={"limit": 25, "offset": 0}).json(),
        "/api/forecast": client.get("/api/forecast", params={
            "gw_start": gw + 1, "horizon": 5}).json(),
        "/api/team/flags": client.get("/api/team/flags",
                                      params={"gw": gw}).json(),
        "/api/transfers/suggestions": client.get(
            "/api/transfers/suggestions").json(),
        "/api/league/standings": client.get("/api/league/standings").json(),
        "/api/league/standings/report": client.get(
            "/api/league/standings/report").json(),
        "/api/league/ownership": client.get("/api/league/ownership").json(),
        "/api/team/history": client.get("/api/team/history").json(),
        "/api/team/performance": client.get(
            "/api/team/performance").json(),
    }

def render_smoke(client, payloads_path, *, timeout: int = 120):
    """Run the headless DOM-render harness over the client's payloads —
    shared by the synthetic and real-data gate tests."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    payloads_path.write_text(json.dumps(_capture_payloads(client)),
                             encoding="utf-8")
    js_dir = Path(web_app.__file__).parent / "static" / "js"
    return subprocess.run(
        [node, str(js_dir / "render_smoke.mjs"), str(payloads_path)],
        capture_output=True, text=True, timeout=timeout, cwd=js_dir)

