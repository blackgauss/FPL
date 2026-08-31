"""API contract tests over a synthetic data root (tmp Store, no disk repo data)."""

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

def test_meta(client: TestClient) -> None:
    r = client.get("/api/meta")
    assert r.status_code == 200
    body = r.json()
    assert body["season"] == SEASON
    assert body["current_gw"] == 1
    assert body["max_forecast_gw"] == 2  # features at gw=1 target gw=2
    assert body["live"]["available"] is True
    assert body["live"]["fetched_at"]
    assert body["live"]["age_seconds"] >= 0
    assert {"name", "mtime", "size"} <= set(body["artifacts"][0])


def test_meta_missing_data(empty_client: TestClient) -> None:
    body = empty_client.get("/api/meta").json()
    assert body["current_gw"] == 0
    assert body["live"] == {"available": False, "fetched_at": None,
                            "age_seconds": None}
    assert body["artifacts"] == []


# -- players ----------------------------------------------------------------

def test_players_rows(client: TestClient) -> None:
    r = client.get("/api/players")
    assert r.status_code == 200
    body = r.json()
    assert body["season"] == SEASON
    assert body["current_gw"] == 1
    assert body["total"] == N_PLAYERS
    row = next(x for x in body["rows"] if x["web_name"] == "Salah")
    assert row["status"] == "a" and "now_cost" in row and "pred_next" in row


def test_players_search_and_position(client: TestClient) -> None:
    body = client.get("/api/players", params={"search": "salah"}).json()
    assert body["total"] == 1
    assert body["rows"][0]["web_name"] == "Salah"

    body = client.get("/api/players", params={"position": "GKP"}).json()
    assert body["total"] == N_PLAYERS // 4
    assert all(r["position"] == "GKP" for r in body["rows"])


def test_players_max_price(client: TestClient) -> None:
    body = client.get("/api/players", params={"max_price": 64}).json()
    assert body["total"] == 4  # now_cost 61..64


def test_players_pred_next(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_forecast(monkeypatch)
    body = client.get("/api/players").json()
    assert all(isinstance(r["pred_next"], (int, float)) for r in body["rows"])
    assert next(r for r in body["rows"] if r["web_name"] == "P01")["pred_next"] == 6.0


def test_players_missing_data(empty_client: TestClient) -> None:
    body = empty_client.get("/api/players").json()
    assert body["available"] is False
    assert body["rows"] == []


# -- forecast -----------------------------------------------------------------

def test_forecast_rows(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_forecast(monkeypatch)
    codes = "1001,1002"
    body = client.get("/api/forecast",
                      params={"player_codes": codes, "gw_start": 3,
                              "horizon": 2}).json()
    assert body["gw_start"] == 3 and body["gw_end"] == 4
    assert {r["player_code"] for r in body["rows"]} == {1001, 1002}
    assert {r["gw"] for r in body["rows"]} == {3, 4}
    qkeys = {f"q{int(q * 100)}" for q in QS}
    assert set(body["rows"][0]["quantiles"]) == qkeys


def test_forecast_default_window_and_position(client: TestClient,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    fake_forecast(monkeypatch)
    body = client.get("/api/forecast", params={"position": "GKP"}).json()
    assert body["gw_start"] == 2 and body["gw_end"] == 6  # current_gw + 1, +5
    assert all(r["web_name"] in {"P04", "P08", "P12", "P16", "P20"}
               for r in body["rows"])


def test_forecast_horizon_capped(client: TestClient) -> None:
    assert client.get("/api/forecast", params={"horizon": 11}).status_code == 422


def test_forecast_unbuildable_is_503(empty_client: TestClient) -> None:
    r = empty_client.get("/api/forecast")
    assert r.status_code == 503
    assert "forecast build failed" in r.json()["detail"]


# -- team flags ------------------------------------------------------------

def test_team_flags(client: TestClient) -> None:
    r = client.get("/api/team/flags")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["gw"] == 1 and body["entry_id"] == ENTRY  # defaults from data
    assert len(body["rows"]) == 15
    assert body["captain"]["player_id"] == 5
    assert body["vice_captain"]["player_id"] == 9
    by_id = {row["player_id"]: row for row in body["rows"]}
    assert by_id[1]["flag"] == "ok"
    assert "INJURED" in by_id[3]["flag"]
    assert body["comparison"]["xscore"] == 40.2
    assert body["comparison"]["players"][0]["actual_points"] == 8


def test_team_flags_explicit_params(client: TestClient) -> None:
    body = client.get("/api/team/flags",
                      params={"gw": 1, "entry_id": ENTRY}).json()
    assert body["available"] is True
    body = client.get("/api/team/flags", params={"gw": 9}).json()
    assert body["available"] is False


def test_team_flags_missing(empty_client: TestClient) -> None:
    assert empty_client.get("/api/team/flags").json()["available"] is False


# -- transfers ---------------------------------------------------------------

def test_transfers_suggestions(client: TestClient) -> None:
    r = client.get("/api/transfers/suggestions")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["gw"] == 2 and body["source"] == "gw2_plan.json"
    sug = body["suggestions"][0]
    assert sug["transfer_in"] == "New Guy"
    assert sug["transfer_out_code"] == 1013
    assert sug["expected_gain"] == 2.4
    assert body["bank_tenths"] == 1


def test_transfers_wrong_gw_and_missing(client: TestClient,
                                        empty_client: TestClient) -> None:
    assert client.get("/api/transfers/suggestions",
                      params={"gw": 9}).json()["available"] is False
    assert empty_client.get("/api/transfers/suggestions").json()["available"] is False


# -- league -----------------------------------------------------------------

def test_league_standings(client: TestClient) -> None:
    r = client.get("/api/league/standings", params={"entry_id": ENTRY})
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True and body["current_gw"] == 1
    assert len(body["rows"]) == 3
    mine = next(r for r in body["rows"] if r["entry_id"] == ENTRY)
    assert mine["is_self"] is True
    assert mine["gw_points"] == 61  # from collected team_history, latest event
    assert mine in body["rows"][1:]


def test_league_missing(empty_client: TestClient) -> None:
    body = empty_client.get("/api/league/standings").json()
    assert body["available"] is False and body["rows"] == []


# -- research -----------------------------------------------------------------

def test_research_metrics_default(client: TestClient) -> None:
    r = client.get("/api/research/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["run"] == "ranking.metrics.json"
    assert body["metrics"]["lgbm_all"]["spearman_rho"] == 0.67
    assert body["config"] == {"models": ["lgbm"]}


def test_research_metrics_named_and_missing(client: TestClient) -> None:
    body = client.get("/api/research/metrics",
                      params={"run": "search.metrics.json"}).json()
    assert body["available"] is True and body["metrics"]["best"] == 61.0
    assert client.get("/api/research/metrics",
                      params={"run": "nope.metrics.json"}
                      ).json()["available"] is False


def test_research_metrics_rejects_traversal(client: TestClient) -> None:
    r = client.get("/api/research/metrics", params={"run": "../secret.json"})
    assert r.status_code == 400


def test_research_metrics_missing(empty_client: TestClient) -> None:
    assert empty_client.get("/api/research/metrics").json()["available"] is False


# -- headless render smoke: real view JS against real captured payloads -------

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


def test_headless_render_smoke(client: TestClient, tmp_path: Path,
                               monkeypatch: pytest.MonkeyPatch) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    fake_forecast(monkeypatch)
    payloads = tmp_path / "payloads.json"
    payloads.write_text(json.dumps(_capture_payloads(client)), encoding="utf-8")
    js_dir = Path(web_app.__file__).parent / "static" / "js"
    r = subprocess.run([node, str(js_dir / "render_smoke.mjs"), str(payloads)],
                       capture_output=True, text=True, timeout=120, cwd=js_dir)
    assert r.returncode == 0, f"render smoke failed:\n{r.stdout}\n{r.stderr}"


def test_static_assets_revalidate(client: TestClient) -> None:
    """Deployed JS must never be served from a heuristic browser cache:
    a stale explorer.js once made a shipped fix look like a live bug."""
    for path in ("/", "/js/views/explorer.js"):
        r = client.get(path)
        assert r.status_code == 200
        assert r.headers["cache-control"] == "no-cache"
        assert "etag" in r.headers


# -- sorting + CDF ------------------------------------------------------------

def test_players_sort_price_desc(client: TestClient) -> None:
    body = client.get("/api/players",
                      params={"sort": "now_cost", "dir": "desc"}).json()
    costs = [r["now_cost"] for r in body["rows"]]
    assert costs == sorted(costs, reverse=True)


def test_players_sort_nulls_last(client: TestClient) -> None:
    # no forecast warmed => all pred_next null; sort must still work
    body = client.get("/api/players",
                      params={"sort": "pred_next", "dir": "desc"}).json()
    assert all(r["pred_next"] is None for r in body["rows"])


def test_players_sort_rejects_unknown_column(client: TestClient) -> None:
    assert client.get("/api/players",
                      params={"sort": "salary_usd"}).status_code == 400


def test_forecast_cdf_unscoreable_window_explains(client: TestClient) -> None:
    """GWs past the last scored one must say *why*, not 'no forecast row' —
    the transfers CDF compare depends on this to distinguish 'not yet'
    from 'this player has no forecast'."""
    r = client.get("/api/forecast/cdf",
                   params={"player_code": 1001, "gw": 99})
    assert r.status_code == 404
    assert "scoreable" in r.json()["detail"]


def test_forecast_cdf(client: TestClient,
                      monkeypatch: pytest.MonkeyPatch) -> None:
    fake_forecast(monkeypatch)
    body = client.get("/api/forecast/cdf", params={
        "player_code": 1001, "gw": 3}).json()
    assert body["gw"] == 3
    assert all(0.0 <= p <= 1.0 for p in body["cdf"])
    assert body["cdf"] == sorted(body["cdf"])   # monotone
    assert body["cdf"][0] <= 0.01 and body["cdf"][-1] >= 0.99


def test_forecast_cdf_missing_player(client: TestClient,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
    fake_forecast(monkeypatch)
    r = client.get("/api/forecast/cdf",
                   params={"player_code": 999999, "gw": 3})
    assert r.status_code == 404


def test_players_have_team_names(client: TestClient) -> None:
    rows = client.get("/api/players").json()["rows"]
    assert all(r["team"] and r["team"].startswith("T") for r in rows)
    asc = client.get("/api/players",
                     params={"sort": "team", "dir": "asc"}).json()["rows"]
    assert [r["team"] for r in asc] == sorted(r["team"] for r in asc)


# -- difficulty, league ownership, form, history --------------------------------

def test_players_has_xdg_and_league_ownership(client: TestClient) -> None:
    rows = client.get("/api/players", params={"limit": 50}).json()["rows"]
    by_id = {r["player_id"]: r for r in rows}
    assert 0.0 <= by_id[1]["xdg_next"] <= 100.0   # GW2 foe rated from GW1 ELO
    assert by_id[1]["xdg_next5"] is not None
    assert by_id[1]["own_league"] == 66.7         # me + rival 111 of 3 entries
    assert by_id[20]["own_league"] == 33.3        # only rival 333 owns element 20
    assert client.get("/api/players", params={
        "sort": "xdg_next", "dir": "desc"}).json()["rows"][0]


def test_league_report_derives_results(client: TestClient) -> None:
    body = client.get("/api/league/standings/report").json()
    assert body["available"] and body["events"] == [1, 2]
    mine = body["managers"][str(ENTRY)]
    assert mine["1"]["result"] == "W" and mine["1"]["points"] == 55.0
    assert mine["2"]["result"] == "W" and mine["2"]["opponent_points"] == 48


def test_league_ownership_endpoint(client: TestClient) -> None:
    body = client.get("/api/league/ownership").json()
    assert body["available"]
    row = next(r for r in body["rows"] if r["player_code"] == 1001)
    assert row["own_league"] == 66.7 and row["managers"] == 2
    assert all(r["own_league"] > 0 for r in body["rows"])


def test_team_history_endpoint(client: TestClient) -> None:
    body = client.get("/api/team/history").json()
    assert body["available"]
    assert [r["gw"] for r in body["rows"]] == [1, 2]
    assert body["rows"][0]["xscore"] == 40.2   # joined from gw1 comparison
    assert body["rows"][1]["bench_points"] == 6


def test_team_performance_percentiles(client: TestClient,
                                      monkeypatch: pytest.MonkeyPatch) -> None:
    fake_forecast(monkeypatch)
    body = client.get("/api/team/performance").json()
    assert body["available"] and body["gw"] == 1
    rows = body["rows"]
    assert len(rows) == 2
    p1 = next(r for r in rows if r["player_code"] == 1001)
    p12 = next(r for r in rows if r["player_code"] == 1012)
    # actual 8 sits mid-digest (q5 ~5, q99 ~10.4): plausible, not flagged
    assert 0.05 < p1["p_exceed"] < 0.95 and 5 < p1["percentile"] < 95
    # actual 1 sits below q05: left-tail — underperform
    assert p12["percentile"] <= 5.0 and p12["p_exceed"] >= 0.95
    assert body["summary"]["below_5th"] == ["P12"]
    assert all(r["q05"] <= r["q95"] for r in rows)


def test_latest_account_picks_highest_gw_not_mtime(tmp_path: Path) -> None:
    """Recency of file mtime must never win over game-week number — a stale
    re-gw2 run after a gw5 plan must not resurrect it."""
    (tmp_path / "account").mkdir()

    def w(name: str, gw: int) -> None:
        (tmp_path / "account" / name).write_text(json.dumps({"gw": gw}))
    w("gw5_plan.json", 5)
    w("gw12_plan.json", 12)
    w("gw3_comparison.json", 3)
    store = Store(root=tmp_path, account_dir="account")
    assert store.latest_account("plan")["file"] == "gw12_plan.json"
    assert store.latest_account("comparison")["file"] == "gw3_comparison.json"
    assert store.latest_account("plan")["gw"] == 12
    assert store.latest_account("matches") is None


def test_entry_id_nested_collection_variant(tmp_path: Path) -> None:
    (tmp_path / "account").mkdir()
    (tmp_path / "account" / "collection.json").write_text(
        json.dumps({"entry": {"id": 123}}))
    assert Store(root=tmp_path, account_dir="account").entry_id() == 123


def test_clock_single_source_of_gw_truth(client: TestClient,
                                         empty_client: TestClient) -> None:
    """Meta fields and clock must never disagree, and an empty store yields
    the documented zeros instead of raising (routers rely on this)."""
    assert client.app.state.store.clock() == {"current": 1, "next": 2,
                                              "scoreable": 2}
    assert empty_client.app.state.store.clock() == {"current": 0, "next": 1,
                                                    "scoreable": 0}
    meta = client.get("/api/meta").json()
    assert (meta["current_gw"], meta["max_forecast_gw"]) == (1, 2)


def test_forecast_cdf_tails_are_server_interpolated(
        client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Views get tail probabilities, not raw grids to interpolate twice in
    JS — and thresholds echo back as the client sent them (dict keys)."""
    fake_forecast(monkeypatch)
    body = client.get("/api/forecast/cdf", params={
        "player_code": 1001, "gw": 3, "at": "5,10,0.5"}).json()
    tails = body["tails"]
    assert set(tails) == {"5", "10", "0.5"}
    for t in tails.values():
        assert t["p_le"] + t["p_gt"] == pytest.approx(1.0, abs=1e-3)
    assert tails["5"]["p_gt"] >= tails["10"]["p_gt"]   # farther right = rarer
    assert tails["0.5"]["p_le"] <= tails["5"]["p_le"]
    bad = client.get("/api/forecast/cdf",
                     params={"player_code": 1001, "gw": 3, "at": "5,lol"})
    assert bad.status_code == 422


def test_with_live_single_join_implementation(tmp_path: Path,
                                              client: TestClient) -> None:
    """`Store.with_live` is the only legal picks/live seam: element-keyed
    (picks) and player_code-keyed (dimensions) callers both work; no
    snapshot passes the frame through untouched."""
    picked = pl.DataFrame({"element": [1, 12, 999_999]})
    joined = client.app.state.store.with_live(picked, on="element")
    row_by = {r["element"]: r for r in joined.to_dicts()}
    assert row_by[1]["web_name"] and row_by[1]["status"]
    assert row_by[999_999]["web_name"] is None           # left join, no drop
    assert Store(root=tmp_path).with_live(picked, on="element").columns == ["element"]
