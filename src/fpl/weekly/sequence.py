"""One-gameweek transfer and lineup planning."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import polars as pl

from fpl.dist import QS
from fpl.domain import Player, PlayerIdentity, PlayerState, Position, Squad, squad_from_frame
from fpl.live.live import fetch_bootstrap, to_live_frame
from fpl.model.inference import load_model
from fpl.model.train import load_training
from fpl.team.scoring import score_players
from fpl.weekly.captain import set_captains
from fpl.weekly.transfer import apply_transfer


def _current_squad(picks: pl.DataFrame, players: pl.DataFrame,
                   live: pl.DataFrame, *, gw: int,
                   entry_id: int | None = None,
                   prices: pl.DataFrame | None = None) -> Squad:
    selected = picks.filter(pl.col("gw") == gw)
    if "entry_id" in selected.columns:
        ids = selected["entry_id"].unique().to_list()
        if entry_id is None:
            if len(ids) > 1:
                raise ValueError("entry_id is required when picks contain multiple teams")
            entry_id = int(ids[0])
        selected = selected.filter(pl.col("entry_id") == entry_id)
    joined = selected.join(
        players.select("player_id", "player_code", "web_name", "position",
                       "team_code"),
        left_on="element", right_on="player_id", how="inner",
    ).join(
        live.select("player_id", "now_cost"),
        left_on="element", right_on="player_id", how="left",
    )
    # Clubs come from the collected snapshot, NOT live: the 3-per-club cap is
    # enforced at deadline, and mid-window club transfers can leave a legal
    # squad with 4 players of one club in the live bootstrap.
    if prices is not None:
        # GW-frozen prices when supplied: live now_cost drifts with the
        # market, which falsely breaks the budget for older-GW squads.
        joined = joined.join(
            prices.select("player_id", pl.col("now_cost").alias("now_cost_snap")),
            left_on="element", right_on="player_id", how="left",
        ).with_columns(
            pl.coalesce(["now_cost_snap", "now_cost"]).alias("now_cost_eff"))
    else:
        joined = joined.with_columns(pl.col("now_cost").alias("now_cost_eff"))
    joined = joined.with_columns(
        pl.col("position_right").alias("player_position"),
    )
    frame = joined.select(
        "player_code", "web_name", "player_position", "team_code",
        "now_cost_eff",
    ).rename({"player_position": "position", "now_cost_eff": "price_tenths"})
    base = squad_from_frame(frame, gw=gw)
    ordered = joined.sort("position")
    result = replace(
        base,
        starters=tuple(ordered.filter(pl.col("position") <= 11)
                        ["player_code"].to_list()),
        bench=tuple(ordered.filter(pl.col("position") > 11)
                    ["player_code"].to_list()),
        captain=int(ordered.filter(pl.col("is_captain"))["player_code"].item()),
        vice_captain=int(ordered.filter(pl.col("is_vice_captain"))
                         ["player_code"].item()),
    )
    problems = result.validate()
    if problems:
        raise ValueError("collected team is invalid: " + "; ".join(problems))
    return result


def _effective_expected(expected: dict[int, float], live: pl.DataFrame) -> dict[int, float]:
    values = {}
    for row in live.iter_rows(named=True):
        code = int(row["player_code"])
        chance = row.get("chance_of_playing_next_round")
        probability = 0.0 if chance is None and row.get("status") in ("i", "s", "u", "n") \
            else (float(chance) / 100.0 if chance is not None else 1.0)
        values[code] = expected.get(code, 0.0) * probability
    return values


def _best_lineup(squad: Squad, values: dict[int, float]) -> Squad:
    """Choose the highest-value legal XI and a substitution-safe bench order."""
    by_pos = {position: [p for p in squad.players if p.position.value == position]
              for position in ("GKP", "DEF", "MID", "FWD")}
    best: tuple[float, tuple[Player, ...]] | None = None
    for gk in combinations(by_pos["GKP"], 1):
        for n_def in range(3, 6):
            for n_fwd in range(1, 4):
                n_mid = 10 - n_def - n_fwd
                if not 2 <= n_mid <= 5:
                    continue
                for defs in combinations(by_pos["DEF"], n_def):
                    for mids in combinations(by_pos["MID"], n_mid):
                        for fwds in combinations(by_pos["FWD"], n_fwd):
                            xi = (*gk, *defs, *mids, *fwds)
                            score = sum(values.get(p.code, 0.0) for p in xi)
                            if best is None or score > best[0]:
                                best = (score, xi)
    if best is None:
        raise ValueError("squad has no legal starting XI")
    starters = tuple(p.code for p in best[1])
    remaining = [p for p in squad.players if p.code not in starters]
    bench_gk = sorted((p for p in remaining if p.position.value == "GKP"),
                      key=lambda p: -values.get(p.code, 0.0))
    bench_outfield = sorted((p for p in remaining if p.position.value != "GKP"),
                            key=lambda p: -values.get(p.code, 0.0))
    return replace(squad, starters=starters,
                   bench=tuple(p.code for p in (*bench_gk, *bench_outfield)),
                   captain=None, vice_captain=None)


def digest_mean(quantiles: Sequence[float]) -> float:
    """E[X] read off a quantile vector by trapezoidal integration over the
    QS levels — our model's own mean, not FPL's ep_next."""
    area = 0.0
    for i in range(len(QS) - 1):
        area += (QS[i + 1] - QS[i]) * (quantiles[i] + quantiles[i + 1]) / 2
    # small open intervals [0, q1) and (q99, 1] carry the tails; treat the
    # digest endpoints as the practical bounds (matches the web CDF display)
    area += QS[0] * quantiles[0] + (1 - QS[-1]) * quantiles[-1]
    return area


def _lineup_digest(squad: Squad,
                   distributions: Mapping[int, Sequence[float]]) -> list[float] | None:
    """Squad-total points distribution, quantile by quantile (independent
    approximation, same convention as the multi-GW web charts). The captain
    slot contributes its digest twice (2x multiplier)."""
    grids = []
    for code in squad.starters:
        grid = distributions.get(code)
        if grid is None or len(grid) != len(QS):
            return None
        grids.append(grid)
    if squad.captain is not None:
        cap = distributions.get(squad.captain)
        if cap is None or len(cap) != len(QS):
            return None
        grids.append(cap)
    return [round(float(sum(col)), 3) for col in zip(*grids, strict=True)]


def prob_greater(a: Sequence[float], b: Sequence[float]) -> float:
    """P(X > Y) from two quantile vectors: quantile-quantile cell average
    (u and v independent uniform) — 0.5 ties split."""
    wins = 0.0
    for va in a:
        for vb in b:
            wins += 1.0 if va > vb else (0.5 if va == vb else 0.0)
    return round(wins / (len(a) * len(b)), 4)


def plan_one_week(
    *, picks: pl.DataFrame, history: pl.DataFrame, players: pl.DataFrame,
    live: pl.DataFrame, expected: dict[int, float], gw: int,
    bank_tenths: int = 0, top: int = 10, entry_id: int | None = None,
    distributions: Mapping[int, Sequence[float]] | None = None,
    prices: pl.DataFrame | None = None,
) -> dict:
    """Rank hold and legal one-transfer plans for the next gameweek.

    `distributions` maps player_code -> the target GW's quantile vector
    (our t-digest forecast at QS levels). When present, every option also
    carries the XI-total distribution and P(this lineup outscores the
    no-transfer lineup) — decisions read the whole shape, not just
    expected_score."""
    current = _current_squad(picks, players, live, gw=gw, entry_id=entry_id,
                             prices=prices)
    values = _effective_expected(expected, live)
    ownership = _league_ownership(picks, players)
    weighted_values = {
        code: value * (1.0 - ownership.get(code, 0.0))
        for code, value in values.items()
    }
    by_code = current.by_code()
    official = {
        int(row["player_code"]): float(row["ep_next"] or 0.0)
        for row in live.iter_rows(named=True)
        if row.get("ep_next") is not None
    }
    candidates: list[Player] = []
    snap = {} if prices is None else dict(
        zip(prices["player_id"], prices["now_cost"], strict=True))
    for row in live.iter_rows(named=True):
        code = int(row["player_code"])
        if code in by_code or not row.get("can_select", True):
            continue
        position = {
            1: Position.GKP, 2: Position.DEF, 3: Position.MID, 4: Position.FWD,
        }.get(int(row["element_type"]))
        if position is None:
            continue
        cost = snap.get(int(row["player_id"]), row["now_cost"])
        candidates.append(Player(
            identity=PlayerIdentity(code, row["web_name"], position),
            state=PlayerState(int(row["team_code"]), int(cost)),
        ))
    # The explicit constructor above is kept behind this boundary because live
    # frames use FPL's numeric element_type while the domain uses Position.
    options = []
    transfer_choices = [(None, None)]
    for out in current.players:
        for new in candidates:
            if new.position != out.position:
                continue
            if new.cost_tenths > out.cost_tenths + bank_tenths:
                continue
            transfer_choices.append((out, new))
    for out, new in transfer_choices:
        try:
            planned = current if out is None else apply_transfer(
                current, out, new, gw=gw + 1)
        except ValueError:
            continue
        planned = _best_lineup(planned, weighted_values)
        planned = set_captains(planned, weighted_values)
        xi_dist = _lineup_digest(planned, distributions or {})
        score = sum(weighted_values.get(code, 0.0) for code in planned.starters)
        score += (weighted_values.get(planned.captain, 0.0)
                  if planned.captain else 0.0)
        options.append({
            "transfer_out": out.name if out else None,
            "transfer_in": new.name if new else None,
            "transfer_out_code": out.code if out else None,
            "transfer_in_code": new.code if new else None,
            "expected_score": round(score, 3),
            "expected_delta": 0.0,
            "ownership_in": round(ownership.get(new.code, 0.0), 3)
            if new else None,
            "ownership_out": round(ownership.get(out.code, 0.0), 3)
            if out else None,
            "model_expected_in": round(values.get(new.code, 0.0), 3)
            if new else None,
            "model_expected_out": round(values.get(out.code, 0.0), 3)
            if out else None,
            "official_ep_next_in": official.get(new.code) if new else None,
            "official_ep_next_out": official.get(out.code) if out else None,
            "xi_quantiles": xi_dist,
            "squad": planned,
        })
    hold_score = next(option["expected_score"] for option in options
                       if option["transfer_out"] is None)
    hold_dist = next(option["xi_quantiles"] for option in options
                     if option["transfer_out"] is None)
    for option in options:
        option["expected_delta"] = round(option["expected_score"] - hold_score, 3)
        if distributions and option["xi_quantiles"] and hold_dist:
            option["prob_beat_hold"] = prob_greater(
                option["xi_quantiles"], hold_dist)
            option["xi_q10"] = option["xi_quantiles"][QS.index(0.10)]
            option["xi_q50"] = option["xi_quantiles"][QS.index(0.50)]
            option["xi_q90"] = option["xi_quantiles"][QS.index(0.90)]
    options.sort(key=lambda option: -option["expected_score"])
    return {
        "gw": gw + 1,
        "bank_tenths": bank_tenths,
        "current_squad": list(current.codes()),
        "ownership_basis": "unique league entries selecting the player",
        "options": [_serialize_option(option) for option in options[:top]],
    }


def _serialize_option(option: dict) -> dict:
    squad = option["squad"]
    by_code = squad.by_code()
    return {**{key: value for key, value in option.items() if key != "squad"},
            "starters": list(squad.starters), "bench": list(squad.bench),
            "starter_names": [by_code[code].name for code in squad.starters],
            "bench_names": [by_code[code].name for code in squad.bench],
            "captain": squad.captain, "vice_captain": squad.vice_captain}


def run_from_files(
    *, picks_path: str, history_path: str, processed: str, season: str,
    model_path: str, gw: int, bank_tenths: int, top: int, out: str | None = None,
    entry_id: int | None = None,
) -> dict:
    """Build a one-week plan. Values come from OUR model: full t-digest
    distributions when their GW window is settled enough, else the point
    model, else (per-player, tagged) official ep_next."""
    players = pl.read_parquet(f"{processed}/players_{season}.parquet")
    live = to_live_frame(fetch_bootstrap())
    try:
        gs = pl.read_parquet(f"{processed}/gw_stats_{season}.parquet")
        prices = (gs.filter(pl.col("gw") <= gw)
                  .sort("player_id", "gw").group_by("player_id")
                  .last().select("player_id", "now_cost"))
    except Exception:
        prices = None
    distributions: dict[int, list[float]] = {}
    expected: dict[int, float] = {}
    source = "official_ep"
    try:
        from fpl.team.distribution import distributional_forecast
        dfg = distributional_forecast(processed, season, gw + 1, gw + 1)
        for row in dfg.to_dicts():
            struct = row.get("quantiles_struct") or {}
            grids = [float(struct[f"q{int(q * 100)}"]) for q in QS]
            distributions[int(row["player_code"])] = grids
        expected = {code: round(digest_mean(vals), 3)
                    for code, vals in distributions.items()}
        source = "model_digest"
    except Exception:
        distributions = {}
    if not distributions:
        try:
            td = load_training(processed, [season], require_target=False)[season]
            model = load_model(model_path)
            _, per_gw = score_players(td, model, gw_start=gw + 1,
                                      gw_end=gw + 1, players=players,
                                      detail=True)
            expected = dict(zip(per_gw["player_code"],
                                per_gw["expected_points"], strict=False))
            source = "model_point"
        except Exception:
            expected = {}
    official = {
        int(r["player_code"]): float(r["ep_next"] or 0.0)
        for r in live.to_dicts() if r.get("ep_next") is not None
    }
    if not expected:
        expected, source = dict(official), "official_ep"
    elif len(expected) < len(official):
        expected |= {c: v for c, v in official.items() if c not in expected}
        source = f"{source}+ep_fallback"
    result = plan_one_week(
        picks=pl.read_parquet(picks_path), history=pl.read_parquet(history_path),
        players=players, live=live, expected=expected, gw=gw,
        bank_tenths=bank_tenths, top=top, entry_id=entry_id,
        distributions=distributions, prices=prices,
    )
    result["expected_source"] = source
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _league_ownership(picks: pl.DataFrame, players: pl.DataFrame) -> dict[int, float]:
    """Return squad ownership as a fraction of unique collected league entries."""
    if "entry_id" not in picks.columns:
        return {}
    entries = picks["entry_id"].n_unique()
    if entries == 0:
        return {}
    counts = picks.select("entry_id", "element").unique().group_by("element").agg(
        pl.len().alias("owners"))
    mapped = counts.join(players.select("player_id", "player_code"),
                         left_on="element", right_on="player_id", how="inner")
    return {
        int(row["player_code"]): float(row["owners"]) / entries
        for row in mapped.iter_rows(named=True)
    }
