"""PoC: ownership-weighted opponent squads + marginal quantile H2H MC.

Uses live ``selected_by_percent`` to sample valid opponent squads and the
existing quantile artifact for player outcomes. Current 2026-27 players not
covered by the 2025-26 artifact use a position-level quantile fallback, so
this is an opponent-risk PoC, not a final GW1 forecast.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import polars as pl

from fpl.domain import (
    Player,
    PlayerIdentity,
    PlayerState,
    players_to_frame,
    squad_from_frame,
)
from fpl.live.current import construction_input
from fpl.live.filters import available, position_for_element_type
from fpl.live.live import load_live_state
from fpl.model.inference import load_model
from fpl.model.train import load_training
from fpl.team.enumerate import greedy_teams
from fpl.team.filtering import filter_pool
from fpl.team.harness import basket_squads
from fpl.weekly.captain import set_captains
from fpl.weekly.opponents import sample_opponents

N = 1000
SEED = 42


def main() -> None:
    live, fetched = load_live_state("data/raw/fpl_api/live.json", max_age_seconds=3600)
    players = pl.read_parquet("data/processed/players_2026-2027.parquet")
    stats = pl.read_parquet("data/processed/gw_stats_2026-2027.parquet")
    model = load_model("data/processed/points_lgbm.txt")
    td = load_training("data/processed", ["2026-2027"], require_target=False)["2026-2027"]
    mask = td.gw == 1
    scored = td.meta.filter(pl.col("gw") == 1).with_columns(
        pl.Series("expected_total", model.predict(td.X[mask])))
    scored = scored.join(players.select("player_id", "player_code", "web_name", "position"),
                         on="player_id")
    avail = (stats.join(players.select("player_id", "player_code"), on="player_id")
             .select("player_code", "now_cost")
             .with_columns(pl.lit(1).alias("minutes_in_window")))
    pool_frame = construction_input(
        scored.with_columns(pl.lit(1).alias("minutes_in_window")), live, available(live))
    basket = greedy_teams(filter_pool(pool_frame, avail, top_k_per_position=25,
                                      max_per_team=4, reserve_top=20),
                          n_teams=6, seed=1)
    expected = dict(zip(scored["player_code"].to_list(),
                        scored["expected_total"].to_list(), strict=False))
    candidates = []
    for _, squad in basket_squads(basket, pool_frame, gw=1):
        squad = replace(squad, bench=tuple(p.code for p in squad.players
                                           if p.code not in squad.starters))
        candidates.append(set_captains(squad, expected))
    ours = max(candidates, key=lambda s: sum(expected.get(c, 0) for c in s.starters))

    ownership = {int(r["player_code"]): float(r["selected_by_percent"] or 0) / 100
                 for r in live.iter_rows(named=True)}
    pool: list[Player] = []
    for r in live.filter(available(live)).iter_rows(named=True):
        p = Player(
            PlayerIdentity(int(r["player_code"]), r["web_name"],
                           position_for_element_type(r["element_type"])),
            PlayerState(int(r["team_code"]), int(r["now_cost"])),
        )
        pool.append(p)
    dist = pl.read_parquet("data/processed/dist_2025-2026.parquet")
    qcols = [f"q{int(q * 100):02d}" for q in (0.01, 0.05, 0.10, 0.25,
                                               0.50, 0.75, 0.90, 0.95, 0.99)]
    dist = dist.with_columns(
        pl.col("quantiles_struct").struct.rename_fields(qcols).alias("q"))
    q_by_code = {}
    for r in dist.group_by("player_code").agg(
        [pl.col("q").struct.field(c).mean().alias(c) for c in qcols]
    ).iter_rows(named=True):
        q_by_code[int(r["player_code"])] = np.asarray([r[c] for c in qcols])
    q_by_pos = {}
    for r in dist.group_by("position").agg(
        [pl.col("q").struct.field(c).mean().alias(c) for c in qcols]
    ).iter_rows(named=True):
        q_by_pos[r["position"]] = np.asarray([r[c] for c in qcols])

    def quantiles_for(player):
        q = q_by_code.get(player.code, q_by_pos.get(player.position.value))
        if q is None:
            return np.full(len(qcols), expected.get(player.code, 0.0))
        # Align the previous-season marginal center to the current forecast.
        return q + (expected.get(player.code, float(np.median(q))) - np.median(q))

    def sample_score(squad, rng):
        starters = list(squad.starters)
        draws = {}
        for p in squad.players:
            q = quantiles_for(p)
            draws[p.code] = np.interp(rng.random(N),
                                      np.asarray([0.01, 0.05, 0.10, 0.25,
                                                  0.50, 0.75, 0.90, 0.95, 0.99]), q)
        score = sum(draws[c] for c in starters)
        if squad.captain is not None:
            score = score + draws[squad.captain]
        return score
    rng = np.random.default_rng(SEED)
    counts: dict[int, int] = {}
    wins = draws = losses = 0
    ours_codes = set(ours.codes())
    for opponent in sample_opponents(rng, pool, ownership, N):
        op = squad_from_frame(players_to_frame(opponent), gw=1)
        op = replace(op, bench=tuple(p.code for p in op.players
                                     if p.code not in op.starters))
        op = set_captains(op, expected)
        op_score = sample_score(op, rng)
        own_score = sample_score(ours, rng)
        wins += int((own_score > op_score).sum())
        draws += int((own_score == op_score).sum())
        losses += int((own_score < op_score).sum())
        for code in op.codes():
            counts[code] = counts.get(code, 0) + 1
    popular = sorted(counts.items(), key=lambda x: -x[1])
    names = {int(r["player_code"]): r["web_name"] for r in live.iter_rows(named=True)}
    print(f"live={fetched} sampled={N} ownership-weighted valid opponents")
    total = wins + draws + losses
    print(f"our candidate captain={ours.by_code()[ours.captain].name} "
          f"H2H MC wins/draws/losses={wins/total:.1%}/{draws/total:.1%}/"
          f"{losses/total:.1%} over {total} paired draws")
    print("opponent coverage among sampled teams for players we do not own:")
    for code, count in popular:
        if code not in ours_codes and count / N >= 0.20:
            print(f"  {names.get(code, code):<22} {count / N:.1%}")


if __name__ == "__main__":
    main()
