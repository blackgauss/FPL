"""GW1 candidate teams: 2026-27 pre-season selection with live reconcile.

The 2026-27 feature store's GW1 rows describe the current (pre-season) world:
carryover form, live prices, ep_next, status. Scoring those rows directly
gives each player's expected points for the upcoming gameweek (GW1). Then the
live snapshot reconciles clubs/injuries into the input BEFORE construction
(fpl.live.current), so `max_per_club`, the pool, and enumeration all see the
current world. The basket is hydrated into typed Squad objects and live
problems (injured / missing from the live roster / transferred / price-moved)
come from `flag_squad` — no inline sixth-price-unit math.

Usage:
    python scripts/live_gw1_teams.py [--cache data/raw/fpl_api/live.json]
"""

from __future__ import annotations

import argparse

import polars as pl

from fpl.domain import position_sort_key
from fpl.live.current import construction_input
from fpl.live.filters import flag_squad, suggest
from fpl.live.live import load_live_state
from fpl.model.inference import load_model
from fpl.model.train import load_training
from fpl.team.enumerate import greedy_teams
from fpl.team.filtering import filter_pool
from fpl.team.harness import basket_squads
from fpl.units import to_millions


def main() -> None:
    ap = argparse.ArgumentParser(description="2026-27 GW1 candidate teams, live-reconciled")
    ap.add_argument("--season", default="2026-2027")
    ap.add_argument("--gw", type=int, default=1)
    ap.add_argument("--cache", default="data/raw/fpl_api/live.json")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--n-teams", type=int, default=20)
    args = ap.parse_args()

    live, fetched = load_live_state(args.cache, max_age_seconds=3600)
    players = pl.read_parquet(f"{args.processed}/players_{args.season}.parquet")
    gw_stats = pl.read_parquet(f"{args.processed}/gw_stats_{args.season}.parquet")
    model = load_model(f"{args.processed}/points_lgbm.txt")

    # GW1 rows: features are the pre-season/current world -> expected GW1 points
    td = load_training(args.processed, [args.season],
                       require_target=False)[args.season]
    mask = td.gw == args.gw
    scored = td.meta.filter(pl.col("gw") == args.gw).with_columns(
        pl.Series("expected_total", model.predict(td.X[mask]))
    )
    scored = scored.join(players.select("player_id", "player_code", "web_name",
                                        "position"),
                         on="player_id", how="left")
    # availability frame for filter_pool (price from gw_stats)
    avail = gw_stats.join(players.select("player_id", "player_code"),
                          on="player_id", how="left").select(
        "player_code", "now_cost").with_columns(
        pl.col("now_cost").cast(pl.Float64),
        pl.lit(1).alias("minutes_in_window"))
    scored = scored.with_columns(pl.lit(1).alias("minutes_in_window"))

    # reconcile the current world BEFORE construction (clubs/injuries/missing)
    scored_live = construction_input(scored, live, suggest(live))
    pool = filter_pool(scored_live, avail, top_k_per_position=25,
                       max_per_team=4, reserve_top=20)
    basket = greedy_teams(pool, n_teams=args.n_teams, seed=1)

    # hydrate the typed interface once — from the RECONCILED frame so the squad
    # clubs are current-world (greedy's ≤3/club caps were live-aware)
    squads = basket_squads(basket, scored_live, gw=args.gw)
    expected = dict(zip(scored["player_code"], scored["expected_total"],
                        strict=False))
    teams = pl.read_parquet(f"{args.processed}/teams_{args.season}.parquet")
    teams_names = dict(zip(teams["code"], teams["name"], strict=False))
    status_of = {c: s for c, s in live.select("player_code", "status").iter_rows()}

    print(f"\nlive {fetched} | 2026-27 GW{args.gw} candidates "
          f"(scored {scored_live.height} players, pool {pool.height}, "
          f"{len(squads)} squads)\n"
          f"{'='*104}")

    # pick the first few DISTINCT squads (unique player sets) for display
    shown: set[tuple[int, ...]] = set()
    for tid, squad in squads:
        sig = tuple(sorted(p.code for p in squad.players))
        if sig in shown:
            continue
        if len(shown) >= 3:
            break
        shown.add(sig)
        exp = sum(expected.get(p.code, 0.0) for p in squad.players)
        cost = to_millions(squad.cost_tenths())
        flags = flag_squad(squad, live)
        print(f"\n### squad {tid} | £{cost:.1f}m | expected GW{args.gw} {exp:.1f}")
        for p in sorted(squad.players, key=lambda p: position_sort_key(p.position)):
            club = teams_names.get(p.club, "?")
            status = status_of.get(p.code) or "-"
            print(f"  {p.name:<26} {club:<14} status={status:>1}  "
                  f"{flags[p.code][:56]}")

    keep = suggest(live).sum()
    print(f"\n  {int(keep)}/{live.height} live players are selectable "
          f"(suggest mask)")
    nd = {tuple(sorted(p.code for p in s.players)) for _, s in squads}
    print(f"  basket distinctness: {len(nd)} of {len(squads)} squads")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
