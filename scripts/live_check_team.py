"""Check a candidate team basket against live FPL API state.

Reproduces the team-search basket the way `inspect_teams.py` does, then flags
every squad from the typed interface — `basket_squads` hydrates the basket
into Squad objects and `flag_squad` reports players who are injured /
suspended / unavailable, transferred clubs, or price-moved. The "my candidate
team had missing/injured players" problem, made detectable.

Usage:
    python scripts/live_check_team.py \
        --season 2025-2026 --gw 2 --gw-end 4 \
        --cache data/raw/fpl_api/live.json
"""

from __future__ import annotations

import argparse

import polars as pl

from fpl.domain import position_sort_key
from fpl.live.filters import flag_squad
from fpl.live.live import load_live_state
from fpl.model.inference import load_model
from fpl.model.train import load_training
from fpl.team.filtering import availability_from_gw_stats
from fpl.team.scoring import score_players
from fpl.team.selection import pool_and_squads


def _expected(squad, expected) -> float:
    return sum(expected.get(p.code, 0.0) for p in squad.players)


def main() -> None:
    ap = argparse.ArgumentParser(description="Detect missing/injured players in a team basket")
    ap.add_argument("--season", default="2025-2026")
    ap.add_argument("--gw", type=int, default=2)
    ap.add_argument("--gw-end", type=int, default=None)
    ap.add_argument("--cache", default="data/raw/fpl_api/live.json")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--n-teams", type=int, default=20, help="how many squads to build")
    args = ap.parse_args()
    gw_end = args.gw_end or args.gw

    # 1) reproducibly build the candidate basket
    players = pl.read_parquet(f"{args.processed}/players_{args.season}.parquet")
    gw_stats = pl.read_parquet(f"{args.processed}/gw_stats_{args.season}.parquet")
    teams = pl.read_parquet(f"{args.processed}/teams_{args.season}.parquet")
    model = load_model(f"{args.processed}/points_lgbm.txt")
    td = load_training(args.processed, [args.season])[args.season]
    scored, _ = score_players(td, model, gw_start=args.gw, gw_end=gw_end,
                              players=players, detail=True)
    avail = availability_from_gw_stats(gw_stats, players,
                                       gw_start=args.gw, gw_end=gw_end)
    selection = pool_and_squads(scored, scored, avail, gw=args.gw,
                                n_teams=args.n_teams, seed=1)
    squads = list(zip(selection.team_ids, selection.squads, strict=False))
    expected = selection.expected
    teams_names = dict(zip(teams["code"], teams["name"], strict=False))

    # 2) live state (rate-limit-safe; cached)
    live, fetched = load_live_state(args.cache, max_age_seconds=3600)
    print(f"\nlive snapshot {fetched} | {live.height} players | "
          f"{len(squads)} squads\n")
    print("=" * 100)

    status_of = {c: s for c, s in live.select("player_code", "status").iter_rows()}

    # rank squads by expected total; show the best, then a problem rollup
    ranked = sorted(squads, key=lambda kv: _expected(kv[1], expected), reverse=True)
    best = ranked[0]
    flags = flag_squad(best[1], live)

    print("=== best squad + live-detected problems ===")
    for p in sorted(best[1].players, key=lambda p: position_sort_key(p.position)):
        club = teams_names.get(p.club, "?")
        status = status_of.get(p.code) or "-"
        print(f"  {p.name:<28} {club:<14} status={status:>1}  "
              f"{flags[p.code][:52]}")

    print("\n=== problem rollup across all squads ===")
    counts: dict[tuple[str, str], int] = {}
    flagged_starts = 0
    for _, squad in squads:
        sf = flag_squad(squad, live)
        for p in squad.players:
            if sf[p.code] != "ok":
                flagged_starts += 1
                counts[(p.name, sf[p.code])] = counts.get((p.name, sf[p.code]), 0) + 1
    for (name, problem), n in sorted(counts.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {n:>3}  {name:<20}  {problem}")
    total_starts = sum(len(s.players) for _, s in squads)
    print(f"\n{flagged_starts} of {total_starts} team-starts have a live problem "
          f"(injured/suspended/transferred/price-moved)")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()