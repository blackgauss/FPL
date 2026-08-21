# Searching Over Teams

## Goal

From a per-player expected-points forecast, build a set of **within-budget,
valid FPL teams**, simulate head-to-heads between them, and assign each team a
**value** (aggregate) plus an understanding of its **weaknesses**. This is the
"step above selecting players" — a filtering stage prunes the player pool so
team search stays tractable.

Source: `docs/journal/Week1.md` → "Searching Over Teams".

## Pipeline

```
feat-store + model  ──►  1. score EVERY player for the GW horizon  (expected_points_horizon)
                         2. filter: keep a diverse subset (position/team/price)
                         3. enumerate within-budget teams from the subset
                         4. simulate H2H over the team basket -> value + weaknesses
                         5. (later) captain mechanic + free transfers (few-week plan)
```

## Stage 1 — Player scoring for the horizon

Reuse `expected_points_horizon(td, model, gw_start, gw_end)` (already exists).
Output: one row per (player_code, gw) for the planning window. Aggregate to a
per-player expected total over the window for ranking.

## Stage 2 — Filtering (keep diversity, bound the search)

Goal: shrink ~500 eligible candidates to a tractable subset while keeping
enough **position, team, and price diversity**.

- Drop players "expected to never feature": `had_match == 0` across the window,
  `status != 'a'`, or negligible minutes.
- Keep top-N by expected total, but **stratify**: ensure GKP/DEF/MID/FWD
  representation, club spread, and a low/mid/high price band in each slot.
- Target subset size ~60-90 players (tractable) — concrete bound TBD below.

## Stage 3 — Team enumeration (the search space)

A valid FPL team: 15 players, budget <= £100m, position counts
(2/5/5/3), max 3 per club; starting XI among them.

- From the subset of size N, teams are roughly "N choose 15" weighted by
  position quotas (count per position choose the required slots).
- Design goal: large enough for complexity, small enough to be tractable.
- Method options:
  - (a) brute-force over a small filtered pool (if N small enough),
  - (b) beam/greedy by position (pick top-K per slot, then combine),
  - (c) random sampling / MCMC over the space (journal's MCMC idea).

## Stage 4 — H2H simulation + team value

- Simulate head-to-heads between teams in the basket using per-GW expected
  points (and, when we add variance, sampling).
- Assign each team a **value**: aggregate win/loss/draw record vs the field,
  not just expected total — the journal's "90% percentile team losing in H2H"
  framing.
- **Weaknesses**: report the team's exposure — e.g. depends on one player,
  weak over a fixture cluster, no good captaincy option, bench thinness.
  Black-box is fine; a few interpretable breakdowns are the goal.

## Stage 5 (later) — Captain + free transfers

- Captain (2x) and vice selection per GW.
- One free transfer per GW; plan a few weeks ahead (no need for full-season
  planning — there are wildcard opportunities mid-season).

## First milestone (this branch)

1. ✅ `score_players(td, model, gw_start, gw_end) -> aggregated scoring frame`
   (825 players scored over a 3-GW horizon on real 2025-26 data)
2. ✅ `filter_pool(pool, availability, top_k_per_position, max_per_team) ->
   diverse subset` (73 players, 20 teams, 4 positions; drops 0-minute
   "never-features")
3. ✅ `greedy_teams(pool, n_teams, seed)` + `search_space_size(pool)`:
   - **Search space on the filtered pool: ~10^13.3 OOM** — the explicit size
     before optimization (budget + club caps shrink the feasible set below
     this bound)
   - Greedy with per-team "missed star" jitter → a basket of valid squads
4. ✅ **Reusable harness** (`fpl/team/harness.py`): fixed skeleton score →
   filter → enumerate → value; the enumeration and value strategies are
   injected from a REGISTRY (greedy / h2h today; MCMC / utility later). This
   is deliberately NOT tied to one algorithm — swap a stage to experiment.
   `scripts/search_teams.py` runs it end-to-end.
5. ✅ `simulate_h2h` + `weaknesses`: per-team win_ratio/avg_edge over the
   round-robin field; worst-GW and star-dependence exposure per squad.

Each step lands with black-box tests (106 total passing).

## Status

All stages of this branch done. The harness exposes a real limitation worth
eyeing: the greedy basket contains near-clones (identical per-GW totals), so
H2H can't rank them — better enumeration diversity is the lever, and the
harness lets us measure it.

## Distributional layer (mean averaging ≠ value)

Mean-only H2H makes teams look identical (both win_ratio AND avg_edge equal
for near-clone squads). The journal's own critique: expected points average
over the very variance (contextual factors) that separates teams.

`src/fpl/dist.py` + `src/fpl/team/distribution.py` add distribution to the
forecast:
- residual CDFs from held-out model errors, **binned by position** (forwards
  are far more volatile than defenders); `fit_residual_cdfs`
- per-player-GW CDF = point prediction + position's residual quantiles
  (`distributional_forecast`), stored as a 9-quantile struct per row —
  a cheap CDF estimate; t-digest would be a drop-in if merging ever needs it
- `scripts/fit_dist.py` writes `data/processed/dist_{season}.parquet`
- `simulate_squad_distributions` MC-samples squad GW totals from player CDFs;
  `simulate_h2h_dist` (registry `value.h2h_dist`) turns those into per-team
  win_ratio / exp_wins / avg_edge

Result on real data: distributional H2H separates **all 20 squads** into
distinct win_ratios (0.49–0.56) where mean-only H2H collapsed to ~5 identical
groups. Higher-variance squads (forwards-heavy, tail-dependent) are now
distinguishable — the risk/return axis from the journal.

Storage note: the per-player-GW CDF is a struct column (quantile vector). If
we ever need to *merge* many distributions compactly (e.g. squad-total CDF
without MC), a t-digest is the right drop-in, but the current vector is
sufficient for MC sampling at these volumes.

## Open decisions

- H2H value metric — win_ratio today; variance/utility (risk-return from the
  journal) to add as a `value` registry entry
- Better basket diversity: swap in an MCMC/beam enumerator (REGISTRY entry)
- Captain / free-transfer mechanics (Stage 5) — need per-GW, not just
  window-sum, expected points (already kept in `per_gw` + now `dist`)