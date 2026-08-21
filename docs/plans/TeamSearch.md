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
   - **Search space on the filtered pool: 5.1 × 10¹² teams, log10 = 12.71
     OOM** — the explicit size before optimization (budget + club caps shrink
     the feasible set below this bound)
   - Greedy with per-team "missed star" jitter → a basket of 16 distinct
     valid squads (real data), all 15 players, all ≤ £100m
4. ⏳ `simulate_h2h(basket, gwhorizon) -> value + weaknesses report`

Each step lands with black-box tests (101 total passing).

## Status

Stages 1-3 done on this branch (real 2025-26 GW31-33 window). Stage 4 (H2H +
value/weaknesses) and Stage 5 (captain/transfers) are next.

## Open decisions

- H2H value metric (win% vs field; expected edge; tail risk)
- Whether to keep basket = top teams by enumeration + a few diverse roll-ups
- Search-space OOM (12.7) is with the current filter knobs; widening the pool
  grows it ~log-linearly — the tractability knob is `top_k_per_position` + the
  enumeration method (greedy now; MCMC/beam later if we want to search wider)