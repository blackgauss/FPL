# Week 1 Plan: Data Prep + First Team

## Scope

Data preparation pipeline (Polars) + transparent scorer + constrained optimizer to produce a GW1 team in 2 days.

## Data Sources

- **FPL API** (`bootstrap-static`, `fixtures`): live prices, availability, teams, expected metrics
- **FPL-Core-Insights**: historical discrete per-GW stats (2025-26 + 2026-27)
- **Vaastav**: deferred post-GW1

## ID Linkage

- Players: `elements.code` (stable cross-season), not `id`
- Teams: `teams.code` (stable), not `id` (1-20 volatile)

## Two Outputs

1. `data/processed/train.parquet` — historical training set for ML (2025-26 discrete GWs, target `next_GW total_points` via `shift(-1)`)
2. `data/processed/inference_gw1.parquet` — one row per active 2026-27 player: carryover 2025-26 rolling features (minutes, xG, xA, bps) via `code` join + live bootstrap (`now_cost`, `status`, `chance_of_playing_*`, `ep_next`, `news`)

## Module Layout

```
config/data.yaml
src/fpl/
  data/{config,loaders,clean,features,dataset}.py
  scoring/expected_points.py
  optim/squad.py
  cli.py
```

## Scorer

Weighted blend (weights in `data.yaml`):
- `ep_next`
- carryover `form`
- rolling `xGI`
- `minutes_reliability` (last-season starts)
- `fixture_difficulty` (FPL `strength_overall_*`)
- `availability_penalty` (`status != 'a'` or `chance < 75%`)

No ML — transparent baseline to beat.

## Optimizer

Greedy fill + local swap under constraints:
- 15 players, £100m, max 3/club
- 2 GKP, 5 DEF, 5 MID, 3 FWD
- Starting XI: 1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD
- Captain (x2) + vice + bench order

## Day-by-Day

### Day 1

- AM: `pyproject.toml` (polars, pyyaml, requests), `.gitignore`, `config/data.yaml`, skeleton
- AM: bootstrap + fixtures snapshotter (timestamped), FPL-Core loaders (`players`, `teams`, `player_gameweek_stats`)
- PM: `clean.py` (cast string numerics, normalize positions, filter `removed`), `features.py` (rolling 3/5 over `code`, opponent strength, target shift)
- PM: `train.parquet` output + basic validation (row counts, null %, ID orphans)
- EVE: hybrid `inference_gw1.parquet` (carryover + live bootstrap join)

### Day 2

- AM: scorer (`expected_points.py`)
- AM: optimizer (`squad.py`)
- PM: run `select` → GW1 team, sanity check (injuries, starters, set pieces)
- PM: optional backtest on 2025-26 GWs
- EVE: journal entry + README with run instructions

## Explicitly Cut

- Vaastav integration
- `playermatchstats` deep join
- Full `validate.py` test suite
- JAX model
- Per-90 normalization (everywhere)
- Elo deep integration (use `teams.csv.elo` as-is only if trivial)

## Escape Hatches

- If Day 1 PM runs over: fall back to live FPL API only for GW1 features (drop carryover), re-add post-GW1
- FPL-Core 2026-27 GW1 data won't exist until after GW1: fine, uses 2025-26 carryover + live bootstrap
- Greedy optimizer may miss global optima: acceptable for GW1, upgrade to LP later
