# Tree Models for Point Forecasting

## Goal

Forecast a player's next-GW FPL points with tree models (LightGBM), built on the
player-GW feature store from `data-tools`. The forecast feeds later team-selection
(taking averages of future GWs, not just GW1).

## Data

- **Features**: `data/processed/features_{season}.parquet` (produced by
  `scripts/features.py`): per (player_id, gw) — team, opponent, venue, elos,
  prev_points, pts_avg_3/5, total_points, next_points.
- **Augmented at assemble time**: position (players), now_cost/ep_next (gw_stats).
- **Seasons**: 2024-25 (26,052 rows) + 2025-26 (28,297 rows). 2026-27 has no
  played matches yet → 0 rows.

## Correctness requirements (why this setup is safe)

- **No leakage**: the feature store shifts the target per player; the model split
  is *by game week* (train GW 1..30, eval GW 31..38), so a held-out GW is never
  in the training window. `gw` is NOT a feature (a time index would leak).
- **Rows with no PL match** (opponent/venue null — 409 in 2025-26) are kept with
  an explicit `had_match` flag and zero-filled venue/elo: a blank GW is itself
  signal (0 points), not missing data.

## Pipeline

```
scripts/ingest.py  -> parquet dataset
scripts/features.py -> features_{season}.parquet
src/fpl/model/train.py  assemble(features, players, gw_stats, season) -> TrainingData
src/fpl/model/eval.py   mae/rmse + baselines
scripts/train_tree.py   train + held-out eval
```

## Held-out baseline (GW 31..38 of 2025-26, 5802 rows)

| Model | MAE | RMSE |
|---|---|---|
| LightGBM | 0.959 | 1.957 |
| FPL ep_next | 1.038 | 2.159 |
| persist last GW | 1.240 | 2.760 |
| constant mean | 1.547 | 2.376 |

LightGBM beats FPL's own forecast and both trivial baselines on the held-out
windows. Top features: ep_next, now_cost, pts_avg_5, pts_avg_3.

## Notable bug fixed during this work

The legacy 2024-25 `playerstats.csv` `total_points` is **cumulative** per player
(GW 14 → 24 → 41...), unlike the modern discrete per-GW files. The legacy loader
now derives discrete points from `event_points` (verified: their per-GW sum equals
the season-end cumulative total). Caught by the model producing nonsense targets
(mean ~23, max 344); a contract test guards the discrete-vs-cumulative trap.

## Next steps

- Per-position models (FWD/MID/DEF/GKP benefit differently from features)
- Predict *future GWs* for team value (bag/horizon aggregation)
- Distribution not just mean (needed for the journal's risk/return framing)