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
- **Stable identity across seasons**: `player_id` is a season-local FPL element
  id that gets REUSED for a different player each season — 803 of 804 shared
  IDs collide between 2024-25 and 2025-26. The feature store therefore carries
  `player_code` (the stable cross-season key); all player-metadata joins in
  `assemble` use `player_code`, and per-season players frames are loaded
  separately. Leakage validation (`src/fpl/model/leakage.py`) runs before any
  training and would fail a pipeline that joined on `player_id`.

## Leakage validation (journal "Data" section)

`src/fpl/model/leakage.py` encodes the guarantees, each a pure check returning
violations; `validate()` raises before a model fits if any fail:
- **identity**: features carry `player_code`; no row keyed by a `player_id`
  whose code resolves to a different player;
- **causality**: `next_points` equals the following GW's `total_points` for the
  same player (statistical check on the target shift);
- **split**: train max GW < test min GW.

The checks caught a real contamination during this fork: the original version
joined 2025-26 player metadata onto 2024-25 rows via `player_id`, silently
attaching the wrong players' positions/prices to every older-season row.

## Pipeline

```
scripts/ingest.py  -> parquet dataset
scripts/features.py -> features_{season}.parquet
src/fpl/model/train.py  assemble(features, players, gw_stats, season,
                                  feature_columns=subset) -> TrainingData
src/fpl/model/eval.py   mae/rmse + baselines
scripts/train_tree.py   train + held-out eval (single model)
scripts/run_experiments.py  fit multiple candidates on the same held-out GWs
src/fpl/model/leakage.py    pre-training leakage gates
```

## Rapid experiments

`config/experiments.yaml` declares candidates (model, params, feature subset);
`scripts/run_experiments.py` scores them all on the shared held-out window
(GW 31..38 of 2025-26). The harness rejects a leaky split (fit max >= test min)
and the registry (`lgbm`, `hist_gb`, `ridge`) makes swapping estimators a dict
change. Held-out results (GW31..38, all comparable; `features` reflects the actual
assembled set):

| model | MAE | RMSE | features |
|---|---|---|---|
| hist_gb | 0.942 | 1.924 | ALL |
| lgbm_all | 0.951 | 1.931 | ALL |
| lgbm_basic | 1.058 | 2.053 | position, now_cost, pts_avg_3/5 |
| ridge | 1.068 | 1.966 | ALL |
| lgbm_no_ep | 1.069 | 2.069 | ALL minus ep_next |

Reading: boosting > linear; `ep_next` is load-bearing (removing it costs
~+0.12 MAE). Note `assemble(feature_columns=...)` ablate feature sets, and the
NaN guard means a null `team_code` (legacy seasons, no team_history) is
backfilled from `players.team_code`.

# Pipeline

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

## Serving: expected points for a player collection + GW

`src/fpl/model/inference.py` turns a trained model into a claim: given a list
of `player_code`s and a gameweek G, report expected points. Row semantics: the
feature-store row at gw=k predicts points in gw=k+1, so GW G uses rows where
gw = G-1. `scripts/train_tree.py` saves the booster to
`data/processed/points_lgbm.txt`; `scripts/predict.py` loads and serves it.

**Model-family independence:** the tooling is estimator-agnostic. Serving only
requires `model.predict(X)`; persistence dispatches on filename suffix via
`SERIALIZERS` (`.txt` lightgbm, `.pkl`/`.joblib` any pickleable estimator), so
adding a new model family is one registry entry — no change to leakage gates,
the experiment harness, or `expected_points`. Sklearn estimators already
round-trip through inference in the tests.

```
python scripts/train_tree.py                     # writes the model
python scripts/predict.py --season 2025-2026 --gw 31
```

Example (2025-26 GW31, expected vs actual): Haaland 1.99↔2, Palme 2.79↔4,
Gabriel 2.63↔4, Foden 1.95↔1, Bowen 3.65↔14. Per-GW noise is high — horizon
aggregation (mean of next 3-5 GWs) is the intended use for team value.

## ep_next data artifact (fixed)

FPL-Core sometimes scrapes `ep_next` as 0.0 for available players whose
`ep_this > 0` (910 rows in 2025-26, concentrated in later GWs — i.e. the held
out window). `load_gw_stats_csv` now repairs those: `ep_next = ep_this` when
the player is available and `ep_this > 0`. This removed the systematic
star-under-prediction (Haaland -0.23 -> 1.99 expected).