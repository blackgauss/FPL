# Rapid Experimentation Workflow

How to try a new modeling idea and compare it fairly, in minutes. This is the
"rapid experimentation loop" from `docs/journal/Week1.md`.

## The loop (30 seconds per idea)

1. **Declare** a candidate in `config/experiments.yaml` (model, params, feature
   subset). No code changes.
2. **Run** `python scripts/run_experiments.py`.
3. **Read** the table — every candidate scores on the *same* held-out window
   (GW 31..38 of 2025-26), so MAE/RMSE are directly comparable.

## Prerequisites (one-time)

```bash
uv venv && uv pip install -e ".[dev,notebook]"
# fetch/refresh the FPL-Core clone if not present
git clone --depth 1 https://github.com/olbauday/FPL-Core-Insights.git external/fpl_core
# build the dataset -> feature store -> trained model (idempotent, all of these)
python scripts/ingest.py --config config/data.yaml
python scripts/features.py --config config/data.yaml
python scripts/train_tree.py        # writes data/processed/points_lgbm.txt
```

`git pull` inside `external/fpl_core` + re-run ingest/features whenever the
source data updates (FPL-Core refreshes twice daily).

## Experiment config

`config/experiments.yaml`:

```yaml
fit_gw_max: 30      # train on GWs <= this
test_gw_min: 31     # score on GWs >= this (shared held-out window)
seasons: ["2024-2025", "2025-2026"]

experiments:
  lgbm_all:            # name -> shown in results table
    model: lgbm        # must exist in src/fpl/model/experiment.py REGISTRY
    # params: {learning_rate: 0.1}       # optional estimator params
    # features: [position, now_cost, pts_avg_3]   # optional subset
    note: free-text
```

- To **switch estimator**: change `model:` to any REGISTRY key.
- To **ablate a feature**: list the subset under `features:` (curated
  categoricals are re-encoded automatically).
- To **test the whole window**: nothing — `fit_gw_max`/`test_gw_min` are shared
  by every experiment; you can tighten them by editing the top-level keys.

## Adding a new model family

`src/fpl/model/experiment.py` — the registry maps a name to a factory that
returns a `make(X, y, categorical) -> predict` callable:

```python
def _poisson(params: dict):
    def make(X, y, categorical=None):
        from sklearn.linear_model import PoissonRegressor
        return PoissonRegressor(**params).fit(X, y).predict
    return make

REGISTRY["poisson"] = _poisson
```

Serving also needs persistence — add the format in
`src/fpl/model/inference.py` `SERIALIZERS` (e.g. `".pkl": (_save_pickle,
_load_pickle)`); sharing the pattern is one dict entry. Nothing else changes.

## Serving a candidate

Train and persist, then ask for expected points for a player collection + GW:

```bash
python scripts/train_tree.py
python scripts/predict.py --season 2025-2026 --gw 31
```

`expected_points(td, model, gw=G, players, code_filter=[...])` predicts on rows
where `gw == G-1` (feature-row semantics: row gw=k predicts points in k+1).
`scripts/predict.py` has a `CODES` list to edit for a target squad.

## Guardrails (what "fair" means)

- **No leakage**: the split is by gameweek; the harness rejects
  `fit_gw_max >= test_gw_min`. Held-out GWs are never in any fit.
- **Identity**: player metadata joins on stable `player_code` (player_id is
  season-local and reused). `scripts/train_tree.py` runs the leakage gates
  before fitting.
- **Same slice**: this is why results are comparable — every row scored the
  same 5802 player-GW test rows.

## Interpreting results

Zero-inflated FPL points mean overall MAE (~1.0) is dominated by 0-point bench
rows. Whether a candidate is *better for team selection* depends on the tail —
pair the harness with per-position breakdowns and the horizon aggregation
(mean expected points over the next 3-5 GWs) before deciding.

## Files to touch for a new idea

| Want to... | Edit |
|---|---|
| Try a new estimator | `experiment.py` REGISTRY + `inference.py` SERIALIZERS |
| Ablate/add a feature | `assemble(...feature_columns=...)` in `train.py` or the config |
| New feature engineering | `src/fpl/data/features.py` → rebuild features |
| Change the held-out window | `config/experiments.yaml` top-level keys |
| Evaluate per position | (next) drill into `expected_points` report by `position` |