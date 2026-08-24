# Experiments

Runnable model/variant experiments for the investigations in
[`analysis/Investigations.md`](../analysis/Investigations.md).

Each experiment lives in its own script/notebook and follows the repo
conventions: synthetic or tracked data inputs, deterministic seeds, and —
where model quality is being judged — evaluation through the **gym**
(`fpl.gym.Eval`, paired toggles), never a bare training-loss number.

Ran so far:

- `ab_matchup_model.py` — matchup-feature A/B (baseline vs `+ opponent_team_code`
  categorical); holdout MAE + gym forecast-vs-actual. Small but real gain on
  calibration; raw ids are informative but weak.
- `ab_team_covariance.py` — structural same-team/opponent covariance model
  (low-rank, learned from train residuals) vs IID; gym arbitrated via squad-
  week z-calibration. Covariance is second-order; the gap is variance scale.
- `ab_match_state_factor.py` — match-state factor (conditional independence),
  model-free + artifact-cached; vs IID. A weak state (final goals) loses:
  conditioning sharpens past reality and fails to reproduce same-team r.
- `ab_tail_calibration.py` — global residual quantiles vs position + predicted-
  level residual quantiles. Tests q50/q90/q95 pinball loss and coverage on a
  leakage-safe holdout; this is tail calibration, not mean accuracy.
- `ab_factor_sampling.py` — existing marginal quantiles with IID sampling vs
  learned shared team/fixture shocks, integrated through `fpl.gym.Eval`.
  Current result is not a production win (tiny mean|z| gain, slightly worse
  dispersion); keep it experimental until larger rolling holdouts improve it.
- `ab_first_team_regularity.py` — all-player training vs adding leakage-safe
  trailing appearance/start/minutes features. The first definition slightly
  improves overall MAE but worsens the regular-player cohort; not a win yet.
- `ab_position_shrinkage.py` — global model plus validation-fitted position
  corrections. Independent position models overfit; shrinkage keeps global
  strength while allowing position-specific behavior.

Velocity conventions (see `analysis/Investigations.md` → Velocity notes):
run a *leakage gate + deterministic seed* first, prefer model-free arms and
reuse the stored model for squads (no retraining), cache train-derived tables
in `experiments/artifacts/` (keyed by season+window), keep `n_teams` small,
and always grade paired toggles.

For standard player-model comparisons, use the declared YAML registry:

```bash
python -m fpl.stages.experiments \
  --config config/experiments.yaml \
  --output experiments/artifacts/model_experiments.json
```

The first migrated A/B is also declared in `config/experiments_matchup.yaml`:

```bash
python -m fpl.stages.experiments \
  --config config/experiments_matchup.yaml \
  --output experiments/artifacts/matchup.json
```

The runner validates leakage before fitting, reuses assembled training data
for duplicate feature sets, and writes a `status: complete` JSON artifact only
after every declared experiment finishes. Printed tables are for humans;
the JSON artifact is the reproducible result contract.

## How to add an experiment

Declare it as YAML — no per-experiment code. A point experiment:

```yaml
seasons: ["2024-2025", "2025-2026"]
split:                       # source-GW windows (target = source + 1)
  fit_gw_max: 30
  cal_start: 31
  cal_end: 30                # empty calibration (no curation slice)
  test_start: 31
  test_end: null
experiments:
  my_exp:
    model: lgbm              # registry: lgbm | hist_gb | ridge
    features: [position, now_cost, pts_avg_3, pts_avg_5]
    categorical_columns: [position]
```

An experiment with a gym arm (same candidate squads, replay + observability):

```yaml
experiments:
  my_exp_gym:
    model: lgbm              # gym requires the default feature set
    gym:
      season: "2025-2026"
      gw_start: 31
      gw_end: 33
      n_teams: 4
      seed: 1
      top: 2
```

Run it and compare:

```bash
python -m fpl.stages.experiments --config config/my_exp.yaml --output experiments/artifacts/my_exp.json
python scripts/compare_experiments.py experiments/artifacts/A.json experiments/artifacts/B.json
```

The runner applies the leakage gate, cohorts (all / top10 / top10-per-position),
point metrics, and optional gym metrics, and writes `complete` — or `failed`
with a traceback if anything broke. A `play_prob(squad, gw)` hook (passed in
`fpl.experiments.run`) switches the gym to predicted settlement.

Every point result reports model performance as two separable stages:

- **ranking** (`rank@...`): scale-free ordering quality — `spearman_rho`,
  `topk_hit_rate` (actual top decile found in the predicted top decile per GW),
  `pairwise_concordance`.
- **calibration** (`cal@...`): magnitude trust — `mae`, `rmse`, `ece`, the
  `actual ~ slope*pred + intercept` line (slope 1 / intercept 0 = calibrated),
  and `variance_ratio`.

Gym observability is a single canonical schema per candidate run
(`EvalResult.observability()`): `totals` + per-week `weeks` rows, emitted
verbatim into the artifact and surfaced by `compare_experiments.py`.

## Infra profiling

Focused, project-specific guidance on how to profile this codebase is in
[`docs/Profiling.md`](../docs/Profiling.md) — quickstart, the two APIs, the
JSON report schema, and which workflow to use when.

`fpl.profiling` makes any workload observable with stdlib cProfile:

- `time_phases(...)` - labelled wall-clock per stage.
- `profile_call(fn, name=..., out_dir=...)` - cProfile a callable, persist
  `*.prof` (git-ignored) and a structured `*.json` (per-function + per-module)
  an agent can read directly.

```bash
python scripts/profile_pipeline.py   # data prep / training / inference / harness / candidates
```

It prints a per-scope wall-clock table and function/module summaries, and
writes generated `experiments/artifacts/profile/*.json` (gitignored,
reproducible — regenerate on demand). Reference pipeline: training
dominates; data prep and candidate search are small.

Findings and conclusions are recorded in `analysis/Investigations.md`, and
only experiment code + recorded outputs live here.

Expected contents (as they land):

- position-model variants (global vs global + per-player correction),
- covariance / team-latent experiments (IID baseline vs covariance-aware),
- momentum feature ablations,
- matchup × position interaction checks.

Findings and conclusions are recorded in `analysis/Investigations.md`, and
only experiment *code + recorded outputs* live here.
