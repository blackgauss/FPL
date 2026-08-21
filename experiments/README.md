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

Velocity conventions (see `analysis/Investigations.md` → Velocity notes):
run a *leakage gate + deterministic seed* first, prefer model-free arms and
reuse the stored model for squads (no retraining), cache train-derived tables
in `experiments/artifacts/` (keyed by season+window), keep `n_teams` small,
and always grade paired toggles.

For standard player-model comparisons, use the declared YAML registry:

```bash
python scripts/run_experiments.py \
  --config config/experiments.yaml \
  --output experiments/artifacts/model_experiments.json
```

The first migrated A/B is also declared in `config/experiments_matchup.yaml`:

```bash
python scripts/run_experiments.py \
  --config config/experiments_matchup.yaml \
  --output experiments/artifacts/matchup.json
```

The runner validates leakage before fitting, reuses assembled training data
for duplicate feature sets, and writes a `status: complete` JSON artifact only
after every declared experiment finishes. Printed tables are for humans;
the JSON artifact is the reproducible result contract.

Findings and conclusions are recorded in `analysis/Investigations.md`, and
only experiment code + recorded outputs live here.

Expected contents (as they land):

- position-model variants (global vs global + per-player correction),
- covariance / team-latent experiments (IID baseline vs covariance-aware),
- momentum feature ablations,
- matchup × position interaction checks.

Findings and conclusions are recorded in `analysis/Investigations.md`, and
only experiment *code + recorded outputs* live here.
