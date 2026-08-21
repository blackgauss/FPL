# Model Ensembles — Investigations

Status: **open** — being scaffolded; conclusions TBD as experiments land.

## Goal

Player models, one per position (`GKP`/`DEF`/`MID`/`FWD`), producing
per-player, per-GW predictions of the form:

    prediction(player) = global(position model) + correction(player)

The correction must absorb what a single global model cannot: a player's own
steady-level, team context, and momentum. We are explicitly NOT modeling each
player as I.I.D.

## Why I.I.D. is the wrong assumption

Team search sums per-player expected totals into a squad score. If the per-player
forecasts are independent but the *realized* outcomes are correlated within a
squad, the correlated forecast errors accumulate. Two failure modes we must
design against:

- **Destructive interference / catastrophic collapse**: a squad loaded with
  players from the same congested fixture cluster (or same team) double-counts
  a shared risk/upside. When the shared factor is a realized miss, all those
  players miss together and the squad collapses below the sum of its parts.
- **False diversification**: mean-only models treat +2 across two teammates as
  worth more than +4 on one star; but if teammate returns co-move, the +2/+2 is
  closer to a single exposure.

Fixes must account for **matchup** (per-position opponent/venue strength) and
**within-squad covariance**, and should capitalize on **team momentum**.

## Key design questions (to resolve in analysis/)

1. **Which fields correlate most strongly among players — and across teammates?**
   - Within a player: features in the current store (`team_code`, `position`,
     `now_cost`, `ep_next`, venue, `opponent_elo`, `home_elo`, `pts_avg_3/5`,
     `prev_points`) and their residual correlation with realized points.
   - Across teammates / same squad: co-movement of *residuals* (actual minus
     predicted) among players sharing a club, a fixture, or an opponent.
2. **Where does the correlation come from?**
   - Shared fixture / opponent strength (same match, both teams affected).
   - Team-level latent (attack/defence form, momentum) moving all teammates.
   - Position cross-effects (strong midfield suppresses defender upside).
3. **How to model it** (candidates, not choices yet):
   - Hierarchical / random-effects: per-player and per-team offsets (EBLUP-style),
     i.e. "global + correction".
   - Team-level latent factor shared across teammates (low-rank / factor model),
     giving an explicit covariance to the squad-scoring step.
   - Per-position matchup terms (e.g., additive opponent/venue × position
     interaction) rather than one global row.
4. **Evaluation discipline (the gym is the judge).** Model variants are run
   through `fpl.gym.Eval` on ACTUAL gameweeks and compared as *paired toggles*:
   fix everything, vary one thing (IID baseline vs covariance-aware; global vs
   per-player correction; momentum on/off). The attributable delta of each
   change is the difference between paired runs — never a single gap.

## Experiment 1 — matchup feature A/B (holdout + gym)

`experiments/ab_matchup_model.py`; only difference is the feature set, same
data split / params / seed / squads / actuals: baseline `FEATURE_COLUMNS` vs
`+ opponent_team_code` (categorical).

- **Holdout (GW 31–38, MAE):** baseline 1.008 vs matchup 1.010 — tied.
  A bare opponent id adds nothing to per-player point prediction error.
- **Gym (fixed squad, fixed actuals, forecast-only toggle):** matchup lands a
  little closer to settled points — |gap| 86.6→85.7 and 105.7→103.0 over two
  squads (still ~3–4% under). `opponent_team_code` ranks 5th by gain (between
  `opponent_elo` and `home_elo`): the model uses it without it moving the
  needle.

**Read:** matchup effects are real (§5–6 diagnostics) but a single categorical
id is too sparse to express them, and it cannot touch within-squad covariance
(that lives in the selection/value layer). This motivates denser representation
(team attack/defence latents, position-gated interactions) — and, precisely, a
feature search rather than hand-picking them.

## Experiment 2 — team-flag covariance for squad evaluation

`experiments/ab_team_covariance.py`; leakage gate first (train gw ≤ 29, holdout
gw 31–38, nothing from the holdout touches any fit). Estimates from TRAIN
residuals: pooled per-player `σ²`, `α` (same-team residual covariance), `β`
(shared-fixture opponent covariance), then arbitrates via the gym with IID vs
covariance-aware squad variance on 48 hold-out squad-weeks.

- **Learned structure is sensible**: `α ≈ 0.042` (matches the §4 same-team
  correlation r≈0.054); `β ≈ 0.006`.
- **Gym arbitration (z = (actual − mean)/σ):** IID `mean|z| = 2.20, std(z) =
  2.42` vs covariance-aware `2.20 / 2.41`. The covariance term is barely
  detectable against the dominating problem.

**Read:** acknowledging within-squad covariance is correct but *second-order*.
The real calibration gap is the VARIANCE SCALE — hold-out prediction errors are
~2.4× the in-sample residual σ² (std(z) ≈ 2.4 vs 1). Priorities: (1) calibrate
per-player variance out-of-sample (scale σ² to hold-out), (2) keep the team-flag
covariance term for *selection* (penalising same-team stacks / pricing shared
fixtures) and for correlated MC sampling — it changes re-ranking and
diversification even where it barely moves a z-calibration number.

## Experiment 3 — match-state factor model (conditional independence)

`experiments/ab_match_state_factor.py`; leakage gate first. Tests the "cheat
code": `P(state) × ∏ P(Xᵢ | state)` with state = (team goals, conceded) from
TRAIN matches, permissions empirical per-position conditional tables
(own-goals × conceded bins, plus a fresh state per draw so the shared factor
varies). IID vs factor on 32 hold-out squad-weeks, 200 draws each.

- **Gym arbitration:** factor is WORSE — mean|z| 6.62 / std(z) 3.59 vs IID
  2.46 / 2.25. Conditioning on the goals state *sharpens* the predictive
  distribution below reality; actual squad totals are dominated by heavy rare
  tails (goals/cards/bonus) the final-score state barely carries.
- **Co-movement reproduction:** same-team simulated r ≈ −0.005 (vs real ~0.05);
  the goals factor does NOT reproduce our measured coupling. (A subtle trap
  caught by the gym: the state must be re-sampled per draw or the covariance
  vanishes entirely.)

**Read:** the factor *approach* is the right cheat (no 22-dim/copula), but the
*state* must actually carry the match variance — final goals alone is too thin
and conditioning on it rushes past reality. The co-movement we measured
(+0.42 own-team / −0.57 in-match) needs a richer shared state (team/opponent
strength latents, form, defensive structure) and, before anything, per-player
variance calibration (Experiment 2's dominant gap). Roadmap: calibrate σ² →
stronger factor state / feature search → correlated sampling into the search.

## Experiment 4 — variance calibration (single scalar)

`experiments/ab_variance_calibration.py`; leakage gate first. Tests the
highest-value fix from Exps 2–3: is the over-confidence mostly a variance
SCALE problem? A single scalar k is fitted on a validation window (gw 28..30)
and applied to the sampled squad-total variance; the arbiter is the untouched
gw 31..38 window. Same marginals/mean for both arms — only variance handling
differs.

- Validation (8 squad-weeks): calibration factor `k = 1.198`.
- Gym arbitration (32 squad-weeks, gw 31..38): raw `mean|z| = 2.55, std(z) =
  1.64` vs calibrated `mean|z| = 2.12, std(z) = 1.37`.

**Read:** calibration works — a single scalar shrinks both metrics toward 1
(std(z) 1.64 → 1.37, ~17%), confirming variance SCALE is the head of the
house. It does not close the gap: the remainder is the mean drifting off on
spike weeks plus heavy tails (rare 10+ point events we under-represent). So
the calibration fix is cheap and worth taking; after it, the residual gap is
fat/tail + mean fidelity, not scale. (k≈1.2 here — small, because the
empirical-marginal sampler's variance is already nearer reality than the
in-sample residual σ² used in Exp 2.)

**Performance vs baseline (gw 31–38 arbiter):** calibration improves every
predictive-distribution metric — mean|z| 2.55→2.12, std(z) 1.64→1.37, 1σ
coverage 0.19→0.26, 2σ coverage 0.37→0.48, CRPS 22.5→21.7. Confirmed a real
model-performance win, leakage-safe (k from gw 28–30 only).

## Experiment 5 — position/prediction-bucket tail calibration

`experiments/ab_tail_calibration.py`; same leakage discipline (fit mean model
through source GW 27, residual quantiles on GWs 28–30, arbitrate GWs 31–38).
The baseline adds one global residual quantile to the mean; the variant uses
position × predicted-level residual quantiles.

- q50 pinball: **0.495 → 0.475**
- q90 pinball: **0.446 → 0.377**
- q95 pinball: **0.328 → 0.274**
- Coverage moved slightly down (q90 **0.902 → 0.892**, q95 **0.953 →
  0.941**), so the improvement is proper tail loss, not yet perfectly
  calibrated coverage.

**Read:** stratifying residual tails by position and predicted level improves
the proper scoring rule, especially q90/q95, without changing the mean model.
Coverage needs a second calibration pass (or conformal adjustment); this is a
useful tail layer before event-specific models.

**Decision-surface rerun:** all-player metrics are not the primary objective,
so the experiment also ranks test rows by pre-GW prediction and reports the
top 10% (plus top 10% within each position). On overall top-decile rows, q90
pinball improves **1.078→0.951** and coverage improves **0.754→0.844** toward
0.90. q95 pinball is nearly unchanged (**0.634→0.630**) while coverage
improves **0.863→0.896**; position-balanced top-decile q90 pinball improves
**1.041→0.947**. The important q90 region improves clearly; q95 remains a
separate extreme-tail calibration problem.

## Experiment 6 — integrated factor sampling A/B

`experiments/ab_factor_sampling.py` uses the existing `dist_2025-2026.parquet`
marginals and `fpl.gym.Eval`: IID samples each player's quantiles independently;
the factor arm preserves those marginals but adds learned shared team and
signed fixture shocks in Gaussian-copula space. The rhos are learned from
pre-holdout residuals only (`team=0.0144`, `fixture=0.0028`); 12 holdout
squad-weeks and 300 draws per week.

- IID: `mean|z| = 3.354`, `std(z) = 1.986`
- factor: `mean|z| = 3.334`, `std(z) = 2.002`

**Read:** the integrated factor has a tiny mean|z| improvement (~0.6%) but
worsens dispersion calibration slightly. It is not a production win yet.
The marginals are preserved and the gym path is correct, but the learned
shared shocks are too small / too weakly identified to improve uncertainty on
12 squad-weeks. Keep this as an experimental seam; do not wire it into team
ranking until it beats IID on a larger, rolling holdout.

## Experiment 7 — first-team regularity features

`experiments/ab_first_team_regularity.py` tests the corrected availability
hypothesis: retain all players in training, but add leakage-safe trailing
role context (`appear_rate_5`, `start_rate_5`, `minutes_share_5`) calculated
from GWs through the source row. A first-team regular is a role cohort, not
simply someone who appeared in the target GW.

- Baseline vs role features on all holdout rows: MAE **0.986 → 0.991**.
- On pre-GW first-team regulars: MAE **2.270 → 2.340**.
- On non-regulars: MAE **0.537 → 0.519**.

**Read:** this first three-feature definition is not a win for the relevant
regular cohort. Filtering training data to regulars was also slightly worse
in the earlier check. Keep all players; investigate a better role/selection
representation (starts, expected lineup role, team depth, and shrinkage)
before adopting it.

## Velocity notes (experiment loop)

- **Model-free where possible**: experiment 3 needs no model fit — empirical
  marginals/tables. Squads reuse the stored `points_lgbm.txt` (no training).
- **Artifact caching**: train-derived tables cached to `experiments/artifacts/`
  (keyed by season+window) so re-runs skip rebuild. Stale-cache risk is kept
  to experiments (investigations), not inference code.
- deltas/gates: leakage gate first, deterministic seeds (`SEED=42`), small
  `n_teams` (greedy feasibility), paired-toggles only. Gym arbitration output
  goes straight into this file.

## Work plan

1. **Scaffold** (this branch): `analysis/` + `experiments/` + this doc.
2. **Diagnostics** — `analysis/ModelEnsemblesInvestigation.ipynb` (build:
   `python scripts/build_investigation.py`): position baselines, driver-feature
   correlation with realized points, driver collinearity, and the **in-squad
   co-movement check** quantifying the I.I.D. violation (2025-26: same-team
   r ≈ 0.054 vs cross-team r ≈ 0.003, ~17× — teammates are NOT independent).
   Drives design questions 1–2; record follow-ups here.
3. **Experiments** (`experiments/`): runnable comparisons, one per question,
   wired through the gym; each stores its paired toggles + observability output.
4. **Model** : position models + per-player correction landing in
   `src/fpl/model/`, evaluated via the gym before any team-search integration.

## Infra requirements

- The feature store / `TrainingData` already key rows by `player_code` + `gw` —
  enough for correlation tables and hierarchical fits.
- `analysis/` = investigation notes + computed tables (language-agnostic, YAML
  config, CSV/parquet outputs).
- `experiments/` = runnable experiment code + recorded evals; conventional
  commits; deterministic seeds so toggles are reproducible.
- Evaluation always goes through `fpl.gym.Eval` (real rules, observability) so a
  "better model" claim is a paired-gym delta, not a training loss.

## Open questions / risks

- Data volume per position/season for player-specific corrections (shrinkage
  needed for low-data players).
- Momentum is confounded with schedule strength — need opponent-adjusted form.
- Covariance estimates are sparse per squad-fix; priors / pooling required.
