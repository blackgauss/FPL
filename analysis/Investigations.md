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

## Work plan

1. **Scaffold** (this branch): `analysis/` + `experiments/` + this doc.
2. **Diagnostics** (`analysis/`): correlation/co-movement tables from the
   feature store — driver features vs realized points; residual co-movement
   among squad-mates/fixture-sharers. Record findings here.
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