# Performance diagnostics

Where the time goes in this repo's reference pipeline, the inefficiencies
found, and what has been addressed.

## Method & reproducibility

```bash
python scripts/profile_pipeline.py
```

Each stage is cProfiled into `experiments/artifacts/profile/*.json`
(gitignored, regenerated on demand); the per-scope wall-clock is printed.

## Current numbers (machine-load dependent)

| Scope            | wall (typical) |
|------------------|----------------|
| training / fit   | ~7-18s         |
| run_experiment   | ~9s (mostly the fit) |
| inference/predict| ~0.1s          |
| data prep        | ~0.2s          |
| candidate search | ~0.5s          |

## Inefficiencies found and addressed

1. **The inference profile was re-fitting.** The diagnostic's "inference"
   stage trained LightGBM a third time just to predict on the holdout. Fixed:
   fit once, profile predict-only -> that stage went from ~9s to ~0.1s, and
   the tooling no longer hides the real inference cost.

2. **No reuse across experiments sharing a config.** Every `run_experiment`
   re-loaded the feature store and re-fitted the same model, even when
   experiments shared seasons/features/splits. Added
   `fpl.experiments.cache` (in-process memo for loaded TrainingData and fitted
   predictors, keyed by config) with full reset + call/hit counters, wired
   into `run_experiment`. This is a per-process cache; cross-invocation reuse
   is listed below.

3. **Committed profile noise.** Profile JSON (up to ~35k lines) was being
   committed. Made `experiments/artifacts/profile/` gitignored and
   reproducible.

## Where the time is actually spent (why)

`pairwise_concordance` was a ~6s O(n²) hotspot earlier; vectorized. Beyond
that, almost all wall-clock is LightGBM training (dataset construction +
boosting on ~46k rows); every other stage is sub-second. Training is the
lever, not IO or metrics.

## Recommended next actions (not yet done)

| Action | Expected effect | Cost |
|---|---|---|
| Fewer/adaptive boosting rounds for exploration runs | Linear cut in training | trivial |
| Freeze categoricals via int8 numpy (avoid per-fit value-counts) | Cuts `lgb.Dataset` build | low |
| Cross-process model/training disk cache keyed by config-hash | Kills refit across CLI invocations | medium |
| Parallel candidate/experiment arms (drop per-arm refit) | Cuts wall for multi-arm runs | medium |
| Switch recommender to warm-start/keep baseline | Long-term | high |

## Rules of thumb

- Profile answers *where time goes*, never *which model is better*.
- Compare within a run or across runs on the same config/load; absolute walls
  vary.
- cProfile adds ~20-50% overhead; use times directionally.