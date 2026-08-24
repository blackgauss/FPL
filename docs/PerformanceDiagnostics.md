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

## Training-time deep dive (is it expected?)

Reproduce: `python scripts/train_time_analysis.py` (writes
`experiments/artifacts/profile/train_time.json`, gitignored).

Measured decomposition, raw wall-clock (no profiler), reference config:

| component | time | note |
|---|---|---|
| assemble X/y | 0.016s | vstack of numpy matrices |
| lgb.Dataset build | ~0s | lazy — cost is inside train |
| **lgb.train 200 rounds** | **~6.5s** | 49,069 rows x 10 features, 63 leaves, MAE |
| per round | ~26ms | |
| without categoricals | ~5.9s | categoricals add ~10% |

**Verdict: expected.** ~6.5s for 200 rounds on ~49k x 10 is normal LightGBM;
it is the largest stage but not pathological. The earlier "13-35s" profile
numbers were inflated by cProfile overhead (~2x) plus machine load. The real,
addressable levers remain: fewer boosting rounds for exploration, and
cross-process fit/forecast caching (the in-process cache already exists).

## Memory / copies: already handled by the libraries

Moving data around less is mostly a solved problem at our layer:

- `polars.to_numpy()` returns a **non-owning view** (measured 0.45ms,
  `owndata=False`) — zero copy across the frame boundary.
- `numpy` ops vectorize; LightGBM consumes our ndarray buffer directly and
  does its own (optimized) histogram construction.
- What *we* were copying unnecessarily was recompute-generated, not
  memory-resident: the combined X/y `vstack` ran on every fit attempt, even
  on cache hits. Fixed — matrix assembly now lives inside the cached fitter,
  so a cache hit skips the stack AND the training. Forecast frames per
  (fit, window) are also cached (skip re-prediction) with call/hit counters.

Measured: assemble ~0.02s, vstack copy ~0.26ms for a few hundred rows — so
even the avoided copies are small; the win is skipping LightGBM itself.

## Rounds lever (profiled) — the cheap win

Reproduce: `python scripts/rounds_sensitivity.py` (writes
`experiments/artifacts/profile/rounds_sensitivity.json`, gitignored).

Sweep on the same split (learning_rate 0.05, 63 leaves); timings are
machine-load dependent (earlier clean run: ~26ms/round), so read the
QUALITY knee and scale time linearly:

| rounds | mae | top10_mae | spearman | rank_topk |
|---|---|---|---|---|
| 25 | 1.115 | 2.80 | **0.704** | 0.355 |
| 50 | 1.028 | 2.97 | **0.705** | 0.346 |
| **100** | **1.001** | 3.05 | 0.683 | **0.355** |
| 200 | 1.009 | 3.11 | 0.680 | 0.343 |

- **100 rounds is the knee**: best bulk MAE, ~half the training time of 200.
- 200 adds nothing (marginal overfit); 25–50 give better ranking/top10 but
  pay ~0.1 MAE.
- Guidance: default experiments at ~100 rounds; exploration at ~50; reserve
  200 for final calibrations.

## Cross-process + parallel levers (implemented)

- **Cross-process disk cache**: `python -m fpl.stages.experiments --cache-dir
  experiments/artifacts/.cache` persists each fitted LightGBM Booster as
  `fit-<content-id>.txt` (stable content-hash filename, NOT builtin `hash()`
  which is per-process randomized). A later CLI run loads instead of
  re-fitting the same config. Same-config second process shows `fit_hits = 1`.
- **Parallel arms**: `--parallel N` runs declared experiments in a
  `ThreadPoolExecutor`. This only pays when per-arm LightGBM threads are
  throttled (`num_threads = cores // N`): untrottled, concurrent fits contend
  and parallel was SLOWER than sequential. Heavy imports are warmed before
  the pool.
- **Measured (ranking config, 3 experiments):**

  | run | wall |
  |---|---:|
  | sequential, no disk | 30.9s |
  | parallel 3 (throttled), cold | 13.6s |
  | parallel 3 (throttled), warm disk | 10.3s |

  i.e. parallel ≈ 2.3×, plus disk ≈ 3× vs sequential.

- Both were the "cross-process" and "parallel" levers from the earlier list;
  training round count + these are the practical wins.

## Training time: before vs now

| What | Before | Now |
|---|---|---|
| Raw LightGBM fit (200 rounds, 49k x 10) | ~6.5s | ~6.5s (expected; internals untouched) |
| Same-config refit within one process | re-fits 6.5s every time | ~0 (in-process cache; matrix assembly skipped too) |
| Same-config refit across CLI runs | re-fits 6.5s | ~0.1-1s Booster disk load (`fit_hits = 1`) |
| Ranking config wall (3 experiment arms) | 30.9s sequential | 10.3s (parallel 3 throttled + warm disk) |

Absolute single runs vary with machine load (6-26s for the same fit across
this session); the durable wins are the workflow ones above.

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
