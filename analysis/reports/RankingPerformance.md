# Ranking Ability Report

Generated: 2026-08-24T20:33:44.191331+00:00 (git 4a28b0d39b72b4dba95a5125457c880fd3e17402)
Frame: ranking stage (spearman / topk_hit_rate / concordance)

Reproduce with:

```bash
python -m fpl.stages.ranking --config config/experiments_ranking.yaml --output analysis/reports/RankingPerformance.md
```

## Method

Leakage-safe split on 2025-2026: fit <= source GW 30, test from source GW 31. Per-gameweek ranking metrics computed on the same test rows for each model and for naive references (persist last snapshot, FPL `ep_next`).

## Table

| ranker | spearman_rho | topk_hit_rate | pairwise_concordance |
|---|---|---|---|
| lgbm_all | 0.6790 | 0.3619 | 0.4634 |
| ridge | 0.6354 | 0.3533 | 0.4505 |
| hist_gb | 0.6840 | 0.3499 | 0.4646 |
| FPL_ep_next | 0.6366 | 0.3396 | 0.4317 |
| persist(prev_points) | 0.5898 | 0.2762 | 0.3375 |

## Read

- Best model by hit-rate: **lgbm_all** (topk-hit 0.362, spearman 0.679).
- FPL's own `ep_next` reference ranks at topk-hit 0.340; persist-last-GW at 0.276.
- Ranking is materially better than naive persistence but still recovers only ~1 in 3 of the true top-decile performers per GW (topk-hit ~0.35), and the edge over FPL's official forecast is thin.
- Dropping `ep_next` from the model degrades ranking sharply (measured separately), so the official forecast carries real ranking signal.

Ranking and calibration are reported separately in every experiment artifact (`rank@...`, `cal@...`); this report focuses on ranking.
