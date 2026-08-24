# Ranking Ability Report

Generated: 2026-08-24T20:00:10.206279+00:00 (git dea1dc4f1ac0500d50c61268145287ab6be40b80)
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
| lgbm_all | 0.6781 | 0.3568 | 0.4631 |
| ridge | 0.6351 | 0.3516 | 0.4504 |
| hist_gb | 0.6820 | 0.3499 | 0.4638 |
| FPL_ep_next | 0.6366 | 0.3396 | 0.4317 |
| persist(prev_points) | 0.5898 | 0.2762 | 0.3375 |

## Read

- Best model by hit-rate: **lgbm_all** (topk-hit 0.357, spearman 0.678).
- FPL's own `ep_next` reference ranks at topk-hit 0.340; persist-last-GW at 0.276.
- Ranking is materially better than naive persistence but still recovers only ~1 in 3 of the true top-decile performers per GW (topk-hit ~0.35), and the edge over FPL's official forecast is thin.
- Dropping `ep_next` from the model degrades ranking sharply (measured separately), so the official forecast carries real ranking signal.

Ranking and calibration are reported separately in every experiment artifact (`rank@...`, `cal@...`); this report focuses on ranking.
