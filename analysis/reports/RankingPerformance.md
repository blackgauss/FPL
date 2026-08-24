# Ranking Ability Report

Generated: 2026-08-24T18:29:51.525310+00:00 (git 9251f260cd4c1559cc2c16fc8d9e430017ddcbc9)
Frame: ranking stage (spearman / topk_hit_rate / concordance)

Reproduce with:

```bash
python scripts/ranking_report.py --config config/experiments_ranking.yaml --output analysis/reports/RankingPerformance.md
```

## Method

Leakage-safe split on 2025-2026: fit <= source GW 30, test from source GW 31. Per-gameweek ranking metrics computed on the same test rows for each model and for naive references (persist last snapshot, FPL `ep_next`).

## Table

| ranker | spearman_rho | topk_hit_rate | pairwise_concordance |
|---|---|---|---|
| ridge | 0.6357 | 0.3568 | 0.4506 |
| lgbm_all | 0.6822 | 0.3551 | 0.4643 |
| hist_gb | 0.6855 | 0.3499 | 0.4654 |
| FPL_ep_next | 0.6366 | 0.3396 | 0.4317 |
| persist(prev_points) | 0.5898 | 0.2762 | 0.3375 |

## Read

- Best model by hit-rate: **ridge** (topk-hit 0.357, spearman 0.636).
- FPL's own `ep_next` reference ranks at topk-hit 0.340; persist-last-GW at 0.276.
- Ranking is materially better than naive persistence but still recovers only ~1 in 3 of the true top-decile performers per GW (topk-hit ~0.35), and the edge over FPL's official forecast is thin.
- Dropping `ep_next` from the model degrades ranking sharply (measured separately), so the official forecast carries real ranking signal.

Ranking and calibration are reported separately in every experiment artifact (`rank@...`, `cal@...`); this report focuses on ranking.
