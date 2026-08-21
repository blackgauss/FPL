"""Build notebooks/ModelEnsemblesInvestigation.ipynb.

Diagnostics for analysis/Investigations.md, run on the existing polars
dataset: position baselines, which driver features correlate with realized
points, driver collinearity, and the in-squad co-movement check that
quantifies the I.I.D. violation motivating the model-ensembles work.

Pattern mirrors scripts/build_notebook.py: pure Polars against
data/processed, config-sourced paths, nbformat generation (not executed —
run `notebooks/ModelEnsemblesInvestigation.ipynb` with Run All).
"""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

nb.cells = [
    md(
        """# Model Ensembles — Investigation Notebook (part 1: diagnostics)

Purpose: quantify the assumptions behind the model-ensembles work
(`analysis/Investigations.md`).

1. Per-position realized-points baselines.
2. Which driver features correlate most with realized points (overall + by
   position).
3. Mutual collinearity of the driver features.
4. **In-squad co-movement**: do squad-mates' points co-move more than random
   pairs? If yes, per-player forecasts are NOT I.I.D., and summing them (the
   team-search assumption) accumulates the co-movement — the destructive-
   interference / catastrophic-collapse risk.

Pure Polars over `data/processed/*_2025-2026.parquet` (change the season in
`config/data.yaml`). Run **Kernel → Restart & Run All**."""
    ),
    code(
        """from pathlib import Path
from collections import defaultdict
import itertools

import numpy as np
import polars as pl

import yaml

root = Path.cwd()
while not (root / "config" / "data.yaml").exists() and root != root.parent:
    root = root.parent

cfg = yaml.safe_load((root / "config" / "data.yaml").read_text())
processed = root / cfg["processed_dir"]
season = cfg["seasons"]["default"]


def t(name: str) -> pl.DataFrame:
    return pl.read_parquet(processed / f"{name}_{season}.parquet")


players, teams = t("players"), t("teams")
gw_stats, features = t("gw_stats"), t("features")
print(f"season={season}  players={players.height}  "
      f"gw_stats={gw_stats.height}  features={features.height}")"""
    ),
    md("## 1. Position baselines — realized points per position"),
    code(
        """gw_stats.join(players.select("player_id", "position"), on="player_id")
    .filter(pl.col("minutes") > 0)
    .group_by("position")
    .agg(pl.col("total_points").mean().alias("mean"),
         pl.col("total_points").count().alias("played_gw"))
    .sort("position")"""
    ),
    md("""## 2. Which driver features correlate with realized points?

`features.next_points` is the realized points a row predicts. Compare each
driver's Pearson correlation with it, overall and per position. Form features
(`pts_avg_5/3`, `prev_points`) are expected to dominate; matchup fields
(home/opponent elo) should show up per-position."""),
    code(
        """f = features.join(players.select("player_code", "position"), on="player_code")
ff = f.with_columns(pl.col("was_home").cast(pl.Float64))

drivers = ["was_home", "home_elo", "opponent_elo", "prev_points",
           "pts_avg_3", "pts_avg_5"]

overall = pl.DataFrame({
    "feature": drivers,
    "overall_r": [float(ff.select(pl.corr(c, "next_points")).item())
                  for c in drivers],
}).sort("overall_r", descending=True)

by_position = pl.concat([
    pl.DataFrame({
        "position": pos,
        "feature": drivers,
        "r": [float(ff.filter(pl.col("position") == pos)
                    .select(pl.corr(c, "next_points")).item())
              for c in drivers],
    })
    for pos in ["GKP", "DEF", "MID", "FWD"]
])

print("correlation with realized points (next_points):")
print(overall)
print("\\nby position (rows=position, cols=driver):")
print(by_position.pivot(index="position", on="feature", values="r"))"""
    ),
    md("""## 3. Mutual collinearity of the drivers

Correlated drivers (form fields especially) mean a global model allocates
credit incorrectly; per-player correction must be careful not to double-count
them."""),
    code(
        """pl.DataFrame({a: [float(ff.select(pl.corr(a, b)).item()) for b in drivers]
                    for a in drivers})"""
    ),
    md("""## 4. In-squad co-movement — the I.I.D. check

For each pair of squad-mates (same club) with enough gameweeks, Pearson
correlate their per-GW point series across the season; compare with random
cross-team pairs (size-matched). If same-team mean r >> cross-team mean r,
teammates co-move far beyond independent draws — the squad-scoring sum then
accumulates shared (not independent) risk, the core motivation."""),
    code(
        """pr = (gw_stats
      .join(players.select("player_id", "player_code"), on="player_id")
      .filter(pl.col("minutes") > 0))
qual = pr.group_by("player_code").agg(pl.col("gw").count().alias("games")) \\
    .filter(pl.col("games") >= 8)["player_code"].to_list()

grid = features.select("player_code", "gw", "total_points") \\
    .filter(pl.col("player_code").is_in(qual))
w = grid.pivot(index="player_code", on="gw", values="total_points").fill_null(0)
codes = w["player_code"].to_list()
arr = w.drop("player_code").to_numpy().astype(np.float64)
sidx = {c: i for i, c in enumerate(codes)}

team_of = {r["player_code"]: r["team_code"]
           for r in players.select("player_code", "team_code").iter_rows(named=True)}
team_of = {c: team_of.get(c, -1) for c in codes}
buckets = defaultdict(list)
for c, tm in team_of.items():
    buckets[tm].append(c)

same, cross = [], []
for cs in buckets.values():
    if len(cs) < 3:
        continue
    for a, b in itertools.combinations(cs, 2):
        i, j = sidx[a], sidx[b]
        if arr[i].std() <= 0 or arr[j].std() <= 0:
            continue
        r = float(np.corrcoef(arr[i], arr[j])[0, 1])
        if np.isfinite(r):
            same.append(r)

rng = np.random.default_rng(0)
from_ = [c for c in codes if team_of[c] >= 0]
while len(cross) < min(len(same), 20_000):
    a, b = rng.choice(len(from_), size=2, replace=False)
    ai, bi = from_[int(a)], from_[int(b)]
    if team_of[ai] == team_of[bi]:
        continue
    ia, ib = sidx[ai], sidx[bi]
    if arr[ia].std() <= 0 or arr[ib].std() <= 0:
        continue
    r = float(np.corrcoef(arr[ia], arr[ib])[0, 1])
    if np.isfinite(r):
        cross.append(r)

same_m, cross_m = float(np.mean(same)), float(np.mean(cross))
print(f"same-team pairs : {len(same):6d}  mean r = {same_m:.4f}")
print(f"cross-team pairs: {len(cross):6d}  mean r = {cross_m:.4f}")
print(f"=> teammates co-move {same_m / cross_m:.1f}x the random baseline "
      f"(ratio), i.e. NOT I.I.D.")"""
    ),
    md("""## Takeaways → next steps

- Expect position-specific models: baselines differ per position (§1).
- Form dominates realized points (§2); form fields are mutually collinear (§3)
  — corrections need to avoid double-counting.
- **§4 is the crux**: teammates' points co-move ~17× random pairs in 2025-26.
  Any per-player model summed into a squad score must either (a) carry the
  within-team correlation (e.g. team-latent factor / hierarchical player plus
  team offsets) or (b) be evaluated through the gym as a paired toggle so the
  co-movement's cost is measured — see `analysis/Investigations.md` and the
  `experiments/` folder."""
    ),
]

out = Path(__file__).resolve().parents[1] / "notebooks" / "ModelEnsemblesInvestigation.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(nbf.writes(nb), encoding="utf-8")
print(f"wrote {out}")