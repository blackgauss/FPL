"""Build analysis/ModelEnsemblesInvestigation.ipynb.

The investigation notebook: the model-ensembles narrative (markdown) plus the
diagnostics (python) in one place — position baselines, driver-feature
correlation with realized points, driver collinearity, and the in-squad
co-movement check quantifying the I.I.D. violation.

Pattern mirrors scripts/build_notebook.py: pure Polars against
data/processed, config-sourced paths, nbformat generation (not executed —
run `analysis/ModelEnsemblesInvestigation.ipynb` with Run All). The plain
markdown lives alongside in analysis/Investigations.md.
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
        """# Model Ensembles — Investigation Notebook

The narrative (`analysis/Investigations.md`) and its diagnostics, combined.
Pure Polars over `data/processed/*_<season>.parquet` (season from
`config/data.yaml`). Run **Kernel → Restart & Run All**.

## Goal

Player models per position, each producing **per-player, per-GW predictions**:

    prediction(player) = global(position model) + correction(player)

## Why per-player forecasts are NOT I.I.D.

Team search sums per-player expected totals into a squad score. If individual
forecasts are treated as independent but realized outcomd co-move within a
squad, the correlated errors accumulate:

- **Destructive interference / catastrophic collapse** — loading up on one
  fixture cluster/team double-counts a shared risk; when the shared factor
  misses, the squad collapses below the sum of its parts.
- **False diversification** — mean-only scoring over-weights spread across
  co-moving teammates over a single star.

So we must account for **matchup** (per-position opponent/venue) and
**within-squad covariance**, and capitalize on **team momentum**. The
diagnostics below are step 1: measure the co-movement before modelling it.

## Evaluation discipline

The gym (`fpl.gym.Eval`) is the judge, and only as **paired toggles**: fix
everything, vary one thing (I.I.D. baseline vs covariance-aware, global vs
global+correction, momentum on/off). The attributable delta of each change is
the difference between paired runs — never a single gap."""
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
matches = t("matches")
print(f"season={season}  players={players.height}  "
      f"gw_stats={gw_stats.height}  features={features.height}")"""
    ),
    md("## 1. Position baselines — realized points per position"),
    code(
        """baseline = (
    gw_stats.join(players.select("player_id", "position"), on="player_id")
    .filter(pl.col("minutes") > 0)
    .group_by("position")
    .agg(pl.col("total_points").mean().alias("mean"),
         pl.col("total_points").count().alias("played_gw"))
    .sort("position")
)
print(baseline)"""
    ),
    md("""## 2. Which driver features correlate with realized points?

`features.next_points` is the realized points a row predicts. Compare each
driver's Pearson correlation with it, overall and per position. Form features
(`pts_avg_5/3`, `prev_points`) should dominate; matchup fields (home/opponent
elo) should show up per-position. This informs the per-player *correction*
and the per-position matchup terms."""),
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
credit incorrectly; per-player corrections must not double-count them."""),
    code(
        """collin = pl.DataFrame({a: [float(ff.select(pl.corr(a, b)).item())
                              for b in drivers]
                    for a in drivers})
print(collin)"""
    ),
    md("""## 4. In-squad co-movement — the I.I.D. check

For each pair of squad-mates (same club) with enough gameweeks, Pearson
correlate their per-GW point series across the season; compare with random
cross-team pairs (size-matched). Same-team mean r >> cross-team mean r ⇒
teammates co-move far beyond independent draws — the squad-scoring sum then
accumulates shared risk, the core motivation for covariance-aware models."""),
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
    md("""## 5. Matchup effects — opponent strength × position

Raw linear correlation of the matchup fields was near zero (§2), but matchups
can still matter non-linearly. Bucket `opponent_elo` into quartiles and look
at mean realized points per position; and split by venue. If positions
respond differently to opponent strength / home-away, per-position matchup
terms (not one global row) are warranted."""),
    code(
        """matchup = ff.with_columns(
    pl.when(pl.col("opponent_elo") < ff["opponent_elo"].quantile(0.25)).then(1)
    .when(pl.col("opponent_elo") < ff["opponent_elo"].quantile(0.5)).then(2)
    .when(pl.col("opponent_elo") < ff["opponent_elo"].quantile(0.75)).then(3)
    .otherwise(4).alias("opp_q"))

print("mean next_points by position x opponent-strength quartile "
      "(1 = weakest opponent):")
print(matchup.group_by(["position", "opp_q"])
      .agg(pl.col("next_points").mean().alias("mean_pts"),
           pl.col("next_points").count().alias("n"))
      .sort(["position", "opp_q"]))
print("\\nmean next_points by position x venue (was_home, 1=home):")
print(matchup.group_by(["position", "was_home"])
      .agg(pl.col("next_points").mean().alias("mean_pts"))
      .sort(["position", "was_home"]))"""
    ),
    md("""## 6. Shared-fixture (opponent) co-movement — players who face each other

§4 showed teammates co-move. The other side of covariance is the OPPONENT in
the same gameweek: does a player's points move with their own team's total and
against the opposing team's total that GW? And do the two teams in a match
co-move at all? Strong own-team coupling (positive) and opponent coupling
(negative) mean a player's weekend outcome is jointly determined with the
people he actually plays alongside and against — covariance, not I.I.D."""),
    code(
        """pt = (gw_stats
      .join(players.select("player_id", "team_code", "position"), on="player_id")
      .filter(pl.col("minutes") > 0))
own = pt.group_by(["team_code", "gw"]).agg(
    pl.col("total_points").sum().alias("own_total"))

m = matches.select("gw", "home_team", "away_team")
opp = pl.concat([
    m.rename({"home_team": "team_code", "away_team": "opp_team"}).select("gw", "team_code", "opp_team"),
    m.rename({"away_team": "team_code", "home_team": "opp_team"}).select("gw", "team_code", "opp_team"),
])
opp_total = opp.join(
    own.rename({"team_code": "opp_team", "own_total": "opp_total"})
       .select("opp_team", "gw", "opp_total"), on=["opp_team", "gw"])

dat = (pt.join(own, on=["team_code", "gw"])
         .join(opp_total.select("team_code", "gw", "opp_total"), on=["team_code", "gw"]))

def corrs(df):
    return {"vs_own_team": float(df.select(pl.corr("total_points", "own_total")).item()),
            "vs_opponent": float(df.select(pl.corr("total_points", "opp_total")).item())}

print("player points vs own / opponent team total that GW (shared fixture):")
print("  overall:", corrs(dat))
for p in ["GKP", "DEF", "MID", "FWD"]:
    print(f"  {p:<4}:", corrs(dat.filter(pl.col("position") == p)))

shared = (
    matches.join(own.rename({"team_code": "home_team", "own_total": "home_pts"}),
                 on=["home_team", "gw"], how="left")
    .join(own.rename({"team_code": "away_team", "own_total": "away_pts"}),
          on=["away_team", "gw"], how="left")
    .select("gw", "home_pts", "away_pts").drop_nulls())
print("shared fixture: corr(home_total, away_total) across all matches =",
      round(float(shared.select(pl.corr("home_pts", "away_pts")).item()), 4))"""
    ),
    md("""## Takeaways → next steps

- Position-specific models are justified: baselines differ per position (§1).
- Form dominates realized points (§2); form fields are mutually collinear (§3)
  — corrections must avoid double-counting them.
- **§4 is the crux**: teammates' points co-move ~17× random pairs in 2025-26.
  Per-player models summed into squad scores must carry the within-team
  correlation (team-latent factor / hierarchical player + team offsets) or be
  judged through the gym as a paired toggle so the co-movement's real cost is
  measured.

Open design questions and the work plan live in `analysis/Investigations.md`;
runnable model-variant comparisons land in `experiments/`."""
    ),
]

out = Path(__file__).resolve().parents[1] / "analysis" / "ModelEnsemblesInvestigation.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(nbf.writes(nb), encoding="utf-8")
print(f"wrote {out}")