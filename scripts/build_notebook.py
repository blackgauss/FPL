"""Build notebooks/DataQuery.ipynb — the EDA pattern over the parquet dataset.

Pure Polars over `data/processed/` (the dataset produced by the ingest stage).
No query API imported: this is the "close to the files" pattern from Data.md.
The only config import is for paths.
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
        """# Data Querying — EDA over the parquet dataset

Driving example (from `docs/plans/DataPipeline.md`):

> Report a player's performance in matches when a certain other player also
> played, vs their overall average.

**Question:** How did Haaland perform in 2025-26 Premier League matches when
Foden also played, vs when Foden didn't?

This notebook reads only `data/processed/*_2025-2026.parquet` (build with
`python -m fpl.stages.ingest --config config/data.yaml`). Everything is plain
Polars — no query API, per `docs/style/Data.md`."""
    ),
    code(
        """from pathlib import Path

import polars as pl

import yaml

# locate the project root regardless of the kernel's cwd
root = Path.cwd()
while not (root / "config" / "data.yaml").exists() and root != root.parent:
    root = root.parent

cfg = yaml.safe_load((root / "config" / "data.yaml").read_text())
processed = root / cfg["processed_dir"]
season = cfg["seasons"]["default"]

def t(name: str) -> pl.DataFrame:
    return pl.read_parquet(processed / f"{name}_{season}.parquet")

players, teams = t("players"), t("teams")
gw_stats, match_stats, matches = t("gw_stats"), t("match_stats"), t("matches")

print(f"season={season}  players={players.height}  gw_stats={gw_stats.height}  "
      f"match_stats={match_stats.height}  matches={matches.height}")"""
    ),
    md(
        """## 1. Resolve players by name

Name resolution is just a filter over `players.parquet` — the data is the
index (no `find_player` API)."""
    ),
    code(
        """players.filter(pl.col("web_name").str.to_lowercase().str.contains("haaland")).join(
    teams.select(pl.col("code").alias("team_code"), pl.col("name").alias("team")), on="team_code"
).select("player_code", "web_name", "team", "position")"""
    ),
    md("## 2. Per-GW points for Haaland (his season, five rows)"),
    code(
        """HAALAND = 223094  # stable player_code across seasons

gw_stats.filter(
    pl.col("player_id").is_in(
        players.filter(pl.col("player_code") == HAALAND).select("player_id").to_series().implode()
    )
).sort("gw").head(5)"""
    ),
    md(
        """## 3. Haaland's played Premier League matches with/without Foden

Semantics: "played" = `minutes_played > 0`; FPL-points stats are Premier League
only — a cup match carries a folder-`gw` and would wrongly join a `gw_stats`
row, so we filter `matches.tournament == 'prem'` first."""
    ),
    code(
        """prem = matches.filter(pl.col("tournament") == "prem")["match_id"].to_list()
haaland = players.filter(pl.col("player_code") == HAALAND)["player_id"].item()
foden = players.filter(pl.col("web_name") == "Foden")["player_id"].item()

played = match_stats.filter(
    (pl.col("minutes_played") > 0) & pl.col("match_id").is_in(prem)
)
foden_played = played.filter(pl.col("player_id") == foden)["match_id"].to_list()

haaland_matches = played.filter(pl.col("player_id") == haaland).select("match_id", "gw")
perf = haaland_matches.join(
    gw_stats.filter(pl.col("player_id") == haaland).select("gw", "total_points"),
    on="gw",
).with_columns(pl.col("match_id").is_in(foden_played).alias("with_foden"))

def summarize(df: pl.DataFrame) -> None:
    n = df.height
    mean = df["total_points"].mean()
    print(f"   n={n}  mean={mean:.3f}" if n else "   n=0")

print("Haaland FPL points (prem 2025-26):")
print("  with Foden:  ", end="")
summarize(perf.filter(pl.col("with_foden")))
print("  without Foden:", end="")
summarize(perf.filter(~pl.col("with_foden")))
print("  overall:     ", end="")
summarize(perf)"""
    ),
    md(
        """## Reading the result

- Means come with match counts — a 4-match `without` bucket is a small sample;
  don't over-conclude from it.
- The same join works for any `gw_stats` stat; for match-level stats (xG etc.)
  drop the `gw_stats` join and read `match_stats` directly.
- `player_code` (223094) is stable across seasons; `player_id` is season-local."""
    ),
]

out = Path(__file__).resolve().parent.parent / "notebooks" / "DataQuery.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print(f"wrote {out}")
