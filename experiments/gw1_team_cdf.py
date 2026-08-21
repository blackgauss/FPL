"""CDF report for the logged GW1 squad using player quantile marginals.

This is an IID marginal CDF: all starters are assumed to play, Rice is
captain (2x), and bench substitutions are not sampled because no 2026-27
playing-probability distribution exists yet. It uses the available 2025-26
quantile artifact, shifted to the current GW1 model means; uncovered current
players fall back to their position marginal. Treat as a distribution PoC,
not a final 2026-27 probability forecast.

Outputs:
  experiments/artifacts/gw1_team_cdf.png
  experiments/artifacts/gw1_team_cdf.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from fpl.live.current import construction_input
from fpl.live.filters import available
from fpl.live.live import load_live_state
from fpl.model.inference import load_model
from fpl.model.train import load_training
from fpl.team.enumerate import greedy_teams
from fpl.team.filtering import filter_pool
from fpl.team.harness import basket_squads
from fpl.weekly.captain import set_captains

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "experiments" / "artifacts"
N_DRAWS = 100_000
SEED = 42
QS = np.asarray([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
TARGET_NAMES = {
    "Martinez", "Pedro Porro", "Lacroix", "Gabriel", "Justin", "Rice",
    "Cherki", "Mbeumo", "Rogers", "Wirtz", "Havertz",
}


def write_cdf_svg(values: np.ndarray, path: Path) -> None:
    """Write a dependency-free step CDF plot as SVG."""
    width, height, left, top = 900, 520, 70, 30
    right, bottom = width - 30, height - 60
    lo, hi = float(values.min()), float(values.max())
    span = max(hi - lo, 1.0)
    full = np.sort(values)
    sample_idx = np.linspace(0, len(full) - 1, min(2000, len(full))).astype(int)
    ordered = full[sample_idx]
    points = []
    for i, value in zip(sample_idx + 1, ordered, strict=False):
        x = left + (float(value) - lo) / span * (right - left)
        y = bottom - i / len(full) * (bottom - top)
        points.append(f"{x:.2f},{y:.2f}")
    mean_x = left + (float(values.mean()) - lo) / span * (right - left)
    median_x = left + (float(np.median(values)) - lo) / span * (right - left)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="black"/>
<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="black"/>
<polyline points="{' '.join(points)}" fill="none" stroke="#2563eb" stroke-width="2"/>
<line x1="{mean_x:.2f}" y1="{top}" x2="{mean_x:.2f}" y2="{bottom}"
      stroke="#dc2626" stroke-dasharray="6,4"/>
<line x1="{median_x:.2f}" y1="{top}" x2="{median_x:.2f}" y2="{bottom}"
      stroke="#16a34a" stroke-dasharray="2,4"/>
<text x="{left}" y="20" font-family="sans-serif" font-size="16">
GW1 squad score CDF (IID marginal proxy)</text>
<text x="{left}" y="{height - 18}" font-family="sans-serif" font-size="12">
score {lo:.1f}..{hi:.1f}</text>
<text x="{right - 90}" y="{top + 18}" font-family="sans-serif" font-size="12">CDF</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    live, fetched = load_live_state(ROOT / "data/raw/fpl_api/live.json",
                                    max_age_seconds=3600)
    players = pl.read_parquet(PROCESSED / "players_2026-2027.parquet")
    stats = pl.read_parquet(PROCESSED / "gw_stats_2026-2027.parquet")
    model = load_model(PROCESSED / "points_lgbm.txt")
    td = load_training(str(PROCESSED), ["2026-2027"], require_target=False)["2026-2027"]
    mask = td.gw == 1
    scored = td.meta.filter(pl.col("gw") == 1).with_columns(
        pl.Series("expected_total", model.predict(td.X[mask])))
    scored = scored.join(players.select("player_id", "player_code", "web_name", "position"),
                         on="player_id")
    availability = (stats.join(players.select("player_id", "player_code"), on="player_id")
                    .select("player_code", "now_cost")
                    .with_columns(pl.lit(1).alias("minutes_in_window")))
    pool_frame = construction_input(
        scored.with_columns(pl.lit(1).alias("minutes_in_window")),
        live, available(live))
    basket = greedy_teams(
        filter_pool(pool_frame, availability, top_k_per_position=25,
                    max_per_team=4, reserve_top=20), n_teams=6, seed=1)
    expected = dict(zip(scored["player_code"].to_list(),
                        scored["expected_total"].to_list(), strict=False))
    candidates = []
    for _, squad in basket_squads(basket, pool_frame, gw=1):
        squad = squad.__class__(
            players=squad.players, gw=1, starters=squad.starters,
            bench=tuple(p.code for p in squad.players if p.code not in squad.starters),
            captain=squad.captain, vice_captain=squad.vice_captain,
        )
        candidates.append(set_captains(squad, expected))
    squad = next(
        s for s in candidates
        if {s.by_code()[c].name for c in s.starters} == TARGET_NAMES
    )

    dist = pl.read_parquet(PROCESSED / "dist_2025-2026.parquet")
    qcols = [f"q{int(q * 100):02d}" for q in QS]
    dist = dist.with_columns(
        pl.col("quantiles_struct").struct.rename_fields(qcols).alias("q"))
    q_by_code = {}
    for row in dist.group_by("player_code").agg(
        [pl.col("q").struct.field(c).mean().alias(c) for c in qcols]
    ).iter_rows(named=True):
        q_by_code[int(row["player_code"])] = np.asarray([row[c] for c in qcols])
    q_by_pos = {}
    for row in dist.group_by("position").agg(
        [pl.col("q").struct.field(c).mean().alias(c) for c in qcols]
    ).iter_rows(named=True):
        q_by_pos[row["position"]] = np.asarray([row[c] for c in qcols])

    rng = np.random.default_rng(SEED)
    draws = np.zeros(N_DRAWS)
    direct = fallback = 0
    for player in squad.players:
        q = q_by_code.get(player.code)
        if q is None:
            q = q_by_pos.get(player.position.value,
                             np.full(len(QS), expected.get(player.code, 0.0)))
            fallback += 1
        else:
            direct += 1
        q = q + (expected.get(player.code, float(np.median(q))) - np.median(q))
        player_draws = np.interp(rng.random(N_DRAWS), QS, q)
        if player.code in squad.starters:
            draws += player_draws
            if player.code == squad.captain:
                draws += player_draws

    quantiles = np.quantile(draws, QS)
    report = pl.DataFrame({
        "stat": ["mean", "std", "q01", "q05", "q10", "q25", "q50", "q75",
                 "q90", "q95", "q99", "p_ge_80", "p_ge_90"],
        "value": [float(draws.mean()), float(draws.std()), *quantiles.tolist(),
                  float(np.mean(draws >= 80)), float(np.mean(draws >= 90))],
    })
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    report.write_csv(ARTIFACTS / "gw1_team_cdf.csv")
    write_cdf_svg(draws, ARTIFACTS / "gw1_team_cdf.svg")
    print(f"live={fetched} captain={squad.by_code()[squad.captain].name} "
          f"direct_quantiles={direct} position_fallback={fallback}")
    print(report)
    print(f"wrote {ARTIFACTS / 'gw1_team_cdf.csv'}")
    print(f"wrote {ARTIFACTS / 'gw1_team_cdf.svg'}")


if __name__ == "__main__":
    main()
