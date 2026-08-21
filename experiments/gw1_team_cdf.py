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
    x_ticks, y_ticks = [], []
    for frac in np.linspace(0, 1, 6):
        x = left + frac * (right - left)
        value = lo + frac * span
        x_ticks.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" '
            f'stroke="#e5e7eb"/><line x1="{x:.2f}" y1="{bottom}" '
            f'x2="{x:.2f}" y2="{bottom + 6}" stroke="black"/>'
            f'<text x="{x:.2f}" y="{bottom + 23}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12">{value:.1f}</text>'
        )
    for frac in np.linspace(0, 1, 5):
        y = bottom - frac * (bottom - top)
        value = frac
        y_ticks.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" '
            f'stroke="#e5e7eb"/><line x1="{left - 6}" y1="{y:.2f}" '
            f'x2="{left}" y2="{y:.2f}" stroke="black"/>'
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="12">{value:.2f}</text>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
{"".join(x_ticks)}
{"".join(y_ticks)}
<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="black"/>
<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="black"/>
<polyline points="{' '.join(points)}" fill="none" stroke="#2563eb" stroke-width="2"/>
<line x1="{mean_x:.2f}" y1="{top}" x2="{mean_x:.2f}" y2="{bottom}"
      stroke="#dc2626" stroke-dasharray="6,4"/>
<line x1="{median_x:.2f}" y1="{top}" x2="{median_x:.2f}" y2="{bottom}"
      stroke="#16a34a" stroke-dasharray="2,4"/>
<text x="{left}" y="20" font-family="sans-serif" font-size="16">
GW1 squad score CDF (IID marginal proxy)</text>
<text x="{(left + right) / 2:.2f}" y="{height - 12}" text-anchor="middle"
 font-family="sans-serif" font-size="13">settled squad points</text>
<text x="18" y="{(top + bottom) / 2:.2f}" text-anchor="middle"
 transform="rotate(-90 18 {(top + bottom) / 2:.2f})" font-family="sans-serif"
 font-size="13">CDF</text>
<text x="{right - 180}" y="{top + 18}" font-family="sans-serif" font-size="12"
 fill="#dc2626">dashed: mean</text>
<text x="{right - 180}" y="{top + 34}" font-family="sans-serif" font-size="12"
 fill="#16a34a">dotted: median</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def write_histogram_svg(values: np.ndarray, path: Path) -> None:
    """Write a dependency-free histogram as SVG."""
    width, height, left, top = 900, 520, 70, 30
    right, bottom = width - 30, height - 60
    counts, edges = np.histogram(values, bins=24)
    max_count = max(int(counts.max()), 1)
    span = max(float(edges[-1] - edges[0]), 1.0)
    bars = []
    for i, count in enumerate(counts):
        x = left + (edges[i] - edges[0]) / span * (right - left)
        x2 = left + (edges[i + 1] - edges[0]) / span * (right - left)
        y = bottom - count / max_count * (bottom - top)
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(x2 - x - 1, 1):.2f}" '
            f'height="{bottom - y:.2f}" fill="#2563eb" opacity="0.75"/>'
        )
    ticks = []
    for frac in np.linspace(0, 1, 6):
        x = left + frac * (right - left)
        value = edges[0] + frac * span
        ticks.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" '
            f'stroke="#e5e7eb"/><text x="{x:.2f}" y="{bottom + 23}" '
            f'text-anchor="middle" font-family="sans-serif" font-size="12">'
            f'{value:.1f}</text>'
        )
    for count in np.linspace(0, max_count, 5).astype(int):
        y = bottom - count / max_count * (bottom - top)
        ticks.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" '
            f'stroke="#e5e7eb"/><text x="{left - 10}" y="{y + 4:.2f}" '
            f'text-anchor="end" font-family="sans-serif" font-size="12">'
            f'{count}</text>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
{"".join(ticks)}
<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="black"/>
<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="black"/>
{"".join(bars)}
<text x="{left}" y="20" font-family="sans-serif" font-size="16">
GW1 squad score histogram (IID marginal proxy)</text>
<text x="{(left + right) / 2:.2f}" y="{height - 12}" text-anchor="middle"
 font-family="sans-serif" font-size="13">settled squad points</text>
<text x="18" y="{(top + bottom) / 2:.2f}" text-anchor="middle"
 transform="rotate(-90 18 {(top + bottom) / 2:.2f})" font-family="sans-serif"
 font-size="13">draw count</text>
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
    write_histogram_svg(draws, ARTIFACTS / "gw1_team_histogram.svg")
    print(f"live={fetched} captain={squad.by_code()[squad.captain].name} "
          f"direct_quantiles={direct} position_fallback={fallback}")
    print(report)
    print(f"wrote {ARTIFACTS / 'gw1_team_cdf.csv'}")
    print(f"wrote {ARTIFACTS / 'gw1_team_cdf.svg'}")
    print(f"wrote {ARTIFACTS / 'gw1_team_histogram.svg'}")


if __name__ == "__main__":
    main()
