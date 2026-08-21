"""Stage 4: H2H simulation + team value/weaknesses.

Operationally, the team search produces a basket of squads (each a set of
player_codes). Given per-GW expected points for every player, each squad gets
a per-GW total. Head-to-head: every squad plays every other once per GW; a
squad wins the GW pair if its total is higher. Summing across the GWs and all
opponents gives the aggregate record — the journal's "value of a team".
"""

from __future__ import annotations

import polars as pl


def squad_gw_totals(
    basket: pl.DataFrame,
    per_gw: pl.DataFrame,
) -> pl.DataFrame:
    """Per-squad total expected points per gameweek.

    basket: (team_id, player_code, ...); per_gw: (player_code, gw,
    expected_points, ...). Returns (team_id, gw, gw_total, n_players).
    """
    return (
        basket.select("team_id", "player_code")
        .join(
            per_gw.select("player_code", "gw", "expected_points"),
            on="player_code",
            how="inner",
        )
        .group_by("team_id", "gw")
        .agg(
            pl.col("expected_points").sum().alias("gw_total"),
            pl.col("player_code").count().alias("n_players"),
        )
    )


def simulate_h2h(gw_totals: pl.DataFrame) -> pl.DataFrame:
    """Round-robin H2H over the team basket.

    Every pair of teams plays in every GW; a team wins the pair-match if its
    gw_total is strictly higher. Returns per-team (team_id, played, wins,
    losses, draws, win_ratio, avg_edge, edge_std) — avg_edge = mean
    (own_total - opp_total) across all matches.
    """
    totals = gw_totals.select("team_id", "gw", "gw_total")
    pairs = totals.join(
        totals.rename({"team_id": "opp", "gw_total": "opp_total"}),
        on="gw",
        how="inner",
    ).filter(pl.col("team_id") != pl.col("opp"))

    agg = (
        pairs.with_columns(
            pl.when(pl.col("gw_total") > pl.col("opp_total")).then(1).otherwise(0)
            .alias("win"),
            pl.when(pl.col("gw_total") < pl.col("opp_total")).then(1).otherwise(0)
            .alias("loss"),
            pl.when(pl.col("gw_total") == pl.col("opp_total")).then(1).otherwise(0)
            .alias("draw"),
            (pl.col("gw_total") - pl.col("opp_total")).alias("edge"),
        )
        .group_by("team_id")
        .agg(
            (pl.col("win").sum() + pl.col("loss").sum() + pl.col("draw").sum())
            .alias("played"),
            pl.col("win").sum().alias("wins"),
            pl.col("loss").sum().alias("losses"),
            pl.col("draw").sum().alias("draws"),
            pl.col("edge").mean().alias("avg_edge"),
            pl.col("edge").std().alias("edge_std"),
        )
        .with_columns(
            ((pl.col("wins") + 0.5 * pl.col("draws")) / pl.col("played"))
            .alias("win_ratio")
        )
    )
    return agg.sort("win_ratio", descending=True)


def weaknesses(basket: pl.DataFrame, gw_totals: pl.DataFrame) -> pl.DataFrame:
    """Interpretable exposure per team.

    - window_total: sum of expected points across the GW window
    - worst_gw / worst_gw_total: the single gameweek the team is most exposed
    - star_dependence: fraction of the window total coming from the best player
    """
    per_team_total = gw_totals.group_by("team_id").agg(
        pl.col("gw_total").sum().alias("window_total"),
        pl.col("gw_total").min().alias("worst_gw_total"),
    )
    # the gw at which the min total occurred (first one by sort)
    worst_gw = (
        gw_totals.sort("gw_total", "gw")
        .group_by("team_id")
        .first()
        .select("team_id", pl.col("gw").alias("worst_gw"))
    )
    per_team_total = per_team_total.join(worst_gw, on="team_id", how="left")

    top_star = (
        basket.group_by("team_id")
        .agg((pl.col("expected_total").max() / pl.col("expected_total").sum())
             .alias("star_dependence"))
    )
    return (
        per_team_total.join(top_star, on="team_id")
        .select("team_id", "window_total", "worst_gw", "worst_gw_total",
                "star_dependence")
        .sort("window_total", descending=True)
    )


# --- distributional simulation ----------------------------------------------

def simulate_squad_distributions(
    basket: pl.DataFrame,
    dist_forecast: pl.DataFrame,
    *,
    n_samples: int = 100,
    seed: int = 0,
) -> pl.DataFrame:
    """MC-sample each squad's GW total from per-player point CDFs.

    basket: (team_id, player_code, ...). dist_forecast: per (player_code, gw)
    a `quantiles_struct` (points at QS). For every squad-GW a squad total is
    sampled `n_samples` times (sum over the squad's players of one CDF draw
    per sample). Returns (team_id, gw, sample_mean, sample_std).
    """
    import numpy as np

    from fpl.dist import QS

    rng = np.random.default_rng(seed)
    qs = np.asarray(QS)

    def _sample(quantiles: np.ndarray) -> float:
        # inverse-CDF sampling from a stored quantile vector
        return float(np.interp(rng.uniform(), qs, quantiles))

    # flatten the quantile struct into columns q<..>, then to a list per row
    qcols = [f"q{int(q*100)}" for q in QS]
    d = dist_forecast.with_columns(
        pl.col("quantiles_struct").struct.rename_fields(qcols).alias("q")
    ).select("player_code", "gw", *[pl.col("q").struct.field(c) for c in qcols])
    # one array per player-GW
    pg = d.with_columns(
        pl.concat_list(*qcols).alias("cdf_list")
    ).select("player_code", "gw", "cdf_list")

    rows = []
    for (team_id, gw), g in (
        basket.select("team_id", "player_code")
        .join(pg, on="player_code", how="inner")
        .group_by(["team_id", "gw"], maintain_order=True)
    ):
        cdfs = [np.asarray(x, dtype=float) for x in g["cdf_list"].to_list()]
        tot = np.zeros(n_samples)
        for c in cdfs:
            tot += np.fromiter((_sample(c) for _ in range(n_samples)), dtype=float)
        rows.append((team_id, gw, float(tot.mean()), float(tot.std())))

    return pl.DataFrame(rows, schema=["team_id", "gw", "sample_mean", "sample_std"],
                        orient="row")


def simulate_h2h_dist(
    squad_dist: pl.DataFrame,
    n_samples: int = 100,
) -> pl.DataFrame:
    """Round-robin H2H over squad GW-total distributions.

    For a squad GW pair, the win probability is P(total_i > total_j) using
    a Normal-of-samples approximation clipped to [0,1] (samples are iid).
    Returns per-team win_ratio + expected edge across all matches.
    """
    x = squad_dist  # team_id, gw, sample_mean, sample_std
    pairs = x.join(
        x.rename({"team_id": "opp", "sample_mean": "om", "sample_std": "os"}),
        on="gw",
        how="inner",
    ).filter(pl.col("team_id") != pl.col("opp"))

    w = pairs.with_columns(
        # P(X1 - X2 > 0) with indep Normals
        pl.struct(["sample_mean", "sample_std", "om", "os"]).map_elements(
            lambda r: _win_prob(float(r["sample_mean"]), float(r["sample_std"]),
                                float(r["om"]), float(r["os"])),
            return_dtype=pl.Float64,
        ).alias("winp"),
    )
    agg = (w.group_by("team_id").agg(
        pl.col("winp").count().alias("played"),
        pl.col("winp").sum().alias("exp_wins"),
        (pl.col("sample_mean") - pl.col("om")).mean().alias("avg_edge"),
    ).with_columns(
        (pl.col("exp_wins") / pl.col("played")).alias("win_ratio"),
    ))
    return agg.sort("win_ratio", descending=True)


def _win_prob(m1: float, s1: float, m2: float, s2: float) -> float:
    """P(N(m1,s1) > N(m2,s2)) = Phi((m1-m2)/sqrt(s1^2+s2^2))."""
    import math

    from scipy import stats

    d = m1 - m2
    v = math.sqrt(s1 * s1 + s2 * s2)
    if v == 0:
        return 0.5 if d == 0 else (1.0 if d > 0 else 0.0)
    return float(stats.norm.cdf(d / v))