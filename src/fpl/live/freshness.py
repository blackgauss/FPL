"""Data freshness gate: fail fast when inputs are too stale/drifted to trust.

Building a team on stale or drifted data is worse than not building one. Later
stages (captain, transfers, horizon sims) all assume the underlying snapshot is
current. These checks raise before construction so a misconfigured / never-
refreshed run is caught loudly instead of silently producing junk.

Two dimensions:
- staleness: live snapshot age vs its TTL; and whether the requested season
  even has feature rows (e.g. pre-season 2026-27 -> GW1 store may be empty).
- drift: live world diverges from the dataset (prices moved, players
  transferred/removed) beyond a tolerance we refuse to paper over.
"""

from __future__ import annotations

import time

import polars as pl


class FreshenError(RuntimeError):
    """Raised when the data is too stale or drifted to trust."""


def check_snapshot_age(
    fetched_at_epoch: float, *, max_age_seconds: int = 3600
) -> None:
    """Raise if the live snapshot is older than `max_age_seconds`."""
    age = time.time() - fetched_at_epoch
    if age > max_age_seconds:
        raise FreshenError(
            f"live snapshot is {age:.0f}s old (limit {max_age_seconds}s); "
            "fetch a fresh one before selecting a team")


def check_season_has_rows(processed: str, season: str) -> None:
    """Raise if the feature store for `season` is empty (season-start hazard)."""
    import polars as pl

    path = f"{processed}/features_{season}.parquet"
    try:
        n = pl.scan_parquet(path).select(pl.len()).collect()[0, 0]
    except Exception as exc:  # missing/broken file
        raise FreshenError(f"no feature store for {season} at {path}: {exc}") from exc
    if n == 0:
        raise FreshenError(
            f"feature store for {season} is empty; use carryover or wait for data")


def check_drift(
    live: pl.DataFrame,
    dataset: pl.DataFrame,
    *,
    dataset_price_col: str = "now_cost",
    dataset_team_col: str = "team_code",
    price_scale: float = 10.0,
    max_price_moved: float = 0.25,
    max_team_moved: float = 0.10,
    max_removed: float = 0.05,
) -> None:
    """Raise when live drifts from the dataset beyond tolerance.

    Tolerance is a fraction of matched players (e.g. >25% price-moved, >10%
    transferred, >5% removed) — thresholds below which the dataset is broadly
    current, above which it's stale. Cheap sanity, not a model.
    """
    from fpl.live.agreement import hygiene_summary

    s = hygiene_summary(live, dataset, dataset_price_col=dataset_price_col,
                        dataset_team_col=dataset_team_col, price_scale=price_scale)
    matched = int(s["matched_to_dataset"].item())
    if matched == 0:
        return  # no overlap: drift numbers meaningless -> let construction decide

    price = int(s["price_moved"].item()) / matched
    team = int(s["team_transferred"].item()) / matched
    removed = int(s["not_available"].item()) / matched

    problems = []
    if price > max_price_moved:
        problems.append(f"price drift {price:.0%} > {max_price_moved:.0%}")
    if team > max_team_moved:
        problems.append(f"team transfers {team:.0%} > {max_team_moved:.0%}")
    if removed > max_removed:
        problems.append(f"removed/unavailable {removed:.0%} > {max_removed:.0%}")
    if problems:
        raise FreshenError(
            "dataset too stale vs live: " + "; ".join(problems) +
            "; refresh the dataset (ingest/features) before selecting a team")