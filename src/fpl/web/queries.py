"""The data window for the web layer — the ONLY fpl.web module that reads
files (parquet, model artifacts, JSON dumps).

Everything is lazy, read-only and memoized twice: in-process for the hot
path, and on disk under `forecast_cache` for distributional forecasts (the
only expensive compute — one cold fit per (season, GW window), then instant).

The UI must NEVER trigger FPL-API calls: the live snapshot is read from the
disk cache only (refresh of that cache is a CLI concern, see fpl.live.live).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import polars as pl


class Store:
    """Paths + lazy/memoized frames for the web routers. No other layer reads disk."""

    def __init__(self, *, root: str | Path = ".", season: str = "2026-2027",
                 processed: str = "data/processed",
                 live_cache: str = "data/raw/fpl_api/live.json",
                 account_dir: str = "data/raw/fpl_api/account",
                 artifacts_dir: str = "experiments/artifacts",
                 forecast_cache: str = "data/webcache") -> None:
        self.root = Path(root)
        self.season = season
        self.processed = self.root / processed
        self.live_cache = self.root / live_cache
        self.account_dir = self.root / account_dir
        self.artifacts_dir = self.root / artifacts_dir
        self.forecast_cache = self.root / forecast_cache
        self._live: tuple[pl.DataFrame, str] | None = None
        self._teams: dict[int, str] | None = None
        self._ratings: dict[int, float] | None = None
        self._fixtures_df: pl.DataFrame | None = None
        self._fixtures_read = False
        self._forecast: dict[tuple[int, int], pl.DataFrame | None] = {}

    # -- live snapshot (disk cache ONLY, never the network) --------------------

    def live(self) -> tuple[pl.DataFrame, str] | None:
        """(live-state frame, fetched_at ISO) from the cached snapshot, or
        None when the cache is missing/corrupt/expired-but-not-refreshed."""
        if self._live is not None:
            return self._live
        from fpl.live.live import to_live_frame

        try:
            record = json.loads(self.live_cache.read_text(encoding="utf-8"))
            frame = to_live_frame(record["payload"])
        except (OSError, ValueError, KeyError):
            return None
        self._live = (frame, str(record.get("fetched_at", "")))
        return self._live

    # -- season / GW ----------------------------------------------------------

    def current_gw(self) -> int:
        """Latest gameweek with settled points (event_live), else features."""
        ev = self.account("event_live")
        if ev is not None and ev.height:
            return int(ev.get_column("gw").max())
        try:
            feats = pl.read_parquet(self.processed / f"features_{self.season}.parquet")
            if feats.height:
                return int(feats.get_column("gw").max())
        except OSError:
            pass
        return 0

    def max_forecast_gw(self) -> int:
        """Highest GW the feature store can still feed (features row at gw=k
        targets GW k+1); 0 when the store is empty. Forecast windows above
        this return no rows until the source GW's data lands."""
        feats = self._safe_parquet(
            self.processed / f"features_{self.season}.parquet")
        if feats is None or not feats.height:
            return 0
        return int(feats.get_column("gw").max()) + 1

    def entry_id(self) -> int | None:
        """Our manager id from the last collection (None if never collected)."""
        try:
            return json.loads(
                (self.account_dir / "collection.json").read_text()
            ).get("entry_id")
        except (OSError, ValueError):
            return None

    # -- players (explorer table) -----------------------------------------------

    def team_names(self) -> dict[int, str]:
        """FPL club code -> short name, from the cached bootstrap payload."""
        if self._teams is None:
            try:
                record = json.loads(self.live_cache.read_text(encoding="utf-8"))
                self._teams = {
                    int(t["code"]): str(t.get("short_name") or t.get("name") or "")
                    for t in record["payload"].get("teams", [])
                    if isinstance(t, dict) and t.get("code") is not None}
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                self._teams = {}
        return self._teams

    # -- fixture difficulty + league-owned derived data ----------------------------

    def club_ratings(self) -> dict[int, float]:
        """Club code -> 0..100 strength on the last KNOWN ELO per club (this
        season fills first, older seasons cover clubs without one; unrated
        clubs get the mean). ELO is null until a match settles, hence the
        cross-season fallback."""
        if self._ratings is None:
            elo: dict[int, float] = {}
            clubs: set[int] = set()
            for path in sorted(self.processed.glob("matches_*.parquet"),
                               reverse=True):
                frame = self._safe_parquet(path)
                if frame is None or "home_team_elo" not in frame.columns:
                    continue
                if frame.get_column("home_team").dtype != pl.Int64:
                    frame = frame.with_columns(
                        pl.col("home_team", "away_team").cast(pl.Int64))
                frame = frame.sort("gw")
                for row in frame.iter_rows(named=True):
                    for club, col in ((row.get("home_team"),
                                       "home_team_elo"),
                                      (row.get("away_team"),
                                       "away_team_elo")):
                        if club is None:
                            continue
                        clubs.add(int(club))
                        val = row.get(col)
                        if val is not None and elo.get(int(club)) is None:
                            elo[int(club)] = float(val)
            if elo:
                lo, hi = min(elo.values()), max(elo.values())
                span = (hi - lo) or 1.0
                mid = sum(elo.values()) / len(elo)
                self._ratings = {
                    c: round((elo.get(c, mid) - lo) / span * 100.0, 1)
                    for c in clubs}
            else:
                self._ratings = {}
        return self._ratings

    def fixtures_frame(self) -> pl.DataFrame | None:
        """(gw, team_code, opponent_code) both-venues view of this season's
        fixture list, or None when the season's matches parquet is absent."""
        if not self._fixtures_read:
            self._fixtures_read = True
            frame = self._safe_parquet(
                self.processed / f"matches_{self.season}.parquet")
            if frame is not None:
                home = frame.select(
                    "gw", pl.col("home_team").alias("team_code"),
                    pl.col("away_team").alias("opponent_code"))
                away = frame.select(
                    "gw", pl.col("away_team").alias("team_code"),
                    pl.col("home_team").alias("opponent_code"))
                self._fixtures_df = pl.concat([home, away]).drop_nulls()
        return self._fixtures_df

    def difficulty_frame(self, gw_from: int, gw_to: int) -> pl.DataFrame:
        """(player_code, gw, xdg) fixture difficulty 0..100 (stronger foe =
        higher) for future GWs; empty when fixtures/ratings are unknown."""
        fixtures, ratings = self.fixtures_frame(), self.club_ratings()
        if fixtures is None or not ratings:
            return pl.DataFrame(schema={
                "player_code": pl.Int64, "gw": pl.Int64, "xdg": pl.Float64})
        players = self._safe_parquet(
            self.processed / f"players_{self.season}.parquet")
        if players is None:
            return pl.DataFrame(schema={
                "player_code": pl.Int64, "gw": pl.Int64, "xdg": pl.Float64})
        out = (
            fixtures.filter(
                (pl.col("gw") >= gw_from) & (pl.col("gw") <= gw_to))
            .join(players.select("player_code", "team_code"),
                  on="team_code", how="inner")
            .join(
                pl.DataFrame({"opponent_code": list(ratings),
                              "xdg": list(ratings.values())}),
                on="opponent_code", how="inner")
            .sort(["player_code", "gw"], maintain_order=True)
            .group_by(["player_code", "gw"], maintain_order=True)
            .first().select("player_code", "gw",
                            pl.col("xdg").cast(pl.Float64))
        )
        return out

    def league_ownership(self) -> dict[int, dict]:
        """player_id -> {pct, managers} among collected league picks (latest
        GW); entry-per-unique-manager basis, same as the weekly planner."""
        picks = self.account("team_picks")
        if picks is None or not picks.height:
            return {}
        gw = int(picks.get_column("gw").max())
        latest = picks.filter(pl.col("gw") == gw)
        n = latest.get_column("entry_id").n_unique()
        if not n:
            return {}
        counts = (latest.unique(["entry_id", "element"])
                  .group_by("element").agg(pl.len().alias("managers")))
        return {int(r["element"]): {"pct": round(100.0 * r["managers"] / n, 1),
                                    "managers": int(r["managers"])}
                for r in counts.iter_rows(named=True)}

    def league_report(self) -> dict | None:
        """Per-manager GW scores + head-to-head results derived from
        collected league matches (FPL's own winner fields are unsettled, so
        results fall back to the score comparison resolve_h2h uses)."""
        matches = self.account("league_matches")
        if matches is None or not matches.height:
            return None
        series: dict[int, dict[int, dict]] = {}
        for row in matches.sort("event").iter_rows(named=True):
            if row.get("is_bye"):
                continue
            sides = [
                (row.get("entry_1_entry"), row.get("entry_1_points"),
                 row.get("entry_2_entry"), row.get("entry_2_points")),
                (row.get("entry_2_entry"), row.get("entry_2_points"),
                 row.get("entry_1_entry"), row.get("entry_1_points")),
            ]
            for eid, pts, opp, opp_pts in sides:
                if eid is None or pts is None:
                    continue
                result = ("W" if opp_pts is None or pts > opp_pts
                          else "D" if pts == opp_pts else "L")
                series.setdefault(int(eid), {})[int(row["event"])] = {
                    "points": float(pts), "opponent": opp,
                    "opponent_points": opp_pts, "result": result}
        events = sorted({gw for per in series.values() for gw in per})
        return {"events": events, "managers": {
            str(eid): {str(gw): cell for gw, cell in per.items()}
            for eid, per in series.items()}}

    SORTABLE = {"web_name", "position", "team", "team_code", "player_code",
                "now_cost", "status", "selected_by_percent", "pred_next",
                "xdg_next", "xdg_next5", "own_league"}

    def players(self, *, search: str | None = None, position: str | None = None,
                club: int | None = None, status: str | None = None,
                max_price: int | None = None, limit: int = 100,
                offset: int = 0, sort: str | None = None,
                descending: bool = False) -> dict:
        """Filterable/sortable player table joined with live status/price/
        ownership and the model's next-GW mean (`pred_next`, null when the
        forecast cache is cold-unbuilt; trees sort nulls last)."""
        players = pl.read_parquet(
            self.processed / f"players_{self.season}.parquet")
        df = players
        if search:
            pat = f"(?i){re.escape(search)}"
            df = df.filter(
                pl.col("web_name").str.contains(pat)
                | pl.col("first_name").str.contains(pat)
                | pl.col("second_name").str.contains(pat))
        if position:
            df = df.filter(pl.col("position") == position)
        if club is not None:
            df = df.filter(pl.col("team_code") == club)
        if max_price is not None:
            gw_stats = self._safe_parquet(
                self.processed / f"gw_stats_{self.season}.parquet")
            if gw_stats is not None:
                latest = gw_stats.sort("player_id", "gw").group_by(
                    "player_id", maintain_order=True).last()
                df = df.join(
                    latest.select("player_id", pl.col("now_cost")),
                    on="player_id", how="left")
                df = df.filter(pl.col("now_cost") <= max_price)

        lv = self.live()
        if lv is not None:
            live_keep = [c for c in
                         ["player_code", "now_cost", "status", "news",
                          "chance_of_playing_next_round", "selected_by_percent",
                          "ep_next"]
                         if c == "player_code" or c not in df.columns]
            df = df.join(lv[0].select(live_keep), on="player_code", how="left")
            if "selected_by_percent" in df.columns:
                df = df.with_columns(pl.col("selected_by_percent").cast(
                    pl.Float64, strict=False))
            if status:
                df = df.filter(pl.col("status") == status)

        names = self.team_names()
        if "team" not in df.columns:
            df = df.with_columns(
                (pl.col("team_code").cast(pl.Int64, strict=False)
                 .replace_strict(names, default=None, return_dtype=pl.String)
                 if names else pl.lit(None, dtype=pl.String)).alias("team"))

        gw = self.current_gw()
        diff = self.difficulty_frame(gw + 1, gw + 5)
        if diff.height:
            nxt = diff.filter(pl.col("gw") == gw + 1).select(
                "player_code", pl.col("xdg").alias("xdg_next"))
            nxt5 = diff.group_by("player_code").agg(
                pl.col("xdg").mean().alias("xdg_next5"))
            df = df.join(nxt, on="player_code", how="left").join(
                nxt5, on="player_code", how="left")
        else:
            df = df.with_columns(
                pl.lit(None, dtype=pl.Float64).alias("xdg_next"),
                pl.lit(None, dtype=pl.Float64).alias("xdg_next5"))

        ownership = self.league_ownership()
        if ownership and "player_id" in df.columns:
            df = df.with_columns(
                pl.col("player_id").cast(pl.Int64, strict=False)
                .replace_strict(
                    {k: v["pct"] for k, v in ownership.items()},
                    default=None, return_dtype=pl.Float64
                ).alias("own_league"))
        elif "own_league" not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(
                "own_league"))

        pred = self.predicted_next(self.current_gw() + 1)
        if pred is not None:
            df = df.join(pred.rename({"pred": "pred_next"}),
                         on="player_code", how="left")
        else:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(
                "pred_next"))
        if sort and sort in df.columns:
            df = df.sort(sort, descending=descending, nulls_last=True)
        total = df.height
        df = df.slice(offset, limit)
        return {"season": self.season, "total": total, "rows": df.to_dicts()}

    # -- distributional forecasts (memoized) -------------------------------------

    def forecast(self, gw_start: int, gw_end: int) -> pl.DataFrame:
        """Per player-GW {pred, q1..q99} over [gw_start, gw_end]; the
        t-digest-derived CDFs. Cold build fits models once and is persisted
        to forecast_cache; later calls read it. Raises if data/model missing."""
        key = (int(gw_start), int(gw_end))
        if key in self._forecast:
            assert self._forecast[key] is not None
            return self._forecast[key]
        if gw_end < gw_start or gw_end - gw_start > 10:
            raise ValueError("forecast window must span 1..11 gameweeks")
        path = self.forecast_cache / (
            f"fc_v2_{self.season}_{gw_start}_{gw_end}.parquet")
        if path.is_file():
            frame = pl.read_parquet(path)
        else:
            from fpl.dist import QS
            from fpl.team.distribution import distributional_forecast

            # params ARE target gameweeks (it shifts rows by -1 internally)
            raw = distributional_forecast(str(self.processed), self.season,
                                          key[0], key[1])
            frame = raw.with_columns(
                *(pl.col("quantiles_struct").struct.field(f"q{int(q * 100)}")
                    .alias(f"q{int(q * 100)}") for q in QS),
            )
            # GW points are >= 0 by rule; the Gaussian-tail quantiles are not
            frame = frame.with_columns(
                *(pl.col(f"q{int(q * 100)}").clip(lower_bound=0.0)
                    for q in QS)).drop("quantiles_struct")
            self.forecast_cache.mkdir(parents=True, exist_ok=True)
            frame.write_parquet(path)
        self._forecast[key] = frame
        return frame

    def predicted_next(self, gw: int) -> pl.DataFrame | None:
        """(player_code, pred) for target GW `gw`, or None if unavailable."""
        try:
            return (self.forecast(gw, gw)
                    .select("player_code", "pred").unique("player_code"))
        except Exception:
            return None

    # -- collected account + research artifacts ----------------------------------

    def account(self, name: str) -> pl.DataFrame | None:
        """One collected parquet (allowlisted; name is a stem not a path)."""
        allowed = {"team_picks", "team_history", "league_standings",
                   "league_matches", "event_live", "resolved_standings"}
        if name not in allowed:
            raise ValueError(f"unknown collected table {name!r}")
        return self._safe_parquet(self.account_dir / f"{name}.parquet")

    def account_json(self, pattern: str) -> list[dict]:
        """All account JSON matching pattern (e.g. 'gw.*_plan'), newest first,
        each tagged with its filename."""
        return self._json_dir(self.account_dir, pattern)

    def artifacts(self) -> list[dict]:
        """Inventory of research artifacts (json only) with update times."""
        out = []
        for p in sorted(self.artifacts_dir.glob("*.json")):
            stat = p.stat()
            out.append({"name": p.name, "mtime": stat.st_mtime,
                        "size": stat.st_size})
        return out

    def artifact_json(self, name: str) -> dict | None:
        """One research artifact by exact basename (no traversal)."""
        if "/" in name or "\\" in name or not name.endswith(".json"):
            raise ValueError("artifact name must be a plain *.json basename")
        path = self.artifacts_dir / name
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    # -- helpers -----------------------------------------------------------------

    @staticmethod
    def _safe_parquet(path: Path) -> pl.DataFrame | None:
        try:
            return pl.read_parquet(path)
        except (OSError, RuntimeError):
            return None

    @staticmethod
    def _json_dir(dir_: Path, pattern: str) -> list[dict]:
        rx = re.compile(pattern)
        found = []
        if dir_.is_dir():
            for p in sorted(dir_.glob("*.json")):
                if rx.search(p.name):
                    try:
                        found.append({"file": p.name, **json.loads(
                            p.read_text(encoding="utf-8"))})
                    except (OSError, ValueError):
                        continue
        # natural order: gw2 before gw10 (numeric runs compare as numbers)
        found.sort(
            key=lambda d: [int(t) if t.isdigit() else t
                           for t in re.split(r"(\d+)", d["file"])])
        return found
