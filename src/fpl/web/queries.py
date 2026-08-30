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

    def entry_id(self) -> int | None:
        """Our manager id from the last collection (None if never collected)."""
        try:
            return json.loads(
                (self.account_dir / "collection.json").read_text()
            ).get("entry_id")
        except (OSError, ValueError):
            return None

    # -- players (explorer table) -----------------------------------------------

    def players(self, *, search: str | None = None, position: str | None = None,
                club: int | None = None, status: str | None = None,
                max_price: int | None = None, limit: int = 100,
                offset: int = 0) -> dict:
        """Filterable player table joined with live status/price/ownership;
        `pred_next` (GW mean for next GW) present when the forecast cache is
        warm — computing it lazily is the caller's (router's) choice."""
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
            if status:
                df = df.filter(pl.col("status") == status)
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
