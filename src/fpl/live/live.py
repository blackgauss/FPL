"""Live FPL data: bootstrap-static snapshot with polite, rate-limit-safe access.

The Fantasy Premier League API is unofficial. It throttles and can return 403
(Cloudflare) when hammered. This module is built around that:

- ONE request per refresh: bootstrap-static carries prices, availability,
  news, chance-of-playing for every player. No per-player element-summary
  calls unless explicitly needed.
- Disk-cached snapshots with a TTL: a fresh snapshot is read from
  data/raw/fpl_api/live.json instead of hitting the network; the API is only
  called once per `max_age`. This bounds request rate to ~1/refresh, never 1
  per query.
- A real User-Agent header is set (a bare requests UA often 403s).
- Graceful degradation: if the fetch fails (rate-limited, offline, 5xx) the
  most recent cached snapshot is returned with its timestamp so callers can
  operate on stale-but-present data instead of crashing.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import requests

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
HEADERS = {
    "User-Agent": "fpl-ml/0.1 (personal ML research; one request per refresh)",
    "Accept": "application/json",
}

# json dictionary of a single element -> the subset we surface as "live state".
LIVE_FIELDS = [
    "id", "code", "web_name", "team", "team_code", "element_type",
    "now_cost", "status", "news", "news_added",
    "chance_of_playing_this_round", "chance_of_playing_next_round",
    "selected_by_percent", "minutes", "ep_next", "ep_this",
    "removed", "can_select", "can_transact",
]


class LiveFetchError(RuntimeError):
    """Raised when the API cannot be reached AND no cached snapshot exists."""


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def fetch_bootstrap(session: requests.Session | None = None,
                    timeout: int = 30) -> dict:
    """One bootstrap-static request; returns the raw JSON."""
    session = session or requests.Session()
    r = session.get(BOOTSTRAP_URL, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def to_live_frame(payload: dict) -> pl.DataFrame:
    """Slice bootstrap-static `elements` into the live-state frame.

    Returns one row per player keyed by `code` (stable across seasons; the
    dataset joins on player_code). now_cost stays in FPL tenths (60 = £6.0m)
    so callers keep units explicit.
    """
    rows = [{k: e.get(k) for k in LIVE_FIELDS} for e in payload["elements"]]
    df = pl.DataFrame(rows)
    # stable, numeric keys for dataset joins
    return (
        df.with_columns(
            pl.col("code").cast(pl.Int64).alias("player_code"),
            pl.col("id").cast(pl.Int64).alias("player_id"),
            pl.col("now_cost").cast(pl.Int64),
            pl.col("minutes").cast(pl.Int64),
        )
        .select(["player_code", "player_id", "web_name", "team_code",
                 "element_type", "now_cost", "status", "news", "news_added",
                 "chance_of_playing_this_round", "chance_of_playing_next_round",
                 "selected_by_percent", "minutes", "ep_next", "ep_this",
                 "removed", "can_select", "can_transact"])
    )


def _load_cache_record(cache: Path) -> dict | None:
    """Cached snapshot record, or None when the file is missing/corrupt.

    A truncated or corrupt cache (interrupted write) must degrade like a
    missing one — never crash the caller that expects stale-but-present.
    """
    try:
        record = json.loads(cache.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (not isinstance(record, dict)
            or not {"payload", "fetched_at", "fetched_epoch"} <= record.keys()):
        return None
    return record


def load_live_state(
    cache_path: str | Path,
    *,
    max_age_seconds: int = 3600,
    session: requests.Session | None = None,
    timeout: int = 30,
) -> tuple[pl.DataFrame, str]:
    """Live-state frame + fetched-at ISO timestamp, respecting the rate limit.

    Returns a cached snapshot if it is fresh (< max_age_seconds). Otherwise
    fetches once, writes it to `cache_path`, and returns it. On a fetch
    failure the last cached snapshot is returned (callers see stale-but-present
    data rather than an error) — unless there is no cache at all, in which
    case LiveFetchError is raised.
    """
    cache = Path(cache_path)
    record = _load_cache_record(cache) if cache.exists() else None
    if record is not None:
        age = time.time() - float(record["fetched_epoch"])
        if age < max_age_seconds:
            return to_live_frame(record["payload"]), record["fetched_at"]

    try:
        payload = fetch_bootstrap(session, timeout)
    except (requests.RequestException, OSError) as exc:
        record = _load_cache_record(cache)
        if record is not None:
            return to_live_frame(record["payload"]), record["fetched_at"]
        raise LiveFetchError(
            f"cannot reach FPL API and no cached snapshot at {cache_path}") from exc

    record = {"fetched_at": _iso_now(), "fetched_epoch": time.time(),
              "payload": payload}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(record))
    return to_live_frame(payload), record["fetched_at"]