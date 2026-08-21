"""Ingest job: FPL-Core CSV tree -> typed parquet dataset.

The parquet files written here are the only interface consumers (notebooks,
production jobs) read from. `load_season` remains internal to this job.

Layout: `processed_dir/{table}_{season}.parquet`, flat, labeled by the season
the data corresponds to (event-time, not file mtime). `season` is also kept as
a column inside every file so tables are self-describing.
"""

from __future__ import annotations

from pathlib import Path

from fpl.data.contract import load_season

TABLES = ("players", "teams", "gw_stats", "match_stats", "matches")


def run(root: str | Path, season: str, processed_dir: str | Path) -> list[Path]:
    """Unify one season and write each canonical table to parquet. Idempotent.

    Returns the list of written paths.
    """
    data = load_season(root, season)
    out = Path(processed_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for table in TABLES:
        path = out / f"{table}_{season}.parquet"
        getattr(data, table).write_parquet(path)
        written.append(path)
    return written