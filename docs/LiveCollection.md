# Live Collection

The global player snapshot is collected separately from account-specific state.
Use the package collector for a classic league and one manager entry:

```bash
uv run python -m fpl collect \
  --league-id <classic-league-id> \
  --entry-id <manager-entry-id> \
  --out data/raw/fpl_api/account
```

The manager entry ID is the number in the FPL team URL. The collector writes:

- `league_standings.parquet` — league rank, totals, and manager names.
- `team_history.parquet` — per-gameweek points, ranks, and transfers from the
  manager history endpoint.
- `team_picks.parquet` — selected players, positions, captaincy, and purchase
  prices for each collected gameweek.
- `collection.json` — collection metadata and the requested identity/window.

Add `--league-picks` to collect picks for every entry returned by the league
standings endpoint. This increases API traffic substantially because picks are
one request per entry per gameweek, so use it deliberately and respect the
API's rate limits.

This is an operational input, not a default DVC stage: league membership and
manager state are mutable and account-specific. Once collected, the parquet
outputs are stable inputs for a future personal-team evaluation stage.
