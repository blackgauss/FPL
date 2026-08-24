# Live Collection

The global player snapshot is collected separately from account-specific state.
Use the package collector for a classic league and one manager entry:

```bash
uv run python -m fpl collect \
  --league-id <classic-league-id> \
  --entry-id <manager-entry-id> \
  --out data/raw/fpl_api/account
```

For an H2H league, use `--league-type h2h --resolve-h2h`. This collects the
H2H fixtures and all league picks, then writes `resolved_standings.parquet`
using event-live points plus the domain captain and auto-substitution rules.

The manager entry ID is the number in the FPL team URL. The collector writes:

- `league_standings.parquet` — league rank, totals, and manager names.
- `team_history.parquet` — per-gameweek points, ranks, and transfers from the
  manager history endpoint.
- `team_picks.parquet` — selected players, positions, captaincy, and purchase
  prices for each collected gameweek.
- `event_live.parquet` — official per-player minutes and points for each
  collected gameweek.
- `collection.json` — collection metadata and the requested identity/window.

Add `--league-picks` to collect picks for every entry returned by the league
standings endpoint. This increases API traffic substantially because picks are
one request per entry per gameweek, so use it deliberately and respect the
API's rate limits.

If a new or private league does not expose standings yet, add
`--skip-league`. The manager history and picks will still be collected, and the
standings can be retried later.

This is an operational input, not a default DVC stage: league membership and
manager state are mutable and account-specific. Once collected, the parquet
outputs are stable inputs for a future personal-team evaluation stage.

Compare a collected gameweek with the model forecast:

```bash
uv run python -m fpl compare \
  --picks data/raw/fpl_api/account/team_picks.parquet \
  --history data/raw/fpl_api/account/team_history.parquet \
  --event-live data/raw/fpl_api/account/event_live.parquet \
  --season 2025-2026 --gw 1 \
  --out data/raw/fpl_api/account/gw1_comparison.json
```

The comparison reports player-level actual versus expected points and uses the
official history score for the team total, including FPL captain fallback and
automatic substitutions.

For the current season's opening gameweek, use `--official-forecast`: the
local model has no preseason source-GW0 row yet, so FPL's `ep_this` is the only
available provisional xScore until a preseason forecast snapshot is stored.
