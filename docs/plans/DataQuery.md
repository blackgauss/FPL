# Data Querying — Phase 1 (complete)

## Status

Phase 1 (2026-08-20): proved Polars can query FPL-Core-Insights data end-to-end
against `external/fpl_core/`. All the schema/cast/unification logic is captured in
tested loaders and catalog code.

**Superseded by [`DataPipeline.md`](DataPipeline.md)** per `docs/style/Data.md`: the
query API below is retired — no abstract query interfaces; the data lives as a parquet
dataset; Python only orchestrates. The driving example survives as an EDA notebook
pattern (and must still reproduce 6.645 / 11.5 / 7.2).

## Goal (phase-1 framing)

Use Polars to query the FPL-Core-Insights data (cloned at `external/fpl_core/`, schema
documented in `DataDocumentation.md`). The driving example:

> Report a player's performance in matches when a certain other player also played,
> vs their overall average.

Phase-1 built lightweight abstractions forming a clean contract over the data. That
approach is deliberately retired under the Data.md direction; only the loaders/casts
and the unified catalog semantics carry forward into the ingest job.

## MVP Definition

A minimal working notebook that runs the driving example end-to-end against
`external/fpl_core/`:

1. Load config → build catalog for a season
2. `find_player("Haaland")` → unambiguous candidate
3. `performance_with_without(...)` → mean FPL points with/without the other player,
   with match counts, vs overall

No MCP, no CLI polish, no feature breadth beyond what the example needs. Features
grow after the notebook works.

## Build Sequence (small, verifiable steps)

Each step lands with black-box tests against synthetic fixtures before the next
starts. Tests never depend on `external/` existing.

**Status: all steps complete (2026-08-20).** 28 tests pass, lint clean, notebook
executed against real data and spot-checked via an independent raw-CSV computation
(Haaland with/without Foden 2025-26: n=31/6.645, n=4/11.5, n=35/7.2).

### Step 1 — Project bootstrap (minimal)
- `pyproject.toml`: deps `polars`, `pyyaml`; dev `pytest`, `ruff`. src layout.
- **Verify**: package imports; `ruff check` clean; `pytest` runs (empty).

### Step 2 — Config loading
- `config/query.yaml`: `fpl_core_root`, `seasons.available`, `seasons.default`,
  `defaults.tournament: prem`
- `load_config(path) -> QueryConfig` (typed, pydantic)
- **Verify**: test loads a temp YAML and asserts fields; rejects bad structure.

### Step 3 — Loaders (pure, path-injected)
One function per table, `(path) -> pl.DataFrame`, applying load-time casts:
- `load_players_csv` — `player_code`/`team_code` as int, position normalized
- `load_teams_csv` — `code` as int
- `load_gw_stats_csv` — numerics cast, `now_cost` kept as decimal millions
- `load_match_stats_csv` — `player_id` int, `minutes_played` float
- `load_matches_csv` — `home_team`/`away_team` float→int (`teams.code`), `gw` int
- **Verify**: tests on tiny hand-written CSVs in `tests/fixtures/` assert dtypes
  and the float-team-code cast.

### Step 4 — Season catalog (`SeasonData`)
- Composes loaders over the directory tree; unifies per-GW folders into single
  frames with `season`, `gw`, `tournament` columns
- Frames: `players`, `teams`, `gw_stats`, `match_stats`, `matches`
- **Verify**: fixture tree (2 GW folders, a tournament-mixed match, a 0-minute
  bench row) → assert unified row counts, columns, tournament preserved.

### Step 5 — Queries
- `find_player(name, catalog) -> list[Candidate]` — returns ALL candidates
  (web_name duplicates are common: 25 in 2025-26); case-insensitive substring on
  `web_name` + exact `second_name` match
- `player_match_log(player_code, catalog, tournament) -> DataFrame`
- `performance_with_without(main_code, other_code, catalog, stat) ->
  WithWithout(mean, n) ×3 (with/without/overall)`
- Semantics: "other player played" = `minutes_played > 0`; both must appear in
  `playermatchstats` for the same `match_id`; default tournament `prem`
- **Verify**: fixture with a hand-computed with/without answer; duplicate-name
  disambiguation test; 0-minute player excluded from "played".

### Step 6 — Notebook
- `notebooks/DataQuery.ipynb`: runs the driving example (Haaland + De Bruyne,
  2025-26), prints candidates, match logs, with/without result
- **Verify**: runs clean against real `external/fpl_core/`; numbers spot-checked
  against manual inspection.

## Semantics Decisions (contract fine print)

- "Other player in the match" = `minutes_played > 0` (squads include 0-minute bench
  rows — 79 of 405 rows in 2025-26 GW38)
- Tournament filter defaults to `prem` (By-Gameweek folders mix competitions);
  opt-in parameter for cups
- Season default = latest season **with played matches** (2026-27 is pre-season only)
- Means are always reported with match counts (with/without buckets are small)
- `now_cost`: FPL-Core decimal millions (`15.5`); no unit conversion in queries

## Testing Strategy

Black-box only (per `docs/style/SWE.md`): synthetic CSV fixtures in
`tests/fixtures/`, exercising known traps (duplicate web_names, 0-minute rows,
tournament mixing, float team codes). One integration smoke test marked
`skipif` when `external/` is absent.

## Out of Scope (for now)

- MCP server (deferred — revisit after notebook works)
- shots/lineups/incidents/enrichment tables
- vaastav, cross-season features, ML, anything beyond the example query
