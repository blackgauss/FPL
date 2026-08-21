# Data Pipeline — Parquet Data Layer

## Direction (from `docs/style/Data.md`)

- Polars is the query engine; **parquet is the storage format**.
- **No abstract query interfaces** (no `get_player`, no `performance_with_without`
  API). Python orchestrates; everything stays close to the files.
- Two downstream task types:
  1. **Exploratory Data Analysis** — gathering information during development.
  2. **Production Jobs** — transformations → feature stores → training data.

This plan supersedes `DataQuery.md` (Phase 1, complete). The loaders/casts and
unification semantics built there carry forward into the ingest job.

## Decisions (2026-08-20)

- **Delete the query functions** (`find_player`, `performance_with_without`,
  `Candidate`, `WithWithoutResult`). The driving example becomes a notebook pattern
  in plain Polars over parquet.
- **Park the MCP idea** — it contradicted the "no abstract query interfaces"
  principle. Revisit only if a concrete need appears.
- **Flat parquet layout** under `data/processed/`, filenames labeled by the
  **event-time dimension** (the season the events correspond to — not file mtime),
  `gw` kept as a column for intra-season ordering:
  ```
  data/processed/{table}_{season}.parquet   # e.g. gw_stats_2025-2026.parquet
  ```
  `season` is also retained as a column inside each file (self-describing).

## Target Architecture

```
Layer 1  ingest job (YAML): FPL-Core CSV tree → typed parquet dataset
         data/processed/{players,teams,gw_stats,match_stats,matches}_{season}.parquet
         idempotent full rebuild (cheap at this size; FPL-Core updates 2×/day)

Layer 2  EDA: notebooks scan_parquet + Polars expressions
         library has no query API — patterns live in notebooks

Layer 3  production jobs (future): feature store + training data
         = derived parquet tables built by config-driven, tested transformation jobs
```

The parquet schema **is** the contract between layers. Contracts are declared per
job (input/output schemas), never as a universal Python interface.

## Dataset Contract (canonical schemas)

These are the load-time-cast schemas from Phase 1, now written to parquet once at
ingest time:

| Table | Key columns |
|---|---|
| `players` | `player_code` Int64, `player_id` Int64, `first_name`, `second_name`, `web_name`, `team_code` Int64, `position` GKP/DEF/MID/FWD, `season` |
| `teams` | `code` Int64, `id` Int64, `name`, `short_name`, `strength` Float64?, `elo` Float64?, `season` |
| `gw_stats` | `player_id` Int64, `gw` Int64, `web_name`, `second_name`, `status`, `total_points`/`minutes`/... Int64, `now_cost`/`form`/`ep_next`/`ep_this`/`selected_by_percent` Float64? (`now_cost` decimal millions), `season` |
| `match_stats` | `player_id` Int64, `match_id`, `minutes_played` Float64, `goals`/`assists`/`xg`/`xa` Float64, `gw` Int64, `season` |
| `matches` | `match_id`, `gw` Int64, `kickoff_time`, `home_team`/`away_team` Int64 (`teams.code`), `home_score`/`away_score` Int64?, `tournament`, `finished` bool, `season` |

Ingest-time fixes (applied once, not per query): float team codes → Int64, verbose
positions → short codes, stringified numerics → floats, discrete-per-GW
`player_gameweek_stats` preserved as-is, per-GW folders unified with `gw` column,
postponed-match dedupe, `tournament` column preserved.

## Build Sequence (small, verifiable steps)

Each step lands with black-box tests against synthetic fixtures; tests never depend
on `external/` existing.

**Status: Steps 1-6 complete (2026-08-20).** Package rehomed to `src/fpl/data/`;
`ingest` writes the flat parquet dataset; 27-test black-box contract suite
(TestFileSet/Schema/ReferentialIntegrity/Uniqueness/SpotValues/DataQuality/
Determinism/EdaPattern). Config is `config/data.yaml`, `query.yaml` retired;
`scripts/ingest.py --config --season` builds the dataset. Notebook reads only
parquet, reproduces 6.645/11.5/7.2; `queries.py`+tests deleted. Parquet dataset
documented in `DataDocumentation.md`. Full suite 40 tests green; real dataset
built at `data/processed/`.

**Multi-season extension (2026-08-20, pre stage-2 assessment):** added a legacy
layout adapter so the dataset covers 3 seasons:
- `2025-2026`, `2026-2027`: modern `By Gameweek/GW{n}` layout
- `2024-2025`: legacy per-table layout (`matches/GW{n}/`, long-table
  `playerstats/playerstats.csv`) auto-detected by `detect_layout`

Legacy seasons keep the *identical* shared schema — the 7 modern columns
(`minutes`, `goals_scored`, `assists`, `saves`, `starts`, plus names) that the
2024-25 playerstats lacks are emitted as typed-null. Legacy `matches.csv` has no
`tournament` column (defaults to `prem`); legacy `players.position` includes
`Unknown` (mapped to `UNK`). Real ingests: 2024-25 = 27,657 gw rows; 2026-27 =
595 players + 390 scheduled fixtures (pre-season, 0 matches played).

**Difficulty = seed, not default (decision):** FPL-Core `teams.strength`/`elo`
are empty pre-season but update weekly as the season progresses; therefore no
static difficulty default is baked in. Any pre-season strength is treated as a
seed value that fresh ingests overwrite — no hardcoded ratings.

**Key finding encoded in the suite:** FPL-points stats are Premier-League only —
a cup match carries a folder-`gw` and will wrongly join a `gw_stats` row (fan-out /
phantom points) unless the EDA pattern first filters `matches.tournament == 'prem'`.

### Step 1 — Rehome into `src/fpl/data/`
- Move `src/fpl/querying/` → `src/fpl/data/`; keep `config.py` + `loaders.py` as-is.
- **Verify**: existing loader/config/contract tests pass after the import rename.

### Step 2 — Ingest job writes parquet
- `ingest.run(root, season) -> list[Path]`: compose loaders + unification (from Phase 1
  `contract.py`) and write `{table}_{season}.parquet` under `processed_dir`. Parquet
  becomes the **only** interface consumers (notebook/jobs) read from — `load_season`
  stays internal to this job.
- Include a minimal runnable entrypoint (e.g. `scripts/ingest.py --config <yaml>
  --season <s>`) so the dataset can actually be produced; verify it runs clean on
  real FPL-Core data.
- **Verify**: black-box test — synthetic season tree → read back each parquet,
  assert schema + row counts equal the in-memory unified frames; entrypoint smoke-run
  on `external/`.

### Step 3 — Config
- `config/data.yaml`: `fpl_core_root`, `processed_dir`, `seasons.available`/`default`.
  Retire `config/query.yaml`.
- **Verify**: config parity test on new keys.

### Step 4 — EDA notebook over parquet
- Rewrite `notebooks/DataQuery.ipynb` to `scan_parquet` + plain Polars:
  name resolution = filter on `players`; with/without = `match_stats` join + group/agg.
- **Verify**: executes clean; reproduces the agreed numbers
  (Haaland with/without Foden 2025-26: n=31/6.645, n=4/11.5, n=35/7.2).

### Step 5 — Delete the query module
- Remove `queries.py` and its tests (patterns now live in the notebook).
- **Verify**: full suite green; `ruff check` clean.

### Step 6 — Document the dataset
- Add a "Parquet dataset" section to `DataDocumentation.md`: table schemas, filename
  convention, refresh note, gitignore.
- **Verify**: doc matches `data/processed/` on disk.

## Out of Scope (for now)

- MCP server (parked)
- FPL API snapshotter (2026-27 GW1 pre-season data in FPL-Core already covers prices)
- Feature store / training-data jobs — this plan lands Layer 1 so those are just
  config + tests later
- vaastav history, extra enrichment tables (shots/lineups/incidents)
- public `SeasonData` catalog or any query API — it's an ingest-internal only, and
  deleted once parquet is the read path
- CLI framework — just a thin `scripts/` entrypoint until a real need appears