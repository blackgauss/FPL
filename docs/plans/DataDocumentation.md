# Week 1 Plan: Data Sources Setup + Schema Documentation

## Scope

Get the data sources set up and their schemas documented. Nothing else.

Deliverable: reproducible data acquisition (snapshot/download commands), a gitignored
`data/` directory layout, and a written schema reference so all later work (cleaning,
features, models) builds on a verified foundation.

Out of scope for this plan: cleaning, feature engineering, scoring, squad optimization,
ML training. Those follow once the data is reliably on disk and understood.

## Sources

| Source | Role | Access |
|---|---|---|
| **FPL API** | Source of truth for current season: prices, availability, teams, fixtures, expected metrics | `GET` endpoints, no auth, snapshot to disk |
| **FPL-Core-Insights** (olbauday) | Historical per-GW and per-match stats, Elo, cup/friendly coverage | Cloned at `external/fpl_core/` (gitignored, shallow) |
| **vaastav/Fantasy-Premier-League** | Deep history (2016-2025) | Deferred — do not download this week |

## Setup Steps

1. **Project bootstrap (minimal)**
   - `pyproject.toml`: `polars`, `pyyaml`, `requests` (dev: `pytest`, `ruff`)
   - `.gitignore`: `data/`, `.venv/`, caches
   - All ingestion takes filepaths from config, never hardcoded paths

2. **FPL API snapshotter**
   - Fetch `bootstrap-static` and `fixtures`, write timestamped JSON snapshots:
     ```
     data/raw/fpl_api/<iso_timestamp>/bootstrap.json
     data/raw/fpl_api/<iso_timestamp>/fixtures.json
     ```
   - One entrypoint (CLI or script): `fpl snapshot` — rerunnable, never mutates old snapshots
   - Rationale: prices/injury news change daily; frozen snapshots make any later
     analysis reproducible

3. **FPL-Core-Insights download**
   - Shallow clone into `external/fpl_core/` (gitignored): `git clone --depth 1 https://github.com/olbauday/FPL-Core-Insights.git external/fpl_core`
   - Needed per season: `players.csv`, `teams.csv`,
     `By Gameweek/GW{x}/player_gameweek_stats.csv` for 2025-2026 and 2026-2027
   - Update with `git pull` inside `external/fpl_core/` — repo auto-updates twice
     daily (07:30/17:30 UTC); pull before each work session

4. **Verify the data** (done 2026-08-20 — see Exit Criteria)
   - Load each file, print shape + columns, spot-check known players (e.g. Haaland
     `code=223094`) and cross-source ID agreement
   - This verification is the exit criterion for the plan

## Data Directory Layout

```
external/                    # gitignored — repo clones, external datasets
  fpl_core/                  # shallow clone of FPL-Core-Insights (145MB)
data/                        # gitignored
  raw/
    fpl_api/<iso_timestamp>/ # bootstrap.json, fixtures.json
  processed/                 # future: cleaned/featured parquet
```

## Schema Reference

### FPL API: `bootstrap-static`

Top-level keys (10): `events`, `game_settings`, `game_config`, `phases`, `teams`,
`element_types`, `element_stats`, `elements`, `chips`, `total_players`.

**`elements`** (players, ~110 fields). Key columns:
- Identity/links: `id` (FPL element ID, primary key, unstable across seasons),
  `code` (permanent player code, e.g. Haaland `223094`), `team` (FK -> `teams.id`),
  `team_code` (FK -> `teams.code`), `element_type` (1=GKP 2=DEF 3=MID 4=FWD)
- Availability: `status` (`a`/`d`/`i`/`s`/`u`/`n`), `news`,
  `chance_of_playing_this_round`, `chance_of_playing_next_round` (nullable 0-100)
- Pricing: `now_cost` (int, tenths: `155` = £15.5m), `cost_change_*`, `value_form`
- Performance aggregates: `total_points`, `event_points`, `points_per_game`, `form`,
  `minutes`, `starts`, `goals_scored`, `assists`, `clean_sheets`, `saves`, `bonus`,
  `bps`, `expected_goals`, `expected_assists`, `expected_goal_involvements`,
  `expected_goals_conceded` (+ `_per_90` variants), `influence`, `creativity`,
  `threat`, `ict_index`, `ep_this`, `ep_next`
- Ownership: `selected_by_percent`, `transfers_in`, `transfers_out`
- Set pieces: `penalties_order`, `direct_freekicks_order`,
  `corners_and_indirect_freekicks_order`

**`teams`** (20 clubs): `id` (1-20, volatile), `code` (stable, e.g. ARS=3, MCI=43),
`name`, `short_name`, `strength`, `strength_overall_home/away`,
`strength_attack_home/away`, `strength_defence_home/away`, `pulse_id`, `form`.

**`events`** (38 GWs): `id`, `deadline_time` (UTC ISO), `is_current`, `is_next`,
`is_previous`, `finished`, `average_entry_score`, `highest_score`,
`most_captained`, `top_element`.

**`element_types`**: position definitions incl. `squad_select`, `min_play`/`max_play`
(what the optimizer needs later): GKP 2 (play 1), DEF 5 (3-5), MID 5 (2-5), FWD 3 (1-3).

**`game_config`**: `squad_total_spend: 1000` (£100m in tenths), `squad_squadsize: 15`,
`squad_team_limit: 3`, scoring map per position.

### FPL API: `fixtures` (380 objects)

`id`, `event` (GW number, nullable), `kickoff_time` (UTC ISO), `team_h`, `team_a`
(FK -> `teams.id`), `team_h_score`/`team_a_score` (null until played),
`team_h_difficulty`/`team_a_difficulty` (1-5), `finished`, `started`, `pulse_id`.

Other useful endpoints (later): `/event/{gw}/live/`, `/element-summary/{id}/`.

### FPL-Core-Insights (verified against local clone at `external/fpl_core/`)

Clone: `git clone --depth 1 https://github.com/olbauday/FPL-Core-Insights.git external/fpl_core`
(~145MB, seasons 2024-2025 / 2025-2026 / 2026-2027, updated twice daily 07:30/17:30 UTC).

#### Layout per season (`data/{season}/`)

- **Master files** (current state): `players.csv`, `teams.csv`, `playerstats.csv`,
  `gameweek_summaries.csv`, `team_history.csv`
- **`By Gameweek/GW{x}/`**: snapshot at end of GW x. **Mixes all competitions** that
  fell in that GW (verified: 2025-26 GW38 has 10 `prem` + 1 `europa-league` match) —
  filter rows on the `tournament` column for PL-only analysis
- **`By Tournament/{name}/GW{x}/`**: same file set filtered to one competition.
  2026-27 has: Community Shield, EFL Cup, Friendlies, Premier League, Uefa Super Cup
  (Friendlies GW0 currently empty). 2025-26 also has Champions/Europa/Conference League
- **`supplemental/`** (2025-26): `incidents_quarantined.csv`

GW folders accumulate extra tables through the season: early GWs have 7 core files;
later GWs (2025-26 GW38) have 15 — the core set plus `lineups`, `shots`, `incidents`,
`momentum`, `average_positions`, `match_enrichment`, `player_match_enrichment`,
`xg_by_minute` (details below).

#### Core tables

**`players.csv`** (master + per-GW; 7 cols): `player_code`, `player_id`, `first_name`,
`second_name`, `web_name`, `team_code`, `position`.
- 595 rows pre-season 2026-27; ~841 rows for completed 2025-26 (in-season additions)
- `player_id` == FPL API `elements.id` **for that season** (verified: Haaland 411 in
  26-27, 430 in 25-26); `player_code` == FPL API `elements.code` (permanent, 223094)

**`teams.csv`** (21 rows): `code` (stable, = FPL `teams.code`), `id` (season-local),
`name`, `short_name`, `strength`, `strength_overall_home/away`,
`strength_attack_home/away`, `strength_defence_home/away`, `pulse_id`, `elo` (ClubElo),
`fotmob_name`.
- Pre-season gaps (verified 2026-27): `strength` and `elo` empty;
  `strength_attack_*`/`strength_defence_*` all 0. Use FPL API bootstrap for difficulty
  ratings at season start

**`playerstats.csv`** (master + per-GW): FPL API bootstrap-`elements` equivalent,
**cumulative** at snapshot time (~80 cols). Master = current; GW folder = up to GW x.

**`player_gameweek_stats.csv`** (per-GW only) — the key training table. Same column
set as `playerstats` but **discrete per GW**. Verified: Haaland 2025-26 GW1 = 13pts /
72min / 2goals, GW2 = 2pts / 90min / 0goals (not cumulative).
- Per-GW calculated: `total_points`, `minutes`, `goals_scored`, `assists`, `bonus`,
  `bps`, `saves`, `starts`, … (previous GW total subtracted)
- Deadline snapshot: `now_cost`, `selected_by_percent`, `form`, `ep_next`, `ep_this`,
  `status`, `news`, `chance_of_playing_*`, `transfers_*`
- One row per player per GW (all ~841/595 players, including non-players — 0 minutes)
- **`now_cost` is decimal millions here** (`15.5`), unlike the API's tenths (`155`)
- 2026-27 GW1 file already carries the live pre-season snapshot (595 rows: prices,
  status, `ep_next` 4.0 for Haaland) — usable for GW1 selection today

**`matches.csv` + `fixtures.csv`** (per-GW, per-tournament; 103 cols, same schema).
`matches.csv` in GW folders **includes unplayed fixtures** (2026-27 GW1: 10 rows, all
`finished=False`, scores empty) — check `finished` before treating as results.
- Links: `home_team`/`away_team` -> `teams.code` (**NOT `teams.id`** — the upstream
  README says `id`, but verified against both seasons it is `code`), stored as floats
  (`3.0`) — cast needed
- `match_id` is a slug string (`25-26-prem-manchester-city-vs-aston-villa`), not an int
- `tournament` values seen: `prem`, `europa-league` (filter `== 'prem'` for PL)
- Other cols: `gameweek`, `kickoff_time`, `home/away_team_elo`, scores, `finished`,
  `fotmob_id`, `stats_processed`, `player_stats_processed`, plus ~60 team stats
  (xG/possession/shots/duels/tackles, split home/away)

**`playermatchstats.csv`** (per-GW, per-tournament; 63 cols): one row per player per
match, **all competitions in that folder** (cup minutes appear here but not in
`player_gameweek_stats`, which is FPL-API/PL-scoped). `player_id`, `match_id`,
`minutes_played`, `start_min`, `finish_min`, `goals`, `assists`, `penalties_scored/
missed`, `xg`, `xa`, `xgot`, `shots_on_target`, `big_chances_missed`,
`touches_opposition_box`, `chances_created`, duels/tackles/interceptions/recoveries/
blocks/clearances, GK metrics (`saves`, `goals_prevented`, `xgot_faced`,
`sweeper_actions`, `high_claim`, `saves_inside_box`), plus physical metrics
(`top_speed`, `distance_covered`, `walking/running/sprinting_distance`,
`number_of_sprints`) and `defensive_contributions`.

**`team_history.csv`** (master): `player_id`, `gw`, `team_code` — player-to-club
mapping per GW (~841×38 rows for 2025-26). Catches mid-season transfers.

**`gameweek_summaries.csv`** (master): FPL API `events` equivalent + `snapshot_time`.

#### Extra per-GW tables (2025-26 full season; accumulate during 2026-27)

| Table | Grain | Key columns |
|---|---|---|
| `lineups.csv` | player×match | `match_id`, `team_side`, `team_code`, `player_id`, `is_starting`, `formation`, `lineup_status` |
| `shots.csv` | shot | `match_id`, `minute`, `player_id`, `outcome`, `situation`, `body_part`, `xg`, `xgot`, `start_x/start_y`, goal-mouth coords |
| `incidents.csv` | event | `match_id`, `incident_type`, `minute`, `player_id`, `secondary/assist_player_id`, `card_type`, `goal_type`, scores |
| `momentum.csv` | minute×match | `match_id`, `minute`, `value` |
| `average_positions.csv` | player×match | `match_id`, `player_id`, `position`, `x`, `y` |
| `match_enrichment.csv` | match | `match_id`, `travel_distance_km`, `weather_description`, `temperature_c`, `wind_speed`, `is_local_derby`, `is_neutral_ground`, shot-model xG |
| `player_match_enrichment.csv` | player×match | `player_id`, `match_id`, `rating`, `possession_lost`, pass/dribble/duel aggregates |
| `xg_by_minute.csv` | minute×match | `match_id`, `minute`, home/away xG + cumulative |

### ID Linkage (the critical bit)

| ID | Stable across seasons? | Use |
|---|---|---|
| `elements.code` / FPL-Core `player_code` | **Yes** | Cross-season player joins |
| `elements.id` / FPL-Core `player_id` / `playerstats.id` | No (within-season ok) | Intra-season joins (verified: FPL-Core `player_id` == API `elements.id` per season) |
| `teams.code` / FPL-Core `team_code` / `matches.home_team`+`away_team` | **Yes** | Cross-season team joins (matches refs verified = `teams.code`, not `id`) |
| `teams.id` / FPL API `fixtures.team_h/a` | No (reordered yearly) | Intra-season only |

Rule: join via `code` whenever data persists across seasons; `id` only within one.
`match_id` (FPL-Core) is a per-season slug — fine as an intra-season join key only.

### Known Pitfalls

- **`now_cost` units differ by source**: FPL API uses tenths (`155` = £15.5m);
  FPL-Core CSVs use decimal millions (`15.5`). Normalize explicitly, convert at
  display only.
- **Stringified numerics** in bootstrap: `form`, `influence`, `creativity`, `threat`,
  `ict_index`, all `expected_*`, `selected_by_percent`, `ep_*` are strings — cast.
- **`team` vs `team_code`**: joining on the volatile id across seasons silently
  corrupts; always check which is meant.
- **FPL-Core `matches.home_team`/`away_team` are `teams.code` stored as floats**
  (`3.0`) — the upstream README wrongly documents them as `teams.id`.
- **By Gameweek folders mix competitions** — filter `tournament == 'prem'` for
  PL-only analysis.
- **`matches.csv` includes unplayed fixtures** — gate on `finished` for results.
- **Cumulative vs discrete**: FPL-Core `playerstats.csv` is cumulative; only
  `player_gameweek_stats.csv` is per-GW (discreteness verified). Using the wrong one
  double-counts.
- **`player_gameweek_stats` is PL-scoped; `playermatchstats` covers all comps** in
  the folder — cup minutes appear only in the latter.
- **Pre-season gaps in FPL-Core `teams.csv`**: `strength`/`elo` empty,
  `strength_attack_*`/`strength_defence_*` zeroed — use the API bootstrap for early
  difficulty ratings.
- **vaastav `xP` lookahead** (when we get to it): scraped post-match; must `shift(1)`
  or drop.
- **FPL API rate limits**: no auth needed for bootstrap/fixtures, but be polite —
  snapshot once per session, cache to disk.
- **New-player nulls**: `ep_this` null before GW1, `chance_of_playing_*` can be null
  even when `status` is set — handle both.

## Exit Criteria

- [ ] `fpl snapshot` writes timestamped bootstrap+fixtures JSON; rerun creates new snapshot
- [x] FPL-Core 2025-2026 + 2026-2027 CSVs on disk under `external/fpl_core/`
- [x] Verification pass: all key files load, headers/row-counts checked, Haaland
      spot-checks pass (`player_code=223094`; `player_id` 411 in 26-27, 430 in 25-26)
- [ ] Cross-source ID check vs live bootstrap snapshot (needs the snapshotter first)
- [x] This schema doc matches what's on disk (verified 2026-08-20; doc corrected where
      the upstream README was wrong — team linkage in `matches.csv`, `now_cost` units,
      tournament mixing in `By Gameweek`)

---

## Parquet Dataset (the ingest contract)

Built by `python -m fpl.stages.ingest --config config/data.yaml [--season <s>]`
(idempotent full rebuild). This is the **only** interface EDA notebooks and
production jobs read from — per `docs/style/Data.md`, no query API, close to files.

### Layout

```
data/processed/{table}_{season}.parquet     # e.g. gw_stats_2025-2026.parquet
```

Flat files, filenames labeled by the **event-time dimension** (the season the
events correspond to — not file mtime). `season` is also a column inside every
file (self-describing). Tables: `players`, `teams`, `gw_stats`, `match_stats`,
`matches`.

### Schemas (canonical, enforced by the contract test suite `test_ingest_contract.py`)

| Table | Columns |
|---|---|
| `players` | `player_code` Int64, `player_id` Int64, `first_name`, `second_name`, `web_name`, `team_code` Int64, `position` GKP/DEF/MID/FWD, `season` |
| `teams` | `code` Int64, `id` Int64, `name`, `short_name`, `strength` Float64?, `elo` Float64?, `season` |
| `gw_stats` | `player_id` Int64, `gw` Int64, `web_name`, `second_name`, `status`, `total_points`/`minutes`/`goals_scored`/`assists`/`bonus`/`bps`/`saves`/`starts` Int64, `now_cost`/`form`/`ep_next`/`ep_this`/`selected_by_percent` Float64? (`now_cost` decimal millions), `season` |
| `match_stats` | `player_id` Int64, `match_id`, `minutes_played` Float64, `goals`/`assists`/`xg`/`xa` Float64, `gw` Int64, `season` |
| `matches` | `match_id`, `gw` Int64, `kickoff_time`, `home_team`/`away_team` Int64 (`teams.code`), `home_score`/`away_score` Int64?, `tournament`, `finished` bool, `season` |

### Refresh

FPL-Core auto-updates twice daily (07:30/17:30 UTC). Rebuild with the same command —
full overwrite, idempotent. `data/` is gitignored.

### Invariants the suite enforces (any conforming implementation must pass)

- Exactly the 5 files, nothing extra
- Exact schema per table (columns + dtypes)
- Referential integrity: `gw_stats`/`match_stats` player_ids ⊆ `players`;
  `match_stats` match_ids ⊆ `matches`; player team codes ⊆ `teams.code`
- Uniqueness: `(season, player_id, gw)` / `(season, player_id, match_id)` /
  `(season, match_id)` / player_code / team code
- Spot values pin the load-time casts and discrete-per-GW semantics
- `minutes_played >= 0`, zero-minute squad rows preserved, `now_cost > 0`
- Ingest twice → identical content (determinism)
- **EDA gotcha (encoded in the suite):** FPL-points stats are Premier-League only.
  A cup match carries a folder-`gw`, so joining `gw_stats` without first filtering
  `matches.tournament == 'prem'` attaches phantom points to cup games.
