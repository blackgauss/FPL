# WebUI (PoC)

Goal: a minimal, modular web UI over existing `fpl.*` capabilities.
Constraints: tiny dependency footprint (FastAPI + uvicorn under the `web`
extra; no node/npm/build step), snappy (in-process compute + parquet/response
memoization), standard machine-readable API (FastAPI auto-publishes OpenAPI at
`/openapi.json`, docs at `/docs`) so the frontend is programmed straight off
the contract.

## Layering / separation of concerns

```
src/fpl/web/
  __init__.py
  __main__.py     # uvicorn launcher: python -m fpl.web --port 8000
  app.py          # app factory; wires routers + static; zero logic
  queries.py      # Store: the ONLY web module that reads parquet/models.
                  # Lazy + memoized (process cache + parquet forecast cache).
  api/            # one thin router per resource: params -> Store -> pydantic
    meta.py players.py forecast.py team.py transfers.py league.py research.py
  static/         # frontend: index.html, js/api.js, js/views/*.js, vendored uPlot
```
Rules: routers never touch each other or `fpl.stages`; all file access and
polars work lives in `queries.Store`; domain logic stays in `fpl.*` (the web
layer is a view, not a second brain).

## Data sources (all on-disk, refreshed by CLI/DVC outside the UI)

| source | contents |
|---|---|
| `data/raw/fpl_api/live.json` | cached bootstrap snapshot (TTL via `load_live_state`) |
| `data/raw/fpl_api/account/*`   | collected: `team_picks.parquet`, `team_history.parquet`, `league_standings.parquet`, `event_live.parquet`, `league_matches.parquet`, `gw{N}_comparison.json`, `gw{N}_plan.json`, `collection.json` |
| `data/processed/{players,features,gw_stats,matches}_{season}.parquet`, `points_lgbm.txt` | models + features (player expectations) |
| `data/webcache/fc_*.parquet` | memoized per-window distributional forecasts |
| `experiments/artifacts/*.json` | research runs + `*.metrics.json` |

## Endpoints (v1, all GET, read-only)

| path | query | returns |
|---|---|---|
| `/api/meta` | — | season, derived current GW, live snapshot age, artifact inventory |
| `/api/players` | `search, position, club, status, max_price, limit, offset` | total + player rows w/ live status, ownership, price, `pred_next` (GW+1 mean when forecast cache is warm) |
| `/api/forecast` | `player_codes (csv) \| position, gw_start, horizon<=10` | per player-gw: `pred` + `quantiles` (q1..q99 from `distributional_forecast`, i.e. t-digest residual shape scaled by learned sigma). First call cold-fits (~10-20 s), persisted to `data/webcache/`. |
| `/api/team/flags` | `gw, entry_id` | per-player live flags (`fpl.live.filters.flag_squad`) + team forecast summary from `gw{N}_comparison.json` |
| `/api/transfers/suggestions` | `gw, entry_id` | from latest `gw{N}_plan.json` / weekly planner: in/out, expected gain, penalty |
| `/api/league/standings` | `entry_id` | `league_standings.parquet` rows + collected event_live points |
| `/api/research/metrics` | `run (artifact basename)` | parsed `*.metrics.json` (+ run config) |

## Views (vanilla ES modules, uPlot 50 KB vendored)

1. **Overview** — GW banner, snapshot age, recent comparison/plan one-liners, artifact freshness
2. **Explorer** — filterable player table (facets + search box); sparkline band (q05–q95) inline; click opens drawer: means-per-GW line + quantile-band chart (uPlot), status/news
3. **My team** — flag table (injured/transferred/price moved), captain box, per-GW team CDF
4. **Transfers** — suggested in/out cards with expected gain/penalty
5. **League** — standings table, highlight own team
6. **Research** — artifact browser; metrics tables

## Run

```
uv sync --extra web
python -m fpl.web            # http://127.0.0.1:8000  (docs at /docs)
```
No UI-side network calls to the FPL API ever; stale cache beats live fetch
(see `fpl.live.live` module docstring). UI is localhost, single-user, no auth.

## Later (v2)
Job runner (thread + `/api/jobs`) to trigger `search`/`gym`/collect from the
browser; DVC-integrated freshness; ownership differentials view.
