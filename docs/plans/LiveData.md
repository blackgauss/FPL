# Live Data: FPL API access + on-the-fly player filters

## Goal (journal "Fresh Data")

Team selection must use up-to-date prices/injuries/availability or it can't
work in real life. The FPL API (or another source) provides the live state;
we fetch and cache it, and expose it as a **separate filter layer** applied to
datasets on the fly — never baked into the feature store.

## Design

`src/fpl/live/` (rates-limit-first):

- `live.py` — one `bootstrap-static` fetch per refresh carries every player's
  price (tenths: 155 = £15.5m), `status` (a/d/i/s/u), `news`, chance-of-playing,
  team, `removed`. Disk-cached with a TTL (`max_age_seconds`) so queries never
  hammer the API; a fresh snapshot is served from cache; on network failure the
  last cache is returned (stale-but-present) instead of crashing. Only raises
  `LiveFetchError` when there is no cache at all.
- `filters.py` — composable masks over the live frame: `available`,
  `not_injured_suspended`, `chance_of_playing`, `no_news`, `in_league`,
  `not_transferred`, `price_unchanged`, and a `suggest` composite.
  `filter_frame_by_code(frame, live, mask)` applies a mask to any player-keyed
  dataset on the fly (unknown players kept).
- `agreement.py` — data-hygiene checks: does live agree with the dataset?
  reports per-player price/team/status diffs + a `hygiene_summary`.

## Critical correctness: price units

Live `now_cost` is in **tenths** (155). Our local dataset stores **decimal
millions** (15.5). Comparing them without normalization makes every player look
like a price move — a real bug caught when live_apply showed `price_moved: 460`
(= all matched players). Fix: `to_tenths(price, scale)` / `price_scale`
(param, default 10 for decimal-millions datasets, 1 for already-tenths); the
diff is computed on normalized tenths. Unit-mismatch and true-move paths are
both pinned by tests.

## Wiring

`scripts/live_apply.py`: fetch live → build the search pool (existing
pipeline, unchanged) → drop excluded players via `suggest` → print a
`hygiene_summary`. Real 2025-26 GW31-32 run: pool 90 → 84 after live filters;
hygiene shows genuine `price_moved=403`, `team_transferred=45`,
`not_available=92` (dataset is a 2025-26 snapshot vs current live — the drift
is real, which is the point of this layer).

### Reconcile BEFORE construction (current.py)

Team construction must never see a stale world, so live state is reconciled
into the *input* to the pipeline (`construction_input(scored, live, mask)`):
- **transfers**: each player's `team_code` is overwritten with the live club,
  so `max_per_club` and enumeration use the current club;
- **missing players**: players absent from the live roster (transferred out of
  FPL, de-listed) are dropped;
- **availability**: injured/suspended/unavailable players excluded via
  `suggest()`.

Real 2025-26 GW2-4 reconcile: scored 753 → 366 (387 removed — missing from
live / unavailable), pool rebuilt on current clubs. This is the "updates the
team for transferred players before construction" guarantee; `current.py` has
dedicated tests for transfer-update, missing-drop, and mask-exclusion.

`scripts/live_check_team.py`: reproduces the candidate team basket and flags
**per-squad live problems** — the "my candidate team had missing/injured
players" case. Uses the shared `flag_squad_player` helper, which detects
injured/suspended/unavailable status, absence from the live roster
(missing/transferred out), a club transfer, and price moves. Real 2025-26
GW2-4 basket: the best squad shows Bowen NOT IN LIVE ROSTER, Livramento
INJURED, Lacroix + Bruno G. TRANSFERRED, and price moves on most others.

## Tests

- `tests/live/` (26) — fetch/cache/TTL/fallback with a mocked session; every
  filter's semantics; `flag_squad_player`; hygiene agreement incl. the
  no-scale unit trap.
- `tests/integration/` (8) — black-box **stage-interface** tests: run the real
  pipeline (features→assemble→score→filter→enumerate→reconcile→hygiene) on a
  dense synthetic season and assert each interface's output contract:
  - assemble(require_target=False) predicts a pre-season window (was 0 rows);
  - enumerate yields only complete 15-man, in-budget, 4-position squads;
  - greedy raises informatively on an infeasible pool;
  - live reconcile updates transferred clubs + drops unavailable BEFORE
    filter_pool;
  - price units agree under price_scale=10.
  These caught four real integration bugs during this work (null-target path,
  units mismatch, empty-squad overflow, silent infeasible-pool failure).

## Follow-ons

- element-summary per player for deeper injury text/timelines (rate-limit
  aware: cache, few calls)
- price/transfer diffs surfaced inside the team-results report, not just
  summary counts