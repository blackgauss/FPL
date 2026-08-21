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

## Tests (21 in `tests/live/`)

- fetch/cache/TTL/fallback with a mocked session (no network, no rate-limit
  pressure)
- every filter's semantics on a synthetic payload
- agreement vs an agreeing dataset, true price/team moves, decimal->tenths
  conversion, and **the no-scale unit trap** (pinned as a test)

## Follow-ons

- element-summary per player for deeper injury text/timelines (rate-limit
  aware: cache, few calls)
- price/transfer diffs surfaced inside the team-results report, not just
  summary counts