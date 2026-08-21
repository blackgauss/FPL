# Weekly Operations: captain, vice-captain, and free transfers

## Goal

A team persists week to week; each week we make small decisions on top of it —
who to captain, who to bench, which transfers to make given the one free
transfer per Gameweek. The journal ("Team Week 1 Selection") calls these out
explicitly, and asks that "candidate teams can change week to week with
captain, vice-captain, and transfers." This plan adds those as first-class,
**separately-optimizable** steps with clean metadata.

We can only plan a few weeks ahead: there are wildcard / free-hit
opportunities that let us reset, so we plan a short horizon, not the season.

## Abstraction rationale

The current `basket` treats a squad as a flat list of 15 `player_code`s. But a
club's weekly identity is richer: 15 players, a starting XI, a bench priority
order, a captain, a vice-captain, and (implicitly) which players we kept vs
changed since last week. Downstream reads are much clearer if this is a typed
object, not an ad-hoc long frame.

Delivery is phased:

- **Phase 1 (current)** — the abstraction + its interface: `Player`, `Squad`,
  construction from the basket frame, and validity/prefix rules with tests.
  This is the "easy to use interface" optimizers will be built against later.
- **Phase 2** — the optimizers themselves (captain, transfer), each its own
  discussion.

Two types, deliberately small:

- **`Player`** — a stable player identity: `code`, `name`, `position`,
  `club`, `cost_tenths`. Value/hashable; the unit of transfer in/out.
- **`Squad`** — the persistent team: 15 `Player`s plus weekly config
  (`starters`, `bench` priority, `captain`, `vice_captain`). Immutable;
  each weekly decision returns a *new* Squad. Carries metadata
  (`gw`, `transfers_in_this_gw`) so we can log a week's decisions.

Why typed Squad (not a dict / long frame): the optimizers operate on *it*,
validity is checked once (15 players, budget, position caps), and the Captain
and Transfer optimizers are decoupled — each takes a Squad + forecasts and
returns one decision, keeping them independently testable and readable.

## Optimizers (separate)

### 1. Captain & Vice-Captain (`weekly/captain.py`)

Inputs: `Squad`, per-player distribution (mean + optionally quantiles) for the
upcoming GW.

- EV rule: captain = the player maximizing expected points **added** by the 2x
  (i.e. max predicted mean among likely starters). Vice = second-best, ideally
  a different club so a postponed fixture doesn't cost both picks.
- Optional distribution-aware: use quantiles to avoid a bench/0-min player;
  penalty for low minutes.

Decision: `(captain_code, vice_code)`.

### 2. Weekly Transfer (`weekly/transfer.py`)

Inputs: `Squad`, candidate player pool (likely from a fresh forecast + live
filters), per-GW forecasts, allowed number of free transfers (default 1).

- Enumerate feasible `(out, in)` swaps that keep the Squad valid; cost paid at
  `4 points per transfer beyond the free allowance` (FPL rules, documented in
  `docs/rules.md`).
- Choose the swap maximizing Δ(horizon expected points) − transfer cost.
- Ties / indifference: prefer no transfer (don't spend a transfer for ~0 gain).

Decision: `transfer choice (out, in) | none`, plus resulting Squad.

### 3. Composition (`weekly/plan.py`)

`optimize_week(squad, forecasts, captain_fn, transfer_fn) -> Squad_next` applies
transfer then captain for one GW, and `plan_horizon` runs it over N upcoming
GWs using the per-GW forecast slice. This is the "week to week simulation"
driver; each step logs its metadata so the season history is auditable.

## Data flow

```
team-search basket  ──►  Squad
forecast (h2h/dist) ──►  per-player per-GW distribution
live reconcile      ──►  candidate transfer pool (current clubs / availability)
captain optimizer   ──►  (captain, vice)
transfer optimizer  ──►  (out, in) + updated Squad
plan_horizon        ──►  list[Squad snapshot] over upcoming GWs
```

Constructing a `Squad` from the basket reuses `fpl.team.enumerate` validity
rules; the optimizers should not re-encode budget/position/club logic — they
reuse the same constants from `fpl.units`.

## Tests

Phase 1 pins the interface:

- Player/Squad invariants: construction from a 15-row basket frame; exactly 15
  players, position counts == SQUAD_COUNTS, budget, ≤3/club, and each
  violation is reported by validate().
- Legal starting XI from squad_from_frame (1 GKP, ≥3 DEF, 1 FWD, 11 starters);
  captain/vice must be starters and different clubs (once their API lands).

Phase 2 adds (later): captain EV + vice different-club; transfer
cost model, validity after swap, "no transfer if indifferent".