# FPL Official Rules — concise reference

Condensed from the official rules (fantasy.premierleague.com/help/rules). The
intent is a machine-readable summary the data pipeline and selection stages
depend on — squad constraints, scoring, transfers, chips, deadlines.

## Squad constraints

- **15 players**: 2 GKP / 5 DEF / 5 MID / 3 FWD.
- **Budget**: start ≤ £100m total.
- **Per club**: ≤ 3 players from one Premier League team.
- **Starting XI**: pick 11 by the GW deadline; formation flexible but must have
  1 GKP, ≥3 DEF, ≥1 FWD at all times.
- **Bench priority**: auto-substitutes act at Gameweek end in bench order —
  GK is replaced by the bench GK if he played; an outfield non-starter is
  replaced by the highest-priority bench player who played and keeps formation
  legal. "Playing" = appeared on pitch OR got a yellow/red card.

## Scoring (per player per match)

| Action | Points |
|---|---|
| Plays up to 60 min | 1 |
| Plays 60+ min (excl. stoppage) | 2 |
| Goal — GK / DEF / MID / FWD | 10 / 6 / 5 / 4 |
| Assist | 3 |
| Clean sheet — GK/DEF / MID | 4 / 1 |
| Every 3 GK saves | 1 |
| 10+ CBI+tackles (DEF) | 2 |
| 12+ CBI+tackles+recoveries (MID/FWD) | 2 |
| Penalty save / miss | +5 / −2 |
| Bonus points (BPS top 3) | 1–3 |
| Every 2 goals conceded (GK/DEF) | −1 |
| Yellow / red card | −1 / −3 |
| Own goal | −2 |

Notes:
- **Clean sheet** requires playing ≥60 min and not conceding while on the pitch.
- **Defensive contribution** points do not stack (one bonus per threshold).
- **Red card** player continues to be penalised for goals conceded; the −3
  includes the yellow-card deduction if both.
- Assists: final touch before the goal (pass/touch/shot), incl. own-goal
  scoring actions; no assist for winning a corner/throw-in that leads to a
  goal; penalties/free-kicks won then scored directly give an assist to the
  fouled player. Final assist decisions made by Opta + FPL.
- **BPS**: stats-based score; 3/2/1 to the three best players; ties share the
  points (e.g. two-way tie for 1st → both 3, next gets 1).

## Transfers

- Unlimited transfers at no cost until the first deadline.
- After that: **1 free transfer per GW**; each extra transfer deducts **4
  points** (classic and H2H), applied at the start of the next GW.
- Unused free transfers roll over, capped at **5 stored**.
- Max **20 transfers in a single GW** (not applied when using Wildcard or Free
  Hit).
- **Player prices** change by transfer-market popularity, not before the season
  starts. Selling price keeps half of any rise since purchase, rounded down to
  nearest £0.1m (e.g. bought £7.5m, rose to £7.8m → sell at £7.6m).

## Chips (one chip per GW)

| Chip | Effect |
|---|---|
| Bench Boost | bench points count this GW |
| Free Hit | unlimited free transfers for one GW; squad reverts at next deadline (cannot be used in consecutive GWs) |
| Triple Captain | captain points ×3 instead of ×2 |
| Wildcard | all transfers free this GW (incl. already made) |

- Two of each chip per season; split at the GW19 deadline (first half available
  GW1→GW19, second half GW20→end; first Free Hit available after GW1).
- Wildcard/Free Hit do not consume saved transfers (they carry over).
- Free Hit and Wildcard are played on confirming transfers (can't be cancelled);
  Bench Boost/Triple Captain can be cancelled before the deadline.

## Deadlines

- All team changes take effect only if made by the GW deadline.
- Deadline = 90 minutes before kick-off of the GW's first match.
- A deadline will not move within 24 hours of the scheduled time.
- (Publication of the table here is skipped; details change weekly.)

## Leagues

- **Classic**: ranked by total points; tiebreak = fewest transfers made
  (wildcard/free-hit transfers don't count). Runs over phases (Overall GW1-38,
  plus monthly phases).
- **Head-to-head**: each GW you play one opponent; win 3pts / draw 1; tiebreak
  = most game points. Score used = GW score minus transfer-point deduction for
  that GW. H2H fixture list generated at league open then locked; odd team count
  gets an "average team" that always scores the GW average.
- Private (≤30) / public (≤5) / global leagues; the Fantasy Cup is automatic.