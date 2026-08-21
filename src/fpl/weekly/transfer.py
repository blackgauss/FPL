"""Basic one-transfer-per-Gameweek policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from fpl.domain import Player, Squad


def apply_transfer(squad: Squad, out: Player, new: Player, *, gw: int) -> Squad:
    """Return a new Squad with one player replaced and transfer metadata logged."""
    players = tuple(new if p.code == out.code else p for p in squad.players)
    starters = list(squad.starters)
    bench = list(squad.bench)
    if out.code in starters:
        starters[starters.index(out.code)] = new.code
    if out.code in bench:
        bench[bench.index(out.code)] = new.code
    return Squad(
        players=players,
        gw=gw,
        starters=tuple(starters),
        bench=tuple(bench),
        captain=None if squad.captain == out.code else squad.captain,
        vice_captain=None if squad.vice_captain == out.code else squad.vice_captain,
        transfers_in=squad.transfers_in + (new.code,),
    )


def choose_transfer(
    squad: Squad,
    candidates: Sequence[Player],
    expected: Mapping[int, float],
    *,
    free_transfers: int = 1,
    transfer_penalty: float = 4.0,
    min_gain: float = 0.0,
) -> tuple[Player | None, Player | None, float]:
    """Choose one feasible position-for-position transfer by expected gain.

    The baseline considers one swap, preserves all Squad invariants, and
    charges four points when no free transfer remains. Returns ``(out, in,
    net_gain)`` or ``(None, None, 0.0)`` when no transfer clears `min_gain`.
    """
    existing = {p.code for p in squad.players}
    best: tuple[Player | None, Player | None, float] = (None, None, min_gain)
    cost = 0.0 if free_transfers > 0 else transfer_penalty
    for out in squad.players:
        for new in candidates:
            if new.code in existing or new.position != out.position:
                continue
            proposal = apply_transfer(squad, out, new, gw=squad.gw)
            if proposal.validate():
                # non-empty problem list => candidate swap is invalid; skip.
                continue
            gain = expected.get(new.code, 0.0) - expected.get(out.code, 0.0) - cost
            if gain > best[2]:
                best = (out, new, gain)
    return best if best[0] is not None else (None, None, 0.0)
